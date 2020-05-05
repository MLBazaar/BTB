# -*- coding: utf-8 -*-
import argparse
import importlib
import json
import logging
import os
import sys
from io import StringIO

import boto3
import tabulate
from dask.distributed import Client
from dask_kubernetes import KubeCluster

RUN_TEMPLATE = """
/bin/bash <<'EOF'

{}

/usr/bin/prepare.sh dask-worker --no-dashboard --memory-limit 0 --death-timeout 0

EOF
"""


def import_function(config):
    function = config['function']
    function = function.split('.')
    function_name = function[-1]
    package = '.'.join(function[:-1])
    module = importlib.import_module(package)

    return getattr(module, function_name)


def get_extra_setup(setup_dict):
    extra_packages = []

    script = setup_dict.get('script')
    if script:
        extra_packages.append('exec {}'.format(script))

    apt_packages = setup_dict.get('apt_packages')
    if apt_packages:
        extra_packages.append('apt get install {}'.format(' '.join(apt_packages)))

    pip_packages = setup_dict.get('pip_packages')
    if pip_packages:
        extra_packages.append('pip install {}'.format(' '.join(pip_packages)))

    git_repository = setup_dict.get('git_repository')
    if git_repository:
        url = git_repository.get('url')
        reference = git_repository.get('reference', 'master')
        install = git_repository.get('install')

        git_clone = 'git clone {} repo && cd repo'.format(url)
        git_checkout = 'git checkout {}'.format(reference)
        extra_packages.append('\n '.join([git_clone, git_checkout, install]))

    if len(extra_packages) > 1:
        return '\n '.join(extra_packages)

    return extra_packages[0]


def generate_cluster_spec(dask_cluster):
    extra_setup = ''

    worker_config = dask_cluster.get('worker_config')
    if worker_config.get('setup'):
        extra_setup = get_extra_setup(worker_config['setup'])

    run_commands = RUN_TEMPLATE.format(extra_setup)

    spec = {
        'metadata': {},
        'spec': {
            'containers': [{
                'args': ['-c', run_commands],
                'command': ['tini', '-g', '--', '/bin/sh'],
                'image': worker_config.get('image', 'daskdev/dask:latest'),
                'name': 'dask-worker',
                'resources': worker_config.get('resources', {})
            }]
        }
    }

    return spec


def df_to_csv_str(df):
    with StringIO() as sio:
        df.to_csv(sio)
        return sio.getvalue()


def upload_to_s3(bucket, output_path, results, aws_key=None, aws_secret=None):
    client = boto3.client('s3', aws_access_key_id=aws_key, aws_secret_access_key=aws_secret)
    client.put_object(Bucket=bucket, Key=output_path, Body=df_to_csv_str(results))


def run_on_kubernetes(config):
    """Run benchmarking on a kubernetes cluster with the given configuration.

    Talks to kubernetes to create `n` amount of new `pods` with a dask worker inside of each
    forming a `dask` cluster. Then, a function specified from `config` is being imported and
    run with the given arguments. The tasks created by this `function` are being run on the
    `dask` cluster for distributed computation.

    The config dict must contain the following sections:
        * run
        * dask_cluster
        * output

    Within the `run` section you need to specify:
        * function:
            The complete python path to the function to be run.
        * args:
            A dictionary containing the keyword args that will be used with the given function.

    Within the `dask_cluster` section you can specify:
        * workers:
            The amount of workers to use. If `int`, that amount of workers will be created. If
            a python dict with `minimum` and `maximum` keywords or `workers` is not provided,
            an adaptive cluster will be used.
        * worker_config:
            A dictionary with the following keys:
            * resources:
                A dictionary containig the following keys:
                * memory:
                    The amount of RAM memory.
                * cpu:
                    The amount of cpu's to use.
            * image:
                A docker image to be used (optional).
            * setup:
                A dictionary containing the following keys:
                    * script:
                        Location to bash script from the docker container to be run.
                    * git_repository:
                        A dictionary containing the following keys:
                            * url:
                                Link to the github repository to be cloned.
                            * reference:
                                A reference to the branch or commit to checkout at.
                            * install:
                                Command to install the repository.
                    * pip_packages:
                        A list of pip packages to be installed.
                    * apt_packages:
                        A list of apt packages to be installed.

    Within the `output` section you can specify:
        * path:
            The path to a local file or s3 were the file will be saved.
        * bucket:
            If given, the path specified previously will be saved as `s3://bucket/path`
        * key:
            AWS authentication key to access the bucket.
        * secret_key:
            AWS secrect authentication key to access the bucket.

    This is an example of this dictionary in Yaml format::

        run :
            function: btb_benchmark.main.run_benchmark
            args:
                iterations: 10
                sample: 4
        dask_cluster:
            workers: 1
            worker_config:
                resources:
                    memory: 2G
                    cpu: 1
                image: mlbazaar/btb_benchmark:latest

    Args:
        config (dict):
            Config dictionary.
    """
    output_conf = config.get('output')
    if output_conf:
        output_path = output_conf.get('output_path')
        if not output_path:
            raise ValueError('An output path must be provided when providing `output`.')

    dask_cluster = config['dask_cluster']
    cluster_spec = generate_cluster_spec(dask_cluster)
    cluster = KubeCluster.from_dict(cluster_spec)

    workers = dask_cluster.get('workers')

    if not workers:
        cluster.adapt()
    elif isinstance(workers, int):
        cluster.scale(workers)
    else:
        cluster.adapt(**workers)

    client = Client(cluster)
    client.get_versions(check=True)

    try:
        run = import_function(config['run'])
        kwargs = config['run']['args']
        results = run(**kwargs)

    finally:
        client.close()
        cluster.close()

    if output_conf:
        bucket = output_conf.get('bucket')

        try:
            if bucket:
                aws_key = output_conf.get('key')
                aws_secret = output_conf.get('secret_key')
                upload_to_s3(bucket, output_path, results, aws_key, aws_secret)
            else:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                results.to_csv(output_path)

        except Exception:
            print('Error storing results. Falling back to console dump.')
            print(df_to_csv_str(results))

    else:
        return results


def _get_parser():
    parser = argparse.ArgumentParser(description='Run on Kubernetes Command Line Interface')

    parser.add_argument('config', help='Path to the JSON config file.')
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help='Be verbose. Use -vv for increased verbosity.')

    return parser


def main():
    # Parse args
    parser = _get_parser()
    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    # Logger setup
    log_level = (3 - args.verbose) * 10
    fmt = '%(asctime)s - %(process)d - %(levelname)s - %(name)s - %(module)s - %(message)s'
    logging.basicConfig(level=log_level, format=fmt)

    with open(args.config) as config_file:
        config = json.load(config_file)

    results = run_on_kubernetes(config)

    if results is not None:
        print(tabulate.tabulate(
            results,
            tablefmt='github',
            headers=results.columns
        ))


if __name__ == '__main__':
    main()
