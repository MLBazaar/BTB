import importlib

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
        install = git_repostiroy.get('install')

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


def run_on_kubernetes(config):
    """Run benchmarking on a kubernetes cluster with the given configuration.

    Talks to kubernetes to create `n` amount of new `pods` with a dask worker inside of each
    forming a `dask` cluster. Then, a function specified from `config` is being imported and
    run with the given arguments. The tasks created by this `function` are being run on the
    `dask` cluster for distributed computation.

    The config dict must contain the following sections:
        * run
        * dask
        * install

    Within the `run` section you need to specify:
        * function:
            The complete python path to the function to be run.
        * args:
            A dictionary containing the keyword args that will be used with the given function.

    Within the `install` section you can specify:
        * install:
            A dictionary containing the following keys:
                * repository:
                    The link to the repository that has to be used by the workers.
                * checkout:
                    The branch or commit to be used.
                * install_commands:
                    The command used to install the repository.

    Within the `dask` section you can specify:
        * image:
            The docker image that you would like to use.
        * workers:
            The amount of workers to use.
        * resources:
            A dictionary containig the following keys:
                * limits:
                    A dictionary containing the following keys:
                        * memory:
                            The amount of RAM memory.
                        * cpu:
                            The amount of cpu's to use.

    This is an example of this dictionary in Yaml format::

        run :
            function: btb_benchmark.main.run_benchmark
            args:
                iterations: 10
                sample: 4
        install:
            repository: https://github.com/HDI-Project/BTB
            checkout: stable
            install_commands: make install-develop

    Args:
        config (dict):
            Config dictionary.
    """
    dask_cluster = config['dask_cluster']
    cluster_spec = generate_cluster_spec(dask_cluster)
    cluster = KubeCluster.from_dict(cluster_spec)
    cluster.scale(dask_cluster['workers'])
    client = Client(cluster)

    try:
        run = import_function(config['run'])
        kwargs = config['run']['args']
        results = run(**kwargs)

    finally:
        client.close()
        cluster.close()

    return results
