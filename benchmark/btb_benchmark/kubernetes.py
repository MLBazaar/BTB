import importlib

from dask.distributed import Client
from dask_kubernetes import KubeCluster

RUN_TEMPLATE = """
/bin/bash <<'EOF'

{extra_commands}

/usr/bin/prepare.sh dask-worker --nthreads 2 --no-dashboard --memory-limit 6GB --death-timeout 60

EOF
"""


def import_function(config):
    function = config['function']
    function = function.split('.')
    function_name = function[-1]
    package = '.'.join(function[:-1])
    module = importlib.import_module(package)

    return getattr(module, function_name)


def generate_cluster_spec(install_config, dask_config):
    repository = install_config.get('repository')

    if repository:
        repository = 'git clone {} repo && cd repo'.format(repository)
        reference = install_config.get('reference', 'master')
        reference = 'git checkout {}'.format(reference)
        install_commands = install_config.get('install_commands', '')

        extra_commands = '\n'.join([repository, reference, install_commands])

    run_commands = RUN_TEMPLATE.format(
        extra_commands=(extra_commands or ''),
    )

    spec = {
        'metadata': {},
        'spec': {
            'containers': [{
                'args': ['-c', run_commands],
                'command': ['tini', '-g', '--', '/bin/sh'],
                'image': dask_config.get('image', 'daskdev/dask:latest'),
                'name': 'dask-worker',
                'resources': dask_config['resources']
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
    install_config = config['install']
    dask_config = config['dask']
    cluster_spec = generate_cluster_spec(install_config, dask_config)
    cluster = KubeCluster.from_dict(cluster_spec)
    cluster.scale(dask_config['workers'])
    client = Client(cluster)

    run = import_function(config['run'])
    kwargs = config['run']['args']
    results = run(**kwargs)

    client.close()
    cluster.close()

    return results
