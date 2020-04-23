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

    Within the `dask` section you can specify:
        * workers:
            The amount of workers to be run.
        * worker_spec:
            A dictionary containing the hardware specifications to be used.
        * install:
            A dictionary containing the following keys:
                * repository:
                    The link to the repository that has to be used by the workers.
                * checkout:
                    The branch or commit to be used.

    This is an example of this dictionary in Yaml format::

        run :
            function: btb_benchmark.main.run_benchmark
            args:
                iterations: 10
                sample: 4
        dask:
            workers: 4
            workers_spec:
                memory_limit: 4G
                cpu_limit: 2
        install:
            repository: https://github.com/HDI-Project/BTB
            checkout: stable

    Args:
        config (dict):
            Config dictionary.
    """
