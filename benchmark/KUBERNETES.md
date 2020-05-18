# Kubernetes

Running the complete BTB Benchmarking suite can take a long time.

For this reason, it comes prepared to be executed distributedly over a dask cluster created using Kubernetes.

## Table of Contents

* [Requirements](#Requirements)
    * [Kubernetes Cluster](#Kubernetes-Cluster)
    * [Admin Acces](#Admin-Access)
    * [Dask Function](#Dask-Function)
* [Benchmark Configuration](#Benchmark-Configuration)
    * [Configuration Format](#Configuration-Format)
    * [Configuration Examples](#Configuration-Examples)
* [Run a function on Kubernetes](#Run-a-function-on-Kubernetes)
    * [Usage Example](#Usage-Example)
* [Results](#Results)

## Requirements

### Kubernetes Cluster

The current version of the BTB benchmark is only prepared to run on kubernetes clusters for which direct access is enabled from the system that is triggering the commands, such as self-hosted clusters or AWS EKS clusters created using `eksctl`.

You can easily make sure of this by running the following command:

```bash
kubectl --version
```

If the output does not show any errors, you should be good to go!

### Admin Access

For the current version, you need to execute the BTB benchmark from a POD inside the cluster within a namespace for which admin access is granted.

If you are running your benchmark POD inside the default workspace, you can create the necessary roles using the following yml config:

```yaml=
kind: ClusterRole
apiVersion: rbac.authorization.k8s.io/v1
metadata:
  name: dask-admin
rules:
- apiGroups:
    - ""
  resources:
    - pods
    - services
  verbs:
    - list
    - create
    - delete
---
kind: ClusterRoleBinding
apiVersion: rbac.authorization.k8s.io/v1
metadata:
  name: dask-admin
subjects:
- kind: ServiceAccount
  name: default
  namespace: default
roleRef:
  kind: ClusterRole
  name: dask-admin
  apiGroup: rbac.authorization.k8s.io
```

NOTE: A yml file named `dask-admin.yml` with this configuration can be found inside the `benchmark/kubernetes` folder.

Once you have downloaded or created this file, run `kubectl apply -f dask-admin.yml`.

### Dask Function

The Kubernetes framework allows running any function on a distributed cluster, as far as it uses Dask to distribute its tasks and its output is a `pandas.DataFrame`.

In particular, the `run_benchmark` function from the BTB Benchmarking framework already does it, so all you need to do is execute it.

## Benchmark Configuration

In order to run a dask function on Kubernetes you will need to create a
config dictionary to indicate how to setup the cluster and what to run in it.

### Configuration Format

The config dict that needs to be provided to the `run_dask_function` function has the following entries:

* `run`: specification of what function needs to be run and its arguments.
* `dask_cluster`: configuration to use when creating the dask cluster.
* `output`: where to store the output from the executed function.

#### run

Within the `run` section you need to specify:

* `function`: The complete python path to the function to be run. When runing `BTB`, this value must be set to `btb_benchmark.main.run_benchmark`.
* `args`: A dictionary containing the keyword args that will be used with the given function.

#### dask_cluster

Within the `dask_cluster` section you can specify:
* `workers`: The amount of workers to use. This can be specified as a fixed integer value or as a subdictionary specifying a range, so dask-kubernetes can adapt the cluster size to the work load:
    * `minimum`: minumum number of dask workers to create.
    * `maximum`: maximum number of dask workers to create.
* `worker_config`: specification about how to setup each worker

##### worker_config

* `resources`: A dictionary containig the following keys:
    * `memory`: The amount of RAM memory.
    * `cpu`: The amount of cpu's to use.
* `image`: A docker image to be used. If not specified, you must specify the `git_repository`.
* `setup`: (Optional) spectification of any additional things to install or run to initialize the container before starting the dask-cluster.

###### setup

* `script`: Location to bash script from the docker container to be run.
* `git_repository`: A dictionary containing the following keys:
    * `url`: Link to the github repository to be cloned.
    * `reference`: A reference to the branch or commit to checkout at.
    * `install`: command run to install this repository.
* `pip_packages`: A list of pip packages to be installed.
* `apt_packages`: A list of apt packages to be installed.

#### output

* `path`: The path to a local file or s3 were the file will be saved.
* `bucket`: If given, the path specified previously will be saved as s3://bucket/path
* `key`: AWS authentication key to access the bucket.
* `secret_key`: AWS secrect authentication key to access the bucket.

### Configuration Examples

Here is an example of such a config dictionary that uses a custom image:

```python
config = {
    'run': {
        'function': 'btb_benchmark.main.run_benchmark',
        'args': {
            'iterations': 10,
            'challenge_types': 'xgboost',
            'sample': 10,
        }
    },
    'dask_cluster': {
        'workers': 8,
        'worker_config': {
            'resources': {
                'memory': '4G',
                'cpu': 4
            },
            'image': 'mlbazaar/btb:latest',
        },
    },
    'output': {
        'path': 'results/my_results.csv',
        'bucket': 'my-s3-bucket',
        'key': 'myawskey',
        'secret_key': 'myawssecretkey'
    }
}
```

And this one is using a github repository and some additional pip packages:

```python
config = {
    'run': {
        'function': 'btb_benchmark.main.run_benchmark',
        'args': {
            'iterations': 10,
            'challenge_types': 'xgboost',
            'sample': 10,
        }
    },
    'dask_cluster': {
        'workers': 8,
        'worker_config': {
            'resources': {
                'memory': '4G',
                'cpu': 4
            },
            'setup': {
                'git_repository': {
                    'url': 'https://github.com/HDI-Project/BTB',
                    'reference': 'kubernetes',
                    'install': 'make install-develop'
                },
                'pip_packages': ['mlblocks', 'sdv']
            },
        },
    },
}
```

## Run a function on Kubernetes.

Create a pod, using the local kubernetes configuration, that starts a Dask Cluster using dask-kubernetes and runs a function specified within the `config` dictionary. Then, this pod talks to kubernetes to create `n` amount of new `pods` with a dask worker inside of each forming a `dask` cluster. Then, a function specified from `config` is being imported and run with the given arguments. The tasks created by this `function` are being run on the `dask` cluster for distributed computation.

Arguments:

* `config`: config dictionary.
* `namespace`: namespace where the dask cluster will be created.

### Usage Example

In this usage example we will create a config dictionary that will run the `btb_benchmark.main.run_benchmark` function. For our `dask_cluster` we will be requesting 2 workers and giving them 4 cores / cpu's to each one to work with and the docker image `mlbazaar/btb:latest`. Then we will call `run_on_benchmark` to create the pods and we will see the logs of the pod that created the workers.

1. First write your config dict following the [instructions above](#benchmark-configuration).
2. Once you have your *config* dict you can import the `run_on_kubernetes` function to create the first pod.

```python
from btb_benchmark.kubernetes import run_on_kubernetes


run_on_kubernetes(config)
```

3. If everything proceeded as expected, a message `Pod created` should be displayed on your end. Then, you can check the pod's state or logs by runing the following commands in your console:

```bash
kubectl get pods
```

*Note*: bear in mind that this pod will create more pods, with the config that we provided there should be a total of 3 pods (the one that launched the task and the two workers that we specified).

4. Once you have the name of the pod (it's usually the name of the image used with a unique extension and with the lowest runing time if you run the command after your code finished executing) you can run the following command to see its logs:

```bash
kubectl logs -f <name-of-the-pod>
```

## Results

The result generated by `run_on_kubernetes` is a `pandas.DataFrame` with one row per Challenge and one column per Tuner, if `detailed_output` was `False` (by default), containing the best scores obtained by each combination. This results will be stored as specified by config's `output`, if you decide to upload your results to `s3`, you can find them in the bucket that you specified. If those are not set-up correctly, a fallback print inside the `log` output of the pod will be generated.


```
                                                       BTB.GPEiTuner  ...  HyperOpt.rand.suggest  HyperOpt.tpe.suggest
0              XGBoostChallenge('PizzaCutter1_1.csv')       0.664602  ...               0.617456              0.658357
1        XGBoostChallenge('analcatdata_apnea3_1.csv')       0.812482  ...               0.800539              0.821088
2                       XGBoostChallenge('ar4_1.csv')       0.675651  ...               0.622220              0.633233
3                       XGBoostChallenge('ar5_1.csv')       0.547937  ...               0.445195              0.445195
4                     XGBoostChallenge('ecoli_1.csv')       0.728677  ...               0.705936              0.724864
5             XGBoostChallenge('eye_movements_1.csv')       0.834001  ...               0.820084              0.824663
```

Otherwise, if you have specified the `detailed_output` argument to `True`, a `pandas.DataFrame` will be returned with the following columns:

* `challenge`: The name of the challenge that has been evaluated.
* `tuner`: The name of the tuner that has been used.
* `score`: The best score obtained.
* `iterations`: The tuning iterations used to obtain such score.
* `elapsed_time`: The total amount of time spent for the iterations.
* `host`: The name of the host where this task has been run.

```
                                           challenge             tuner     score  iterations         elapsed hostname
0              XGBoostChallenge('newton_hema_1.csv')       BTB.GPTuner  0.333333           1 00:00:01.573637      lgn
1   XGBoostChallenge('Click_prediction_small_1.csv')       BTB.GPTuner  0.509740           1 00:02:18.636728      lgn
2                   XGBoostChallenge('vinnie_1.csv')       BTB.GPTuner  0.587278           1 00:00:01.723117      lgn
3              XGBoostChallenge('newton_hema_1.csv')     BTB.GPEiTuner  0.361667           1 00:00:00.741206      lgn
4   XGBoostChallenge('Click_prediction_small_1.csv')     BTB.GPEiTuner  0.534609           1 00:01:47.558414      lgn
5                   XGBoostChallenge('vinnie_1.csv')     BTB.GPEiTuner  0.817457           1 00:00:04.198422      lgn
```
