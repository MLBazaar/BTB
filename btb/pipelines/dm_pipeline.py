from btb.hyper_parameter import HyperParameter, ParamTypes

class DmPipeline(object):
    """Pipeline abstract class based on that of DeepMining.
        Will be subclassed for each type of pipeline
        (eg simple image, image with keras, etc)

    Class variables:
        HYPERPARAMETER_RANGES: The default hyperparam_ranges.
            List of HyperParameter objects

    Attributes:
        hyperparam_ranges: List of HyperParameter objects, can be fine tuned from
            class defauls based on cpus/ram/dataset.
        d3m_primitives: list of strings of d3m primitives used in pipeline
        recommended_scoring: string represengint suggested scoring metric
        hyperparams: current hyperparameters
        d3mds: object of D3M dataset for pipeline
        cpus: cpus information of system that pipeline is run on. Used to fine
            tune hyperparam ranges
        ram: ram of system that pipeline is run on. Used to fine tune hyperparam
            ranges.

    """

    HYPERPARAMETER_RANGES = None
    D3M_PRIMITIVES = None

    def __init__(self, cpus=None, ram=None):
        self.hyperparams = None
        self.recommended_scoring = None
        self.cpus = cpus
        self.ram = ram

    @classmethod
    def get_d3m_primitives(cls):
        return cls.D3M_PRIMITIVES[:]

    @classmethod
    def get_hyperparam_ranges(cls):
        return cls.HYPERPARAMETER_RANGES[:]

    def cv_score(self, d3mds, cv_scoring = None, cv = None):
        """splits and scores and returns cv score"""
        pass

    def fit(self, d3mds):
        "Fits model to given dataset for given hyperparameters"""
        pass

    def predict(self, d3mds):
        """Make predictions on fitted pipeline"""
        pass

    def set_hyperparams(self, hyperparams):
        """Sets hyperparams.

        Must be called before fit if pipeline has hyperparamers"""
        params = {}
        for i in range(len(self.HYPERPARAMETER_RANGES)):
            is_int = self.HYPERPARAMETER_RANGES[i][1].type == ParamTypes.INT
            value = int(hyperparams[i]) if is_int else hyperparams[i]
            params[self.HYPERPARAMETER_RANGES[i][0]] = value
        self.hyperparams = params
