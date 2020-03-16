from hyperopt import Trials, fmin, hp, tpe


def search_space_from_dict(dict_hyperparams):
    hyperparams = {}

    if not isinstance(dict_hyperparams, dict):
        raise TypeError('Hyperparams must be a dictionary.')

    for name, hyperparam in dict_hyperparams.items():
        hp_type = hyperparam['type']

        if hp_type == 'int':
            hp_range = hyperparam.get('range') or hyperparam.get('values')
            hp_min = min(hp_range) if hp_range else None
            hp_max = max(hp_range) if hp_range else None
            hp_instance = hp.uniformint(name, hp_min, hp_max)

        elif hp_type == 'float':
            hp_range = hyperparam.get('range') or hyperparam.get('values')
            hp_min = min(hp_range)
            hp_max = max(hp_range)
            hp_instance = hp.uniform(name, hp_min, hp_max)

        elif hp_type == 'bool':
            hp_instance = hp.choice(name, [True, False])

        elif hp_type == 'str':
            hp_choices = hyperparam.get('range') or hyperparam.get('values')
            hp_instance = hp.choice(name, hp_choices)

        hyperparams[name] = hp_instance

    return hyperparams


def sanitize_scoring_function(scoring_function):
    def sanitized(args):
        return -scoring_function(**args)

    return sanitized


def hyperopt_tuning_function(scoring_function, tunable_hyperparameters, iterations):

    # convert scoring to minimize
    sanitized_scorer = sanitize_scoring_function(scoring_function)

    search_space = search_space_from_dict(tunable_hyperparameters)
    trials = Trials()

    fmin(sanitized_scorer, search_space, algo=tpe.suggest, max_evals=iterations, trials=trials)

    # normalize best score to match other tuners
    best_score = -1 * trials.best_trial['result']['loss']

    return best_score
