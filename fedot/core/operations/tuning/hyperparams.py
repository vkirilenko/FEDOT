import numpy as np
from hyperopt import hp, fmin, tpe, space_eval

params_by_operation = {
    'kmeans': ['n_clusters'],
    'adareg': ['n_estimators', 'learning_rate', 'loss'],
    'gbr': ['n_estimators', 'loss', 'learning_rate', 'max_depth', 'min_samples_split',
            'min_samples_leaf', 'subsample', 'max_features', 'alpha'],
    'logit': ['C'],
    'rf': ['n_estimators', 'criterion', 'max_features', 'min_samples_split',
           'min_samples_leaf', 'bootstrap'],
    'lasso': ['alpha'],
    'ridge': ['alpha'],
    'dtreg': ['max_depth', 'min_samples_split', 'min_samples_leaf'],
    'treg': ['n_estimators', 'max_features', 'min_samples_split', 'min_samples_leaf', 'bootstrap'],
    'dt': ['max_depth', 'min_samples_split', 'min_samples_leaf'],
    'knnreg': ['n_neighbors', 'weights', 'p'],
    'knn': ['n_neighbors', 'weights', 'p'],
    'rfr': ['n_estimators', 'max_features', 'min_samples_split',
            'min_samples_leaf', 'bootstrap'],
    'xgbreg': ['n_estimators', 'max_depth', 'learning_rate', 'subsample',
               'min_child_weight', 'objective'],
    'xgboost': ['n_estimators', 'max_depth', 'learning_rate', 'subsample',
                'min_child_weight', 'nthread'],
    'svr': ['loss', 'tol', 'C', 'epsilon'],
    'arima': ['p', 'd', 'q'],
    'ar': ['lag_1', 'lag_2'],

    'pca': ['n_components', 'svd_solver'],
    'kernel_pca': ['n_components'],
    'ransac_lin_reg': ['min_samples', 'residual_threshold',
                       'max_trials', 'max_skips'],
    'ransac_non_lin_reg': ['min_samples', 'residual_threshold',
                           'max_trials', 'max_skips'],
    'rfe_lin_reg': ['n_features_to_select', 'step'],
    'rfe_non_lin_reg': ['n_features_to_select', 'step'],
    'poly_features': ['degree', 'interaction_only'],
    'lagged': ['window_size'],
    'smoothing': ['window_size'],
    'gaussian_filter': ['sigma'],
}


def __get_range_by_parameter(label, parameter_name):
    """
    Function prepares appropriate labeled dictionary for desired operation

    :param label: label to assign in hyperopt pyll
    :param parameter_name: name of hyperparameter of particular operation

    :return : dictionary with appropriate range
    """

    range_by_parameter = {
        'kmeans | n_clusters': hp.choice(label, [2, 3, 4, 5, 6]),

        'adareg | n_estimators': hp.choice(label, [100]),
        'adareg | learning_rate': hp.uniform(label, 1e-3, 1.0),
        'adareg | loss': hp.choice(label, ["linear", "square", "exponential"]),

        'gbr | n_estimators': hp.choice(label, [100]),
        'gbr | loss': hp.choice(label, ["ls", "lad", "huber", "quantile"]),
        'gbr | learning_rate': hp.uniform(label, 1e-3, 1.0),
        'gbr | max_depth': hp.choice(label, range(1, 11)),
        'gbr | min_samples_split': hp.choice(label, range(2, 21)),
        'gbr | min_samples_leaf': hp.choice(label, range(1, 21)),
        'gbr | subsample': hp.uniform(label, 0.05, 1.0),
        'gbr | max_features': hp.uniform(label, 0.05, 1.0),
        'gbr | alpha': hp.uniform(label, 0.75, 0.99),

        'logit | C': hp.uniform(label, 1e-2, 10.0),

        'rf | n_estimators': hp.choice(label, [100]),
        'rf | criterion': hp.choice(label, ["gini", "entropy"]),
        'rf | max_features': hp.uniform(label, 0.05, 1.01),
        'rf | min_samples_split': hp.choice(label, range(2, 10)),
        'rf | min_samples_leaf': hp.choice(label, range(1, 15)),
        'rf | bootstrap': hp.choice(label, [True, False]),

        'lasso | alpha': hp.uniform(label, 0.01, 10.0),
        'ridge | alpha': hp.uniform(label, 0.01, 10.0),

        'rfr | n_estimators': hp.choice(label, [100]),
        'rfr | max_features': hp.uniform(label, 0.05, 1.01),
        'rfr | min_samples_split': hp.choice(label, range(2, 21)),
        'rfr | min_samples_leaf': hp.choice(label, range(1, 21)),
        'rfr | bootstrap': hp.choice(label, [True, False]),

        'xgbreg | n_estimators': hp.choice(label, [100]),
        'xgbreg | max_depth': hp.choice(label, range(1, 11)),
        'xgbreg | learning_rate': hp.choice(label, [1e-3, 1e-2, 1e-1, 0.5, 1.]),
        'xgbreg | subsample': hp.choice(label, np.arange(0.05, 1.01, 0.05)),
        'xgbreg | min_child_weight': hp.choice(label, range(1, 21)),
        'xgbreg | objective': hp.choice(label, ['reg:squarederror']),

        'xgboost | n_estimators': hp.choice(label, [100]),
        'xgboost | max_depth': hp.choice(label, range(1, 7)),
        'xgboost | learning_rate': hp.uniform(label, 0.1, 0.9),
        'xgboost | subsample': hp.uniform(label, 0.05, 0.99),
        'xgboost | min_child_weight': hp.choice(label, range(1, 21)),
        'xgboost | nthread': hp.choice(label, [1]),

        'svr | loss': hp.choice(label, ["epsilon_insensitive",
                                        "squared_epsilon_insensitive"]),
        'svr | tol': hp.choice(label, [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]),
        'svr | C': hp.uniform(label, 1e-4, 25.0),
        'svr | epsilon': hp.uniform(label, 1e-4, 1.0),

        'dtreg | max_depth': hp.choice(label, range(1, 11)),
        'dtreg | min_samples_split': hp.choice(label, range(2, 21)),
        'dtreg | min_samples_leaf': hp.choice(label, range(1, 21)),

        'treg | n_estimators': hp.choice(label, [100]),
        'treg | max_features': hp.uniform(label, 0.05, 1.0),
        'treg | min_samples_split': hp.choice(label, range(2, 21)),
        'treg | min_samples_leaf': hp.choice(label, range(1, 21)),
        'treg | bootstrap': hp.choice(label, [True, False]),

        'dt | max_depth': hp.choice(label, range(1, 11)),
        'dt | min_samples_split': hp.choice(label, range(2, 21)),
        'dt | min_samples_leaf': hp.choice(label, range(1, 21)),

        'knnreg | n_neighbors': hp.choice(label, range(1, 50)),
        'knnreg | weights': hp.choice(label, ["uniform", "distance"]),
        'knnreg | p': hp.choice(label, [1, 2]),

        'knn | n_neighbors': hp.choice(label, range(1, 50)),
        'knn | weights': hp.choice(label, ["uniform", "distance"]),
        'knn | p': hp.choice(label, [1, 2]),

        'arima | p': hp.choice(label, [1, 2, 3, 4, 5, 6]),
        'arima | d': hp.choice(label, [0, 1]),
        'arima | q': hp.choice(label, [1, 2, 3, 4]),

        'ar | lag_1': hp.uniform(label, 2, 200),
        'ar | lag_2': hp.uniform(label, 2, 800),

        'pca | n_components': hp.uniform(label, 0.1, 0.99),
        'pca | svd_solver': hp.choice(label, ['full']),

        'kernel_pca | n_components': hp.choice(label, range(1, 20)),

        'ransac_lin_reg | min_samples': hp.uniform(label, 0.1, 0.9),
        'ransac_lin_reg | residual_threshold': hp.choice(label, [0.1, 1.0, 100.0, 500.0, 1000.0]),
        'ransac_lin_reg | max_trials': hp.uniform(label, 50, 500),
        'ransac_lin_reg | max_skips': hp.uniform(label, 50, 500000),

        'ransac_non_lin_reg | min_samples': hp.uniform(label, 0.1, 0.9),
        'ransac_non_lin_reg | residual_threshold': hp.choice(label, [0.1, 1.0, 100.0, 500.0, 1000.0]),
        'ransac_non_lin_reg | max_trials': hp.uniform(label, 50, 500),
        'ransac_non_lin_reg | max_skips': hp.uniform(label, 50, 500000),

        'rfe_lin_reg | n_features_to_select': hp.choice(label, [0.5, 0.7, 0.9]),
        'rfe_lin_reg | step': hp.choice(label, [0.1, 0.15, 0.2]),

        'rfe_non_lin_reg | n_features_to_select': hp.choice(label, [0.5, 0.7, 0.9]),
        'rfe_non_lin_reg | step': hp.choice(label, [0.1, 0.15, 0.2]),

        'poly_features | degree': hp.choice(label, [2, 3, 4]),
        'poly_features | interaction_only': hp.choice(label, [True, False]),

        'lagged | window_size': hp.uniform(label, 10, 500),

        'smoothing | window_size': hp.uniform(label, 2, 20),

        'gaussian_filter | sigma': hp.uniform(label, 1, 5),
    }

    return range_by_parameter.get(parameter_name)


def get_node_params(node_id, operation_name):
    """
    Function for forming dictionary with hyperparameters for considering
    operation as a part of the whole chain

    :param node_id: number of node in chain.nodes list
    :param operation_name: name of operation in the node

    :return params_dict: dictionary-like structure with labeled hyperparameters
    and their range per operation
    """

    # Get available parameters for operation
    params_list = params_by_operation.get(operation_name)

    if params_list is None:
        params_dict = None
    else:
        params_dict = {}
        for parameter_name in params_list:
            # Name with operation and parameter
            parameter_name = ''.join((operation_name, ' | ', parameter_name))

            # Name with node id || operation | parameter
            labeled_parameter_name = ''.join((str(node_id), ' || ', parameter_name))

            # For operation get range where search can be done
            space = __get_range_by_parameter(label=labeled_parameter_name,
                                             parameter_name=parameter_name)

            params_dict.update({labeled_parameter_name: space})

    return params_dict


def convert_params(params):
    """
    Function removes labels from dictionary with operations

    :param params: labeled parameters
    :return new_params: dictionary without labels of node_id and operation_name
    """
    operation_parameters = list(params.keys())

    new_params = {}
    for operation_parameter in operation_parameters:
        value = params.get(operation_parameter)

        # Remove right part of the parameter name
        parameter_name = operation_parameter.split(' | ')[-1]

        if value is None:
            pass
        else:
            new_params.update({parameter_name: value})

    return new_params


def get_new_operation_params(operation_name):
    """ Function return a dictionary with new

    :param operation_name: name of operation to get hyperparameters for
    """

    # Function to imitate objective
    def fake_objective(fake_params):
        return 0

    # Get available parameters for operation
    params_list = params_by_operation.get(operation_name)

    if params_list is None:
        params_dict = None
    else:
        params_dict = {}
        for parameter_name in params_list:
            # For operation get range where search can be done
            new_parameter_name = ''.join((operation_name, ' | ', parameter_name))

            space = __get_range_by_parameter(label=parameter_name,
                                             parameter_name=new_parameter_name)

            # Get parameters values for chosen parameter
            small_dict = {parameter_name: space}
            best = fmin(fake_objective,
                        small_dict,
                        algo=tpe.suggest,
                        max_evals=1,
                        show_progressbar=False)
            best = space_eval(space=small_dict, hp_assignment=best)
            params_dict.update(best)

    return params_dict
