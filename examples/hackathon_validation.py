import timeit
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score

from fedot.core.chains.chain import Chain
from fedot.core.chains.node import PrimaryNode, SecondaryNode
from fedot.core.data.data import InputData
from fedot.core.repository.dataset_types import DataTypesEnum
from fedot.core.repository.tasks import Task, TaskTypesEnum, TsForecastingParams

warnings.filterwarnings('ignore')
np.random.seed(2020)


def make_forecast(chain):
    """
    Function for predicting values in a time series

    :param chain: Chain object

    :return predicted_values: numpy array, forecast of model
    """

    # Fit it
    start_time = timeit.default_timer()
    chain.fit_from_scratch()
    amount_of_seconds = timeit.default_timer() - start_time

    print(f'\nIt takes {amount_of_seconds:.2f} seconds to train chain\n')

    # Predict
    predicted_values = chain.predict()
    predicted_values = predicted_values.predict

    return predicted_values


def prepare_input_data(len_forecast, train_data_features, train_data_target,
                       test_data_features):
    """ Function return prepared data for fit and predict

    :param len_forecast: forecast length
    :param train_data_features: time series which can be used as predictors for train
    :param train_data_target: time series which can be used as target for train
    :param test_data_features: time series which can be used as predictors for prediction

    :return train_input: Input Data for fit
    :return predict_input: Input Data for predict
    :return task: Time series forecasting task with parameters
    """

    task = Task(TaskTypesEnum.ts_forecasting,
                TsForecastingParams(forecast_length=len_forecast))

    train_input = InputData(idx=np.arange(0, len(train_data_features)),
                            features=train_data_features,
                            target=train_data_target,
                            task=task,
                            data_type=DataTypesEnum.ts)

    # Determine indices for forecast
    start_forecast = len(train_data_features)
    end_forecast = start_forecast + len_forecast
    predict_input = InputData(idx=np.arange(start_forecast, end_forecast),
                              features=test_data_features,
                              target=None,
                              task=task,
                              data_type=DataTypesEnum.ts)

    return train_input, predict_input, task


def divide_into_train_and_test(array, len_forecast):
    # Let's divide our data on train and test samples
    train_data = array[:-len_forecast]
    test_data = array[-len_forecast:]

    return train_data, test_data


def run_exogenous_experiment(path_to_file, len_forecast=12,
                             with_visualisation=True,
                             dict_with_lags=None) -> None:
    """ Function with example how time series forecasting can be made with using
    exogenous features

    :param path_to_file: path to the csv file with dataframe
    :param len_forecast: forecast length
    :param with_visualisation: is it needed to make visualisations
    :param dict_with_lags: dictionary with lags for forecasting
    """

    df = pd.read_csv(path_to_file)
    target = np.array(df['sick'])

    # The features
    first_period_h = np.array(df['first_period_h'])
    worked_days = np.array(df['worked_days'])
    working_hours_monthly = np.array(df['working_hours_monthly'])
    working_hours_daily = np.array(df['working_hours_daily'])
    age = np.array(df['age'])
    second_period_h = np.array(df['second_period_h'])
    third_period_h = np.array(df['third_period_h'])

    # Divide our data on train and test samples
    train_target, test_target = divide_into_train_and_test(target, len_forecast)

    # For features we don't need to have second (test) part
    train_first_period_h, _ = divide_into_train_and_test(first_period_h, len_forecast)
    train_worked_days, _ = divide_into_train_and_test(worked_days, len_forecast)
    train_working_hours_monthly, _ = divide_into_train_and_test(working_hours_monthly, len_forecast)
    train_working_hours_daily, _ = divide_into_train_and_test(working_hours_daily, len_forecast)
    train_age, _ = divide_into_train_and_test(age, len_forecast)
    train_second_period_h, _ = divide_into_train_and_test(second_period_h, len_forecast)
    train_third_period_h, _ = divide_into_train_and_test(third_period_h, len_forecast)

    # Source time series
    train_input, predict_input, task = prepare_input_data(len_forecast=len_forecast,
                                                          train_data_features=train_target,
                                                          train_data_target=train_target,
                                                          test_data_features=train_target)

    # Exogenous time series - first_period_h
    train_input_first_period_h, predict_input_first_period_h, _ = \
        prepare_input_data(len_forecast=len_forecast,
                           train_data_features=train_first_period_h,
                           train_data_target=train_target,
                           test_data_features=train_first_period_h)

    # Exogenous time series - worked_days
    train_input_worked_days, predict_input_worked_days, _ = \
        prepare_input_data(len_forecast=len_forecast,
                           train_data_features=train_worked_days,
                           train_data_target=train_target,
                           test_data_features=train_worked_days)

    # Exogenous time series - working_hours_monthly
    train_input_working_hours_monthly, predict_input_working_hours_monthly, _ = \
        prepare_input_data(len_forecast=len_forecast,
                           train_data_features=train_working_hours_monthly,
                           train_data_target=train_target,
                           test_data_features=train_working_hours_monthly)

    # Exogenous time series - working_hours_daily
    train_input_working_hours_daily, predict_input_working_hours_daily, _ = \
        prepare_input_data(len_forecast=len_forecast,
                           train_data_features=train_working_hours_daily,
                           train_data_target=train_target,
                           test_data_features=train_working_hours_daily)

    # Exogenous time series - age
    train_input_age, predict_input_age, _ = \
        prepare_input_data(len_forecast=len_forecast,
                           train_data_features=train_age,
                           train_data_target=train_target,
                           test_data_features=train_age)

    # Exogenous time series - second_period_h
    train_input_second_period_h, predict_input_second_period_h, _ = \
        prepare_input_data(len_forecast=len_forecast,
                           train_data_features=train_second_period_h,
                           train_data_target=train_target,
                           test_data_features=train_second_period_h)

    # Exogenous time series - third_period_h
    train_input_third_period_h, predict_input_third_period_h, _ = \
        prepare_input_data(len_forecast=len_forecast,
                           train_data_features=train_third_period_h,
                           train_data_target=train_target,
                           test_data_features=train_third_period_h)

    # Define chain
    # NODE 1 !!!
    node_target = PrimaryNode('lagged', node_data={'fit': train_input,
                                                   'predict': predict_input})
    node_target.custom_params = {'window_size': dict_with_lags.get('target')}

    # NODE 2 !!!
    node_first_period_h = PrimaryNode('lagged', node_data={'fit': train_input_first_period_h,
                                                           'predict': predict_input_first_period_h})
    node_first_period_h.custom_params = {'window_size': dict_with_lags.get('first_period_h')}

    # NODE 3 !!!
    node_worked_days = PrimaryNode('lagged', node_data={'fit': train_input_worked_days,
                                                           'predict': predict_input_worked_days})
    node_worked_days.custom_params = {'window_size': dict_with_lags.get('worked_days')}

    # NODE 4 !!!
    node_working_hours_monthly = PrimaryNode('lagged', node_data={'fit': train_input_working_hours_monthly,
                                              'predict': predict_input_working_hours_monthly})
    node_working_hours_monthly.custom_params = {'window_size': dict_with_lags.get('working_hours_monthly')}

    # NODE 5 !!!
    node_working_hours_daily = PrimaryNode('lagged', node_data={'fit': train_input_working_hours_daily,
                                                                'predict': predict_input_working_hours_daily})
    node_working_hours_daily.custom_params = {'window_size': dict_with_lags.get('working_hours_daily')}

    # NODE 6 !!!
    node_age = PrimaryNode('lagged', node_data={'fit': train_input_age,
                                                'predict': predict_input_age})
    node_age.custom_params = {'window_size': dict_with_lags.get('age')}

    # NODE 7 !!!
    node_second_period_h = PrimaryNode('lagged', node_data={'fit': train_input_second_period_h,
                                                            'predict': predict_input_second_period_h})
    node_second_period_h.custom_params = {'window_size': dict_with_lags.get('second_period_h')}

    # NODE 8 !!!
    node_third_period_h = PrimaryNode('lagged', node_data={'fit': train_input_third_period_h,
                                                           'predict': predict_input_third_period_h})
    node_third_period_h.custom_params = {'window_size': dict_with_lags.get('third_period_h')}

    node_final = SecondaryNode('dtreg', nodes_from=[node_target,
                                                    node_first_period_h,
                                                    node_worked_days,
                                                    node_working_hours_monthly,
                                                    node_working_hours_daily,
                                                    node_age,
                                                    node_second_period_h,
                                                    node_third_period_h])
    chain = Chain(node_final)

    predicted = make_forecast(chain)
    predicted = np.ravel(np.array(predicted))
    test_data = np.ravel(test_target)

    print(f'Predicted values: {predicted}')
    print(f'  Actual  values: {test_data}')

    try:
        f1_metric = f1_score(test_data, predicted)
        print(f'F1 - {f1_metric:.4f}')
    except Exception:
        pass

    if with_visualisation:
        plt.plot(range(0, len(target)), target, label='Actual time series')
        plt.plot(range(len(train_target), len(target)), predicted, label='Forecast')
        # Plot black line which divide our array into train and test
        plt.plot([len(train_target), len(train_target)],
                 [min(target)-0.2, max(target)+0.2], c='black',
                 linewidth=2)
        plt.xlabel('Time index', fontsize=14)
        plt.legend()
        plt.grid()
        plt.show()


if __name__ == '__main__':
    run_exogenous_experiment(path_to_file='../cases/data/hackathon/sick.csv',
                             len_forecast=12,
                             dict_with_lags={'target': 2, 'first_period_h': 15,
                                             'worked_days': 10, 'working_hours_monthly': 10,
                                             'working_hours_daily': 1, 'age': 1,
                                             'second_period_h': 14, 'third_period_h': 10})
