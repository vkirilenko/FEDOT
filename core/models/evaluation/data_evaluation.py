import numpy as np
from scipy import signal
from statsmodels.tsa.seasonal import seasonal_decompose

from core.models.data import InputData
from core.models.evaluation.evaluation import EvaluationStrategy

from functools import partial

def get_data(predict_data: InputData):
    return predict_data.features


def get_difference(predict_data: InputData):
    number_of_inputs = predict_data.features.shape[1]
    if number_of_inputs != 1:
        raise ValueError('Too many inputs for the differential model')
    return predict_data.features[:, 0] - predict_data.target


def get_sum(predict_data: InputData):
    if predict_data.features.shape[1] != 2:
        raise ValueError('Wrong number of inputs for the additive model')
    return np.sum(predict_data.features, axis=1)


def _estimate_period(variable):
    analyse_ratio = 10
    f, pxx_den = signal.welch(variable, fs=1, scaling='spectrum',
                              nfft=int(len(variable) / analyse_ratio),
                              nperseg=int(len(variable) / analyse_ratio))
    period = int(1 / f[np.argmax(pxx_den)])
    return period


def get_trend(predict_data: InputData, period: int):
    target = predict_data.target
    decomposed_target = seasonal_decompose(target, period=period, extrapolate_trend='freq')
    return decomposed_target.trend


def get_residual(predict_data: InputData, period: int):
    target_trend = get_trend(predict_data, period)
    target_residual = predict_data.target - target_trend
    return target_residual


class DataModellingStrategy(EvaluationStrategy):
    _model_functions_by_type = {
        'direct_data_model': get_data,
        'diff_data_model': get_difference,
        'additive_data_model': get_sum,
        'trend_data_model': get_trend,
        'residual_data_model': get_residual
    }

    def __init__(self, model_type: str):
        self._model_specific_predict = self._model_functions_by_type[model_type]
        if model_type in ['trend_data_model', 'residual_data_model']:
            self.period = None

    def fit(self, train_data: InputData):
        if not hasattr(self, 'period'):
            return self._model_specific_predict

        if hasattr(train_data.task.task_params, 'period'):
            period = train_data.task.task_params.period
        else:
            period = _estimate_period(train_data.target)
        return partial(self._model_specific_predict, period=period)

    def predict(self, trained_model, predict_data: InputData):
        return trained_model(predict_data)

    def fit_tuned(self, train_data: InputData, iterations: int = 30):
        return None, None
