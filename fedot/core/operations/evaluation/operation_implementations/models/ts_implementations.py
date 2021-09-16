from typing import Optional
from copy import copy

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from scipy.special import inv_boxcox, boxcox
from statsmodels.tsa.api import STLForecast
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA

from fedot.core.data.data import InputData
from fedot.core.log import Log

from fedot.core.operations.evaluation.operation_implementations.data_operations.ts_transformations import _ts_to_table, \
    LaggedImplementation, LaggedTransformationImplementation, _prepare_target
from fedot.core.operations.evaluation. \
    operation_implementations.implementation_interfaces import ModelImplementation
from fedot.core.pipelines.ts_wrappers import _update_input, exception_if_not_ts_task, _calculate_number_of_steps
from fedot.core.repository.dataset_types import DataTypesEnum

from fedot.utilities.ts_gapfilling import SimpleGapFiller
from sklearn.preprocessing import StandardScaler


class ARIMAImplementation(ModelImplementation):

    def __init__(self, log: Log = None, **params):
        super().__init__(log)
        self.params = params
        self.arima = None
        self.lambda_value = None
        self.scope = None
        self.actual_ts_len = None
        self.sts = None

    def fit(self, input_data):
        """ Class fit arima model on data

        :param input_data: data with features, target and ids to process
        """
        source_ts = np.array(input_data.features)
        # Save actual time series length
        self.actual_ts_len = len(source_ts)
        self.sts = source_ts

        # Apply box-cox transformation for positive values
        min_value = np.min(source_ts)
        if min_value > 0:
            pass
        else:
            # Making a shift to positive values
            self.scope = abs(min_value) + 1
            source_ts = source_ts + self.scope

        _, self.lambda_value = stats.boxcox(source_ts)
        transformed_ts = boxcox(source_ts, self.lambda_value)

        # Set parameters
        p = int(self.params.get('p'))
        d = int(self.params.get('d'))
        q = int(self.params.get('q'))
        params = {'order': (p, d, q)}
        self.arima = ARIMA(transformed_ts, **params).fit()

        return self.arima

    def predict(self, input_data, is_fit_pipeline_stage: bool):
        """ Method for time series prediction on forecast length

        :param input_data: data with features, target and ids to process
        :param is_fit_pipeline_stage: is this fit or predict stage for pipeline
        :return output_data: output data with smoothed time series
        """
        input_data = copy(input_data)
        parameters = input_data.task.task_params
        forecast_length = parameters.forecast_length
        old_idx = input_data.idx
        target = input_data.target

        # For training pipeline get fitted data
        if is_fit_pipeline_stage:
            fitted_values = self.arima.fittedvalues

            fitted_values = self._inverse_boxcox(predicted=fitted_values,
                                                 lambda_param=self.lambda_value)
            # Undo shift operation
            fitted_values = self._inverse_shift(fitted_values)

            diff = int(self.actual_ts_len - len(fitted_values))
            # If first elements skipped
            if diff != 0:
                # Fill nans with first values
                first_element = fitted_values[0]
                first_elements = [first_element] * diff
                first_elements.extend(list(fitted_values))

                fitted_values = np.array(first_elements)

            _, predict = _ts_to_table(idx=old_idx,
                                      time_series=fitted_values,
                                      window_size=forecast_length)

            new_idx, target_columns = _ts_to_table(idx=old_idx,
                                                   time_series=target,
                                                   window_size=forecast_length)

            # Update idx and target
            input_data.idx = new_idx
            input_data.target = target_columns

        # For predict stage we can make prediction
        else:
            start_id = old_idx[-1] - forecast_length + 1
            end_id = old_idx[-1]
            predicted = self.arima.predict(start=start_id,
                                           end=end_id)

            predicted = self._inverse_boxcox(predicted=predicted,
                                             lambda_param=self.lambda_value)

            # Undo shift operation
            predict = self._inverse_shift(predicted)
            # Convert one-dim array as column
            predict = np.array(predict).reshape(1, -1)
            new_idx = np.arange(start_id, end_id + 1)

            # Update idx
            input_data.idx = new_idx
        # Update idx and features
        output_data = self._convert_to_output(input_data,
                                              predict=predict,
                                              data_type=DataTypesEnum.table)


        return output_data

    def get_params(self):
        return self.params

    def _inverse_boxcox(self, predicted, lambda_param):
        """ Method apply inverse Box-Cox transformation """
        if lambda_param == 0:
            return np.exp(predicted)
        else:
            res = inv_boxcox(predicted, lambda_param)
            res = self._filling_gaps(res)
            return res

    def _inverse_shift(self, values):
        """ Method apply inverse shift operation """
        if self.scope is None:
            pass
        else:
            values = values - self.scope

        return values

    @staticmethod
    def _filling_gaps(res):
        nan_ind = np.argwhere(np.isnan(res))
        res[nan_ind] = -100.0

        # Gaps in first and last elements fills with mean value
        if 0 in nan_ind:
            res[0] = np.mean(res)
        if int(len(res) - 1) in nan_ind:
            res[int(len(res) - 1)] = np.mean(res)

        # Gaps in center of timeseries fills with linear interpolation
        if len(np.ravel(np.argwhere(np.isnan(res)))) != 0:
            gf = SimpleGapFiller()
            res = gf.linear_interpolation(res)

        return res


class AutoRegImplementation(ModelImplementation):

    def __init__(self, log: Log = None, **params):
        super().__init__(log)
        self.params = params
        self.actual_ts_len = None
        self.autoreg = None

    def fit(self, input_data):
        """ Class fit ar model on data

        :param input_data: data with features, target and ids to process
        """

        source_ts = np.array(input_data.features)
        self.actual_ts_len = len(source_ts)
        lag_1 = int(self.params.get('lag_1'))
        lag_2 = int(self.params.get('lag_2'))
        params = {'lags': [lag_1, lag_2]}
        self.autoreg = AutoReg(source_ts, **params).fit()

        return self.autoreg

    def predict(self, input_data, is_fit_pipeline_stage: bool):
        """ Method for time series prediction on forecast length

        :param input_data: data with features, target and ids to process
        :param is_fit_pipeline_stage: is this fit or predict stage for pipeline
        :return output_data: output data with smoothed time series
        """
        input_data = copy(input_data)
        parameters = input_data.task.task_params
        forecast_length = parameters.forecast_length
        old_idx = input_data.idx
        target = input_data.target

        if is_fit_pipeline_stage:
            fitted = self.autoreg.predict(start=old_idx[0], end=old_idx[-1])
            # First n elements in time series are skipped
            diff = self.actual_ts_len - len(fitted)

            # Fill nans with first values
            first_element = fitted[0]
            first_elements = [first_element] * diff
            first_elements.extend(list(fitted))

            fitted = np.array(first_elements)

            _, predict = _ts_to_table(idx=old_idx,
                                      time_series=fitted,
                                      window_size=forecast_length)

            new_idx, target_columns = _ts_to_table(idx=old_idx,
                                                   time_series=target,
                                                   window_size=forecast_length)

            # Update idx and target
            input_data.idx = new_idx
            input_data.target = target_columns

        # For predict stage we can make prediction
        else:
            start_id = old_idx[-1] - forecast_length + 1
            end_id = old_idx[-1]
            predicted = self.autoreg.predict(start=start_id,
                                             end=end_id)

            # Convert one-dim array as column
            predict = np.array(predicted).reshape(1, -1)
            new_idx = np.arange(start_id, end_id + 1)

            # Update idx
            input_data.idx = new_idx

            # Update idx and features
        output_data = self._convert_to_output(input_data,
                                              predict=predict,
                                              data_type=DataTypesEnum.table)
        return output_data

    def get_params(self):
        return self.params


class STLForecastARIMAImplementation(ModelImplementation):
    def __init__(self, log: Log = None, **params: Optional[dict]):
        super().__init__(log)
        self.params = params
        self.model = None
        self.lambda_param = None
        self.scope = None
        self.actual_ts_len = None
        self.sts = None

    def fit(self, input_data):
        """ Class fit STLForecast arima model on data

        :param input_data: data with features, target and ids to process
        """

        source_ts = np.array(input_data.features)
        # Save actual time series length
        self.actual_ts_len = len(source_ts)
        self.sts = source_ts

        if not self.params:
            # Default data
            self.params = {'p': 2, 'd': 0, 'q': 2, 'period': 365}

        p = int(self.params.get('p'))
        d = int(self.params.get('d'))
        q = int(self.params.get('q'))
        period = int(self.params.get('period'))
        params = {'period': period, 'model_kwargs': {'order': (p, d, q)}}
        self.model = STLForecast(source_ts, ARIMA, **params).fit()

        return self.model

    def predict(self, input_data, is_fit_pipeline_stage: bool):
        """ Method for time series prediction on forecast length

        :param input_data: data with features, target and ids to process
        :param is_fit_pipeline_stage: is this fit or predict stage for pipeline
        :return output_data: output data with smoothed time series
        """
        parameters = input_data.task.task_params
        forecast_length = parameters.forecast_length
        old_idx = input_data.idx
        target = input_data.target

        # For training pipeline get fitted data
        if is_fit_pipeline_stage:
            fitted_values = self.model.get_prediction(start=old_idx[0], end=old_idx[-1]).predicted_mean
            diff = int(self.actual_ts_len) - len(fitted_values)
            # If first elements skipped
            if diff != 0:
                # Fill nans with first values
                first_element = fitted_values[0]
                first_elements = [first_element] * diff
                first_elements.extend(list(fitted_values))

                fitted_values = np.array(first_elements)

            _, predict = _ts_to_table(idx=old_idx,
                                      time_series=fitted_values,
                                      window_size=forecast_length)

            new_idx, target_columns = _ts_to_table(idx=old_idx,
                                                   time_series=target,
                                                   window_size=forecast_length)

            # Update idx and target
            input_data.idx = new_idx
            input_data.target = target_columns

        # For predict stage we can make prediction
        else:
            start_id = old_idx[-1] - forecast_length + 1
            end_id = old_idx[-1]
            predicted = self.model.get_prediction(start=start_id, end=end_id).predicted_mean

            # Convert one-dim array as column
            predict = np.array(predicted).reshape(1, -1)
            new_idx = np.arange(start_id, end_id + 1)

            # Update idx
            input_data.idx = new_idx

        # Update idx and features
        output_data = self._convert_to_output(input_data,
                                              predict=predict,
                                              data_type=DataTypesEnum.table)
        return output_data

    def get_params(self):
        return self.params


class CLSTMImplementation(ModelImplementation):
    def __init__(self, log: Log = None, **params):
        super().__init__(log)
        self.params = params
        self.epochs = params.get("num_epochs")
        self.input_size = params.get("input_size")
        self.forecast_length = params.get("forecast_length")
        self.hidden_size = int(params.get("hidden_size"))
        self.batch_size = params.get("batch_size")
        self.learning_rate = params.get("learning_rate")
        self.window_size = int(params.get("window_size"))
        # Where we will perform training: cpu or gpu (if it is possible)
        self.device = self._get_device()
        self.model = LSTMNetwork(input_size=self.input_size,
                                 output_size=1,
                                 hidden_size=self.hidden_size)

        self.scaler = StandardScaler()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

    def fit(self, train_data: InputData):
        self.model = self.model.to(self.device)
        features_scaled, target_scaled = self._fit_transform_scaler(train_data)
        train_data_new = copy(train_data)
        train_data_new.features = features_scaled
        train_data_new.target = target_scaled
        new_idx, lagged_table = _ts_to_table(idx=train_data_new.idx,
                                             time_series=train_data_new.features,
                                             window_size=self.window_size)

        final_idx, features_columns, final_target = _prepare_target(idx=new_idx,
                                                                    features_columns=lagged_table,
                                                                    target=train_data_new.target,
                                                                    forecast_length=1)
        train_data_new.idx = final_idx
        train_data_new.features = features_columns
        train_data_new.target = final_target
        x = torch.from_numpy(train_data_new.features.copy()).float()
        y = torch.from_numpy(train_data_new.target.copy()).float().reshape(-1, 1)
        data_loader = DataLoader(TensorDataset(x, y), batch_size=self.batch_size)

        self.model.train()
        for i in range(self.epochs):
            for x, y in data_loader:
                self.optimizer.zero_grad()
                x = x.to(self.device)
                y = y.to(self.device)
                self.model.init_hidden(x.size(0), self.device)
                output = self.model(x.unsqueeze(1))
                loss = self.criterion(output, y)
                loss.backward()
                print(loss.item())
                self.optimizer.step()
        return self.model

    def predict(self, input_data: InputData, is_fit_pipeline_stage: Optional[bool]):
        self.model.eval()
        input_data_new = copy(input_data)
        old_idx = input_data_new.idx
        if is_fit_pipeline_stage:

            new_idx, lagged_table = _ts_to_table(idx=old_idx,
                                                 time_series=input_data_new.features,
                                                 window_size=self.window_size)

            final_idx, features_columns, final_target = _prepare_target(idx=new_idx,
                                                                        features_columns=lagged_table,
                                                                        target=input_data_new.target,
                                                                        forecast_length=self.forecast_length)
            input_data_new.idx = final_idx
            input_data_new.features = features_columns
            input_data_new.target = final_target
        else:
            input_data_new.features = input_data_new.features[-self.window_size:].reshape(1, -1)
            input_data.idx = input_data_new.idx[-self.window_size:].reshape(1, -1)
        predict = self._out_of_sample_ts_forecast(input_data_new)


        output_data = self._convert_to_output(input_data_new,
                                           predict=predict,
                                           data_type=DataTypesEnum.table)
        return output_data

    def _predict(self, input_data: InputData):
        features_scaled = self._transform_scaler_features(input_data)
        x = torch.Tensor(features_scaled).to(self.device)
        self.model.init_hidden(x.shape[0], self.device)
        pred = self.model(x.unsqueeze(1)).squeeze(0).cpu().detach().numpy()
        return self._inverse_transform_scaler(pred)

    def _out_of_sample_ts_forecast(self, input_data: InputData) -> np.array:
        input_data_new = copy(input_data)
        # Prepare data for time series forecasting
        task = input_data_new.task
        exception_if_not_ts_task(task)

        pre_history_ts = np.array(input_data_new.features)


        # How many elements to the future pipeline can produce
        scope_len = task.task_params.forecast_length
        number_of_iterations = scope_len

        # Make forecast iteratively moving throw the horizon
        final_forecast = None

        for _ in range(0, number_of_iterations):
            with torch.no_grad():
                iter_predict = self._predict(input_data_new)
            if final_forecast is not None:
                final_forecast = np.hstack((final_forecast, iter_predict))
            else:
                final_forecast = iter_predict

            # Add prediction to the historical data - update it
            pre_history_ts = np.hstack((pre_history_ts[:, 1:], iter_predict))
            # Prepare InputData for next iteration
            input_data_new = _update_input(pre_history_ts, scope_len, task)

        return final_forecast

    def _fit_transform_scaler(self, data: InputData):
        f_scaled = self.scaler.fit_transform(data.features.reshape(-1, 1)).reshape(-1)
        t_scaled = self.scaler.transform(data.target.reshape(-1, 1)).reshape(-1)
        return f_scaled, t_scaled

    def _inverse_transform_scaler(self, data: np.ndarray):
        start_shape = data.shape
        return self.scaler.inverse_transform(data.reshape(-1, 1)).reshape(start_shape)

    def _transform_scaler_features(self, data: InputData):
        start_shape = data.features.shape
        return self.scaler.transform(data.features.reshape(-1, 1)).reshape(start_shape)

    def _transform_scaler_target(self, data: InputData):
        return self.scaler.transform(data.target.reshape(-1, 1)).reshape(-1)

    def get_params(self):
        return self.params

    @staticmethod
    def _get_device():
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
        return device


class LSTMNetwork(nn.Module):
    def __init__(self,
                 input_size=1,
                 output_size=1,
                 hidden_size=200,
                 cnn1_kernel_size=5,
                 cnn1_output_size=16,
                 cnn2_kernel_size=3,
                 cnn2_output_size=32,
                 ):
        super().__init__()

        self.hidden_size = hidden_size
        self.cnn2_output_size = cnn2_output_size
        self.conv_block1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=cnn1_output_size, kernel_size=cnn1_kernel_size),
            nn.ReLU()
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv1d(in_channels=cnn1_output_size, out_channels=cnn2_output_size, kernel_size=cnn2_kernel_size),
            nn.ReLU()
        )
        self.lstm = nn.LSTM(cnn2_output_size, self.hidden_size, dropout=0.1)
        self.hidden_cell = None
        self.linear = nn.Linear(self.hidden_size, 1)

    def init_hidden(self, batch_size, device):
        self.hidden_cell = (torch.zeros(1, batch_size, self.hidden_size).to(device),
                            torch.zeros(1, batch_size, self.hidden_size).to(device))

    def forward(self, x):
        if self.hidden_cell is None:
            raise Exception
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = x.permute(2, 0, 1)
        out, self.hidden_cell = self.lstm(x, self.hidden_cell)
        #  hidden_cat = torch.cat([self.hidden_cell[0], self.hidden_cell[1]], dim=2)
        predictions = self.linear(self.hidden_cell[0])

        return predictions
