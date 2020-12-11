from fedot.core.utils import ComparableEnum as Enum


class DataTypesEnum(Enum):
    table = 'feature_table'

    # 2d dataset with timeseries as target and external variables as features
    ts = 'time_series'

    # 2d dataset with time series forecasted by model
    forecasted_ts = 'time_series_forecasted'

    # 2d dataset with lagged features - (n, window_len * features)
    ts_lagged_table = 'time_series_lagged_table'


def get_data_type(data_type: str):
    if data_type == 'feature_table':
        return DataTypesEnum.table
    if data_type == 'time_series':
        return DataTypesEnum.ts
    if data_type == 'time_series_forecasted':
        return DataTypesEnum.forecasted_ts
    if data_type == 'time_series_lagged_table':
        return DataTypesEnum.ts_lagged_table
