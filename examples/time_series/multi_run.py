import numpy as np
import pandas as pd
from pylab import rcParams
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Additional custom functions
from cases.industrial.processing import multi_automl_fit_forecast, plot_results, prepare_multimodal_data

if __name__ == '__main__':
    forecast_length = 2
    df = pd.read_csv('multi_test.csv', parse_dates=['datetime'])

    # Wrap time series data into InputData class
    features_to_use = ['predictor_1', 'predictor_2']
    ts = np.array(df['target'])
    mm_train, mm_test, = prepare_multimodal_data(dataframe=df,
                                                 features=features_to_use,
                                                 forecast_length=forecast_length)

    composer_params = {'max_depth': 6,
                       'max_arity': 3,
                       'pop_size': 20,
                       'num_of_generations': 20,
                       'timeout': 0.5,
                       'preset': 'light',
                       'metric': 'rmse',
                       'cv_folds': None,
                       'validation_blocks': None}
    forecast, obtained_pipeline = multi_automl_fit_forecast(mm_train, mm_test,
                                                            composer_params,
                                                            ts, forecast_length,
                                                            vis=True)
