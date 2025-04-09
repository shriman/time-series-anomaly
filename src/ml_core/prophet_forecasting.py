import itertools
import pandas as pd
import numpy as np
from prophet.diagnostics import cross_validation, performance_metrics
from prophet import Prophet
import pickle


def prophet_train(data_path, model_output_path):
    # train model and grid search
    df = pd.read_csv(data_path)
    df.rename(columns={'transaction_date': 'ds', 'spend': 'y'}, inplace=True)

    param_grid = {
        'changepoint_prior_scale': [0.01, 0.1, 0.5, 1.0, 5.0],
        'seasonality_prior_scale': [0.001, 0.01, 0.1, 1.0],
    }

    cutoffs = pd.to_datetime(['2022-10-10', '2023-04-10', '2023-10-10'])

    # Generate all combinations of parameters
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    rmses = []  # Store the RMSEs for each params here

    # Use cross validation to evaluate all parameters
    for params in all_params:
        m = Prophet(**params).fit(df)  # Fit model with given params
        df_cv = cross_validation(m, cutoffs=cutoffs, horizon='30 days', parallel="processes")
        df_p = performance_metrics(df_cv, rolling_window=1)
        rmses.append(df_p['rmse'].values[0])

    best_params = all_params[np.argmin(rmses)]
    print("min rmse:", np.min(rmses), "max rmse:", np.max(rmses))
    print("best params: ", best_params)

    best_model = Prophet(n_changepoints=20, yearly_seasonality=True, changepoint_prior_scale=best_params['changepoint_prior_scale'],
                         seasonality_prior_scale=best_params['seasonality_prior_scale'])

    best_model.fit(df)

    return best_model


def prophet_forecast(model_path, forecast_horizon=30):
    # read prophet model object
    with open(model_path, mode='rb') as f:
        prophet_model = pickle.load(f)

    future_dates = prophet_model.make_future_dataframe(periods=forecast_horizon, freq='D')
    forecast = prophet_model.predict(future_dates)

    return forecast
