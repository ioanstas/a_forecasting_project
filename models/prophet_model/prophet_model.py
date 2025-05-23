from prophet import Prophet
import pandas as pd


def train_prophet_model(df):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(df)
    return model


def do_forecast(model, test_df) -> pd.DataFrame:
    ts_forecast = model.predict(test_df)
    return ts_forecast
