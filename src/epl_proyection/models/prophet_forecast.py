from prophet import Prophet
import pandas as pd

def train_prophet_model(df, target_column, holidays_df=None, changepoints=None):
    """
    Entrena un modelo Prophet para datos mensuales.

    Args:
        df (pd.DataFrame): DataFrame con columnas 'ds' (datetime) y target_column.
        target_column (str): Variable a predecir.
        holidays_df (pd.DataFrame, optional): Feriados o choques especiales.
        changepoints (list, optional): Fechas de cambios de tendencia esperados.

    Returns:
        model: Modelo Prophet entrenado
    """
    prophet_df = df.rename(columns={target_column: 'y'})[['ds', 'y']]

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        holidays=holidays_df,
        changepoints=changepoints
    )

    model.fit(prophet_df)
    return model

def make_forecast(model, periods, freq='M'):
    """
    Realiza un forecast para series mensuales.

    Args:
        model: Modelo Prophet entrenado
        periods (int): Número de meses a predecir
        freq (str): 'M' para mensual

    Returns:
        pd.DataFrame: Forecast
    """
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast
