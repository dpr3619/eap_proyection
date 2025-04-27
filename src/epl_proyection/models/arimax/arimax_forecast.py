import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

def train_validate_arimax(
    df,
    target_column,
    exog_columns,
    train_start,
    train_end,
    val_start,
    val_end,
    order=(1, 0, 0),
    seasonal_order=(0, 0, 0, 0)
):
    """
    Entrena y valida un modelo ARIMAX.

    Args:
        df (pd.DataFrame): DataFrame con 'ds', target y regresores exógenos.
        target_column (str): Columna objetivo (y).
        exog_columns (list): Lista de columnas regresoras.
        train_start (str): Inicio del entrenamiento (YYYY-MM-DD).
        train_end (str): Fin del entrenamiento (YYYY-MM-DD).
        val_start (str): Inicio de la validación (YYYY-MM-DD).
        val_end (str): Fin de la validación (YYYY-MM-DD).
        order (tuple): (p,d,q) para el ARIMAX.
        seasonal_order (tuple): (P,D,Q,s) para estacionalidad (por defecto sin estacionalidad).

    Returns:
        dict: {
            'model': modelo SARIMAX entrenado,
            'forecast': predicciones,
            'metrics': {'mae': valor, 'mse': valor}
        }
    """

    df = df.copy()
    df = df.rename(columns={target_column: 'y'})

    # 1. Definir conjuntos de entrenamiento y validación
    train = df[(df['ds'] >= train_start) & (df['ds'] <= train_end)]
    val = df[(df['ds'] >= val_start) & (df['ds'] <= val_end)]

    # 2. Extraer X e y
    y_train = train['y']
    X_train = train[exog_columns]
    y_val = val['y']
    X_val = val[exog_columns]

    # 3. Definir y entrenar el modelo
    model = SARIMAX(
        endog=y_train,
        exog=X_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    model_fit = model.fit(disp=False)

    # 4. Forecast
    forecast = model_fit.predict(
        start=val.index[0],
        end=val.index[-1],
        exog=X_val
    )

    # 5. Evaluar
    mae = mean_absolute_error(y_val, forecast)
    mse = mean_squared_error(y_val, forecast)
    mape = mean_absolute_percentage_error(y_val, forecast)

    return {
        'model': model_fit,
        'forecast': forecast,
        'metrics': {'mae': mae, 'mse': mse, 'mape': mape}
    }
