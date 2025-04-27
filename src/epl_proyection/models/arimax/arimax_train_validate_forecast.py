import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def train_validate_forecast_arimax_full(
    df,
    target_column,
    exog_columns,
    train_end="2023-12-01",
    val_start="2024-01-01",
    val_end="2025-02-01",
    future_exog=None,
    future_dates=None,
    order=(1,1,1),
    seasonal_order=(0,1,1,12),
    verbose=True
):
    """
    Entrena ARIMAX hasta train_end, valida en val_start–val_end, y proyecta a futuro.

    Args:
        df (pd.DataFrame): DataFrame con 'ds', target, y regresores.
        target_column (str): Columna objetivo.
        exog_columns (list): Columnas de regresores exógenos.
        train_end (str): Fecha de fin de entrenamiento.
        val_start (str): Inicio de validación.
        val_end (str): Fin de validación.
        future_exog (pd.DataFrame): Regresores futuros (2025–2040).
        future_dates (pd.Series): Fechas correspondientes al futuro.
        order (tuple): (p,d,q) para ARIMAX.
        seasonal_order (tuple): (P,D,Q,s) para estacionalidad.
        verbose (bool): Mostrar métricas.

    Returns:
        dict: {
            'model': modelo entrenado,
            'forecast_val': df con 'ds' y 'forecast',
            'forecast_future': df con 'ds' y 'forecast',
            'mae_validation': error en validación
        }
    """

    df = df.copy()
    df = df.rename(columns={target_column: 'y'})

    # 1. Train
    train = df[df['ds'] <= train_end]
    y_train = train['y']
    X_train = train[exog_columns]

    # 2. Validation
    val = df[(df['ds'] >= val_start) & (df['ds'] <= val_end)]
    y_val = val['y']
    X_val = val[exog_columns]

    # 3. Entrenar modelo
    model = SARIMAX(
        endog=y_train,
        exog=X_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    model_fit = model.fit(disp=False)

    # 4. Forecast en validación
    forecast_val = model_fit.predict(
        start=val.index[0],
        end=val.index[-1],
        exog=X_val
    )

    forecast_val_df = pd.DataFrame({
        "ds": val['ds'].values,
        "forecast": forecast_val.values
    })

    mae_val = mean_absolute_error(y_val, forecast_val)
    if verbose:
        print(f"✅ MAE en Validación ({val_start} - {val_end}): {mae_val:.2f}")

    # 5. Forecast futuro
    steps_future = len(future_exog)
    forecast_future = model_fit.forecast(
        steps=steps_future,
        exog=future_exog
    )

    forecast_future_df = pd.DataFrame({
        "ds": future_dates.values,
        "forecast": forecast_future.values
    })

    return {
        'model': model_fit,
        'forecast_val': forecast_val_df,
        'forecast_future': forecast_future_df,
        'mae_validation': mae_val
    }


from statsmodels.tsa.statespace.sarimax import SARIMAX

def train_forecast_arimax(
    df_labor,
    target_column,
    exog_columns,
    train_end,
    order,
    seasonal_order
):
    """
    Entrena ARIMAX simple y predice el futuro.

    Args:
        df_labor (pd.DataFrame): DataFrame completo.
        target_column (str): Variable target (ya en log si es necesario).
        exog_columns (list): Variables exógenas.
        train_end (str): Fecha fin de entrenamiento.
        order (tuple): (p,d,q).
        seasonal_order (tuple): (P,D,Q,s).

    Returns:
        dict: {
            'model': modelo entrenado,
            'forecast': DataFrame de predicciones
        }
    """

    # Split train/future
    df_train = df_labor[df_labor['ds'] <= train_end].copy()
    df_future = df_labor[df_labor['ds'] > train_end].copy()

    X_train = df_train[exog_columns] if exog_columns else None
    y_train = df_train[target_column]

    X_future = df_future[exog_columns] if exog_columns else None

    # Entrenar modelo
    model = SARIMAX(
        endog=y_train,
        exog=X_train,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    # Forecast futuro
    steps = len(df_future)
    forecast_result = model.get_forecast(steps=steps, exog=X_future)
    forecast_mean = forecast_result.predicted_mean

    forecast_df = df_future[['ds']].copy()
    forecast_df['forecast'] = forecast_mean.values

    return {
        'model': model,
        'forecast': forecast_df
    }


import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error

def rolling_one_step_forecast_arimax(
    df,
    target_column,
    exog_columns,
    train_end="2023-12-01",
    val_start="2024-01-01",
    val_end="2025-02-01",
    order=(1,1,1),
    seasonal_order=(0,1,1,12),
    verbose=True
):
    """
    Rolling one-step forecast para validación respetando temporalidad.

    Args:
        df (pd.DataFrame): DataFrame con 'ds', target y regresores.
        target_column (str): Nombre de la variable objetivo.
        exog_columns (list): Regresores externos.
        train_end (str): Fin de entrenamiento inicial.
        val_start (str): Inicio de validación rolling.
        val_end (str): Fin de validación rolling.
        order (tuple): (p,d,q) para ARIMAX.
        seasonal_order (tuple): Estacionalidad.
        verbose (bool): Mostrar métricas.

    Returns:
        dict: {
            'rolling_forecast': DataFrame con 'ds' y 'forecast',
            'mae_validation': MAE final,
            'model': último modelo entrenado
        }
    """

    df = df.copy()
    df = df.rename(columns={target_column: 'y'})

    forecast_results = []

    # Inicialmente entrenar en datos hasta train_end
    current_train = df[df['ds'] <= train_end].copy()

    for current_date in pd.date_range(val_start, val_end, freq='MS'):
        # Preparar entrenamiento actual
        y_train = current_train['y']
        X_train = current_train[exog_columns]

        # Entrenar modelo
        model = SARIMAX(
            endog=y_train,
            exog=X_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )

        model_fit = model.fit(disp=False)

        # Datos exógenos para el siguiente paso (predicción de un mes)
        next_exog = df[df['ds'] == current_date][exog_columns]

        # Forecast 1 paso adelante
        forecast = model_fit.forecast(steps=1, exog=next_exog)

        forecast_results.append({
            "ds": current_date,
            "forecast": forecast.values[0]
        })

        # Ahora agregar el valor real observado a la base de entrenamiento
        next_real = df[df['ds'] == current_date][['ds', 'y'] + exog_columns]
        current_train = pd.concat([current_train, next_real], ignore_index=True)

    rolling_forecast_df = pd.DataFrame(forecast_results)

    # Comparar rolling forecast vs realidad
    real_val = df[(df['ds'] >= val_start) & (df['ds'] <= val_end)][['ds', 'y']].reset_index(drop=True)
    merged = rolling_forecast_df.merge(real_val, on='ds')
    # mae_val = mean_absolute_error(merged['y'], merged['forecast'])

    #if verbose:
    #    print(f"✅ Rolling forecast terminado. MAE: {mae_val:.2f}")

    return {
        'rolling_forecast': rolling_forecast_df,
        'model': model_fit  # Último modelo entrenado
    }
