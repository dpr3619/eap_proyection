import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from src.epl_proyection.models.arimax.arimax_train_validate_forecast import train_forecast_arimax
from src.epl_proyection.models.arimax.arimax_grid_search import grid_search_arimax
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd


def arimax_full_pipeline(
    df_labor,
    target_column,
    exog_columns,
    fechas,
    p_range=(0,5),
    q_range=(0,5),
    seasonal_order=(0,1,1,12)
):
    """
    Pipeline completo: grid search + entrenar final + forecast futuro.
    
    Args:
        df_labor (pd.DataFrame): DataFrame completo con ds, target y exógenas.
        target_column (str): Target variable (ya en log).
        exog_columns (list): Variables exógenas.
        fechas (dict): Diccionario con train_end, val_start, val_end, future_start, future_end.
        p_range (tuple): Rango de p en grid search.
        q_range (tuple): Rango de q en grid search.
        seasonal_order (tuple): Orden estacional.
        
    Returns:
        dict: {
            'model': modelo entrenado,
            'order': mejor orden,
            'forecast': predicciones de 2024–2040,
            'rmse_val': RMSE validación
        }
    """

    # 1. Grid search para (p,q)
    best_order = grid_search_arimax(
        df=df_labor,
        target_column=target_column,
        exog_columns=exog_columns,
        train_end=fechas['train_end'],
        val_start=fechas['val_start'],
        val_end=fechas['val_end'],
        p_range=p_range,
        q_range=q_range,
        seasonal_order=seasonal_order
    )

    print(f"Mejor orden ARIMAX encontrado: {best_order}")

    # 2. Entrenar modelo final y forecast
    result_forecast = train_forecast_arimax(
        df_labor=df_labor,
        target_column=target_column,
        exog_columns=exog_columns,
        train_end=fechas['train_end'],
        order=best_order,
        seasonal_order=seasonal_order
    )

    # 3. Calcular RMSE en validación
    df_val = df_labor[(df_labor['ds'] >= fechas['val_start']) & (df_labor['ds'] <= fechas['val_end'])]
    df_pred_val = result_forecast['forecast'][(result_forecast['forecast']['ds'] >= fechas['val_start']) & (result_forecast['forecast']['ds'] <= fechas['val_end'])]

    y_true_val = df_val[target_column].values
    y_pred_val = df_pred_val['forecast'].values

    rmse_val = sqrt(mean_squared_error(y_true_val, y_pred_val))

    return {
        'model': result_forecast['model'],
        'order': best_order,
        'forecast': result_forecast['forecast'],
        'rmse_val': rmse_val
    }



