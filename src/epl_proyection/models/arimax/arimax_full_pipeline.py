import pandas as pd
import numpy as np
from src.epl_proyection.models.arimax.arimax_grid_search import grid_search_arimax
from src.epl_proyection.models.arimax.arimax_forecast import train_validate_arimax

def predict_pea_arimax(df_labor):
    """
    Realiza el pipeline completo de predicción de la población en edad de trabajar (PET)
    utilizando ARIMAX, y devuelve el DataFrame actualizado.

    Args:
        df_labor (pd.DataFrame): DataFrame original con variables y fecha 'ds' como índice.

    Returns:
        pd.DataFrame: DataFrame con las columnas 'pred_log_pea' y 'PredPea' añadidas.
    """

    # Definir fechas para PEA
    fechas_pea = {
        'train_end': "2023-12-01",
        'val_start': "2024-01-01",
        'val_end': "2025-01-01",
        'future_start': "2025-02-01",
        'future_end': "2040-12-01"
    }

    # Variables exógenas
    exog_columns_base = ['workdays', 'weekends', 'holidays', 'negative_crashes']

    # Buscar mejor orden ARIMAX
    best_order = grid_search_arimax(
        df_labor,
        target_column='log_población en edad de trabajar (pet)',
        exog_columns=exog_columns_base,
        train_end=fechas_pea['train_end'],
        val_start=fechas_pea['val_start'],
        val_end=fechas_pea['val_end'],
        p_range=(1, 3),
        q_range=(1, 3),
        seasonal_order=(0, 1, 1, 12)
    )

    # Entrenar ARIMAX y hacer predicciones
    result = train_validate_arimax(
        df=df_labor,
        target_column='log_población en edad de trabajar (pet)',
        exog_columns=exog_columns_base,
        train_start='2001-01-01',
        train_end="2024-01-01",
        val_start='2024-02-01',
        val_end="2040-12-01",
        order=best_order,
        seasonal_order=(0, 1, 1, 12)
    )

    # Integrar predicción al DataFrame original
    df_labor_new = df_labor.merge(
        result['forecast'].to_frame(name='pred_log_pea'),
        left_index=True,
        right_index=True,
        how='left'
    )

    # Calcular PredPea (nivel normal)
    df_labor_new['PredPea'] = np.exp(df_labor_new['pred_log_pea'])

    return df_labor_new
