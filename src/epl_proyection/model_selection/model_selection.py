import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from src.epl_proyection.models.arimax.arimax_full_pipeline import arimax_full_pipeline
from src.epl_proyection.models.catboost.catboost_main_pipeline import run_catboost_pipeline
import pandas as pd
from sklearn.metrics import mean_absolute_error
from src.epl_proyection.models.catboost.catboost_main_pipeline import TRAINDATES

fechas_dict = {
    'log_población en edad de trabajar (pet)': {
        'train_end': "2023-12-01",
        'val_start': "2024-01-01",
        'val_end': "2025-02-01",
        'future_start': "2025-03-01",
        'future_end': "2040-12-01"
    },
        'log_población ocupada': {
        'train_end': "2023-12-01",
        'val_start': "2024-01-01",
        'val_end': "2025-02-01",
        'future_start': "2025-03-01",
        'future_end': "2040-12-01"
    },
    'log_agricultura, ganadería, caza, silvicultura y pesca': {
        'train_end': "2023-02-01",  # entrenar hasta 1 año antes para dejar 12 meses de validación
        'val_start': "2023-03-01",
        'val_end': "2024-02-01",
        'future_start': "2024-03-01",
        'future_end': "2040-12-01"
    },
    'log_industrias manufactureras': {
        'train_end': "2023-02-01",
        'val_start': "2023-03-01",
        'val_end': "2024-02-01",
        'future_start': "2024-03-01",
        'future_end': "2040-12-01"
    },
    'log_formales': {
        'train_end': "2024-09-01",  # 5 meses de validación
        'val_start': "2024-10-01",
        'val_end': "2025-02-01",
        'future_start': "2025-03-01",
        'future_end': "2040-12-01"
    },
    'log_informales': {
        'train_end': "2024-09-01",
        'val_start': "2024-10-01",
        'val_end': "2025-02-01",
        'future_start': "2025-03-01",
        'future_end': "2040-12-01"
    }
}



import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from src.epl_proyection.models.arimax.arimax_full_pipeline import arimax_full_pipeline
from src.epl_proyection.models.catboost.catboost_main_pipeline import run_catboost_pipeline


def calculate_rmse(y_true, y_pred):
    return sqrt(mean_squared_error(y_true, y_pred))

def calculate_mae(y_true, y_pred):
    return mean_absolute_error(y_true, y_pred)

def generate_evaluation_table(df_labor):
    """
    Genera tabla de evaluación comparando CatBoost vs ARIMAX y crea DF de predicciones futuras.

    Args:
        df_labor (pd.DataFrame): DataFrame base.

    Returns:
        tuple: (evaluation_table, df_final_predictions)
    """

    # 1. Corre CatBoost
    df_catboost_trained = run_catboost_pipeline(df_labor, n_trials=30)

    results = []
    future_predictions = {}

    vars_to_predict = list(fechas_dict.keys())

    for var in vars_to_predict:
        print(f"\nEvaluando variable: {var}")

        fechas = fechas_dict[var]

        # --- ARIMAX ---
        result_arimax = arimax_full_pipeline(
            df_labor=df_labor,
            target_column=var,
            exog_columns=['workdays', 'weekends', 'holidays', 'negative_crashes'],
            fechas=fechas,
            p_range=(0,5),
            q_range=(0,5),
            seasonal_order=(0,1,1,12)
        )

        forecast_future_arimax = result_arimax['forecast']

        # --- CatBoost ---
        forecast_future_catboost = df_catboost_trained[['ds', f'pred_{var}']].copy()

        # --- Valores reales para validación ---
        mask_val = (df_labor['ds'] >= fechas['val_start']) & (df_labor['ds'] <= fechas['val_end'])
        y_true = df_labor.loc[mask_val, var].values

        # --- Predicciones para validación ---
        y_pred_arimax = result_arimax['forecast'].loc[
            (result_arimax['forecast']['ds'] >= fechas['val_start']) & (result_arimax['forecast']['ds'] <= fechas['val_end']),
            'forecast'
        ].values

        y_pred_catboost = df_catboost_trained.loc[
            (df_catboost_trained['ds'] >= fechas['val_start']) & (df_catboost_trained['ds'] <= fechas['val_end']),
            f'pred_{var}'
        ].values

        # --- Calcular métricas ---
        rmse_arimax = calculate_rmse(y_true, y_pred_arimax)
        mae_arimax = calculate_mae(y_true, y_pred_arimax)

        rmse_catboost = calculate_rmse(y_true, y_pred_catboost)
        mae_catboost = calculate_mae(y_true, y_pred_catboost)

        # --- Decidir mejor modelo ---
        if rmse_arimax <= rmse_catboost:
            best_model = 'ARIMAX'
            pred_future = forecast_future_arimax[['ds', 'forecast']].rename(columns={'forecast': f'forecast_{var}'})
        else:
            best_model = 'CatBoost'
            pred_future = forecast_future_catboost.rename(columns={f'pred_{var}': f'forecast_{var}'})

        # --- Guardar evaluación ---
        results.append({
            'Variable': var,
            'RMSE_ARIMAX': rmse_arimax,
            'MAE_ARIMAX': mae_arimax,
            'RMSE_CatBoost': rmse_catboost,
            'MAE_CatBoost': mae_catboost,
            'Mejor Modelo': best_model
        })

        # --- Guardar predicción futura ---
        future_predictions[var] = pred_future

    # 2. Convertir resultados en tabla
    evaluation_table = pd.DataFrame(results)

    # 3. Armar df_final_predictions
    # Merge todas las predicciones sobre la base de fechas
    df_final_predictions = None

    for var, pred_df in future_predictions.items():
        if df_final_predictions is None:
                df_final_predictions = pred_df.copy()
        else:
            df_final_predictions = df_final_predictions.merge(pred_df, on='ds', how='left')

        return evaluation_table, df_final_predictions


