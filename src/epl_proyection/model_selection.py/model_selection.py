import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from src.epl_proyection.models.catboost.catboost_feature_engineering import make_features
from src.epl_proyection.models.catboost.catboost_optuna_tuning import catboost_optuna_tuning
from src.epl_proyection.models.catboost.catboost_train_catboost import train_catboost_with_params
from src.epl_proyection.models.arimax.arimax_train_validate_forecast import train_validate_forecast_arimax

def train_catboost_and_arimax(
    df,
    target_column,
    exog_columns,
    fechas,
    lags_catboost=[1, 2, 3],
    rolling_windows=[1, 2, 3],
    catboost_trials=30,
    catboost_timeout=600,
    arima_order=(3,1,3),
    arima_seasonal_order=(0,1,1,12)
):
    """
    Entrena CatBoost y ARIMAX para la misma serie y selecciona el mejor modelo basado en RMSE.

    Args:
        df (pd.DataFrame): DataFrame con columna 'ds' y features.
        target_column (str): Variable objetivo.
        exog_columns (list): Variables exógenas.
        fechas (dict): Fechas de corte: train_end, val_start, val_end.

    Returns:
        dict: {
            'best_model': 'catboost' o 'arimax',
            'forecast_val': DataFrame de predicción en validación,
            'rmse_catboost': RMSE en validación CatBoost,
            'rmse_arimax': RMSE en validación ARIMAX
        }
    """

    # --- CatBoost ---
    # 1. Feature Engineering
    df_features = make_features(df, target_column, exog_columns, lags_catboost, rolling_windows)

    feature_columns = [col for col in df_features.columns if col not in ["ds", target_column]]

    # 2. Optuna para mejores hiperparámetros
    catboost_tuning = catboost_optuna_tuning(
        df_train = df_features[df_features['ds'] <= fechas['train_end']],
        df_val = df_features[(df_features['ds'] >= fechas['val_start']) & (df_features['ds'] <= fechas['val_end'])],
        target_column=target_column,
        feature_columns=feature_columns,
        n_trials=catboost_trials,
        timeout=catboost_timeout
    )
    best_params_catboost = catboost_tuning['best_params']

    # 3. Entrenamiento final CatBoost
    catboost_model = train_catboost_with_params(
        df=df_features[df_features['ds'] <= fechas['val_end']],
        target_column=target_column,
        feature_columns=feature_columns,
        params=best_params_catboost
    )

    preds_catboost = catboost_model['predictions']
    y_true_catboost = df_features.loc[df_features['ds'] <= fechas['val_end'], target_column]

    # Extraer solo el rango de validación
    mask_val = (df_features['ds'] >= fechas['val_start']) & (df_features['ds'] <= fechas['val_end'])
    preds_catboost_val = preds_catboost[mask_val]
    y_true_catboost_val = y_true_catboost[mask_val]

    rmse_catboost = sqrt(mean_squared_error(y_true_catboost_val, preds_catboost_val))

    # --- ARIMAX ---
    arimax_result = train_validate_forecast_arimax(
        df=df,
        target_column=target_column,
        exog_columns=exog_columns,
        train_end=fechas['train_end'],
        val_start=fechas['val_start'],
        val_end=fechas['val_end'],
        order=arima_order,
        seasonal_order=arima_seasonal_order
    )

    preds_arimax_val = arimax_result['forecast_val']['forecast'].values
    y_true_arimax_val = df[(df['ds'] >= fechas['val_start']) & (df['ds'] <= fechas['val_end'])][target_column].values

    rmse_arimax = sqrt(mean_squared_error(y_true_arimax_val, preds_arimax_val))

    # --- Selección del mejor modelo ---
    if rmse_catboost <= rmse_arimax:
        best_model = 'catboost'
        forecast_val = df_features.loc[mask_val, ['ds']].copy()
        forecast_val['forecast'] = preds_catboost_val
    else:
        best_model = 'arimax'
        forecast_val = arimax_result['forecast_val']

    return {
        'best_model': best_model,
        'forecast_val': forecast_val,
        'rmse_catboost': rmse_catboost,
        'rmse_arimax': rmse_arimax
    }
