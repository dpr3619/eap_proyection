import pandas as pd
from src.epl_proyection.models.catboost.catboost_feature_engineering import make_features
from src.epl_proyection.models.catboost.catboost_optuna_tuning import catboost_optuna_tuning
from src.epl_proyection.models.catboost.catboost_train_catboost import train_catboost_with_params

import pandas as pd
import numpy as np
from src.epl_proyection.models.catboost.catboost_feature_engineering import make_features
from src.epl_proyection.models.catboost.catboost_optuna_tuning import catboost_optuna_tuning
from src.epl_proyection.models.catboost.catboost_train_catboost import train_catboost_with_params

TRAINDATES = {
    'logdiff_población en edad de trabajar (pet)': {'train_end': "2023-12-01"}
}

def run_catboost_pipeline(df_labor, n_trials=30):
    """
    Entrena CatBoost solo para 'logdiff_población en edad de trabajar (pet)'.

    Args:
        df_labor (pd.DataFrame): DataFrame base.
        n_trials (int): Número de pruebas en Optuna.

    Returns:
        pd.DataFrame: DataFrame con predicciones.
    """

    var = 'logdiff_población en edad de trabajar (pet)'
    df = df_labor.copy()
    dfsinNa = df.dropna(subset=[var])

    # Feature engineering
    dfsinNa = make_features(dfsinNa, target_column=var, exog_columns=['workdays', 'weekends', 'holidays','negative_crashes'], lags=[1,2,3], rolling_windows=[1,2,3])
    df = make_features(df, target_column=var, exog_columns=['workdays', 'weekends', 'holidays','negative_crashes'], lags=[1,2,3], rolling_windows=[1,2,3])

    df_train = dfsinNa[dfsinNa['ds'] <= TRAINDATES[var]['train_end']]
    df_val = dfsinNa[~dfsinNa['ds'].isin(df_train['ds'])].copy()

    feature_columns = ['workdays', 'weekends', 'holidays', 'negative_crashes', 'lag_1', 'lag_2', 'lag_3', 'rolling_mean_1', 'rolling_mean_2', 'rolling_mean_3', 'month']

    # Optuna tuning
    study = catboost_optuna_tuning(
        df_train=df_train,
        df_val=df_val,
        target_column=var,
        feature_columns=feature_columns,
        n_trials=n_trials,
        timeout=600
    )
    best_params = study['best_params']

    # Entrenar modelo
    model_trained = train_catboost_with_params(
        df=df_train,
        target_column=var,
        feature_columns=feature_columns,
        params=best_params
    )

    # Predicciones sobre datos conocidos
    df_to_know_data = dfsinNa[dfsinNa['ds'] <= "2025-02-01"].copy()
    df_to_know_data[f'pred_{var}'] = model_trained['model'].predict(df_to_know_data[feature_columns])

    # Preparar para predicciones futuras
    df_to_future = df.copy()
    df_to_future = df_to_future[df_to_future['ds'] >= "2025-03-01"]
    df_to_future = pd.concat([df_to_know_data, df_to_future], ignore_index=True)

    df_to_future = make_features(df_to_future, target_column=var, exog_columns=['workdays', 'weekends', 'holidays','negative_crashes'], lags=[1,2,3], rolling_windows=[1,2,3])

    # Predicción iterativa para el futuro
    future_dates = df_to_future[df_to_future['ds'] >= "2025-03-01"].copy()

    for index, row in future_dates.iterrows():
        features = row[feature_columns].values.reshape(1, -1)
        predicted_value = model_trained['model'].predict(features)
        df_to_future.at[index, f'pred_{var}'] = predicted_value

        # Actualizar lags y rolling means
        df_to_future.at[index + 1, 'lag_1'] = predicted_value
        if index + 2 in df_to_future.index:
            df_to_future.at[index + 2, 'lag_2'] = predicted_value
        if index + 3 in df_to_future.index:
            df_to_future.at[index + 3, 'lag_3'] = predicted_value

        rolling_vals = df_to_future.loc[max(0, index - 2):index, f'pred_{var}'].dropna().values
        if len(rolling_vals) >= 1:
            df_to_future.at[index + 1, 'rolling_mean_1'] = rolling_vals[-1]
        if len(rolling_vals) >= 2:
            df_to_future.at[index + 1, 'rolling_mean_2'] = rolling_vals[-2:].mean()
        if len(rolling_vals) >= 3:
            df_to_future.at[index + 1, 'rolling_mean_3'] = rolling_vals.mean()

    # Solo regresar las columnas necesarias
    df_final = df_to_future[['ds', var, f'pred_{var}']]
    df_final.dropna(subset=['pred_logdiff_población en edad de trabajar (pet)'], inplace=True)
    df_final = df_final[['ds', 'pred_logdiff_población en edad de trabajar (pet)']].copy()
    df_final = df_labor.merge(df_final, on='ds', how='left')

    for index,row in df_final.iterrows():
        df_final.at[index, 'pred_log_población en edad de trabajar (pet)'] = df_final.at[index, 'log_población en edad de trabajar (pet)']
        if index == 0:
            continue
        else:
            df_final.at[index, 'pred_log_población en edad de trabajar (pet)'] = df_final.at[index-1, 'pred_log_población en edad de trabajar (pet)'] + row['pred_logdiff_población en edad de trabajar (pet)']
            df_final.at[index, 'pred_pea_catboost'] = np.exp(df_final.at[index, 'pred_log_población en edad de trabajar (pet)'] )

    return df_final

