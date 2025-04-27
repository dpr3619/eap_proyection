import pandas as pd
from src.epl_proyection.models.catboost.catboost_feature_engineering import make_features
from src.epl_proyection.models.catboost.catboost_optuna_tuning import catboost_optuna_tuning
from src.epl_proyection.models.catboost.catboost_train_catboost import train_catboost_with_params

def run_catboost_pipeline(df_labor, n_trials = 30):
    """
    Entrena CatBoost para múltiples variables y genera un DataFrame final de resultados.

    Args:
        df_labor (pd.DataFrame): DataFrame base con columnas de fechas, target y exógenas.

    Returns:
        pd.DataFrame: DataFrame final con valores reales + predicciones.
    """

    vars_to_predict = [
        'logdiff_población ocupada',
        'logdiff_agricultura, ganadería, caza, silvicultura y pesca',
        'logdiff_industrias manufactureras',
        'logdiff_formal', 'logdiff_informal'
    ]

    list_df_predictions = []

    for var in vars_to_predict:
        df = df_labor.copy()
        dfsinNa = df.dropna(subset=[var])

        if var == 'logdiff_población ocupada':
            dfsinNa = make_features(dfsinNa, target_column=var, exog_columns=['workdays', 'weekends', 'holidays','negative_crashes'], lags=[1,2,3], rolling_windows=[1,2,3])
            df =  make_features(df, target_column=var, exog_columns=['workdays', 'weekends', 'holidays','negative_crashes'], lags=[1,2,3], rolling_windows=[1,2,3])
            df_train_a = dfsinNa[(dfsinNa['ds'] >= "2001-01-01") & (dfsinNa['ds'] <= "2014-12-01")]
            df_train_b = dfsinNa[(dfsinNa['ds'] >= "2016-01-01") & (dfsinNa['ds'] <= "2023-12-01")]
            df_train = pd.concat([df_train_a, df_train_b], ignore_index=True)
            df_val = dfsinNa[~dfsinNa['ds'].isin(df_train['ds'])].copy()
            feature_columns = ['workdays', 'weekends', 'holidays','negative_crashes','lag_1','lag_2','lag_3',
                                            'rolling_mean_1','rolling_mean_2','rolling_mean_3','month']

        elif var in ['logdiff_agricultura, ganadería, caza, silvicultura y pesca',
                    'logdiff_industrias manufactureras']:
            dfsinNa = make_features(dfsinNa, target_column=var, exog_columns=['workdays', 'weekends', 'holidays','negative_crashes'], lags=[1,2,3], rolling_windows=[1,2,3])
            df = make_features(df, target_column=var, exog_columns=['workdays', 'weekends', 'holidays','negative_crashes'], lags=[1,2,3], rolling_windows=[1,2,3])
            min_date = dfsinNa['ds'].min()
            max_date = dfsinNa['ds'].max() - pd.DateOffset(months=12)
            df_train = dfsinNa[(dfsinNa['ds'] >= min_date) & (dfsinNa['ds'] <= max_date)]
            df_val = dfsinNa[~dfsinNa['ds'].isin(df_train['ds'])].copy()
            feature_columns = ['workdays', 'weekends', 'holidays','negative_crashes','lag_1','lag_2','lag_3',
                                            'rolling_mean_1','rolling_mean_2','rolling_mean_3','month']

        else:
            dfsinNa = make_features(dfsinNa, target_column=var, exog_columns=['workdays', 'weekends', 'holidays','negative_crashes'], lags=[1], rolling_windows=[1])
            df = make_features(df, target_column=var, exog_columns=['workdays', 'weekends', 'holidays','negative_crashes'], lags=[1], rolling_windows=[1])
            min_date = dfsinNa['ds'].min()
            max_date = dfsinNa['ds'].max() - pd.DateOffset(months=5)
            df_train = dfsinNa[(dfsinNa['ds'] >= min_date) & (dfsinNa['ds'] <= max_date)]
            df_val = dfsinNa[~dfsinNa['ds'].isin(df_train['ds'])].copy()
            feature_columns = ['workdays', 'weekends', 'holidays','negative_crashes','lag_1','rolling_mean_1','month']

        study = catboost_optuna_tuning(
            df_train=df_train,
            df_val=df_val,
            target_column=var,
            feature_columns=feature_columns,
            n_trials=n_trials,
            timeout=600
        )
        best_params = study['best_params']

        model_trained = train_catboost_with_params(
            df=df_train,
            target_column=var,
            feature_columns=feature_columns,
            params=best_params
        )

        df_to_know_data = dfsinNa[dfsinNa['ds'] <= "2025-02-01"].copy()
        df_to_know_data[f'pred_{var}'] = model_trained['model'].predict(df_to_know_data[feature_columns])

        df_to_future = df.copy()
        df_to_future = df_to_future[df_to_future['ds'] >= "2025-03-01"]
        df_to_future = pd.concat([df_to_know_data, df_to_future], ignore_index=True)

        if var in ['logdiff_población ocupada', 'logdiff_agricultura, ganadería, caza, silvicultura y pesca', 'logdiff_industrias manufactureras']:
            df_to_future = make_features(df_to_future, target_column=var, exog_columns=['workdays', 'weekends', 'holidays','negative_crashes'], lags=[1,2,3], rolling_windows=[1,2,3])
        else:
            df_to_future = make_features(df_to_future, target_column=var, exog_columns=['workdays', 'weekends', 'holidays','negative_crashes'], lags=[1], rolling_windows=[1])

        future_dates = df_to_future[df_to_future['ds'] >= "2025-03-01"].copy()

        for index, row in future_dates.iterrows():
            features = row[feature_columns].values.reshape(1, -1)
            predicted_value = model_trained['model'].predict(features)

            df_to_future.at[index, f'pred_{var}'] = predicted_value

            if var in ['logdiff_población ocupada', 'logdiff_agricultura, ganadería, caza, silvicultura y pesca', 'logdiff_industrias manufactureras']:
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
            else:
                df_to_future.at[index + 1, 'lag_1'] = predicted_value
                rolling_vals = df_to_future.loc[max(0, index - 1):index, f'pred_{var}'].dropna().values
                if len(rolling_vals) >= 1:
                    df_to_future.at[index + 1, 'rolling_mean_1'] = rolling_vals[-1]
                if len(rolling_vals) >= 2:
                    df_to_future.at[index + 1, 'rolling_mean_2'] = rolling_vals.mean()

        list_df_predictions.append(df_to_future)

    # Ahora unificar todo en un solo DataFrame
    df_final = list_df_predictions[0][['ds']].copy()

    for i, var in enumerate(vars_to_predict):
        df_var = list_df_predictions[i][['ds', var, f'pred_{var}']].copy()
        df_final = df_final.merge(df_var, on='ds', how='left')

    return df_final
