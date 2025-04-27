from catboost import CatBoostRegressor
import pandas as pd

def train_catboost_with_params(
    df,
    target_column,
    feature_columns,
    params
):
    """
    Entrena un modelo CatBoost con parámetros dados y devuelve predicciones sobre el mismo set.

    Args:
        df (pd.DataFrame): DataFrame con 'ds', target y features.
        target_column (str): Nombre de la columna objetivo.
        feature_columns (list): Lista de columnas features.
        params (dict): Parámetros para CatBoost (suelen venir de Optuna).

    Returns:
        dict: {
            'model': modelo entrenado,
            'predictions': predicciones sobre df,
            'feature_importances': importancia de features
        }
    """

    X = df[feature_columns]
    y = df[target_column]

    model = CatBoostRegressor(**params)
    model.fit(X, y)

    preds = model.predict(X, verbose = 0)

    feature_importances = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.get_feature_importance()
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    return {
        'model': model,
        'predictions': preds,
        'feature_importances': feature_importances
    }
