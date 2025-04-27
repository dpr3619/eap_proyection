import optuna
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

def catboost_optuna_tuning(
    df_train,
    df_val,
    target_column,
    feature_columns,
    n_trials=30,
    timeout=600
):
    """
    Optimiza hiperparámetros de CatBoost usando Optuna.

    Args:
        df (pd.DataFrame): DataFrame con 'ds', target y features.
        target_column (str): Variable objetivo.
        feature_columns (list): Features de entrada.
        train_end (str): Fin de entrenamiento inicial.
        val_start (str): Inicio de validación.
        val_end (str): Fin de validación.
        n_trials (int): Número de pruebas de Optuna.
        timeout (int): Tiempo máximo (segundos).

    Returns:
        dict: {
            'best_params': parámetros óptimos,
            'study': objeto Optuna completo
        }
    """

    X_train = df_train[feature_columns]
    y_train = df_train[target_column]
    X_val = df_val[feature_columns]
    y_val = df_val[target_column]

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 300, 1000),
            "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_loguniform("l2_leaf_reg", 1e-3, 10),
            "bagging_temperature": trial.suggest_uniform("bagging_temperature", 0, 1),
            "random_strength": trial.suggest_uniform("random_strength", 0, 1),
            "loss_function": "MAE",
            "verbose": 0
        }

        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        mape = mean_absolute_percentage_error(y_val, preds)

        return mape  # Queremos minimizar el MAPE

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    return {
        'best_params': study.best_params,
        'study': study
    }
