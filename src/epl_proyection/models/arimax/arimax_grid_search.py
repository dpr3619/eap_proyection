import warnings
import itertools
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

warnings.filterwarnings("ignore")

def arimax_grid_search_cv(
    df,
    ds_column,
    target_column,
    exog_columns,
    train_start,
    train_end,
    val_start,
    val_end,
    p_range=(0, 3),
    d_range=(0, 1),
    q_range=(0, 3),
    seasonal_order=(0, 0, 0, 0),
    verbose=True
):
    """
    Realiza grid search de ARIMAX usando error en validación (MAE) como criterio.

    Args:
        df (pd.DataFrame): DataFrame con 'ds', target y regresores.
        target_column (str): Nombre de la columna objetivo.
        exog_columns (list): Lista de columnas regresoras.
        train_start (str): Inicio de entrenamiento (YYYY-MM-DD).
        train_end (str): Fin de entrenamiento (YYYY-MM-DD).
        val_start (str): Inicio de validación (YYYY-MM-DD).
        val_end (str): Fin de validación (YYYY-MM-DD).
        p_range, d_range, q_range (tuples): Rango de hiperparámetros.
        seasonal_order (tuple): Orden estacional.
        verbose (bool): Mostrar progreso.

    Returns:
        dict: {'best_order': (p,d,q), 'best_mae': valor, 'model': fitted model}
    """

    df = df.copy()
    df = df.rename(columns={target_column: 'y'})
    df = df.rename(columns={ds_column: 'ds'})

    # Definir conjuntos
    train = df[(df['ds'] >= train_start) & (df['ds'] <= train_end)]
    val = df[(df['ds'] >= val_start) & (df['ds'] <= val_end)]

    y_train = train['y']
    X_train = train[exog_columns]
    y_val = val['y']
    X_val = val[exog_columns]

    p_values = range(p_range[0], p_range[1]+1)
    d_values = range(d_range[0], d_range[1]+1)
    q_values = range(q_range[0], q_range[1]+1)

    best_mae = float("inf")
    best_order = None
    best_model = None

    for order in itertools.product(p_values, d_values, q_values):
        try:
            model = SARIMAX(
                endog=y_train,
                exog=X_train,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            model_fit = model.fit(disp=False)

            # Forecast para el horizonte de validación
            pred = model_fit.predict(
                start=val.index[0],
                end=val.index[-1],
                exog=X_val
            )

            # Evaluación
            mape = mean_absolute_percentage_error(y_val, pred)

            if verbose:
                print(f"Probing ARIMAX{order} - MAE Validation: {mape:.2f}")

            if mape < best_mae:
                best_mae = mape
                best_order = order
                best_model = model_fit

        except Exception as e:
            if verbose:
                print(f"Model ARIMAX{order} failed. Reason: {e}")

    print("\n🏆 Best model found:")
    print(f"Order: {best_order} - MAE Validation: {best_mae:.2f}")

    return {
        'best_order': best_order,
        'best_mae': best_mae,
        'model': best_model
    }
