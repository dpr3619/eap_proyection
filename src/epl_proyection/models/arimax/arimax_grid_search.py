import warnings
import itertools
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from src.epl_proyection.models.arimax.arimax_train_validate_forecast import train_forecast_arimax

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


from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np

def grid_search_arimax(
    df,
    target_column,
    exog_columns,
    train_end,
    val_start,
    val_end,
    p_range=(0,5),
    q_range=(0,5),
    seasonal_order=(0,1,1,12)
):
    """
    Realiza grid search de ARIMAX para encontrar mejor (p,d,q) basado en RMSE validación.

    Args:
        df (pd.DataFrame): DataFrame completo.
        target_column (str): Target variable.
        exog_columns (list): Variables exógenas.
        train_end (str): Fecha fin de entrenamiento.
        val_start (str): Inicio validación.
        val_end (str): Fin validación.
        p_range (tuple): Rango de p.
        q_range (tuple): Rango de q.
        seasonal_order (tuple): Orden estacional (P,D,Q,s).

    Returns:
        tuple: Mejor orden (p,1,q) encontrado.
    """

    best_rmse = np.inf
    best_order = None

    for p in range(p_range[0], p_range[1] + 1):
        for q in range(q_range[0], q_range[1] + 1):
            try:
                # 1. Entrenar y predecir
                result = train_forecast_arimax(
                    df_labor=df,
                    target_column=target_column,
                    exog_columns=exog_columns,
                    train_end=train_end,
                    order=(p,1,q),
                    seasonal_order=seasonal_order
                )

                forecast = result['forecast']

                # 2. Validar solo en fechas de validación
                df_val = df[(df['ds'] >= val_start) & (df['ds'] <= val_end)]
                df_pred_val = forecast[(forecast['ds'] >= val_start) & (forecast['ds'] <= val_end)]

                y_true_val = df_val[target_column].values
                y_pred_val = df_pred_val['forecast'].values

                rmse = sqrt(mean_squared_error(y_true_val, y_pred_val))

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_order = (p,1,q)

            except Exception as e:
                print(f"Falló para orden ({p},1,{q}): {e}")
                continue

    return best_order
