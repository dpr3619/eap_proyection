import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def adf_test(series):
    """
    Corre el Augmented Dickey-Fuller Test para revisar estacionariedad.

    Args:
        series (pd.Series): Serie temporal.

    Returns:
        dict: {'adf_statistic': valor, 'p_value': valor, 'is_stationary': bool}
    """
    result = adfuller(series.dropna())
    adf_statistic = result[0]
    p_value = result[1]
    is_stationary = p_value <= 0.05

    print(f"ADF Statistic: {adf_statistic:.4f}")
    print(f"p-value: {p_value:.4f}")
    if is_stationary:
        print("✅ La serie es estacionaria.")
    else:
        print("⚠️ La serie NO es estacionaria (se sugiere diferenciación).")
    
    return {
        'adf_statistic': adf_statistic,
        'p_value': p_value,
        'is_stationary': is_stationary
    }

def plot_acf_pacf(series, lags=40, title_suffix=""):
    """
    Grafica el ACF y PACF de una serie temporal.

    Args:
        series (pd.Series): Serie temporal.
        lags (int): Número de rezagos a mostrar.
        title_suffix (str): Texto adicional para los títulos.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16,4))
    plot_acf(series.dropna(), ax=axes[0], lags=lags)
    axes[0].set_title(f"ACF {title_suffix}")
    plot_pacf(series.dropna(), ax=axes[1], lags=lags)
    axes[1].set_title(f"PACF {title_suffix}")
    plt.show()

def full_ts_diagnostics(series, lags=40):
    """
    Corre prueba ADF + grafica ACF y PACF, sugiriendo diferenciación si es necesario.

    Args:
        series (pd.Series): Serie temporal.
        lags (int): Número de rezagos a graficar.
    """
    print("🔎 Diagnóstico de la serie original:")
    result = adf_test(series)

    plot_acf_pacf(series, lags=lags, title_suffix="(Serie original)")

    if not result['is_stationary']:
        print("\n🔄 Aplicando primera diferenciación...")
        diff_series = series.diff()
        print("\n🔎 Diagnóstico de la serie diferenciada:")
        adf_test(diff_series)

        plot_acf_pacf(diff_series, lags=lags, title_suffix="(Serie diferenciada)")

