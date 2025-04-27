import pandas as pd

def make_features(df, target_column, exog_columns=[], lags=[1,3,6,12], rolling_windows=[3,6]):
    """
    Agrega lags, rolling means y features de calendario a un dataframe.

    Args:
        df (pd.DataFrame): DataFrame original con 'ds' y target.
        target_column (str): Nombre de la variable target.
        exog_columns (list): Lista de variables exógenas.
        lags (list): Lags del target a generar.
        rolling_windows (list): Tamaños de ventana para medias móviles.

    Returns:
        pd.DataFrame: DataFrame con nuevas features.
    """

    df = df.copy()

    # Crear lags
    for lag in lags:
        df[f"lag_{lag}"] = df[target_column].shift(lag)

    # Crear rolling means
    for window in rolling_windows:
        df[f"rolling_mean_{window}"] = df[target_column].shift(1).rolling(window=window).mean()

    # Calendario
    df["year"] = df['ds'].dt.year
    df["month"] = df['ds'].dt.month

    # Variables exógenas ya vienen en el df si están.

    return df
