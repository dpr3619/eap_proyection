import pandas as pd
import numpy as np
from typing import List, Dict
import os
from src.epl_proyection.etl import read_clean_geih
from src.epl_proyection.etl import read_informal_geih

def generate_labor_data(path_df2:str, sheet_name_df2:str, sector:List[str]) -> pd.DataFrame:
    """
    Function to generate Labor data from national and sectoral data.
    Args:
        path_df1 (str): Path to the national data file.
        sheet_name_df1 (str): Name of the sheet in the national data file.
        path_df2 (str): Path to the sectoral data file.
        sheet_name_df2 (str): Name of the sheet in the sectoral data file.
        sector (list): List of sectors to filter.
    Returns:
        pd.DataFrame: The final processed DataFrame.
    """
    # Read the data
    df1 = read_clean_geih.run_pipeline(target_year = 2040,
                                    sectors=sector)
    df2 = read_informal_geih.run_pipeline(path_df2, sheet_name_df2)

    # Transform the data

    df2.drop(columns = ['Year','Month','Población ocupada'],
            inplace = True)

    # Join the data
    df_final = pd.merge(df1,
                        df2,
                        how = 'left',
                        on = 'YearMonth',
                        suffixes= ('_total','_sector'))
    
    return df_final

def generate_log_and_logdiff(df:pd.DataFrame, cols:List[str]) -> pd.DataFrame:
    """
    Function to generate logarithms of the specified columns in the DataFrame.
    Args:
        df (pd.DataFrame): The input DataFrame.
        cols (list): List of columns to apply the logarithm transformation.
    Returns:
        pd.DataFrame: The DataFrame with logarithm columns added.
    """
    for col in cols:
        new_col = col.strip().lower()
        df[f'log_{new_col}'] = np.log(df[col])
        df[f'logdiff_{new_col}'] = df[f'log_{new_col}'].diff()
    
    return df
## Choques de productividad
def add_pandemic_impact(df:pd.DataFrame) -> pd.DataFrame:
    """
    Function to add the negative impact of the pandemic
    Args: 
        df (pd.DataFrame): The input DataFrame.
    Returns:
        pd.DataFrame: The DataFrame with the added column of the negative impact
    """
    df['negative_crashes'] = np.where(
        (df['YearMonth'] >= '2020-03-01') & (df['YearMonth'] <= '2021-12-01'),
        1,
        0
    )
    df['CommentBinaryVariable'] = np.where(
        (df['YearMonth'] >= '2020-03-01') & (df['YearMonth'] <= '2021-12-01'),
        'Pandemia',
        ''
    )
    return df

def financial_crisis_impact(df:pd.DataFrame) -> pd.DataFrame:
    """
    Function to add the negative impact of the financial crisis
    Args: 
        df (pd.DataFrame): The input DataFrame.
    Returns:
        pd.DataFrame: The DataFrame with the added column of the negative impact
    """
    df['negative_crashes'] = np.where(
        (df['YearMonth'] >= '2007-01-01') & (df['YearMonth'] <= '2008-07-01'),
        1,
        0
    )
    df['CommentBinaryVariable'] = np.where(
        (df['YearMonth'] >= '2007-01-01') & (df['YearMonth'] <= '2007-01-01'),
        'Crisis Financiera',
        ''
    )
    return df




def dynamic_moving_average(df, columns, cutoff='2025-03-01', window=24):
    """
    Calcula un moving average dinámico alimentándose de las nuevas predicciones a partir del cutoff.

    Args:
        df (pd.DataFrame): DataFrame con 'ds' y las columnas de interés.
        columns (list): Columnas para aplicar el promedio móvil.
        cutoff (str): Fecha desde donde se empieza el cálculo dinámico.
        window (int): Número de meses para el promedio.

    Returns:
        pd.DataFrame: DataFrame con nuevas columnas *_ma24.
    """
    df_result = df.copy()
    df_result['ds'] = pd.to_datetime(df_result['ds'])

    for col in columns:
        ma_values = []
        historical_values = list(df_result.loc[df_result['ds'] < pd.to_datetime(cutoff), col].dropna())

        for idx, row in df_result.iterrows():
            current_date = row['ds']

            if current_date < pd.to_datetime(cutoff):
                ma_values.append(np.nan)
            else:
                # Calcular promedio de los últimos 'window' valores
                window_values = historical_values[-window:]
                avg = np.mean(window_values)
                ma_values.append(avg)

                # Actualizar históricos con el nuevo promedio calculado
                historical_values.append(avg)

        df_result[f'{col}_ma24'] = ma_values

    return df_result







def calculate_proportions(df:pd.DataFrame) -> pd.DataFrame:
    """
    Function to calculate the proportions of formal and informal employment.
    Args:
        df (pd.DataFrame): The input DataFrame.
    Returns:
        pd.DataFrame: The DataFrame with the added columns for proportions.
    """
    df['proportion_formal_PET'] = df['Formal'] / df['Población en edad de trabajar (PET)']
    df['proportion_informal_PET'] = df['Informal'] / df['Población en edad de trabajar (PET)']
    df['porportion_aggriculture_Occupied'] = df['Agricultura, ganadería, caza, silvicultura y pesca'] / df['Población ocupada']
    df['proportion_manufacturing_Occupied'] = df['Industrias manufactureras'] / df['Población ocupada']
    
    return df

def run_preprocessing_pipeline(path_df2:str, sheet_name_df2:str, sector:List[str],
                                cols_to_lag:List[str]) -> pd.DataFrame:
    """
    Function to run the preprocessing pipeline.
    Args:
        path_df2 (str): Path to the sectoral data file.
        sheet_name_df2 (str): Name of the sheet in the sectoral data file.
        sector (list): List of sectors to filter.
        cols_to_lag (list): List of columns to apply lagging.
    Returns:
        pd.DataFrame: The final processed DataFrame.
    """
    # Generate labor data
    df = generate_labor_data(path_df2, sheet_name_df2, sector)

    # Generate log and logdiff columns
    df = generate_log_and_logdiff(df, cols_to_lag)

    # Add pandemic impact
    df = add_pandemic_impact(df)

    # Add financial crisis impact
    df = financial_crisis_impact(df)

    # Calculate proportions
    df = calculate_proportions(df)
    df['ds'] = df['YearMonth']

    # Generate moving averages from proportions
    df = dynamic_moving_average(df, ['proportion_formal_PET', 'proportion_informal_PET'], window=36)
    df = dynamic_moving_average(df, ['porportion_aggriculture_Occupied', 'proportion_manufacturing_Occupied'], window=36)

    return df