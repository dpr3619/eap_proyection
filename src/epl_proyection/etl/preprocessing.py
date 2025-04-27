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

    return df