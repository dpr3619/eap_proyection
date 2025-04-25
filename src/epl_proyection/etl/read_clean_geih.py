"""
File for cleaning and transforming the data from the GEIH survey.
Date: 24/04/2025
"""

## Libraries

import pandas as pd
import numpy as np
import os

# CONSTANTS
MONTHSDICT = {'Ene': '01', 'Feb': '02', 'Mar': '03', 'Abr': '04', 'May': '05', 'Jun': '06',
                'Jul': '07', 'Ago': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dic': '12'}
IDCOLS = ['Year', 'Month','MonthNumber', 'YearMonth']

##### CLEANING OF GLOBAL EAP DATA #####

def read_geih_global_data(path: str) -> pd.DataFrame:
    """
    Reads the GEIH raw data from the specified path and return an unprocessed DataFrame.
    Args:
        path (str): Path to the GEIH data file.
    Returns:
        pd.DataFrame: Unprocessed DataFrame containing the GEIH data.
    """
    # Read the data
    df = pd.read_excel('data/anex-GEIH-feb2025.xlsx', 
                sheet_name = 'Total nacional',
                header = 11)
    # Filtering by neccesary columns
    df = df.iloc[0:18].copy()
    return df


def transform_dataframe_v2(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms a raw DataFrame to a more usable format by performing the following steps:
    1. Creates a transposed copy of the input DataFrame to orient the data.
    2. Extracts the first row of the transposed DataFrame to be used as column names.
    3. Assigns these extracted values as the new column names of the DataFrame.
    4. Removes the original first row, which now contains the former column names.
    5. Resets the index of the DataFrame, creating a new default integer index and a 'Year' column from the old index.
    6. Converts the 'Year' column to a numeric data type, with any non-convertible values becoming NaN.
    7. Fills any NaN values in the 'Year' column using the forward fill method, propagating the last valid observation forward.
    8. Converts the 'Year' column to an integer data type.
    9. Removes any columns where all values are missing (NaN).
    10. Renames any column that has a missing value (NaN) as its name to 'Month'.

    Args:
        df (pd.DataFrame): The raw input Pandas DataFrame.

    Returns:
        pd.DataFrame: The transformed Pandas DataFrame.
    """
    new_df = df.copy()
    new_df = new_df.T
    new_columns = new_df.iloc[0]
    new_df.columns = new_columns
    new_df = new_df.iloc[1:].copy()
    new_df.reset_index(inplace=True, names='Year')
    new_df['Year'] = pd.to_numeric(new_df['Year'], errors='coerce')
    new_df['Year'] = new_df['Year'].ffill()
    new_df['Year'] = new_df['Year'].astype(int)
    new_df.dropna(axis=1, how='all', inplace=True)
    new_df.rename(columns={np.nan: 'Month'}, inplace=True)

    return new_df


def process_dataframe_with_year_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Corrects and adds a 'YearMonth' column to the input DataFrame.

    This function performs the following steps:
    1. Checks if a column named 'Month' exists in the DataFrame. If not, it raises a ValueError.
    2. Creates a 'MonthNumber' column by removing any '*' characters from the 'Month' column.
    3. Maps abbreviated Spanish month names in the 'MonthNumber' column to their corresponding two-digit numerical representation.
    4. Creates a 'YearMonth' column by combining the 'Year' column (converted to string) and the 'MonthNumber' column into a datetime object. The format '%Y-%m' is used for parsing.

    Args:
        df (pd.DataFrame): The input Pandas DataFrame, which is expected to have 'Year' and 'Month' columns. The 'Month' column should contain abbreviated Spanish month names (e.g., 'Ene', 'Feb').

    Returns:
        pd.DataFrame: The modified DataFrame with an added 'YearMonth' column.

    Raises:
        ValueError: If the input DataFrame does not contain a column named 'Month'.
    """
    if 'Month' not in df.columns:
        raise ValueError("The DataFrame must contain a column named 'Month'.")

    df['MonthNumber'] = [x.replace('*', '') for x in df['Month']]
    df['MonthNumber'] = df['MonthNumber'].map(MONTHSDICT)
    df['YearMonth'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['MonthNumber'].astype(str), format='%Y-%m')
    return df

def columns_to_numeric(df:pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Converts specified columns in a DataFrame to numeric data types.

    Args:
        df (pd.DataFrame): The input DataFrame.
        columns (list): A list of column names to be converted to numeric.

    Returns:
        pd.DataFrame: The modified DataFrame with specified columns converted to numeric.
    """
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

#### CLEANING OF SECTOR DATA #####

def read_geih_sector_data(path: str) -> pd.DataFrame:
    """
    Reads the GEIH raw data from the specified path and return an unprocessed DataFrame.
    Args:
        path (str): Path to the GEIH data file.
    Returns:
        pd.DataFrame: Unprocessed DataFrame containing the GEIH data.
    """
    # Read the data
    df = pd.read_excel('data/anex-GEIH-feb2025.xlsx', 
                sheet_name = 'Ocupados TN_T13_rama',
                header = 12)
    # Filtering by neccesary columns
    df = df.iloc[0:18].copy()
    return df



## MAIN PIPELINE ##

def run_pipeline():
    """
    Main function to run the data processing pipeline.
    """
    # Read the data
    df = read_geih_global_data('data/anex-GEIH-feb2025.xlsx')
    df_sector = read_geih_sector_data('data/anex-GEIH-feb2025.xlsx')

    # Transform the data
    df = transform_dataframe_v2(df)
    df_sector = transform_dataframe_v2(df_sector)

    # Process the data
    df = process_dataframe_with_year_month(df)
    df_sector = process_dataframe_with_year_month(df_sector)

    # Convert columns to numeric
    cols_to_correct_total = [x for x in df.columns if x not in IDCOLS]
    cols_to_correct_sector = [x for x in df_sector.columns if x not in IDCOLS]

    df = columns_to_numeric(df, cols_to_correct_total)
    df_sector = columns_to_numeric(df_sector, cols_to_correct_sector)
    df_sector.drop(columns = ['Year','Month','MonthNumber'], inplace = True)

    # Join the dataframes
    df_joined = pd.merge(df,
                    df_sector,
                    how = 'left',
                    on = 'YearMonth',
                    suffixes= ('_total','_sector'))

    return df_joined