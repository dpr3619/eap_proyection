"""
File for cleaning and transforming the data from the GEIH survey.
Date: 25/04/2025
"""

## Libraries

import pandas as pd
import numpy as np
import os

MONTHSDICT = {'Ene': '01', 'Feb': '02', 'Mar': '03', 'Abr': '04', 'May': '05', 'Jun': '06',
                'Jul': '07', 'Ago': '08', 'Sep': '09', 'Oct': '10', 'Nov': '11', 'Dic': '12'}
IDCOLS = ['Year', 'Month','MonthNumber', 'YearMonth']
IDCOLSSECTOR = ['Year','Quarter']
## Read informal GEIH data from 2021 and above ##

def read_geih_informal_data(path: str, sheet_name:str) -> pd.DataFrame:
    """
    Reads the GEIH raw data from the specified path and return an unprocessed DataFrame.
    Args:
        path (str): Path to the GEIH data file.
    Returns:
        pd.DataFrame: Unprocessed DataFrame containing the GEIH data.
    """
    # Read the data
    df = pd.read_excel(io = path, 
                sheet_name = sheet_name,
                header = 10)
    # Filtering by neccesary columns
    df = df.iloc[0:5].copy()
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
    new_df.reset_index(inplace=True, drop = True)
    new_df = new_df.rename_axis(None, axis=1)
    new_df = new_df.rename(columns = {'Concepto':'Year','Total 13 áreas':'Month'})
    new_df['Year'] = pd.to_numeric(new_df['Year'], errors='coerce')
    new_df['Year'] = new_df['Year'].ffill()
    new_df['Year'] = new_df['Year'].astype(int)

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


# GET INFORMAL DATA BY SECTOR #

def read_geih_informal_data_sector(path: str, sheet_name:str) -> pd.DataFrame:
    """
    Reads the GEIH raw data from the specified path and return an unprocessed DataFrame.
    Args:
        path (str): Path to the GEIH data file.
    Returns:
        pd.DataFrame: Unprocessed DataFrame containing the GEIH data.
    """
    # Read the data
    df = pd.read_excel(io = path, 
                sheet_name = sheet_name,
                header = 10)
    # Filtering by neccesary columns
    df = df.iloc[0:47].copy()
    return df

def transform_dataframe_v2_sector(df: pd.DataFrame) -> pd.DataFrame:
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
    new_df.reset_index(inplace=True, drop = True)
    new_df = new_df.rename_axis(None, axis=1)
    new_df = new_df.rename(columns = {'Concepto':'Year',np.nan:'Quarter'})
    new_df['Year'] = pd.to_numeric(new_df['Year'], errors='coerce')
    new_df['Year'] = new_df['Year'].ffill()
    new_df['Year'] = new_df['Year'].astype(int)

    return new_df

def rename_and_filter_columns(df:pd.DataFrame, sector:list) -> pd.DataFrame:
    """
    Renames and filters columns in the DataFrame to keep only relevant ones.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The modified DataFrame with renamed and filtered columns.
    """
    # Loop over columns list and rename them
    new_cols_list = []
    cols_list = df.columns.tolist()
    index_pob_ocupada = cols_list.index('Población ocupada')
    index_formal = cols_list.index('Formal')
    index_informal = cols_list.index('Informal')
    for col in range(len(cols_list)):
        if col <= index_formal:
            new_cols_list.append(cols_list[col])
        elif col > index_formal and col < index_informal:
            new_col = 'Formal_' + cols_list[col]
            new_cols_list.append(new_col)
        elif col == index_informal:
            new_cols_list.append(cols_list[col])
        else:
            new_col = 'Informal_' + cols_list[col]
            new_cols_list.append(new_col)
    
    df.columns = new_cols_list

    ## Filter by specified sectors
    renamed_sector_informal = []
    renamed_sector_formal = []
    for sect in sector:
        new_sect_formal =  'Formal_' + sect
        new_sect_informal = 'Informal_' + sect
        renamed_sector_formal.append(new_sect_formal)
        renamed_sector_informal.append(new_sect_informal)
    
    df = df[['Year', 'Quarter', 'Población ocupada','Formal','Informal'] + sector + renamed_sector_formal + renamed_sector_informal].copy()


    return df

def run_pipeline(path: str, sheet_name:str) -> pd.DataFrame:
    """
    Main function to run the data processing pipeline.

    Args:
        path (str): Path to the GEIH data file.
        sheet_name (str): Name of the sheet in the Excel file.
        sector (list): List of sectors to filter.

    Returns:
        pd.DataFrame: The final processed DataFrame.
    """
    # Informal and Formal labor data
    df = read_geih_informal_data(path, sheet_name)
    df = transform_dataframe_v2(df)
    df = process_dataframe_with_year_month(df)
    df = columns_to_numeric(df, ['Población ocupada','Formal','Informal'])
    
    return df

def run_pipeline_sector(path: str, sheet_name:str, sector:list) -> pd.DataFrame:
    """
    Main function to run the data processing pipeline for sector data.

    Args:
        path (str): Path to the GEIH data file.
        sheet_name (str): Name of the sheet in the Excel file.
        sector (list): List of sectors to filter.

    Returns:
        pd.DataFrame: The final processed DataFrame.
    """
    # Informal and Formal labor data
    df = read_geih_informal_data_sector(path, sheet_name)
    df = transform_dataframe_v2_sector(df)
    df = rename_and_filter_columns(df, sector)
    cols_to_change = [x for x in df.columns if x not in IDCOLSSECTOR]
    df = columns_to_numeric(df, columns = cols_to_change)
    df['Year'] = df['Year'].astype(int)
    df = columns_to_numeric(df, ['Población ocupada','Formal','Informal'])
    
    return df
