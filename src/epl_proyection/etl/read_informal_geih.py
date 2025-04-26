"""
File for cleaning and transforming the data from the GEIH survey.
Date: 25/04/2025
"""

## Libraries

import pandas as pd
import numpy as np
import os


## Read informal GEIH data

def read_geih_informal_data(path: str) -> pd.DataFrame:
    """
    Reads the GEIH raw data from the specified path and return an unprocessed DataFrame.
    Args:
        path (str): Path to the GEIH data file.
    Returns:
        pd.DataFrame: Unprocessed DataFrame containing the GEIH data.
    """
    # Read the data
    df = pd.read_excel(path, 
                sheet_name = 'Grandes dominios ',
                header = 11)
    # Filtering by neccesary columns
    df = df.iloc[0:4].copy()
    return df