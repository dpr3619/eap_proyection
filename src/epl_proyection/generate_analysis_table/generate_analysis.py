from src.epl_proyection.models.arimax.arimax_full_pipeline import predict_pea_arimax
from src.epl_proyection.models.catboost.catboost_main_pipeline import run_catboost_pipeline
from src.epl_proyection.etl import preprocessing
import pandas as pd

def generate_analysis_table(path:str, 
                            sheet_name:str, 
                            sector:str, 
                            cols_to_lag: list) -> pd.DataFrame:
    """
    Function to generate the analysis table for the labor market.
    
    Args:
        path (str): Path to the Excel file.
        sheet_name (str): Name of the sheet in the Excel file.
        sector (str): Sector to filter.
    
    Returns:
        pd.DataFrame: The final analysis table.
    """
    
    # Generate labor data
    df_labor = preprocessing.run_preprocessing_pipeline(path_df2 = path,
                                                        sheet_name_df2 = sheet_name,
                                                        sector = sector,
                                                        cols_to_lag=cols_to_lag)
    
    # Generate log and logdiff columns
    df_labor = predict_pea_arimax(df_labor)

    
    # Add pandemic impact
    df_labor = run_catboost_pipeline(df_labor, n_trials = 30)

    # Rename some columns
    df_labor.rename(columns={'PredPea': 'PredPETArimax',
                        'pred_pea_catboost':'PredPETCatboost'}, inplace=True)
    
    # Calculate Prediction Proportions
    # Informales
    df_labor['PredInformalArimax'] = df_labor['PredPETArimax'] * df_labor['proportion_informal_PET']
    df_labor['PredInformalCatboost'] = df_labor['PredPETCatboost'] * df_labor['proportion_informal_PET']
    # Formales
    df_labor['PredFormalArimax'] = df_labor['PredPETArimax'] * df_labor['proportion_formal_PET_ma24']
    df_labor['PredFormalCatboost'] = df_labor['PredPETCatboost'] * df_labor['proportion_formal_PET_ma24']
    # Ocupados
    df_labor['PredOcupadosArimax'] = df_labor['PredInformalArimax'] + df_labor['PredFormalArimax']
    df_labor['PredOcupadosCatboost'] = df_labor['PredInformalCatboost'] + df_labor['PredFormalCatboost']
    # Sectores
    df_labor['PredAgriculturaArimax'] = df_labor['PredOcupadosArimax'] * df_labor['porportion_aggriculture_Occupied_ma24']
    df_labor['PredAgriculturaCatboost'] = df_labor['PredOcupadosCatboost'] * df_labor['porportion_aggriculture_Occupied_ma24']
    df_labor['PredManufacturaArimax'] = df_labor['PredOcupadosArimax'] * df_labor['proportion_manufacturing_Occupied_ma24']
    df_labor['PredManufacturaCatboost'] = df_labor['PredOcupadosCatboost'] * df_labor['proportion_manufacturing_Occupied_ma24']
    
    return df_labor