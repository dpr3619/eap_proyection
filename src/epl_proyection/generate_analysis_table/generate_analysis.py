from src.epl_proyection.models.arimax.arimax_full_pipeline import predict_pea_arimax
from src.epl_proyection.models.catboost.catboost_main_pipeline import run_catboost_pipeline
from src.epl_proyection.etl import preprocessing
from src.epl_proyection.etl.read_informal_geih import run_pipeline_sector
import pandas as pd

def generate_analysis_table(path_national:str,
                            sheet_name_national:str,
                            path_sector:str,
                            sheet_name_sector:str,
                            path_formal_informal:str,
                            sheet_name_formal_informal:str, 
                            path_activity_formal:str,
                            sheet_name_activity_formal:str,
                            sector:str, 
                            cols_to_lag: list,) -> pd.DataFrame:
    """
    Function to generate the analysis table for the labor market.
    
    Args:
        path_national:str Path For National Data
        sheet_name_national:str Sheet Name for National DaTA
        path_sector:str  Path for economic sector data employement, usually the same as National data
        sheet_name_sector:str Sheet name for economic sector data employement
        path_formal_informal:str Path for formal and informal data
        sheet_name_formal_informal:str Sheet name for formal and informal data 
        path_activity_formal:str Path for information about informal and formal employment data usually the same as forma and informal
        sheet_name_activity_formal:str Sheet name for information about informal and formal employment data
        sector:str, 
        cols_to_lag: list
    
    Returns:
        pd.DataFrame: The final analysis table.
    """
    
    # Generate labor data
    df_labor = preprocessing.run_preprocessing_pipeline(path_national=path_national,
                                                        sheet_name_national=sheet_name_national,
                                                        path_sector=path_sector,
                                                        sheet_name_sector=sheet_name_sector,
                                                        path_formal_informal=path_formal_informal,
                                                        sheet_name_formal_informal=sheet_name_formal_informal,
                                                        sector=sector,
                                                        cols_to_lag=cols_to_lag)
    # Get Sector Mean
    df = run_pipeline_sector(path = path_activity_formal,
                                            sheet_name=sheet_name_activity_formal,
                                            sector=['Agricultura, ganadería, caza, silvicultura y pesca','Industrias manufactureras'])
    df['PropFomralesAgricultura'] = df['Formal_Agricultura, ganadería, caza, silvicultura y pesca']/df['Formal']
    df['PropFomralesIndustrias'] = df['Formal_Industrias manufactureras']/df['Formal']
    mean_agg = df['PropFomralesAgricultura'].mean()
    mean_ind = df['PropFomralesIndustrias'].mean()
    
    # Generate log and logdiff columns
    df_labor = predict_pea_arimax(df_labor)

    
    # Add pandemic impact
    df_labor = run_catboost_pipeline(df_labor, n_trials = 30)

    # Rename some columns
    df_labor.rename(columns={'PredPea': 'PredPETArimax',
                        'pred_pea_catboost':'PredPETCatboost'}, inplace=True)
    
    # Calculate Prediction Proportions
    # Informales
    df_labor['PredInformalArimax'] = df_labor['PredPETArimax'] * df_labor['proportion_informal_PET_ma24']
    df_labor['PredInformalCatboost'] = df_labor['PredPETCatboost'] * df_labor['proportion_informal_PET_ma24']
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

    # Sectores Formal
    df_labor['PredAgriculturaFormalArimax'] = df_labor['PredFormalArimax'] * mean_agg
    df_labor['PredAgriculturaFormalCatboost'] = df_labor['PredFormalCatboost'] * mean_agg
    df_labor['PredManufacturaFormalArimax'] = df_labor['PredFormalArimax'] * mean_ind
    df_labor['PredManufacturaFormalCatboost'] = df_labor['PredFormalCatboost'] * mean_ind
    
    return df_labor