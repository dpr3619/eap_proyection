from src.epl_proyection.models.arimax.arimax_full_pipeline import predict_pea_arimax
from src.epl_proyection.models.catboost.catboost_main_pipeline import run_catboost_pipeline
from src.epl_proyection.etl import preprocessing
from src.epl_proyection.etl.read_informal_geih import run_pipeline_sector
import pandas as pd
import numpy as np

def apply_quadratic_decline(df, date_col, target_col, start_date, end_date, decline_pct, new_col_name='AdjustedForecast'):
    """
    Aplica una caída cuadrática al target_col entre start_date y end_date
    Esta función es para el empleo formal del agro debido a las fuentes encontradas.

    Parameters:
    - df: DataFrame con columnas de fecha y valores a ajustar.
    - date_col: Nombre de la columna de fechas (ej. 'ds').
    - target_col: Columna de proyección base (ej. 'PredEmpleo').
    - start_date: Fecha de inicio de impacto (str, ej. '2030-01-01').
    - end_date: Fecha final del impacto (str, ej. '2040-12-01').
    - decline_pct: Caída total en proporción (ej. -0.03 para -3%).
    - new_col_name: Nombre de la nueva columna con proyección ajustada.

    Returns:
    - DataFrame con nueva columna ajustada.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    # Normalizar tiempo entre 0 y 1
    mask = (df[date_col] >= start_date) & (df[date_col] <= end_date)
    total_months = (end_date.to_period('M') - start_date.to_period('M')).n + 1
    df.loc[mask, 't_norm'] = np.linspace(0, 1, total_months)

    # Función cuadrática de caída
    df['adjustment_factor'] = 1.0
    df.loc[mask, 'adjustment_factor'] = 1.0 + decline_pct * df.loc[mask, 't_norm'] ** 2

    # Aplicar ajuste
    df[new_col_name] = df[target_col] * df['adjustment_factor']

    # Limpieza
    df.drop(columns=['t_norm', 'adjustment_factor'], inplace=True)
    
    return df

def apply_quadratic_growth_agriculture(df, date_col, target_col, start_date, end_date, growth_pct=0.02, new_col_name='AdjustedAgro'):
    """
    Aplica un crecimiento cuadrático del 2% al empleo agrícola proyectado.

    Parámetros:
    - df: DataFrame con fechas y predicción base.
    - date_col: Nombre de la columna de fechas (ej. 'ds').
    - target_col: Columna base de la proyección (ej. 'PredAgricultura').
    - start_date, end_date: Rango de fechas donde se aplica el impacto.
    - growth_pct: Porcentaje de aumento (por defecto 0.02).
    - new_col_name: Nombre de la nueva columna con el ajuste.

    Retorna:
    - DataFrame con columna ajustada.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    mask = (df[date_col] >= start_date) & (df[date_col] <= end_date)
    total_months = (end_date.to_period('M') - start_date.to_period('M')).n + 1
    df.loc[mask, 't_norm'] = np.linspace(0, 1, total_months)

    df['adjustment_factor'] = 1.0
    df.loc[mask, 'adjustment_factor'] = 1.0 + growth_pct * df.loc[mask, 't_norm'] ** 2

    df[new_col_name] = df[target_col] * df['adjustment_factor']

    df.drop(columns=['t_norm', 'adjustment_factor'], inplace=True)
    return df


def apply_quadratic_growth_ia(df, date_col, target_col, start_date, end_date, growth_pct=0.02, new_col_name='AdjustedIA'):
    """
    Aplica un ajuste cuadrático creciente hacia arriba (IA) al target_col.

    Parámetros:
    - df: DataFrame con fechas y columna objetivo.
    - date_col: Columna de fecha (ej. 'ds').
    - target_col: Columna con predicción base.
    - start_date, end_date: Fechas de inicio y fin del impacto.
    - growth_pct: Porcentaje de incremento (por defecto 2%).
    - new_col_name: Nombre de la nueva columna.

    Retorna:
    - DataFrame con columna ajustada.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    mask = (df[date_col] >= start_date) & (df[date_col] <= end_date)
    total_months = (end_date.to_period('M') - start_date.to_period('M')).n + 1
    df.loc[mask, 't_norm'] = np.linspace(0, 1, total_months)

    df['adjustment_factor'] = 1.0
    df.loc[mask, 'adjustment_factor'] = 1.0 + growth_pct * df.loc[mask, 't_norm'] ** 2
    df[new_col_name] = df[target_col] * df['adjustment_factor']

    df.drop(columns=['t_norm', 'adjustment_factor'], inplace=True)
    return df

def apply_quadratic_growth_iiot(df, date_col, target_col, start_date, end_date, growth_pct=0.04, new_col_name='AdjustedIIoT'):
    """
    Aplica un ajuste cuadrático creciente hacia arriba (IIoT) al target_col.

    Parámetros:
    - df: DataFrame con fechas y columna objetivo.
    - date_col: Columna de fecha (ej. 'ds').
    - target_col: Columna con predicción base.
    - start_date, end_date: Fechas de inicio y fin del impacto.
    - growth_pct: Porcentaje de incremento (por defecto 4%).
    - new_col_name: Nombre de la nueva columna.

    Retorna:
    - DataFrame con columna ajustada.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    mask = (df[date_col] >= start_date) & (df[date_col] <= end_date)
    total_months = (end_date.to_period('M') - start_date.to_period('M')).n + 1
    df.loc[mask, 't_norm'] = np.linspace(0, 1, total_months)

    df['adjustment_factor'] = 1.0
    df.loc[mask, 'adjustment_factor'] = 1.0 + growth_pct * df.loc[mask, 't_norm'] ** 2
    df[new_col_name] = df[target_col] * df['adjustment_factor']

    df.drop(columns=['t_norm', 'adjustment_factor'], inplace=True)
    return df


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
    
    # PredictionPEA
    df_labor['PredPETA'] = (df_labor['PredPETArimax'] + df_labor['PredPETCatboost'])/2
    # Calculate Prediction Proportions
    # Informales
    df_labor['PredInformalArimax'] = df_labor['PredPETArimax'] * df_labor['proportion_informal_PET_ma24']
    df_labor['PredInformalCatboost'] = df_labor['PredPETCatboost'] * df_labor['proportion_informal_PET_ma24']
    df_labor['PredInformal'] = df_labor['PredPETArimax'] * df_labor['proportion_informal_PET_ma24']
    # Formales
    df_labor['PredFormalArimax'] = df_labor['PredPETArimax'] * df_labor['proportion_formal_PET_ma24']
    df_labor['PredFormalCatboost'] = df_labor['PredPETCatboost'] * df_labor['proportion_formal_PET_ma24']
    df_labor['PredFormal'] = df_labor['PredPETArimax'] * df_labor['proportion_formal_PET_ma24']
    # Ocupados
    df_labor['PredOcupadosArimax'] = df_labor['PredInformalArimax'] + df_labor['PredFormalArimax']
    df_labor['PredOcupadosCatboost'] = df_labor['PredInformalCatboost'] + df_labor['PredFormalCatboost']
    df_labor['PredOcupados'] = df_labor['PredInformalArimax'] + df_labor['PredFormalArimax']
    # Sectores
    df_labor['PredAgriculturaArimax'] = df_labor['PredOcupadosArimax'] * df_labor['porportion_aggriculture_Occupied_ma24']
    df_labor['PredAgriculturaCatboost'] = df_labor['PredOcupadosCatboost'] * df_labor['porportion_aggriculture_Occupied_ma24']
    df_labor['PredManufacturaArimax'] = df_labor['PredOcupadosArimax'] * df_labor['proportion_manufacturing_Occupied_ma24']
    df_labor['PredManufacturaCatboost'] = df_labor['PredOcupadosCatboost'] * df_labor['proportion_manufacturing_Occupied_ma24']
    df_labor['PredAgricultura'] = df_labor['PredOcupadosArimax'] * df_labor['porportion_aggriculture_Occupied_ma24']
    df_labor['PredManufactura'] = df_labor['PredOcupadosCatboost'] * df_labor['porportion_aggriculture_Occupied_ma24']

    # Sectores Formal
    df_labor['PredAgriculturaFormalArimax'] = df_labor['PredFormalArimax'] * mean_agg
    df_labor['PredAgriculturaFormalCatboost'] = df_labor['PredFormalCatboost'] * mean_agg
    df_labor['PredManufacturaFormalArimax'] = df_labor['PredFormalArimax'] * mean_ind
    df_labor['PredManufacturaFormalCatboost'] = df_labor['PredFormalCatboost'] * mean_ind
    df_labor['PredAgriculturaFormal'] = df_labor['PredFormalArimax'] * mean_agg
    df_labor['PredManufacturaFormal'] = df_labor['PredFormalCatboost'] * mean_agg

    # Adding IOT impact
    df_labor = apply_quadratic_growth_iiot(df_labor, date_col='ds', 
                                            target_col='PredManufacturaFormal', 
                                            start_date='2028-01-01', 
                                            end_date='2040-12-01', 
                                            growth_pct=0.04, 
                                            new_col_name='PredManufacturaFormalIIoT')

    # Adding IA impact
    df_labor = apply_quadratic_growth_ia(df_labor, date_col='ds', 
                                            target_col='PredManufacturaFormal', 
                                            start_date='2028-01-01', 
                                            end_date='2040-12-01', 
                                            growth_pct=0.02, 
                                            new_col_name='PredManufacturaFormalIA')
    # Adding Agriculture impact
    df_labor = apply_quadratic_growth_agriculture(df_labor, 
                                                    date_col='ds', 
                                                    target_col='PredAgriculturaFormal', 
                                                    start_date='2028-01-01', 
                                                    end_date='2040-12-01', 
                                                    growth_pct=0.02, 
                                                    new_col_name='PredManufacturaFormalAgro')
    
    df_labor = apply_quadratic_decline(df=df_labor,
                                        date_col='ds',
                                        target_col='PredAgriculturaFormal',
                                        start_date='2030-01-01',
                                        end_date='2040-12-01',
                                        decline_pct=-0.03,
                                        new_col_name='PredAgriculturaFormalIAConservador')

    df_labor = apply_quadratic_decline(df=df_labor,
                                        date_col='ds',
                                        target_col='PredAgriculturaFormal',
                                        start_date='2030-01-01',
                                        end_date='2040-12-01',
                                        decline_pct=-0.06,
                                        new_col_name='PredAgriculturaFormalIAPesimista'
)
    return df_labor