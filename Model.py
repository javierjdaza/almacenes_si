from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
import logging
logging.getLogger("prophet").setLevel(logging.WARNING)
import warnings
warnings.filterwarnings('ignore')
import logging

# Set the logging level for cmdstanpy to WARNING
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
import pandas as pd
import numpy as np
import json
from datetime import timedelta
from glob import glob
from tqdm import tqdm
import os
import yaml
import warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s :: %(levelname)s :: %(message)s")

class AlmacenesSiModel:
    
    def __init__(self, serialized_models_path: str, campaigns_filepath: str, year_to_forecast: int,prior_year_sales_file_path: str):
        self.serialized_models_path = serialized_models_path
        self.year_to_forecast = year_to_forecast
        self.campaigns_filepath = campaigns_filepath
        
        self.future_regressors = pd.read_csv(self.campaigns_filepath)
        logging.info('Archivo de campañas cargado exitosamente ✅')
        self.prior_year_sales_df = pd.read_csv(prior_year_sales_file_path)
        logging.info(f'Archivo historico de ventas del año {self.year_to_forecast - 1} cargado exitosamente ✅')
        self.future_regressors['date_week'] = pd.to_datetime(self.future_regressors['date_week'])
        logging.info('Cargando datos del Modelo Predictivo ⌛')
        self.models_info = self.get_keys_names_and_model()
        logging.info('Modelo Predictivo cargado exitosamente ✅')
        
        
        
    @staticmethod
    def load_model_from_json(file_path):
        with open(file_path, "rb") as file:
            prophet_model = model_from_json(file.read())
        
        return prophet_model    
    
    @staticmethod
    def make_predictions(model, key_combination: str, future_regressors : pd.DataFrame, year : int)->pd.DataFrame:
        # Obtener la última fecha histórica
        last_date = max(model.history_dates) + timedelta(weeks=1)
        
        # Crear un DataFrame con las fechas futuras hasta el año especificado, por semana comenzando en lunes
        future_dates = pd.date_range(start=last_date, end=f'{year}-12-31', freq='W-MON')

        # Crear el DataFrame con las fechas y los regresores
        future = pd.DataFrame({'ds': future_dates})
        future = future[future['ds'] >= f'{year}-01-01']
        
        # prepara los regresores dada la familia
        familia = str(key_combination[:3])
        df_discount_and_campaings = future_regressors[future_regressors['familia'] == familia]
        df_discount_and_campaings.rename(columns = {'date_week':'ds'}, inplace = True)
        
        # Añadir los regresores al DataFrame futuro
        future = future.merge(df_discount_and_campaings, how='left', on='ds')
        
        # Llenar NaNs con 0 si es necesario 
        future.fillna(0, inplace=True)
        
        # Hacer la predicción
        forecast = model.predict(future)
        forecast['yhat'] = forecast['yhat'].apply(lambda x: 0 if x < 1 else x )
        forecast['yhat'] = forecast['yhat'].apply(lambda x: np.ceil(x) )
        return forecast[['ds', 'yhat']]
    
    def get_keys_names_and_model(self):
        
        models_info = {}
        for model_path in tqdm(glob(f'{self.serialized_models_path}/*.json')):
            
            key_name = str(os.path.basename(model_path).split('.')[0].strip())
            model_temp = self.load_model_from_json(model_path)
            models_info[key_name] = model_temp
            
        return models_info
        
    def get_all_keys_prediction(self):
        
        forecast_df = pd.DataFrame()
        
        logging.info(f'Calculando Predicciones para el año {self.year_to_forecast} ⌛')
        for key, model in tqdm(self.models_info.items()):
            
            PARAMS = {
                'key_combination' : key,
                'model' : model,
                'future_regressors' : self.future_regressors,
                'year' : self.year_to_forecast
            }
            try:
                forecast_temp = self.make_predictions(**PARAMS )
                forecast_temp['llave'] = key
                forecast_df = pd.concat([forecast_df,forecast_temp])
            except Exception as e:
                print(e)
                
        self.forecast_df = forecast_df
        self.forecast_df.rename(columns = {'ds' : 'fecha','yhat' : 'prediccion_demanda'}, inplace = True)
        return forecast_df
    
    def calculate_store_breakdown(self):
        
        forecast_df = self.forecast_df
        prior_year_sales_df = self.prior_year_sales_df
        forecast_df['fecha'] = pd.to_datetime(forecast_df['fecha'])
        forecast_df['semana'] = pd.to_datetime(forecast_df['fecha']).apply(lambda x: x.strftime('%W'))
        forecast_df['semana'] = pd.to_datetime(forecast_df['semana']).apply(lambda x: x.lstrip('0'))

        prior_year_sales_df['date'] = pd.to_datetime(prior_year_sales_df['date'])
        prior_year_sales_df['week'] = prior_year_sales_df['date'].apply(lambda x: x.strftime('%W'))
        prior_year_sales_df['week'] = prior_year_sales_df['week'].apply(lambda x: x.lstrip('0'))
        
        self.demanda_desagrada_por_tienda = pd.DataFrame()
        year_prediction = int(forecast_df['fecha'].max().strftime('%Y'))
        if  self.year_to_forecast - year_prediction != 1:
            for semana in forecast_df['semana'].unique(): # recorro todas las llaves disponibles en la prediccion
                logging.info(f'semana: {semana}')
                llaves_to_explore = forecast_df[forecast_df['semana'] == semana]['llave'].unique()# saco todas las llaves a explorar en la semana T
                df_week_selected =  prior_year_sales_df[(prior_year_sales_df['week'] == semana)] # saco la informacion del ultimo año disponible en la semana T
                # ------------------
                for llave in tqdm(llaves_to_explore): # recorro todas las llaves disponibles en la semana T

                    demanda_forecast = forecast_df[(forecast_df['semana'] == semana) & (forecast_df['llave'] == llave)]['prediccion_demanda'] # calculo de la prediccion de la demanda para X llave
                    demanda_forecast = np.ceil(demanda_forecast.iat[0]) # redondeo de la prediccion
                    df_week_key_selected = df_week_selected[df_week_selected['combination'] == llave] # filtrado de la cantidad de unidades vendidad para la semana T para la llave X en el ultimo año de data
                    if len(df_week_key_selected) > 0:
                        data_grouped_by_store = df_week_key_selected.groupby(['store_id'], as_index=False)['quantity'].sum() # calculo de cuantas unidades se vendieron por tienda en la semana T para la llave X
                        store_proportion = data_grouped_by_store.copy()
                        store_proportion['proportion'] = round(store_proportion['quantity'] / (store_proportion['quantity'].sum()),2) # calculo de porcentaje de las unidades vendidas por tienda, en la semana T para la llave X
                        store_proportion = store_proportion[store_proportion['proportion'] != 0] # remover aquellas tiendas que no vendieron ninguna unidad de la llave X para la semana T
                        store_proportion['forecast_llave'] = demanda_forecast
                        store_proportion['demanda'] = store_proportion['forecast_llave'] * store_proportion['proportion'] # multiplico la proporcion de ventas la llave X en la semana T para cada tienda, segun pronostico
                        store_proportion['demanda'] = store_proportion['demanda'].apply(lambda x: np.ceil(x)) # redondeo de la prediccion
                        temp_store_key_prediction = store_proportion.copy()
                        temp_store_key_prediction = temp_store_key_prediction[['store_id','demanda']]
                        temp_store_key_prediction['llave'] = llave
                        temp_store_key_prediction['week'] = semana
                        temp_store_key_prediction['year'] = year_prediction
                        self.demanda_desagrada_por_tienda = pd.concat([self.demanda_desagrada_por_tienda,temp_store_key_prediction])
            
            self.demanda_desagrada_por_tienda['demanda'] = self.demanda_desagrada_por_tienda['demanda'].apply(lambda x: 0 if x < 0 else x)           
            return self.demanda_desagrada_por_tienda
        else:
            logging.info('No se puede calcular la desagregacion por tienda si la data historica no es del año inmediatamente anterior al que se quiere predecir')
            return 'No se puede calcular la desagregacion por tienda si la data historica no es del año inmediatamente anterior al que se quiere predecir'
        
if __name__ == '__main__':
    

    file_path = './config.yaml'  
    assert os.path.exists(file_path), "El archivo de config.yaml no esta en el directorio"
    with open(file_path, 'r') as file: config = yaml.safe_load(file)
    
    # check files exist
    assert os.path.exists(config['campaigns_filepath']), "El archivo de campañas especificado no existe"
    assert os.path.exists(config['prior_year_sales_file_path']), "El archivo historico de ventas especificado no existe"
    assert os.path.isdir(config['serialized_models_path']), "La carpeta contenedora de los modelos especificada no existe"
    
    PARAMS = {
        'serialized_models_path' : config['serialized_models_path'],
        'campaigns_filepath' : config['campaigns_filepath'],
        'prior_year_sales_file_path' : config['prior_year_sales_file_path'],
        'year_to_forecast' : config['year_to_forecast'],
    }
    almacenes_si = AlmacenesSiModel(**PARAMS)

    # ==============================
    # --- Prediccion de demanda ---
    # ==============================
    forecast = almacenes_si.get_all_keys_prediction()
    forecast_output_path = f'./predicciones/Almacenes_si_prediccion_demanda_{PARAMS["year_to_forecast"]}.csv'
    forecast.to_csv(forecast_output_path, index = False, sep = ',')
    logging.info(f'La Prediccion esta lista y ha sido guardada en {forecast_output_path}✅')
    
    # ======================================================
    # --- Prediccion de demanda desagregado por tienda ---
    # ======================================================
    store_breakdown_output_path = f'./demanda_por_tienda/Almacenes_si_prediccion_demanda_desagregado_por_tienda_{PARAMS["year_to_forecast"]}.csv'
    demanda_desagrada_por_tienda = almacenes_si.calculate_store_breakdown()
    demanda_desagrada_por_tienda.to_csv(store_breakdown_output_path, index=False)
    logging.info(f'Los calculos desagregados por tienda estan listos y han sido guardados en {store_breakdown_output_path}✅')