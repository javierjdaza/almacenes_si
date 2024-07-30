from prophet.serialize import model_from_json
import logging
import pandas as pd
import numpy as np
from datetime import timedelta
from glob import glob
from stqdm import stqdm
import os
import warnings
import streamlit as st
logging.getLogger("prophet").setLevel(logging.WARNING)
warnings.filterwarnings('ignore')
from almacenes_si_db import get_sales_last_year
class AlmacenesSiModel:
    
    def __init__(self, serialized_models_path: str, campaigns_filepath: str, year_to_forecast: int,training_dataset_path: str, price_increment_path: str):
        self.serialized_models_path = serialized_models_path
        self.year_to_forecast = year_to_forecast
        self.campaigns_filepath = campaigns_filepath
        
        self.discount_file = pd.read_csv(self.campaigns_filepath)
        self.discount_file['ds'] = pd.to_datetime(self.discount_file['ds'])
        st.success('Archivo de campañas cargado exitosamente ✅')
        self.price_increment_df = pd.read_csv(price_increment_path)
        st.success('Archivo de incremento de precio cargado exitosamente ✅')
        
        self.training_dataset = pd.read_parquet(training_dataset_path)
        self.training_dataset = self.training_dataset[self.training_dataset['date'].between('2023-01-01','2023-12-31')]
        self.training_dataset['price_taxes_excluded'] = self.training_dataset['price_taxes_excluded'].astype(float)
        st.success('Cargando datos del Modelo Predictivo ⌛')
        self.models_info = self.get_keys_names_and_model()
        st.success('Modelo Predictivo cargado exitosamente ✅')
        
        self.prior_year_sales_df = get_sales_last_year()
        
    @staticmethod
    def load_model_from_json(file_path):
        with open(file_path, "rb") as file:
            prophet_model = model_from_json(file.read())
        
        return prophet_model    
    
    @staticmethod
    def make_predictions(model, key_combination: str, discount_file : pd.DataFrame, year_to_forecast : int, training_dataset : pd.DataFrame, price_increment_df: pd.DataFrame)->pd.DataFrame:
        
        # Crear un DataFrame con las fechas futuras hasta el año especificado, por semana comenzando en lunes
        future_dates = pd.date_range(start=f'{year_to_forecast}-01-01', end=f'{year_to_forecast}-12-31', freq='W-MON')


        # Crear el DataFrame con las fechas y los regresores
        future_df = pd.DataFrame({'ds': future_dates})
        future_df = future_df[future_df['ds'] >= f'{year_to_forecast}-01-01']
        
        # prepara los regresores dada la familia
        familia = str(key_combination[:3])
        df_discount = discount_file[discount_file['familia'] == familia]
        df_discount = discount_file[discount_file['familia'] == int(familia)]
        
        # Añadir campaña de descuento al DataFrame futuro
        future_df = future_df.merge(df_discount, how='left', on ='ds')
        
        # Llenar NaNs con 0 si es necesario 
        future_df['familia'].fillna(int(familia), inplace=True)
        future_df['discount'].fillna(0, inplace=True)
        st.subheader(key_combination)
        st.dataframe(future_df)
        
        # agregar regressor precio, basado en el promedio del precio del año 2023 (ultimo año con el que se entreno)
        df_precio = training_dataset[training_dataset['combination'] == key_combination][['date','combination','price_taxes_excluded']]
        incremento_porcentual = price_increment_df[price_increment_df['familia'] == int(familia)]['incremento_porcentual'].iat[0]
        df_precio['incremento_porcentual'] = incremento_porcentual / 100
        df_precio['price_taxes_excluded'] = (df_precio['price_taxes_excluded'] * df_precio['incremento_porcentual']) + df_precio['price_taxes_excluded'] 

        future_df['price_taxes_excluded'] = df_precio['price_taxes_excluded'].mean()

        # Hacer la predicción
        forecast = model.predict(future_df)
        forecast['yhat'] = forecast['yhat'].apply(lambda x: 0 if x < 1 else x )
        forecast['yhat'] = forecast['yhat'].apply(lambda x: np.ceil(x) )
        
        return forecast[['ds', 'yhat']]
    
    def get_keys_names_and_model(self):
        
        models_info = {}
        for model_path in stqdm(glob(f'{self.serialized_models_path}/*.json')[:5]):
            
            key_name = str(os.path.basename(model_path).split('.')[0].strip())
            model_temp = self.load_model_from_json(model_path)
            models_info[key_name] = model_temp
            
        return models_info
        
    def get_all_keys_prediction(self):
        
        forecast_df = pd.DataFrame()
        
        st.success(f'Calculando Predicciones para el año {self.year_to_forecast} ⌛')
        for key, model in stqdm(self.models_info.items()):
            
            PARAMS = {
                'key_combination' : key,
                'model' : model,
                'discount_file' : self.discount_file,
                'price_increment_df' : self.price_increment_df,
                'training_dataset' : self.training_dataset,
                'year_to_forecast' : self.year_to_forecast
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
        forecast_df['semana'] = forecast_df['fecha'].apply(lambda x: x.strftime('%W'))
        forecast_df['semana'] = forecast_df['semana'].apply(lambda x: x.lstrip('0'))

        prior_year_sales_df['date'] = pd.to_datetime(prior_year_sales_df['date'])
        prior_year_sales_df['week'] = prior_year_sales_df['date'].apply(lambda x: x.strftime('%W'))
        prior_year_sales_df['week'] = prior_year_sales_df['week'].apply(lambda x: x.lstrip('0'))
        
        
        self.demanda_desagrada_por_tienda = pd.DataFrame()
        year_prediction = int(prior_year_sales_df['date'].max().strftime('%Y'))
        if  self.year_to_forecast - year_prediction == 1:
            for semana in forecast_df['semana'].unique(): # recorro todas las llaves disponibles en la prediccion
                st.success(f'semana: {semana}')
                llaves_to_explore = forecast_df[forecast_df['semana'] == semana]['llave'].unique()# saco todas las llaves a explorar en la semana T
                df_week_selected =  prior_year_sales_df[(prior_year_sales_df['week'] == semana)] # saco la informacion del ultimo año disponible en la semana T
                # ------------------
                for llave in stqdm(llaves_to_explore): # recorro todas las llaves disponibles en la semana T

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
                        temp_store_key_prediction['year'] = self.year_to_forecast
                        self.demanda_desagrada_por_tienda = pd.concat([self.demanda_desagrada_por_tienda,temp_store_key_prediction])
            
            self.demanda_desagrada_por_tienda['demanda'] = self.demanda_desagrada_por_tienda['demanda'].apply(lambda x: 0 if x < 0 else x)
            return self.demanda_desagrada_por_tienda
        else:
            return 'No se puede calcular la desagregacion por tienda si la data historica no es del año inmediatamente anterior al que se quiere predecir'
        
