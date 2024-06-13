import streamlit as st 
from streamlit_option_menu import option_menu
import pandas as pd
from tqdm import tqdm
from datetime import date
import numpy as np
import pandas as pd
from utils_web import AlmacenesSiModel
from stqdm import stqdm
import os
import yaml
# ------------------------
# PAGE CONFIGURATION
# ------------------------

st.set_page_config(page_title='Almacenes Si Forecast', page_icon = ':brain:', layout="wide", initial_sidebar_state="collapsed")
# hide_watermarks()

# ------
# LOGO
# ------

a1,a2,a3,a4,a5 = st.columns(5)
with a3:
    st.image('./web_img/logo.png',use_column_width=True)





st.write('---')

def load_params(year_predict):
    file_path = './config.yaml'  
    if os.path.exists(file_path) != True:
        st.error("El archivo de config.yaml no esta en el directorio")

    with open(file_path, 'r') as file: config = yaml.safe_load(file)

    # check files exist
    if os.path.exists(config['campaigns_filepath']) != True:
        st.error("El archivo de campañas especificado no existe")

    if os.path.exists(config['prior_year_sales_file_path']) != True:
        st.error("El archivo historico de ventas especificado no existe")

    if os.path.isdir(config['serialized_models_path']) != True:
        st.error("La carpeta contenedora de los modelos especificada no existe")

    PARAMS = {
        'serialized_models_path' : config['serialized_models_path'],
        'campaigns_filepath' : config['campaigns_filepath'],
        'prior_year_sales_file_path' : config['prior_year_sales_file_path'],
        'year_to_forecast' : int(year_predict),
    }
    return PARAMS

st.markdown("<h1 style='text-align: center;'>Prediccion de Demanda</h1>", unsafe_allow_html=True)
st.write('---')

b1,b2,b3 = st.columns((1,3,1))
with b2:
    st.markdown("<h4 style='text-align: center;'>Archivo de Campañas y descuentos</h4>", unsafe_allow_html=True)
    st.write('')
    file_path = './config.yaml'  
    with open(file_path, 'r') as file: config = yaml.safe_load(file)
    if not os.path.exists(config['campaigns_filepath']):
        st.error("El archivo de campañas especificado no existe")
    else:
        campanias_df = pd.read_csv(config['campaigns_filepath'])
        st.dataframe(campanias_df, use_container_width = True)

    
    t1,t2,t3 = st.columns((1,3,1))
    with t2:
        st.markdown("<h4 style='text-align: center;'>Seleccionar año a predecir</h4>", unsafe_allow_html=True)
        anio_prediccion = st.slider(label= "Año de prediccion", min_value=2024, max_value=2030)
        boton_prediccion = st.button('Realizar Prediccion')
    
    if boton_prediccion:    
        PARAMS = load_params(anio_prediccion)

        almacenes_si = AlmacenesSiModel(**PARAMS)

        # ==============================
        # --- Prediccion de demanda ---
        # ==============================
        forecast = almacenes_si.get_all_keys_prediction()
        forecast_output_path = f'./predicciones/Almacenes_si_prediccion_demanda_{PARAMS["year_to_forecast"]}.csv'
        forecast.to_csv(forecast_output_path, index = False, sep = ',')
        st.success(f'La Prediccion esta lista y ha sido guardada en {forecast_output_path}✅')
        st.markdown("<h4 style='text-align: center;'>Archivo de Prediccion</h4>", unsafe_allow_html=True)
        st.dataframe(forecast, use_container_width=True)
        
        st.write('---')
        st.markdown("<h1 style='text-align: center;'>Calculo Prediccion desagregada</h1>", unsafe_allow_html=True)
        st.write('---')
    
        # ======================================================
        # --- Prediccion de demanda desagregado por tienda ---
        # ======================================================
        store_breakdown_output_path = f'./demanda_por_tienda/Almacenes_si_prediccion_demanda_desagregado_por_tienda_{PARAMS["year_to_forecast"]}.csv'
        # try:
        demanda_desagrada_por_tienda = almacenes_si.calculate_store_breakdown()
        if not isinstance(demanda_desagrada_por_tienda, str):
            st.success(f'Los calculos desagregados por tienda estan listos y han sido guardados en {store_breakdown_output_path}✅')
            st.markdown("<h4 style='text-align: center;'>Prediccion Desagregada por Tienda</h4>", unsafe_allow_html=True)
            demanda_desagrada_por_tienda['store_id']  = demanda_desagrada_por_tienda['store_id'].astype(str)
            st.dataframe(demanda_desagrada_por_tienda, use_container_width=True)
        else:
            st.error(demanda_desagrada_por_tienda)
        # except Exception as e:
        #     st.error('Se debe correr la prediccion, para sacar el calculo desagregado')
        #     st.error(e)

# if menu_bar_selected == 'Calculo Prediccion desagregada':
    # st.markdown("<h1 style='text-align: center;'>Calculo Prediccion desagregada</h1>", unsafe_allow_html=True)
    # st.write('---')
    # c1,c2,c3 = st.columns((1,3,1))
    # with c2:
    #     # ======================================================
    #     # --- Prediccion de demanda desagregado por tienda ---
    #     # ======================================================
    #     store_breakdown_output_path = f'./demanda_por_tienda/Almacenes_si_prediccion_demanda_desagregado_por_tienda_{PARAMS["year_to_forecast"]}.csv'
    #     try:
    #         demanda_desagrada_por_tienda = almacenes_si.calculate_store_breakdown()
    #         if type(demanda_desagrada_por_tienda) != str:
    #             st.success(f'Los calculos desagregados por tienda estan listos y han sido guardados en {store_breakdown_output_path}✅')
    #             st.markdown("<h4 style='text-align: center;'>Prediccion Desagregada por Tienda</h4>", unsafe_allow_html=True)
    #             st.dataframe(demanda_desagrada_por_tienda, use_container_width=True)
    #         else:
    #             st.error(demanda_desagrada_por_tienda)
    #     except:
    #         st.error('Se debe correr la prediccion, para sacar el calculo desagregado')