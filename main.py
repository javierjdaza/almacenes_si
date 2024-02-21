import streamlit as st 
from streamlit_option_menu import option_menu
from utils import get_data_delta_year,pareto_plot,get_data_for_plot_historical_info,plot_line_plot_ytrue_hat
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
# pd.options.display.float_format = '{:.2f}'.format
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np


st.set_page_config(page_title='Football Stats', page_icon = '📊', layout="wide", initial_sidebar_state="collapsed")


st.markdown("<h1 style='text-align: center;'>Almacenes SI</h1>", unsafe_allow_html=True)

st.write('---')

menu_bar_selected = option_menu(None, ["Pareto Ventas por Familia", "Analisis RMSE", 'Llaves Entrenamiento 2022'], 
                                    icons=['graph-up-arrow', 'people-fill', 'person'], 
                                    menu_icon="cast", default_index=0, orientation="horizontal")

st.write('---')
st.write(' ')
st.write(' ')
st.write(' ')
# ------------------
# FUCTIONS HELPERS
# ------------------

@st.cache_data
def get_llaves_new_training_just_2022():
    llaves_new_training_just_2022 = pd.read_excel('./data/llaves_new_training_just_2022.xlsx')
    return llaves_new_training_just_2022

@st.cache_data
def get_data_by_week():
    data_by_week = pd.read_parquet('./data/almacenes_si_curated_by_week.parquet')
    del data_by_week['campaigns_name']
    return data_by_week

@st.cache_data
def get_proporcion_error_porcentual_df():
    proporcion_error_porcentual = pd.read_excel('./data/proporcion_error_porcentual.xlsx')
    return proporcion_error_porcentual
@st.cache_data
def get_final_prediction_2023_with_test_data_df():
    final_prediction_2023_with_test_data = pd.read_excel('./data/final_prediction_2023_with_test_data.xlsx')
    return final_prediction_2023_with_test_data

if menu_bar_selected == 'Pareto Ventas por Familia':
    
    data_by_week = get_data_by_week()

    c1,c2,c3 = st.columns(3)
    with c2:
        year_selected = st.selectbox('Seleccionar año',options=['2018','2019','2020','2021','2022','2023'])
        year_selected_df = get_data_delta_year(data_by_week, year =year_selected)
        year_selected_df.sort_values(by = ['price_taxes_excluded'], inplace = True, ascending=False)
        print(year_selected_df.dtypes)
        st.write(' ')
        st.write(' ')
    a1,a2,a3 = st.columns((1,5,1))
    with a2:
        st.write('---')
        st.dataframe(year_selected_df,use_container_width=True)
        st.write('---')
        st.markdown("<h2 style='text-align: center;'>Pareto Plot 📊</h2>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: left;'>Ventas X Llave, Anualizado</h4>", unsafe_allow_html=True)
        st.pyplot(pareto_plot(year_selected_df))
        
        
if menu_bar_selected == "Analisis RMSE":
    data_by_week = get_data_by_week()
    proporcion_error_porcentual_df = get_proporcion_error_porcentual_df()
    final_prediction_2023_with_test_data_df = get_final_prediction_2023_with_test_data_df()
    quantile_95 = np.quantile(proporcion_error_porcentual_df['proporcion_error_porcentual'], .90)
    proporcion_error_porcentual_more_than_q95_llaves = proporcion_error_porcentual_df[proporcion_error_porcentual_df['proporcion_error_porcentual'] > quantile_95]
    
    st.write(' ')
    e1,e2,e3,e4,e5,e6 = st.columns(6)
    with e2:
        st.success(f"Quartil 25%: {np.quantile(proporcion_error_porcentual_df['proporcion_error_porcentual'], .25)}")
    with e3:
        st.success(f"Quartil 50%: {np.quantile(proporcion_error_porcentual_df['proporcion_error_porcentual'], .5)}")
    with e4:
        st.warning(f"Quartil 75%: {np.quantile(proporcion_error_porcentual_df['proporcion_error_porcentual'], .75)}")
    with e5:
        st.error(f"Quartil 100%: {np.quantile(proporcion_error_porcentual_df['proporcion_error_porcentual'], 1)}")
        
    d1,d2,d3 = st.columns((1,5,1))
    with d2:
        st.dataframe(proporcion_error_porcentual_more_than_q95_llaves,use_container_width=True)
        st.caption('** Los valores por Cuartil son calculados con base en la proporcion del error porcentual')
        st.caption(f'** La anterior tabla muestra todas las llaves que tienen un proporcion de error porcentual > {quantile_95}, este valor corresponde al cuartil 95%')
    st.write('---')
    b1,b2,b3 = st.columns(3)
    with b2:
        llave_selected = st.selectbox('Seleccionar Llave', options= proporcion_error_porcentual_more_than_q95_llaves['llave'].values)
        st.caption(f'** Las llaves presentes en el dropdown menu corresponden a las llaves con un error porcentual > {quantile_95}%, valor del cuartil 95%')
    y_true_historical,y_hat_2023 = get_data_for_plot_historical_info(data_by_week,final_prediction_2023_with_test_data_df, llave_selected)
    
    fig_ = plot_line_plot_ytrue_hat(y_true_historical,y_hat_2023, llave= llave_selected)
    f1,f2,f3 = st.columns((1,5,1))
    with f2:
        st.write(' ')
        st.plotly_chart(fig_, use_container_width=True)
    g1,g2,g3 = st.columns(3)
    with g2:
        st.caption('** La linea ⚫ Negra representa los valores reales')
        st.caption('** La linea 🔴 Roja representa los valores del pronostico del model.')
    
    
if menu_bar_selected == 'Llaves Entrenamiento 2022':
    
    llaves_new_training_just_2022 = get_llaves_new_training_just_2022()
    f1,f2,f3 = st.columns((1,5,1))
    with f2:
        st.markdown("<h3 style='text-align: center;'>Comparativa RMSE nueva estrategia de entrenamiento</h3>", unsafe_allow_html=True)
        st.write(' ')
        st.dataframe(llaves_new_training_just_2022,use_container_width=True)
        st.caption('** Esta es la comparativa entre los RMSE de las llaves con alto drifting, RMSE anterior VS New RMSE (entrenado solo con data del 2022)')
    st.write('---')
    
    