import pandas as pd
import os
from dotenv import load_dotenv
import mysql.connector
from datetime import datetime


def forecast_key_format(row):
    familias_tipo_1 = [278,212,213,225,276,238,270,206,260,261,262,263,275,205,257,258,259] # formato = [cod_fami]
    familias_tipo_2 = [244,245,246,247,202,269,230,252,277,201,264,265,266,267,] # formato = [cod_fami,cod_subg]
    familias_tipo_3 = [239,240,243,] # formato = [cod_fami, cod_subg, cod_dsub]
    familias_tipo_4 = [241, 242, 209] # formato = [cod_fami, cod_grup, cod_subg]
    familias_tipo_5 = [219, 214] # formato = [cod_fami, cod_tipo, cod_subg]
    familias_tipo_6 = [250] # formato = [cod_fami, cod_tipo, cod_grup, cod_subg]
    familias_tipo_7 = [280,251,281,268] # formato = [cod_fami, cod_tipo, cod_grup, cod_subg,cod_marca]
    familias_tipo_8 = [248] # formato = [cod_fami, cod_subg, cod_dsub, cod_tlla]
    familias_tipo_9 = [211] # formato = [cod_fami,cod_grup]

    if int(row['cod_fami']) in familias_tipo_1:
        return row['cod_fami']
    elif int(row['cod_fami']) in familias_tipo_2:
        return row['cod_fami'] + row['cod_subg'] 
    elif int(row['cod_fami']) in familias_tipo_3:
        return row['cod_fami'] + row['cod_subg'] + row['cod_dsub']
    elif int(row['cod_fami']) in familias_tipo_4:
        return row['cod_fami'] + row['cod_grup'] + row['cod_subg']
    elif int(row['cod_fami']) in familias_tipo_5:
        return row['cod_fami'] + row['cod_tipo'] + row['cod_subg']
    elif int(row['cod_fami']) in familias_tipo_6:
        return row['cod_fami'] + row['cod_tipo'] + row['cod_grup'] + row['cod_subg']
    elif int(row['cod_fami']) in familias_tipo_7:
        return row['cod_fami'] + row['cod_tipo'] + row['cod_grup'] + row['cod_subg'] + row['cod_marc']
    elif int(row['cod_fami']) in familias_tipo_8:
        return row['cod_fami'] + row['cod_subg'] + row['cod_dsub'] + row['cod_tlla']
    elif int(row['cod_fami']) in familias_tipo_9:
        return row['cod_fami'] + row['cod_grup']    


def query_all_table_from_mysql(table_name,db_config):
    conn = mysql.connector.connect(**db_config)    
    cursor = conn.cursor()

    query = f'SELECT * FROM {table_name}'

    cursor.execute(query)

    # Fetch the results into a pandas DataFrame
    df = pd.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])

    conn.commit()
    cursor.close()
    conn.close()

    return df

def edited_query_from_table_mysql(db_config,sql_query):
    conn = mysql.connector.connect(**db_config)    
    cursor = conn.cursor()

    query = sql_query

    cursor.execute(query)

    # Fetch the results into a pandas DataFrame
    df = pd.DataFrame(cursor.fetchall(), columns=[i[0] for i in cursor.description])

    conn.commit()
    cursor.close()
    conn.close()

    return df


def get_sales_last_year():
    load_dotenv()
    today = datetime.now()
    year_now = today.year
    last_year = year_now - 1
    min_date = f'{last_year}-01-01'
    max_date = f'{last_year}-12-31'
    # check if the last year sales file exist, if not, create
    if not os.path.exists(f'./datasets/historico_ventas_{last_year}_semanal.csv'):

        #Configurando Datos de Conexión a la DB
        db_config = {
            "host": os.getenv("MYSQL_HOST"),
            "user": os.getenv("MYSQL_USER"),
            "password": os.getenv("MYSQL_PASSWORD"),
            "database": os.getenv("MYSQL_DATABASE")
        }

        df_product_raw = query_all_table_from_mysql("product_information",db_config)
        df_product_raw['combination'] = df_product_raw.apply(lambda x: forecast_key_format(x), axis = 1) # calculate forecastkey (combination)

        
        df_ventas = edited_query_from_table_mysql(db_config,f'SELECT * FROM sales where `Date` between "{min_date}" and "{max_date}"')

        df_product_raw = df_product_raw[['ProductId','cod_fami','combination']]
        df_last_year_sales = df_ventas.merge(df_product_raw, how = 'left', on = 'ProductId')
        df_last_year_sales['Date'] = pd.to_datetime(df_last_year_sales['Date'])
        # Group By Week
        df_last_year_sales_by_week = df_last_year_sales.groupby(['combination', pd.Grouper(key='Date', freq='W-MON')]).agg({'Quantity': 'sum', 'StoreId':'max','ProductId':'max'}).reset_index() 
        df_last_year_sales_by_week.rename(columns = {'Quantity': 'quantity', 'StoreId':'store_id','ProductId':'product_id', 'Date':'date'}, inplace = True)
        df_last_year_sales_by_week.to_csv(f'./datasets/historico_ventas_{last_year}_semanal.csv', sep = ',', index = False)
    else:
        df_last_year_sales_by_week = pd.read_csv(f'./datasets/historico_ventas_{last_year}_semanal.csv')

    return df_last_year_sales_by_week

