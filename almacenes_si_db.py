import pandas as pd
import os
from dotenv import load_dotenv
import mysql.connector
from datetime import datetime




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

    #Configurando Datos de Conexión a la DB
    db_config = {
        "host": os.getenv("MYSQL_HOST"),
        "user": os.getenv("MYSQL_USER"),
        "password": os.getenv("MYSQL_PASSWORD"),
        "database": os.getenv("MYSQL_DATABASE")
    }

    df_product_raw = query_all_table_from_mysql("product_information",db_config)

    today = datetime.now()
    year_now = today.year
    last_year = year_now - 1
    min_date = f'{last_year}-01-01'
    max_date = f'{last_year}-12-31'
    df_ventas = edited_query_from_table_mysql(db_config,f'SELECT * FROM sales where `Date` between "{min_date}" and "{max_date}"')

    df_product_raw = df_product_raw[['ProductId','cod_fami']]
    df_last_year_sales = df_ventas.merge(df_product_raw, how = 'left', on = 'ProductId')

    return df_last_year_sales

