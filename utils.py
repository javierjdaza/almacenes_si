import pandas as pd
import warnings
warnings.filterwarnings('ignore')
pd.options.display.float_format = '{:.2f}'.format
import matplotlib.pyplot as plt



def get_data_delta_year(df, year:str):
    df_temp = df.copy()
    df_filtred_year = df_temp[df_temp['date_week'].between(f'{year}-01-01',f'{year}-12-31')]
    df_filtred_year['fam_cod'] = df_filtred_year['combination'].apply(lambda x: x[:3])
    df_grouped = df_filtred_year.groupby(['fam_cod'], as_index = False)[['quantity','price_taxes_excluded']].sum()
    df_grouped.sort_values(by = ['price_taxes_excluded'], inplace = True, ascending = False)
    sum_total_price = df_grouped['price_taxes_excluded'].sum()
    df_grouped['price_percentage_total'] = df_grouped['price_taxes_excluded'].apply(lambda x: x/sum_total_price * 100)
    df_grouped['cum_price_percentage_total'] = df_grouped['price_percentage_total'].cumsum()
    df_grouped.reset_index(inplace = True, drop = True)
    return df_grouped
    

def pareto_plot(df):
    # Set figure and axis
    # Plot bars (i.e. frequencies)
    fig, ax = plt.subplots(figsize=(18,7))
    # Plot bars (i.e. frequencies)
    ax.bar(df['fam_cod'], df["price_taxes_excluded"], zorder = 1, color = '#414141')
    # X and y labels
    plt.ylabel('Frecuency', fontsize=10, color='#414141',labelpad=15)
    plt.xlabel('Store ID', fontsize=10, color='#414141',labelpad=15)

    # Bolded horizontal line at y=0
    plt.axhline(y=1, color='#414141', linewidth=1, alpha=1)

    # Y-labels to only these
    plt.yticks( fontsize=10, color='#414141');
    plt.xticks( fontsize=10, color='#414141');
    # Second y axis (i.e. cumulative percentage)
    ax2 = ax.twinx()
    ax2.plot(df['fam_cod'], df["cum_price_percentage_total"], color="red", marker=".", ms=20,linewidth=2, alpha=.5, zorder = 2)
    ax2.axhline(80, color="#525CEB", linestyle="dashed",linewidth=2, alpha=.6)
    # ax2.yaxis.set_major_formatter(PercentFormatter())
    plt.ylabel('Cumulative Percentage', fontsize=10, color='#414141',labelpad=15)
    plt.yticks( fontsize=10, color='#414141');
    ax2.grid(False)

    # Title text
    # plt.text(x=-1.6, y=113, s = "Ventas Anuales X Llave", fontsize=22.5, fontweight='bold', color='#414141');

    # Subtitle text
    # plt.text(x=-1.6, y=109, s = 'Pareto Plot', fontsize=20.5, color='#414141');
    return fig


def get_data_for_plot_historical_info(data_by_week, predictions_2023_dataset, llave):
    x = data_by_week[data_by_week['combination'] == llave]
    x['month_year'] = x['date_week'].apply(lambda x: x.strftime('%m-%Y'))
    x = x.groupby(['month_year'], as_index=False)['quantity'].sum()
    x['month_year'] = pd.to_datetime(x['month_year'])
    x.sort_values(by = ['month_year'], inplace=True)
    y_true_historical = x.copy()
    
    
    
    y = predictions_2023_dataset[predictions_2023_dataset['llave'] == llave]
    y['month_year'] = y['ds'].apply(lambda x: x.strftime('%m-%Y'))
    y = y.groupby(['month_year'], as_index=False)['yhat'].sum()
    y['month_year'] = pd.to_datetime(y['month_year'])
    y.sort_values(by = ['month_year'], inplace=True)
    y_hat_2023 = y.copy()
    
    return y_true_historical,y_hat_2023

def plot_line_plot_ytrue_hat(y_true_historical,y_hat_2023, llave):
    import plotly.graph_objects as go


    # Create traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true_historical['month_year'], y=y_true_historical['quantity'],
                        mode='lines+markers',
                        name='lines+markers',line = dict(color='#414141', width=2,)))
    fig.add_trace(go.Scatter(x=y_hat_2023['month_year'], y=y_hat_2023['yhat'],
                        mode='lines+markers',
                        name='lines+markers',line = dict(color='#FF407D', width=2,)))
    fig.update_layout(showlegend=False)

    fig.update_layout(title=f'Comportamiento historico<br>Llave: {llave}',
                    xaxis_title='Month-Year',
                    yaxis_title='Total Venta')
    fig.add_vline(x=pd.to_datetime('01-2023'), line_width=1.5, line_dash="dash", line_color="#FF407D")

    return fig
    
    

    
