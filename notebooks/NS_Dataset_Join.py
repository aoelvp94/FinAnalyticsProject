import pandas as pd
import yfinance as yf
import numpy as np
from pandas_datareader import data as pdr
import datetime
import matplotlib.pyplot as plt

#Cargamos ambos files y los unimos, usando como indices en ambos las fechas.
def load_and_join():
    mtum = yf.Ticker("MTUM")
    stocks = ["MTUM"]
    start = datetime.datetime(2000,11,30)
    end = datetime.datetime(2019,11,30)
    
    yf.pdr_override()
    
    df_etf = pdr.get_data_yahoo(stocks, start=start, end=end)
    
    df = pd.read_excel('https://images.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Century-of-Factor-Premia-Monthly.xlsx',
                      header =18, nrows = 1220)
    
    df['Date'] =  pd.to_datetime(df['Date'])
    
    df = df.set_index('Date')
    df = df[['Equity indices Momentum','Equity indices Momentum','Equity indices Carry','Equity indices Defensive']]
    
    df_final = df_etf.merge(df, how='left',left_index=True,right_index=True)
    
    return df_final

#Extendemos los datos mensuales a los registros diarios. (Ejemplo: todos los datos de enero, tomaran el valor monthly del 31/01)
def fill_joined_missing_fields(df):
    for i in range(1,len(df)+1):
        if np.isnan(df.iloc[-i,9]):
                df.iloc[-i,6] = df.iloc[-i+1,6]
                df.iloc[-i,7] = df.iloc[-i+1,7]
                df.iloc[-i,8] = df.iloc[-i+1,8]
                df.iloc[-i,9] = df.iloc[-i+1,9]
         
#Calculamos las medias de 50 y 200 dias, asi como la señal en base a la posicion.
def calculate_averages_50_200(df):
    df['50_days_average'] = df.iloc[:,3].rolling(window=50).mean()
    df['200_days_average'] = df.iloc[:,3].rolling(window=200).mean()
    df.loc[df['50_days_average'] >= df['200_days_average'], 'Buy/Sell'] = -1 #Si la media de corto plazo supera a la de largo, es posicion de sell
    df.loc[df['50_days_average'] < df['200_days_average'], 'Buy/Sell'] = 1 #Si la media de corto plazo esta por debajo de la de largo, es posicion de buy
 
#Visualizacion de la evolucion de precios y las medias.
def visualize_close_50_200(df):
    plt.plot(df['Close'])
    plt.plot(df['50_days_average'])
    plt.plot(df['200_days_average'])
    plt.legend(['Close','50_days_avg','200_days_avg'])
    plt.title('Evolution of the MTUM over time')
    plt.show()


if __name__ == '__main__':
    df = load_and_join()
    fill_joined_missing_fields(df)
    calculate_averages_50_200(df)
    df = df.iloc[200:,:]
    visualize_close_50_200(df)
