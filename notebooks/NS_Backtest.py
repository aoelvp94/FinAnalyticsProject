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
         

#Introducimos labels al dataset, calculando medias 50 y 200 dias.
'''(REVEER!)'''    
def labeling_df(df):
    df['50_days_average'] = df.iloc[:,3].rolling(window=50).mean()
    df['200_days_average'] = df.iloc[:,3].rolling(window=200).mean()
    df.loc[df['50_days_average'] >= df['200_days_average'], 'Buy/Sell'] = -1 #Si la media de corto plazo supera a la de largo, es posicion de sell
    df.loc[df['50_days_average'] < df['200_days_average'], 'Buy/Sell'] = 1 #Si la media de corto plazo esta por debajo de la de largo, es posicion de buy
    return df
 
#Visualizacion de la evolucion de precios y las medias.
def visualize_close_50_200(df):
    plt.plot(df['Close'])
    plt.plot(df['50_days_average'])
    plt.plot(df['200_days_average'])
    plt.legend(['Close','50_days_avg','200_days_avg'])
    plt.title('Evolution of MTUM ETF over time')
    plt.show()


def back_test(df,mode,start_date,end_date):
    df1 = df.loc[start_date:end_date]
    ''' Modos disponibles:
        1- 'Simple' -> Simple: Toma la performance de una estrategia "Buy and Hold". Compra en t=1, vende en t=T (Ultimo dia)
        2- 'Signal' -> Señal: Compra y vende segun el campo 'Buy/Sell'. Size es 1 siempre.
        3- 'BetS' -> Bet Sizing: Usa la señal del campo Buy/Sell y la pondera por la probabilizada del campo 'BetSize'. '''
        
    if mode == 'Simple':
        print('Modo Simple!')
        opening_price = df1['Close'].first('D')[0]
        closing_price = df1['Close'].last('D')[0]
        result = (closing_price - opening_price) / opening_price
        print('Opening Price: $ {:.2f}'.format(opening_price))
        print('Closing Price: $ {:.2f}'.format(closing_price))
        print('Rate of Return: {:.2%}'.format(result))
        
    elif mode == 'Signal':
        print('Modo via señales!')
        trades = pd.DataFrame(columns=['Opening_Price','Closing_Price','Return','Type'])
        position = 0
        for i in range(len(df1)):
            if position == 0:
                position = df1.iloc[0,12]
                open_price = df1.iloc[0,3]
            elif position == 1 and (df1.iloc[i,12] == -1 or i == (len(df1)-1)):
                # Cambio de Buy a Sell
                result = (df1.iloc[i,3] - open_price) / open_price
                trades = trades.append({'Opening_Price':open_price,'Closing_Price':df1.iloc[i,3],'Return':result,'Type':'Long'}, ignore_index=True)
                open_price = df1.iloc[i,3]
                
            elif position == -1 and (df1.iloc[i,12] == 1 or i == (len(df1)-1)):
                # Cambio de Sell a Buy
                result = (open_price - df1.iloc[i,3]) / open_price
                trades = trades.append({'Opening_Price':open_price,'Closing_Price':df1.iloc[i,3],'Return':result,'Type':'Short'}, ignore_index=True)
                open_price = df1.iloc[i,3]
                
            position = df1.iloc[i,12]
        
        print('*********** Trades ejecutados: **************')
        print('*********************************************')
        for i in range(len(trades)):
            print('Trade {}, tipo: {}'.format(i,trades.iloc[i,3]))
            print('Abrio al precio de $ {:.2f} y cerro en $ {:.2f}'.format(trades.iloc[i,0],trades.iloc[i,1]))
            print('Retorno: {:.2%}'.format(trades.iloc[i,2]))
            print('*********************************************')
        
        print('Resultado del Portfolio: {:.2%}'.format(trades['Return'].sum()))
                                                 
    elif mode == 'BetS':
        print('Modo via señales y bet sizing!')
        trades = pd.DataFrame(columns=['Opening_Price','Closing_Price','Return','Type','Bet Size'])
        position = 0
        for i in range(len(df1)):
            if position == 0:
                position = df1.iloc[0,12]
                open_price = df1.iloc[0,3]
                bet_size = df1.iloc[0,13]
            elif position == 1 and (df1.iloc[i,12] == -1 or i == (len(df1)-1)):
                # Cambio de Buy a Sell
                result = ((df1.iloc[i,3] - open_price) / open_price)*bet_size
                trades = trades.append({'Opening_Price':open_price,'Closing_Price':df1.iloc[i,3],'Return':result,'Type':'Long','Bet Size':bet_size}, ignore_index=True)
                open_price = df1.iloc[i,3]
                
            elif position == -1 and (df1.iloc[i,12] == 1 or i == (len(df1)-1)):
                # Cambio de Sell a Buy
                result = ((open_price - df1.iloc[i,3]) / open_price)*bet_size
                trades = trades.append({'Opening_Price':open_price,'Closing_Price':df1.iloc[i,3],'Return':result,'Type':'Short','Bet Size':bet_size}, ignore_index=True)
                open_price = df1.iloc[i,3]
                
                
            position = df1.iloc[i,12]
            bet_size = df1.iloc[i,13]
        
        print('*********** Trades ejecutados: **************')
        print('*********************************************')
        for i in range(len(trades)):
            print('Trade {}, tipo: {}'.format(i,trades.iloc[i,3]))
            print('Abrio al precio de $ {:.2f} y cerro en $ {:.2f}, con un size de {:.2f}'.format(trades.iloc[i,0],trades.iloc[i,1],trades.iloc[i,4]))
            print('Retorno: {:.2%}'.format(trades.iloc[i,2]))
            print('*********************************************')
        
        print('Resultado del Portfolio: {:.2%}'.format(trades['Return'].sum()))
        
        
    else:
        print('Modo incorrecto!')
            
            
            
if __name__ == '__main__':
    df = load_and_join()
    fill_joined_missing_fields(df)
    df = df['2014-01-31':]
    df['BetSize'] = 0.5
    
   
    #Ploteando series
    df = labeling_df(df)
    plt.figure(2)
    visualize_close_50_200(df)
    
    
    back_test(df,'Simple','2018-01-02','2019-01-02')
    print('***************************************')
    print('---------------------------------------')
    print('***************************************')
    back_test(df,'Signal','2018-01-02','2019-01-02')
    print('***************************************')
    print('---------------------------------------')
    print('***************************************')
    back_test(df,'BetS','2015-01-02','2019-01-02')   