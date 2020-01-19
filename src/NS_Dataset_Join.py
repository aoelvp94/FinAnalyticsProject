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

#Consigue los weights para la diferenciacion!
def getWeights_FFD(d,size):
    w=[1.]
    for k in range(1,size):
        w_=-w[-1]/k*(d-k+1)
        w.append(w_)
    w=np.array(w[::-1]).reshape(-1,1)
    return w

#Funcion aux para pesos de FFD
def plotWeights(dRange,nPlots,size):
    w=pd.DataFrame()
    for d in np.linspace(dRange[0],dRange[1],nPlots):
        w_=getWeights_FFD(d,size=size)
        w_=pd.DataFrame(w_,index=range(w_.shape[0])[::-1],columns=[d])
        w=w.join(w_,how='outer')
    ax=w.plot()
    ax.legend(loc='upper right');plt.show()
    return

#Diferenciamos la serie! d es el orden de diferenciacion. "Thres" (threshold) maneja la acceptabilidad de las exclusiones. No modificar.
def fracDiff(series,d,thres=0.01):

    w=getWeights_FFD(d,series.shape[0])

    w_=np.cumsum(abs(w))
    w_/=w_[-1]
    skip=w_[w_>thres].shape[0]

    df={}
    for name in series.columns:
        seriesF,df_=series[[name]].fillna(method='ffill').dropna(),pd.Series()
        for iloc in range(skip,seriesF.shape[0]):
            loc=seriesF.index[iloc]

            df_[loc]=np.dot(w[-(iloc+1):,:].T,seriesF.loc[:loc])[0,0]
        df[name]=df_.copy(deep=True)
    df=pd.concat(df,axis=1)
    return df

#Funcion para buscar el mejor d
def plotMinFFD(df):
    from statsmodels.tsa.stattools import adfuller
    import numpy.ma as ma
    out=pd.DataFrame(columns=['adfStat','pVal','lags','nObs','95% conf','corr'])
    for d in np.linspace(0,1,21):
        df1=np.log(df[['Close']]).resample('1D').last() # Pasar a observaciones diarias
        df2=fracDiff(df1,d,thres=.01)
        corr = ma.corrcoef(ma.masked_invalid(df1.loc[df2.index,'Close']), ma.masked_invalid(df2['Close']))[0,1]
        df2=adfuller(df2['Close'],maxlag=1,regression='c',autolag=None)
        out.loc[d]=list(df2[:4])+[df2[4]['5%']]+[corr] # Aportar valores criticos
    out[['adfStat','corr']].plot(secondary_y='adfStat')
    plt.axhline(out['95% conf'].mean(),linewidth=1,color='r',linestyle='dotted')
    plt.show()
    return out

#Obtiene los factores de tiempo para restar importancia a las observaciones.
def getTimeDecay(tW,clfLastW):
    clfW=tW.sort_index().cumsum()
    if clfLastW>=0:slope=(1.-clfLastW)/clfW.iloc[-1]
    else:slope=1./((clfLastW+1)*clfW.iloc[-1])
    const=1.-slope*clfW.iloc[-1]
    clfW=const+slope*clfW
    clfW[clfW<0]=0
    # print(const,slope)
    return clfW


if __name__ == '__main__':
    df = load_and_join()
    fill_joined_missing_fields(df)  
    
    #Diferenciacion fraccionaria: Buscando el mejor d, d* = 0.1
    plt.figure(1)
    out = plotMinFFD(df)
    
    #Usando la diferenciacion con d = 0.05
    df_ffd = fracDiff(df,0.05)    
    
    #Ploteando series
    df = labeling_df(df)
    plt.figure(2)
    visualize_close_50_200(df)
    
    df_ffd = labeling_df(df_ffd)
    plt.figure(3)
    visualize_close_50_200(df_ffd)
    
    time_decay = getTimeDecay(df[['Close']],0)
    
    #1 Revisar el labeling!
    #2 Falta aplicar el time_decay (y precisarlo!)
    #3 Elegir si usar ffd (tiene menos obs!)
