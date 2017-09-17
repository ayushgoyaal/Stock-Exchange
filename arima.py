#-----------------------ARIMA MODEL ----------------Feature engineering---------
import sys
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.arima_model import ARMAResults
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller
df = pd.read_csv('2015.csv')
df['Date'] = pd.to_datetime(df['Date'])
#print(df['Date'])
ts = df['Close Price']
ts.dropna(inplace=True)
dDays = 10
#plt.figure(figsize=(18,4))
#original plotting of data
#plt.title('Closing Value vs Date')
#plt.xlabel('Date')
#plt.ylabel('Closing Value')
#plt.plot(df['Date'],df['Close Price'])
#plt.show()
def test_stationarity(timeseries):#checks data is stationary or not
    
    #Determing rolling statistics
    #plt.figure(figsize=(18,4))
    #rolmean = pd.rolling_mean(timeseries, window=dDays)
    #rolstd = pd.rolling_std(timeseries, window=dDays)

    #Plot rolling statistics:
    #orig = #plt.plot(timeseries, color='blue',label='Original')
    #mean = #plt.plot(rolmean, color='red', label='Rolling Mean')
    #std = #plt.plot(rolstd, color='black', label = 'Rolling Std')
    #plt.legend(loc='best')
    #plt.title('Rolling Mean & Standard Deviation')
    #plt.show(block=False)
    print ('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print (dfoutput)
    
test_stationarity(ts)
ts_log = np.log(ts)
ts_log.dropna(inplace=True)
#print(ts_log)
#moving avg
moving_avg = pd.rolling_mean(ts_log,dDays)
#plt.figure(figsize=(18,4))
#plt.plot(ts_log, color='blue',label='logged Original')
#plt.plot(moving_avg, color='red', label='logged Rolling Mean')
#plt.legend(loc='best')
#plt.show()
ts_log_moving_avg_diff = ts_log - moving_avg
ts_log_moving_avg_diff.dropna(inplace=True)
test_stationarity(ts_log_moving_avg_diff)

#weighted avg
expwighted_avg = pd.ewma(ts_log, halflife=12)
#plt.figure(figsize=(18,4))
#plt.plot(ts_log, color='blue',label='logged Original')
#plt.plot(expwighted_avg, color='red', label='logged weighted Mean')
ts_log_ewma_diff = ts_log - expwighted_avg
test_stationarity(ts_log_ewma_diff)


#1st order differencing
ts_log_diff = ts_log - ts_log.shift()
ts_log_diff.dropna(inplace=True)
#plt.figure(figsize=(18,4))
#plt.plot(ts_log_diff, color='blue',label='logged difference')
#plt.legend(loc='best')
#plt.show()
ts_log_diff.dropna(inplace=True)
test_stationarity(ts_log_diff)



ts.to_csv("ts.csv",header=True)
ts_log.to_csv("ts_log.csv",header=True)
ts_log_moving_avg_diff.to_csv("ts_log_moving_avg_diff.csv",header=True)
ts_log_ewma_diff.to_csv("ts_log_ewma_diff.csv",header=True)





data = list()
mydict={}
date=df['Date']
mydict={'Date':date[9:],'ts':ts[9:]}
mydict1={'Date':date[9:],'ts_log':ts_log[9:]}
mydict2={'Date':date[9:],'ts_log_moving_avg_diff':ts_log_moving_avg_diff}
mydict3={'Date':date[9:],'ts_log_ewma_diff':ts_log_ewma_diff[9:]}
#mydict2={,'ts_log':ts_log[9:],'ts_log_moving_avg_diff':ts_log_moving_avg_diff,'ts_log_ewma_diff':ts_log_ewma_diff[9:]}
train_x=pd.DataFrame(mydict)
train_x1=pd.DataFrame(mydict1)
train_x2=pd.DataFrame(mydict2)
train_x3=pd.DataFrame(mydict3)
train_x = train_x.set_index('Date')
train_x1 = train_x1.set_index('Date')
train_x2 = train_x2.set_index('Date')
train_x3 = train_x3.set_index('Date')
data.append(train_x)
data.append(train_x1)
data.append(train_x2)
data.append(train_x3)
#print(data[0])
#print(train_x)





for currentdata in data: 
    TS = currentdata
    final_aic = math.inf
    final_bic=math.inf
    final_order=(0,0,0)
    #print(final_order)
    for p in range(0,3):
        for d in range(1,3):
            for q in range(0,3):
                try:
                    model = ARIMA(TS,order=(p,d,q))
                    #print(p,q,d)
                    results_ARIMA = model.fit(disp=-1)
                    current_aic = results_ARIMA.aic  #compute AIC error on the model formed so far
                    current_bic = results_ARIMA.bic  #compute BIC error on the model formed so far
                    #print(p,d,q)
                    if (current_bic < final_bic and current_aic < final_aic):#if current error is minimum then update all the order,model etc
                        final_aic = current_aic
                        final_bic = current_bic
                        final_order = (p,d,q)
                        '''results_final_ARIMA = final_arima.fit()
                        print(results_final_ARIMA.summary())
                        #final_accuracy = accuracy(model)'''
                except (ValueError,RuntimeError, TypeError, NameError):
                    print ("")
                except np.linalg.linalg.LinAlgError :
                    print("")
                except IndexError:
                    print("")
    print(TS.columns.values)
    final_arima = ARIMA(TS,order=final_order)
    print("AIC=",final_aic)
    print("BIC=",final_bic)
    print("(P,d,q)=",final_order)
    #print(final_arima)

#############ARIMA-------------Result##########################
df = pd.read_csv('2016.csv')
#print(df['Date'])
test = df['Close Price']
test.dropna(inplace=True)
history = [x for x in data[0].values]
#print(history)
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(2, 1, 1))  
    results_ARIMA = model.fit(disp=-1)
    output=results_ARIMA.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    #print(t)
    print('predicted=%f, expected=%f' % (yhat, obs))
    
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
plt.figure(figsize=(18,4))
plt.plot(test,label="original")
plt.plot(predictions, color='red',label="predicted")
plt.legend(loc='best')
plt.show()

