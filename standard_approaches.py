import sys
import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier


def get_data(filename,percent_train):
    days=10
    train_data =pd.read_csv(filename,index_col="Date",parse_dates=True)
    train_data=train_data.dropna()
    close_price=train_data['Close Price']
    low_price=train_data['Low Price']
    high_price=train_data['High Price']
    len_close=len(close_price)
    #########################ROLLING MEAN####################
    SMA=(pd.rolling_mean(close_price,window=days)).dropna()
    SMA = SMA.values
    #print(SMA.values)
    #SMA.plot(label='Rolling_mean',ax=close)
    #plt.show()
    #########################weighted moving average####################
    WMA=np.zeros((len_close-(days-1),1))
    for i in range((days-1),len_close):
        sum1=0.0
        for j in range((i-(days-1)),i+1):
            sum1+=(days-i+j)*close_price[j]
        sum1/=(days*(days+1))/2
        WMA[i-(days-1)]=sum1
    WMA = WMA.ravel()
    #print(WMA)
    #WMA.plot(label='WMA',ax=close)
    #########################Momentum####################
    Momentum=(close_price.shift(days-1)-close_price).dropna()
    Momentum = Momentum.ravel()
    #########################StochasticK####################
    StochasticK=np.zeros((len_close-(days-1),1))
    minn=np.zeros((len_close-(days-1),1))
    maxx=np.zeros((len_close-(days-1),1))
    #print(min(close_price[i-(days-1):i+1]))
    
    for i in range((days-1),len_close):
        minn[i-(days-1)]=min(low_price[i-(days-1):i+1])
        maxx[i-(days-1)]=max(high_price[i-(days-1):i+1])
    #print(minn)
    #print(maxx)
    
    for i in range((days-1),len_close):
        StochasticK[i-(days-1)]=((close_price[i]-minn[i-(days-1)])*100.0/((maxx[i-(days-1)])-minn[i-(days-1)]))
    StochasticK = StochasticK.ravel()
    #print(StochasticK)
    
    #########################StochasticD####################
    StochasticD=np.zeros((len(StochasticK)-(days-1),1))
    for i in range((days-1),len(StochasticK)):
        sum1=0.0
        for j in range(i-(days-1),i+1):
            sum1+=StochasticK[j]                 
        sum1/=days
        StochasticD[i-(days-1)]=sum1
    StochasticD = StochasticD.ravel()
    #print(StochasticD)
    
    #########################Relative Strength Index (RSI)####################
    
    UP=np.zeros((len_close-1,1))
    DW=np.zeros((len_close-1,1))
    for i in range(1,len_close):
        temp=close_price[i]-close_price[i-1]
        if temp<0:
            UP[i-1]=0
            DW[i-1]=-temp
        else:
            UP[i-1]=temp
            DW[i-1]=0
    #print(UP)
    #print(DW)


    RSI=np.zeros((len(UP)-(days-1),1))
    for i in range((days-1),len(UP)):
        RSI[i-(days-1)]=100-(100/(1+(np.average(UP[i-(days-1):i+1])/np.average(DW[i-(days-1):i+1]))))
    RSI=RSI.ravel()
    #print(RSI) #6
    
    
    #########################Moving Average Convergence Divergence (MACD)####################

    EMA12=np.zeros((len_close,1))
    EMA12[0]=close_price[0]
    for i in range(1,len(EMA12)):
        EMA12[i]=EMA12[i-1]+( (2/(1+12)) * (close_price[i]-EMA12[i-1]) )
    EMA26=np.zeros((len_close,1))
    EMA26[0]=close_price[0]
    for i in range(1,len(EMA26)):
        EMA26[i]=EMA26[i-1]+( (2/(1+26)) * (close_price[i]-EMA26[i-1]) )
    DIFF=EMA12-EMA26
    MACD=np.zeros((len_close,1))
    MACD[0]=DIFF[0]
    for i in range(1,len(EMA26)):
        MACD[i]=MACD[i-1]+( (2/(len(MACD)+1)) * (DIFF[i]-MACD[i-1]) )
    MACD = MACD.ravel()
    #print(MACD) #7
    #########################Larry Williams R####################
    LWR=np.zeros((len_close,1))
    for i in range(len_close):
        LWR[i]=(high_price[i]-close_price[i])/(high_price[i]-low_price[i]) if (high_price[i]-low_price[i])!=0 else 0.00000000000001
    LWR = LWR.ravel()
    #print(LWR) #8

    
    #########################A/D (Accumulation/Distribution) Oscillator####################
    ADO=np.zeros((len_close-1,1))
    for i in range(1,len_close):
        ADO[i-1]=(high_price[i]-close_price[i-1])/(high_price[i]-low_price[i]) if (high_price[i]-low_price[i])!=0 else 0.00000000000001
    ADO = ADO.ravel() #9

    #########################CCI (Commodity Channel Index)####################
    M=np.zeros((len_close,1))
    for i in range(len_close):
        M[i]=(high_price[i]+low_price[i]+close_price[i])/3.0
    #print(M)
    SM=np.zeros((len(M)-(days-1),1))
    for i in range((days-1),len(M)):
        SM[i-(days-1)]=np.average(M[i-(days-1):i+1])
    #print(SM)
    D=np.zeros((len(M)-(days-1),1))
    for i in range((days-1),len(M)):
        D[i-(days-1)]=np.average(np.abs(M[i-(days-1):i+1]-SM[i-(days-1)]))
    #print(D)
    CCI=np.zeros((len(SM),1))
    for i in range(len(SM)):
        CCI[i-(days-1)]=(M[i+(days-1)]-SM[i])/(0.015*D[i])
    CCI=CCI.ravel() #10
    #Date=
    #print(Date)
    
    
    #print(len(SMA),len(WMA),len(Momentum),len(StochasticK),len(StochasticD),len(RSI),len(LWR),len(ADO),len(CCI))
    mydict={"SMA":SMA[days-1:],"WMA":WMA[days-1:],"Momentum":Momentum[days-1:],"StochasticK":StochasticK[days-1:],"StochasticD":StochasticD,"RSI":RSI[days-2:],"MACD":MACD[2*(days-1):],"LWR":LWR[2*(days-1):],"ADO":ADO[2*days-3:],"CCI":CCI[days-1:],
            "output":close_price[2*(days-1):]}
    #(mydict)
    train_x=pd.DataFrame(mydict)
    #print(train_x)
    result=train_x.values
    #print("ahjsb")
    #print(result.shape)
    new_data = np.zeros((len(result)-1,12))
    for i in range (0,len(result)-1) :
        for j in range(0,len(result[0])) :
            
            if j == 10 :
                if result[i][j] > result[i+1][j] :
                    new_data[i][j] = 1
                    new_data[i][j+1] = 0
                else :
                    new_data[i][j] = 0
                    new_data[i][j+1] = 1
        
            else :
                if result[i][j] > result[i+1][j] :
                    new_data[i][j] = 0
                else :
                    new_data[i][j] = 1
            
    #print(new_data.shape)

    #print(train_x)
    #date=date[2*]
    return new_data,train_x



features_train,pd_train= get_data('train.csv',1.0)
#print(features_train.shape)
features_test,pd_test= get_data('test.csv',1.0)
#print(features_test.shape)
labels_test=features_test[:,11]
features_test=features_test[:,:10]
labels_train=features_train[:,11]



#################Different Approaches Used With the help of python libraries########

########################## SVM #################################
print("")
clf = SVC(kernel="linear")
clf.fit(features_train[:,:10],labels_train)
pred=clf.predict(features_test)
acc = accuracy_score(pred, labels_test)
print ("SVM accuracy--",acc)
###############Naive Bayes#########
gnb = GaussianNB()
gnb_pred = gnb.fit(features_train[:,:10],labels_train).predict(features_test)
print("")
acc = accuracy_score(gnb_pred, labels_test)
print ("Naive Bayes accuracy--",acc)
############## Bagging meta-estimator#########
bagging = BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5)
bagging_pred=bagging.fit(features_train[:,:10],labels_train).predict(features_test)
print("")
acc = accuracy_score(bagging_pred, labels_test)
print ("Bagging meta-estimator accuracy--",acc)
##############AdaBoost#######################
ada = AdaBoostClassifier(n_estimators=100)
ada_pred=ada.fit(features_train[:,:10],labels_train).predict(features_test)
print("")
acc = accuracy_score(ada_pred, labels_test)
print ("AdaBoost accuracy--",acc)




