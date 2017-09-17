import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def get_data_test(filename,label):
    df = pd.read_csv(path, header=None)
    Ddf=df.shape[1]
    x = df.iloc[:,0:Ddf-2]
    y = df.iloc[:,Ddf-1:Ddf]
    
    df[Ddf] = df.apply(lambda row: 1 if row[Ddf-1] == label else 0, axis=1)
    #df[Ddf] = df.apply(lambda row: 1 if row[Ddf-1] == " g" else 0, axis=1)
    
    yt=np.array(df[Ddf])
   
    X=np.array(x)
   # print(df)
    for i in range(1,x.shape[1]):
        X[:,i]=(X[:,i]-np.amin(X[:,i]))/(np.amax(X[:,i])-np.amin(X[:,i]))
   
    T=np.array(y).T
    T=T.reshape(T.shape[1])
    print((yt!=T).sum())
    N,D=X.shape
    ones = np.ones((N, 1))
    Xb = np.concatenate((ones, X), axis=1)
    
    return Xb, yt



def get_data(filename):
    days=10
    #data = np.genfromtxt('/home/ayush/subj2/post_ml/assign_2/train.csv', delimiter=',')
    #print(data[1]) print nan values for strings
    train_data =pd.read_csv(filename,index_col="Date",parse_dates=True)
    #date=train_data['Date'].values
    train_data=train_data.dropna()
    close_price=train_data['Close Price']
    low_price=train_data['Low Price']
    high_price=train_data['High Price']
    len_close=len(close_price)
    #print(len_close)
    #close=close_price.plot(label='Close_price')
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
    #print(Momentum)
    '''
    Momentum=np.zeros((len_close-(days-1),1))
    for i in range((days-1),len_close):
        Momentum[i-(days-1)]=close_price[i]-close_price[i-(days-1)]
    print(Momentum)
    '''
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
    #########################Larry William####################
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
    return new_data







'''
r = np.zeros((N,1))
for i in list(range(N)):
    r[i] = np.sqrt(X[i,:].dot(X[i,]))
'''

# randomly initialize the weights


def sigmoid(z):
    return 1/(1 + np.exp(-z))




# calculate the cross-entropy error
def cross_entropy(T, Y):
    # E = 0
    # for i in xrange(N):
    #     if T[i] == 1:
    #         E -= np.log(Y[i])
    #     else:
    #         E -= np.log(1 - Y[i])
    # return E
    return (T*np.log(Y) + (1-T)*np.log(1-Y)).sum()

''' Gradeint descet'''
def fit(X,T):
    w = np.random.randn(X.shape[1])
    # calculate the model output
    z = X.dot(w)
    Y = sigmoid(z)
    
    # let's do gradient descent 100 times
    learning_rate = 0.001
    error = []
    for i in list(range(9000)):
        e = cross_entropy(T, Y)
        error.append(e)
        
     
    
        # gradient descent weight udpate with regularization
        # w += learning_rate * ( np.dot((T - Y).T, Xb) - 0.01*w ) # old
        grad=np.array((X.T).dot((T - Y)))
        if i % 10000 == 0:
            print ("entropy " ,e)
            print("gradient norm" , np.linalg.norm(grad))

        if np.linalg.norm(grad)<=0.00000000:
        	break
        #print(T-Y)
        grad_step= np.abs(sigmoid(np.linalg.norm(grad)))-0.40
        w += learning_rate*grad*grad_step- 0.0*w 
    
        # recalculate Y
        Y = sigmoid(X.dot(w))
    plt.plot(error)
    plt.title("Cross-entropy per iteration")
    plt.show()
    return w,Y


'''--------------------------------------------------------'''




  #print(X)
features_train = get_data('train.csv')
Y=features_train[:,11]
X=features_train[:,:10]
  #print(X)
  #Xt,Yt = get_data('/home/vivek/ie613 assigment/Ion_data/ionosphere_test.csv')



Xtrain,  Ytrain   =X, Y
  

T = Y

#print(X.shape,T.sha

wfit,Y=fit(X,T)
print ("Final w:", wfit)
print ("train error rate:",  np.abs(T - np.round(Y)).sum() / X.shape[0])
features_test= get_data('test.csv')
labels_test=features_test[:,11]
features_test=features_test[:,:10]


Yt=sigmoid(features_test.dot(wfit))
print ("test error rate:", np.abs(labels_test - np.round(Yt)).sum() / Yt.shape[0])
print("test entropy",cross_entropy(labels_test, Yt))

