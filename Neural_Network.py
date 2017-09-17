
# coding: utf-8

# In[40]:

import sys
#from class_vis import prettyPicture

import matplotlib.pyplot as plt
#import copy
import numpy as np
#import pylab as pl
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier


# In[41]:

def get_data(filename,percent_train):
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
    #########################Larry William’s R%####################
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


# In[70]:




# In[ ]:




# In[ ]:




# In[66]:

def calculate_loss(model,X,y):
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    examples=len(X);
    # Forward propagation to calculate our predictions
    z1 = np.dot(X,W1) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(a1,W2) + b2
    a2 = np.tanh(z2)
    z3 = np.dot(a2,W3) + b3
    scores = np.exp(z3)
    prbs = scores / np.sum(scores, axis=1, keepdims=True)
    # Calculating the loss
    corect_logprbs = -np.log(prbs[range(examples), y])
    loss = np.sum(corect_logprbs)
   
    return 1./examples * loss

def build_model(train_set):
    # Gradient descent parameters
    learn_rate = 0.00001 # learning rate for gradient descent
    reg_lembde = 0.001 # regularization strength
    nn_hdim1 = 15
    passes = 2000
    nn_output_dim = 2
    nn_hdim2 = 15
    X = train_set [:,0:10]
    nn_input_dim = len (X[0])
    y= train_set [:,11] 
    y=np.asarray(y,dtype=int)
    np.random.seed(0)
    num_examples = len (X)
 
  # Initialize the parameters to random values
    W1 = np.random.randn(nn_input_dim, nn_hdim1) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim1))
    W2 = np.random.randn(nn_hdim1, nn_hdim2) / np.sqrt(nn_hdim1)
    b2 = np.zeros((1, nn_hdim2))
    W3 = np.random.randn(nn_hdim2, nn_output_dim) / np.sqrt(nn_hdim2)
    b3 = np.zeros((1, nn_output_dim))

    model = {}
    
    # Gradient descent. For each batch...
    for i in range(0, passes):

        # Forward propagation
        z1 = np.dot(X,W1) + b1
        a1 = np.tanh(z1)
        z2 = np.dot(a1,W2) + b2
        a2 = np.tanh(z2)
        z3 = np.dot(a2,W3) + b3
        z3 = np.array(z3,dtype=np.float128)
        scores=np.array([[]],dtype=np.float128)
        scores = np.exp(z3)
        prbs = scores / np.sum(scores, axis=1, keepdims=True)
        
        # Backpropagation
        delta4 = prbs
        delta4[range(num_examples), y] -= 1
        dW3=np.dot(a2.T,delta4)
        db3=np.sum(delta4, axis=0, keepdims=True)
        delta3 = delta4.dot(W3.T) * (1 - np.power(a2, 2))
        dW2=np.dot(a1.T,delta3)
        db2=np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)
        
        # Add regularization terms 
        dW3 += reg_lembde * W3
        dW2 += reg_lembde * W2
        dW1 += reg_lembde * W1

        # Gradient descent parameter update
        W1 += -learn_rate * dW1
        b1 += -learn_rate * db1
        W2 += -learn_rate * dW2
        b2 += -learn_rate * db2
        W3 += -learn_rate * dW3
        b3 += -learn_rate * db3

       
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3}
        
        if i % (500) == 0:
           print ("Loss after %i: %f" %(i, calculate_loss(model,X,y)))
    	
    return model


features_train,pd_train= get_data('train.csv',1.0)
trainset_array=np.asarray(features_train)
model = build_model(trainset_array)



###########################################################################################



# In[73]:

def predict(model, x):
    W1, b1, W2, b2, W3, b3 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3']
    # Forward propagation
    z1 = np.dot(x,W1) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(a1,W2) + b2
    a2 = np.tanh(z2)
    z3 = np.dot(a2,W3) + b3
    exp_scores = np.exp(z3)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return np.argmax(probs, axis=1)

features_test,pd_test = get_data('test.csv',1.0)
labels_test=features_test[:,11]
features_test=features_test[:,:10]
features_test=np.asarray(features_test)
predicted = predict(model,features_test)
acc = accuracy_score(predicted, labels_test)
print (acc)


# In[ ]:



