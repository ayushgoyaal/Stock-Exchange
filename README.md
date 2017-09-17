ALL CODE DEVELOPED IN PYTHON

###############Input files################
test.csv
train.csv
2015.csv
2016.csv
arima.py
standard_approaches.py
Neural_Network.py
LogisticReg.py
ts_log.csv
ts_log_ewma_diff.csv
ts_log_moving_avg_diff.csv


##############Libraries Used###############
sys
math
matplotlib.pyplot
numpy
sklearn
pandas
statsmodels


##############standard_approaches.py#######
To run: python standard_approaches.py
Give accuracy for 4 standard approaches:
Naive bayes
bagging meta estimator
adaboost
SVM


##############Neural_Network.py############
To run: python3 Neural_Network.py
Gives accuracy as the output


#############arima.py#####################
to run:python3 arima.py
Gives a graph as output showing relation between test and predicted values
Gives MSE error
Gives Convergence Warning for non-stationary models(while training)---since arima works for stationary models only
Gives three files for traiing purpose ts_log.csv,ts_log_ewma_diff.csv,ts_log_moving_avg_diff.csv
#############LogisticReg.py#################
to run:python3 LogisticReg.py
gives error rate and entropy as answer

###########train.csv######################
contain data of Reliance Industries for 15 years of data i.e. from 2000 to 2015


###########test.csv######################
contain data of Reliance Industries for 2 years of data i.e. from 2016 and 2017

##########2015.csv and 2016.csv#######################
small datasets used for arima---(reliance industries LTD)









