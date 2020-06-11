# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 11:16:47 2020

@author: Værksted Matilde
"""

from sklearn.ensemble import RandomForestClassifier
from POST import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Equal_opportunity import equal_opportunity 
from conf_and_rates import plot_conf
import numpy as np
from randomforrest import train_test_RF
from tensorflow.keras.models import load_model
import pickle
from Permutation_test import load_classifier

from Equalised_odds import equal_odds, estimate, percentile

#Import variables from other scripts
from Process_data import A, ytrue, yhat
from Process_data import y_train, y_test, X_train, X_test, train_index, test_index

#Prepair A anf ytrue for models
A = A.values[test_index]
A = pd.DataFrame(A)
ytrue = ytrue.values[test_index]
ytrue = pd.DataFrame(ytrue)


#%% Random forrest
model_rf = load_classifier("RF")
train_acc, test_acc, yhat_rf = train_test_RF(model_rf, X_train, y_train, X_test, y_test, train = False)
print("Training accuracy, RF: %s" %train_acc)
print("Test accuracy, RF:     %s" %test_acc)

#Define equal class variable with two_year_recid[test_index], race[test_index] and predictions on test set
Equal_rf = equal(A[0], yhat_rf[0], ytrue[0], N=400)

#Equalised odds
group = 'African-American'
p0 = [1,1]
T = np.arange(0,1.001,0.001)
FP_TP_rate_A, FP_TP_rate_C, ACC, conf_odds= equal_odds(T, Equal_rf, group, p0, plot = True)

#Equal opportunity
Sigma = 0.001
max_acc, maxtA, maxtC, rateA, rateC,conf_before, conf_after, acc_before, acc_after, rate_before= equal_opportunity(Sigma, T, Equal_rf, plot = True)
 
##Print accuracy of conf mtrx
#Before 
print("Accuracy before of 'before' classifier, African-American: %s" %acc_before[0])
print("Accuracy before of 'before' classifier, Caucasian: %s" %acc_before[1])
print("                 ")
#equalised odds classifier
print("Accuracy of equalised odds classifier, African-American %s" %ACC[0]) 
print("Accuracy of equalised odds classifier, Caucasian %s" %ACC[1]) 
print("                 ")
#equal opportunity
print("Accuracy of equal opportunity classifier, African-American %s" %acc_after[0]) 
print("Accuracy of equal opportunity classifier, Caucasian %s" %acc_after[1])
print("                 ")
print("                 ")

##Print TPR, FPR 
print("Before, African-American")
print("TPR: %s" %rate_before[0][0])
print("FPR: %s" %rate_before[0][1])
print("                 ")
print("Before, Caucasian")
print("TPR: %s" %rate_before[1][0])
print("FPR: %s" %rate_before[1][1])
print("                 ")
print("Equalised odds classfier, African-American")
print("TPR: %s" %FP_TP_rate_A[0])
print("FPR: %s" %FP_TP_rate_A[1])
print("                 ")
print("Equalised odds classfier, Caucasian")
print("TPR: %s" %FP_TP_rate_C[0])
print("FPR: %s" %FP_TP_rate_C[1])
print("                 ")
print("Equal opportunity classifier, African-American")
print("TPR: %s" %rateA[0])
print("FPR: %s" %rateA[1])
print("                 ")
print("Equal opportunity classifier, Caucasian")
print("TPR: %s" %rateC[0])
print("FPR: %s" %rateC[1])

#Confusion matricer
#Before
plot_conf(conf_before[0])
plt.show()
plot_conf(conf_before[1])
plt.show()
#After equalised odds
plot_conf(conf_odds[0])
plt.show()
plot_conf(conf_odds[1])
plt.show()

#After equal opportunity
plot_conf(conf_after[0])
plt.show()
plot_conf(conf_after[1])
plt.show()

#%% NN
def ROC_NN(A, model, pred, ytrue, X_test):

       pred = model.predict(X_test)

       equal_NN = equal(A[0],pred,ytrue, N =400)

       T = np.arange(0,1.001,0.001)

       FPR, TPR = equal_NN.ROC_(T, models = True)

       return T , equal_NN, FPR, TPR

model_nn = load_classifier("NN")
y_nn = model_nn.predict(X_test)
loss_nn_test, acc_nn_test = model_nn.evaluate(X_test, y_test, verbose = 0)
loss_nn_train, acc_nn_train = model_nn.evaluate(X_train, y_train, verbose = 0)

#Compute FPR and TPR of NN. And define class variable of equal class. 
equal_NN = equal(A[0],y_nn,ytrue, N =400)

#Equalised odds
group = 'African-American'
p0 = [1,1]
#Rate_A_NN, FP_TP_C_NN, ACC_NN, odd_confA_NN, odds_confC_NN = equal_odds(T, equal_NN, group, p0, plot = True)

#hej, hej1, hej2, hej3, hej4 = equal_odds(T, equal_NN, group, p0, plot = True)

rateANNods, rateCNNods, accNN, conf_odds = equal_odds(T, equal_NN, group, p0, plot = True)
