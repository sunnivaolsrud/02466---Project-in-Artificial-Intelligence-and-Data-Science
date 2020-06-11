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
import pickle
from randomforrest import train_test_RF

#Import variables from other scripts
from Process_data import A, ytrue, yhat
from Process_data import y_train, y_test, X_train, X_test, train_index, test_index

#%% Random forrest
model = pickle.load(open("./RF.sav", 'rb'))
train_acc, test_acc, yhat_rf = train_test_RF(model, X_train, y_train, X_test, y_test, train = False)
print("Training accuracy: %s" %train_acc)
print("Test accuracy:     %s" %test_acc)


#Define equal class variable with two_year_recid[test_index], race[test_index] and predictions on test set
A = A.values[test_index]
A = pd.DataFrame(A)
ytrue = ytrue.values[test_index]
ytrue = pd.DataFrame(ytrue)
Equal_rf = equal(A[0], yhat_rf[0], ytrue[0], N=400)

group = 'African-American'
p0 = [1,1]
T = np.arange(0,1.001,0.001)

#Equalised odds
FP_TP_rate_A, FP_TP_rate_C, ACC_A, ACC_C, Fpr_cau, Tpr_cau, Fpr_afri, Tpr_afri, postconf_afri, postconf_cau= equal_odds(T, Equal_rf, group, p0, plot = True)


#Equal opportunity
#rateA and rateC includes both FPR and TPR
Sigma = 0.001
max_acc, maxtA, maxtC, rateA, rateC, conf_before_A, conf_before_C, conf_after_A, conf_after_C = equal_opportunity(Sigma, T, Equal_rf, plot = True)
 

#Confusion matricer

#Before
plot_conf(conf_before_A)
plt.show()
plot_conf(conf_before_C)
plt.show()
#After equalised odds
plot_conf(postconf_afri)
plt.show()
plot_conf(postconf_cau)
plt.show()

#After equal opportunity
plot_conf(conf_after_A)
plt.show()
plot_conf(conf_after_C)
plt.show()