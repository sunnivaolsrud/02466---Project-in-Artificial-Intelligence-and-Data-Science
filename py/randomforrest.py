# -*- coding: utf-8 -*-
"""
Created on Fri May  8 15:00:01 2020

@author: Værksted Matilde
"""

from sklearn.ensemble import RandomForestClassifier
from POST import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Equal_opportunity import equal_opportunity 


#%% Functions
def train_test_RF(model, X_train, y_train, X_test, y_test): 
    
# Fit on training data
    model.fit(X_train, y_train)
    
    #predict test and training 
    rf_predictions_train = model.predict(X_train)
    rf_predictions = model.predict(X_test)
    
    # Probabilities for score = 1, test
    yhat_rf = model.predict_proba(X_test)[:, 1]
    yhat_rf = pd.DataFrame(yhat_rf)
    
    #Training and test accurracy
    train_acc = np.sum(rf_predictions_train ==y_train)/len(y_train)
    test_acc = np.sum(rf_predictions ==y_test)/len(y_test)
    
    return train_acc, test_acc, yhat_rf
    

def percentile(p1,p2,p3):
    l1 = np.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    l2 = np.sqrt((p1[0]-p3[0])**2+(p1[1]-p3[1])**2)

    return l1/l2

def estimate(x,y):
    """
    Computes and returns parameters of ax+b
    x: x-koefficients of both points
    y: y-koefficients of both points    
    """
    a1 = (y[1]-y[0])/(x[1]-x[0]+0.000000000000000000000000000001)
    b1 = y[1]-a1*x[1]      
    return a1, b1


def equal_odds(T, CLVar, group, p0, plot = False): 
    """
    Compute FPR and TPR of both groups of protected attribute (in given class)
    Find equalised odds predictor
    Plot equalised odds predictor with ROC curves
    Compute confusions matrices, FPR/TPR of equalised odds predictor. 
    Compute confusions matrices, FPR/TPR with t = 0.5
    
    Input: 
    T: List of thresholds
    CLVar: Class variable of the class "equal"
    group: The group with the lowest ROC curve
    p0: Whether to hold (0,0) or (1,1)
    
    Output: 
    FP_TP_rate_A
    FP_TP_rate_C
    ACC_A
    ACC_C
    Fpr_cau
    Tpr_cau
    Fpr_afri
    Tpr_afri
    """
    
    Fpr_rf, Tpr_rf= CLVar.ROC_(T, models = True)
    Fpr_cau, Tpr_cau = Fpr_rf['Caucasian'], Tpr_rf['Caucasian']
    Fpr_afri, Tpr_afri = Fpr_rf['African-American'], Tpr_rf['African-American']

    #accuracies of all values of t
    accs_rf = Equal_rf.acc_(T, models = True)

    #Find point with highest acc on "lowest" ROC curve
    accs_rf = accs_rf[group] #all acc
    max_idx = np.argmax(accs_rf) #idx for highest acc
    maxt = T[max_idx] #t for highest acc
    maxp = [Fpr_afri[max_idx], Tpr_afri[max_idx]] #(fpr,tpr) with highest acc

    #Compute parameters of line between p0 and all points on the highest ROC curve (one at a time)
    #For all lines, estimate the y-kooordinate (TPR) of maxp. 
    yall = np.empty(len(Fpr_cau)-2)
    for i in range(len(Fpr_cau)-2):
        a1,b1 = estimate([p0[0],Fpr_cau[i+1]], [p0[1],Tpr_cau[i+1]])
        y = a1*maxp[0]+b1
        yall[i] = y

    #Compute absolute distance between the true TPR of maxp and the estimated. 
    diff = []
    for i in range(len(yall)): 
        diff.append(abs(yall[i]-maxp[1]))
    
    #Find optimal estimate of TPR of maxp
    minidx = np.argmin(diff) + 1
    p1 = [Fpr_cau[minidx], Tpr_cau[minidx]] #add 1 to get the idx that works with all
                                        #except diff and yall

    #Find percentages and thresholds of classifier that satisfies equalised odds. 
    t_afri = T[max_idx]
    
    percent = percentile(p0, maxp, p1)
    if p0 == [0,0]:
        t_cau1 = T[-1] 
    else: 
        t_cau1 = T[0]   
    t_cau2 = T[minidx]
    
    postconf_afri =  CLVar.conf_models(t_afri,0)
    postconf_cau = CLVar.calc_ConfusionMatrix(t_cau1,t_cau2, 1,percent)

    FP_TP_rate_A = CLVar.FP_TP_rate(postconf_afri)
    FP_TP_rate_C = CLVar.FP_TP_rate(postconf_cau)
    
    ACC_A = CLVar.acc_with_conf(postconf_afri)
    ACC_C = CLVar.acc_with_conf(postconf_cau)

    if plot == True: 
        
        #plot with max accu point
        plt.plot(Fpr_rf['Caucasian'], Tpr_rf['Caucasian'],'g', label = 'Caucasian')
        plt.plot(Fpr_rf['African-American'], Tpr_rf['African-American'],'b', label = 'African-american')
        plt.plot(maxp[0], maxp[1], 'bo', label = "Max accu")
        plt.plot(p0[0], p0[1], 'go')
        plt.plot(p1[0], p1[1], 'go')
        #plt.plot([p1[0],p0[0]], [p1[1],p0[1]], 'go')
        plt.plot([p0[0],p1[0]], [p0[1], p1[1]], 'r')
        plt.legend()
        plt.show()
        
    return FP_TP_rate_A, FP_TP_rate_C, ACC_A, ACC_C, Fpr_cau, Tpr_cau, Fpr_afri, Tpr_afri

#%%
    
# Define random forrest model
model = RandomForestClassifier(n_estimators=100,
                               criterion = 'entropy',
                               min_samples_split=2,
                               bootstrap = True,
                               max_features = None)


#Import variables from other scripts
from Process_data import A, ytrue, yhat
from Process_data import y_train, y_test, X_train, X_test, train_index, test_index

#train and test model
train_acc, test_acc, yhat_rf = train_test_RF(model, X_train, y_train, X_test, y_test)
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
FP_TP_rate_A, FP_TP_rate_C, ACC_A, ACC_C, Fpr_cau, Tpr_cau, Fpr_afri, Tpr_afri= equal_odds(T, Equal_rf, group, p0, plot = True)


#Equal opportunity
#rateA and rateC includes both FPR and TPR
Sigma = 0.001
max_acc, maxtA, maxtC, rateA, rateC, conf_before_A, conf_before_C, conf_after_A, conf_after_C = equal_opportunity(Sigma, T, Equal_rf, plot = True)

def plot_conf(conf_mtrx):
    """
    input: confusion matrix of type conf(tp, fp, tn, fn)
    """
    conf = np.empty([2,2])
    conf[0,0] = conf_mtrx[0]
    
    conf[0,1] = conf_mtrx[1]
    conf[1,0] = conf_mtrx[3]
    conf[1,1] = conf_mtrx[2]
    
    df = pd.DataFrame(conf, index = ["Predictive (1)", "Predictive (0)"], columns = ["Actual (1)","Actual (0)"])
    import seaborn as sns

    sns.heatmap(df/np.sum(df), annot=True, 
            fmt='.2%', cmap='Blues')
    return conf 
                

conf = plot_conf(conf_after_C)

     
# np.asarray(conf_after_C).reshape(2,2,order = 'C')

import pickle

filename = 'RF.sav'
pickle.dump(model, open(filename, 'wb'))

# load

