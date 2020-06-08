# -*- coding: utf-8 -*-
"""
Created on Fri May  8 15:00:01 2020

@author: Værksted Matilde
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from Process_data import A, ytrue, yhat
from POST import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#Creat
np.random.seed(10)
# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=100,
                               criterion = 'entropy',
                               min_samples_split=2,
                               bootstrap = True,
                               max_features = None)



#Define X and y
y = twoyears.data['decile_score.1'].values
X = twoyears.data.drop(['decile_score.1'],axis = 1).values

#split dataset 
split = ShuffleSplit(n_splits=1, test_size=0.2)
split.get_n_splits(X, y)
train_index, test_index = next(split.split(X, y)) 
X_train, X_test = X[train_index], X[test_index] 
y_train, y_test = y[train_index], y[test_index]


# Fit on training data
model.fit(X_train, y_train)

#predict
rf_predictions_train = model.predict(X_train)
rf_predictions = model.predict(X_test)


# Probabilities for score = 1
yhat_rf = model.predict_proba(X_test)[:, 1]
yhat_rf = pd.DataFrame( yhat_rf)


#acc
print(np.sum(rf_predictions_train ==y_train)/len(y_train))
print(np.sum(rf_predictions ==y_test)/len(y_test))


#Define equal class variable 
#ytrue = two-years-recidivism
#yhat = predicted probability 
#A: Race
A = A.values[test_index]
A = pd.DataFrame(A)
ytrue = ytrue.values[test_index]
ytrue = pd.DataFrame(ytrue)
Equal_rf = equal(A[0], yhat_rf[0], ytrue[0])

#ROC 
T = np.arange(0,1.1,0.001)
Fpr_rf, Tpr_rf= Equal_rf.ROC_(T)

Fpr_cau, Tpr_cau = Fpr_rf['Caucasian'], Tpr_rf['Caucasian']

Fpr_afri, Tpr_afri = Fpr_rf['African-American'], Tpr_rf['African-American']

plt.plot(Fpr_rf['Caucasian'], Tpr_rf['Caucasian'], label = 'caucasian')
plt.plot(Fpr_rf['African-American'], Tpr_rf['African-American'], label = 'african-american')
plt.legend()
plt.show()


#POST
#accuracies of ROC curve
accs_rf = Equal_rf.acc_(T)

#index of maximum accruracy for both groups
max_idx = [np.argmax(accs_rf[0]), np.argmax(accs_rf[1])]

#threshold with highest accuracy for both groups
maxt_afri = T[max_idx[0]]
maxt_cau = T[max_idx[1]]

#point in (fpt, tpr) for both groups
maxp_afri = [Fpr_afri[max_idx[0]], Tpr_afri[max_idx[0]]]
maxp_cau = [Fpr_cau[max_idx[1]], Tpr_cau[max_idx[1]]]


#plot with point

j = 60 #idx of second point of caucasian
k = 700
plt.plot(Fpr_rf['African-American'], Tpr_rf['African-American'], label = 'african-american')
plt.plot([maxp_cau[0], Fpr_cau[j]], [maxp_cau[1], Tpr_cau[j]],'go', label = 'Caucasian')
plt.plot(Fpr_rf['Caucasian'], Tpr_rf['Caucasian'], 'g',label = 'Caucasian')
plt.plot([Fpr_afri[k],maxp_afri[0]], [Tpr_afri[k],maxp_afri[1]],'bo', label = 'African-American')
#plt.plot(0,0,'ro')
#plt.plot(Fpr_cau[j], Tpr_cau[j],'bo')
plt.plot(inter[0], inter[1],'r*', label = 'Intersection')
plt.legend()
plt.show()

#Compute lines between points
l_afri = line([Fpr_afri[k],Tpr_afri[k]],maxp_afri)
l_cau = line(maxp_cau,[Fpr_cau[j], Tpr_cau[j]])

#Compute intersection between lines
inter = intersection(l_afri, l_cau)

#compute percentage
perc_afri = percentile([Fpr_afri[k], Tpr_afri[k]],[inter[0], inter[1]],maxp_afri)
perc_cau = percentile(maxp_cau,[inter[0], inter[1]],[Fpr_cau[j], Tpr_cau[j]])

#confmatrix and acc before 
preconf_afri = Equal_rf.conf_(0.5,0)
preconf_cau = Equal_rf.conf_(0.5,1)
preaccu = Equal_rf.acc_([0.5])
Equal_rf.FP_TP_rate(preconf_afri)
Equal_rf.FP_TP_rate(preconf_cau)


#conf matrix and acc after
postconf_afri = Equal_rf.calc_ConfusionMatrix(T[k], maxt_afri, 0,perc_afri)
postconf_cau =  Equal_rf.calc_ConfusionMatrix(maxt_cau,T[j], 1,perc_cau)

Equal_rf.FP_TP_rate(postconf_afri)
Equal_rf.FP_TP_rate(postconf_cau)
