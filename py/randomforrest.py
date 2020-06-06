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

# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=1000, 
                               bootstrap = True,
                               max_features = 'sqrt')



#Define X and y
y = twoyears.data['decile_score.1'].values
X = twoyears.data.values

#split dataset 
split = ShuffleSplit(n_splits=1, test_size=0.2)
split.get_n_splits(X, y)
train_index, test_index = next(split.split(X, y)) 
X_train, X_test = X[train_index], X[test_index] 
y_train, y_test = y[train_index], y[test_index]


# Fit on training data
model.fit(X_train, y_train)

#predict
#rf_predictions = model.predict(X_test)

# Probabilities for score = 1
yhat_rf = model.predict_proba(X_test)[:, 1]
yhat_rf = pd.DataFrame( yhat_rf)


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

plt.plot(Fpr_rf['Caucasian'], Tpr_rf['Caucasian'])
plt.plot(Fpr_rf['African-American'], Tpr_rf['African-American'])
plt.show()





