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
from conf_and_rates import plot_conf
import numpy as np
import pickle


#%% Functions
def train_test_RF(model, X_train, y_train, X_test, y_test, train = True): 
    
# Fit on training data
    if train == True: 
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
    
#%%
#Uncomment to train model 
    
"""
# Define random forrest model
model = RandomForestClassifier(n_estimators=100,
                               criterion = 'entropy',
                               min_samples_split=2,
                               bootstrap = True,
                               max_features = None)




#Import variables from other scripts
from Process_data import A, ytrue, yhat
#from Process_data import y_train, y_test, X_train, X_test, train_index, test_index

#train and test model
train_acc, test_acc, yhat_rf = train_test_RF(model, X_train, y_train, X_test, y_test)
          


#save model
filename = 'RF.sav'
pickle.dump(model, open(filename, 'wb'))
"""



