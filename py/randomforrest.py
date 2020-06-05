# -*- coding: utf-8 -*-
"""
Created on Fri May  8 15:00:01 2020

@author: Værksted Matilde
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Create the model with 100 trees
model = RandomForestClassifier(n_estimators=1000, 
                               bootstrap = True,
                               max_features = 'sqrt')



#dataset: 
labels = twoyears.data["decile_score.1"].values
Data = twoyears.data.drop(["decile_score.1"],axis=1).values
train,test,train_labels,y_test=train_test_split(Data, labels,test_size=0.2)
# Fit on training data
model.fit(train, train_labels)

rf_predictions = model.predict(test)
# Probabilities for each class
rf_probs = model.predict_proba(test)[:, 1]
rf_probs = model.predict_proba(test)[:, 0]
