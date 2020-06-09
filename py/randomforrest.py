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
Equal_rf = equal(A[0], yhat_rf[0], ytrue[0], N=400)

#ROC 
T = np.arange(0,1.001,0.001)
Fpr_rf, Tpr_rf= Equal_rf.ROC_(T, models = True)

Fpr_cau, Tpr_cau = Fpr_rf['Caucasian'], Tpr_rf['Caucasian']

Fpr_afri, Tpr_afri = Fpr_rf['African-American'], Tpr_rf['African-American']

plt.plot(Fpr_rf['Caucasian'], Tpr_rf['Caucasian'], label = 'caucasian')
plt.plot(Fpr_rf['African-American'], Tpr_rf['African-American'], label = 'african-american')
plt.legend()
plt.show()


#POST
#accuracies of ROC curve
accs_rf = Equal_rf.acc_(T, models = True)

#race (Lowest ROC)
idx = 'African-American'


#choose (0,0) or (1,1)
p0 = [0,0]

#Point with highest acc on "lowest" ROC curve
accs_rf = accs_rf[idx] #all acc
max_idx = np.argmax(accs_rf) #idx for highest acc
maxt = T[max_idx] #t for highest acc
maxp = [Fpr_afri[max_idx], Tpr_afri[max_idx]] #(fpr,tpr) highest acc



#estimate line between p0 and maxp on african-american ROC
#eatimate koefficients
def estimate(x,y):
        a1 = (y[1]-y[0])/(x[1]-x[0]+0.000000000000000000000000000001)
        b1 = y[1]-a1*x[1]      
        return a1, b1

#Compute y-values of maxp, for of p0 and all point on caucasian ROC (one at a time)
#First and last point are not used, as the cause division by zero
yall = np.empty(len(Fpr_cau)-2)
for i in range(len(Fpr_cau)-2):
    a1,b1 = estimate([p0[0],Fpr_cau[i+1]], [p0[1],Tpr_cau[i+1]])
    y = a1*maxp[0]+b1
    yall[i] = y

#compute absolute distance between y-coordinate of maxp and the estimated y-coordinates of maxp
diff = []
for i in range(len(yall)): 
    diff.append(abs(yall[i]-maxp[1]))

#Find the index wih lowest difference
minidx = np.argmin(diff) + 1

#point on afri ROC
p1 = [Fpr_cau[minidx], Tpr_cau[minidx]] #add 1 to get the idx that works with all
                                        #except diff and yall

#compute a and b for 

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

#Find percentages and thresholds
t_afri = T[max_idx]

percent = percentile(p0, maxp, p1)
if p0 == [0,0]:
    t_cau1 = T[-1] 
else: 
    t_cau1 = T[0]
    
t_cau2 = T[minidx]
#compute conf matrix




#conf matrix and acc after
postconf_afri =  Equal_rf.conf_models(t_afri,0)
postconf_cau = Equal_rf.calc_ConfusionMatrix(t_cau1,t_cau2, 1,percent)


print(Equal_rf.FP_TP_rate(postconf_afri))
print(Equal_rf.FP_TP_rate(postconf_cau))

print(Equal_rf.acc_with_conf(postconf_afri))
