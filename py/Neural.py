import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.models import load_model
import pandas as pd
import datetime
from Process_data import twoyears
import numpy as np
import matplotlib.pyplot as plt
from POST import equal
from Process_data import A, ytrue, yhat


np.random.seed(2)

# load the dataset

dataset = twoyears.data

print(twoyears.data.columns)

attributes = ['age','Female','Male','Less than 25','25 - 45','Greater than 45','African-American','Caucasian', '(CO3)',
       '(F1)', '(F2)', '(F3)', '(F5)', '(F6)', '(F7)', '(M1)', '(M2)', '(MO3)']

# Attributes
X_frame = dataset[attributes]
X = pd.DataFrame.to_numpy(X_frame)

X = twoyears.data.drop(['decile_score.1'],axis = 1).values

input_size = len(attributes)

# binary labels
y_frame = dataset["decile_score.1"]
y = pd.DataFrame.to_numpy(y_frame)

# train and test split
upper = int(np.floor(6907*0.8))

X_train = X[0:upper]
X_test = X[upper:]
y_train = y[0:upper]
y_test = y[upper:]

model = load_model("C:/Users/rasmu/OneDrive/Dokumenter/4. semester/Fagprojekt/02466---Project-in-Artificial-Intelligence-and-Data-Science/py/First.h5")

def train_NN():
       model = Sequential()
       model.add(Dense(input_size, activation='relu'))
       model.add(Dense(32, activation='relu'))
       model.add(Dense(16, activation='relu'))
       model.add(Dense(1, activation='sigmoid'))

       model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

       model.fit(X_train, y_train, epochs=150, batch_size=10)

       return 

_, accuracy = model.evaluate(X_test, y_test)

print('Accuracy: %.2f' % (accuracy*100))

A = A.values[upper:]
A = pd.DataFrame(A)

pred = model.predict(X_test)

equal_NN = equal(A[0],pred,y_test)

T = np.arange(0,1.001,0.001)

FPR, TPR = equal_NN.ROC_(T)

#plt.plot(FPR['Caucasian'], TPR['Caucasian'], label = 'caucasian')
#plt.plot(FPR['African-American'], TPR['African-American'], label = 'african-american')
#plt.legend()
#plt.show()

FPRC = FPR['Caucasian']
FPRA = FPR['African-American']

TPRC = TPR['Caucasian']
TPRA = TPR['African-American']

TP = []
FP = []

for i in range(len(FPRC)):
       if TPRC[i] > TPRA[i]:
              TP.append(TPRA[i])
              FP.append(FPRA[i])
       elif TPRC[i] <= TPRA[i]:
              TP.append(TPRC[i])
              FP.append(FPRC[i])
       

plt.plot(FP,TP)
              
plt.show()


"""
TP = np.array([])

FP = np.array([])

rate = len(pred)

for i in range(0,1000):
       k = i/1000

       pred_t = pred <= k

       TP = np.append(TP,(sum(pred_t == y_test)/rate))
       FP = np.append(FP,(sum(pred_t != y_test)/rate))

print(TP.shape)

plt.plot(FP,TP)
plt.show()

#model.save("First.h5")
"""