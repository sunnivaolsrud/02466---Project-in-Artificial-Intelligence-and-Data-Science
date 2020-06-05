from keras.models import Sequential
from keras.layers import Dense
import pandas as pd

from Process_data import twoyears

# load the dataset
dataset = twoyears.data

attributes = ['age','Female','Male','Less than 25','25 - 45','Greater than 45','African-American','Caucasian', '(CO3)',
       '(F1)', '(F2)', '(F3)', '(F5)', '(F6)', '(F7)', '(M1)', '(M2)', '(MO3)',
       '(F1)', '(F2)', '(F3)', '(F5)', '(F6)', '(F7)', '(M1)', '(M2)', '(MO3)']

X_frame = dataset[attributes]

X = pd.DataFrame.to_numpy(X_frame)

input_size = len(attributes)

y_frame = dataset["decile_score.1"]

y = pd.DataFrame.to_numpy(y_frame)

y = y>5

# define the keras model
model = Sequential()
model.add(Dense(input_size, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=10)

# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

