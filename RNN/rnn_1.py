# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 14:47:29 2018

@author: Sishir
"""

# Part 1 - Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
 
# Importing Training Set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
 
cols = list(dataset_train)[1:5]
 
#Preprocess data for training by removing all commas
 
dataset_train = dataset_train[cols].astype(str)
for i in cols:
    for j in range(0,len(dataset_train)):
        dataset_train[i][j] = dataset_train[i][j].replace(",","")
 
dataset_train = dataset_train.astype(float)
 
 
training_set = dataset_train.as_matrix() # Using multiple predictors.
 
# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
 
sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
 
sc_predict = MinMaxScaler(feature_range=(0,1))
 
sc_predict.fit_transform(training_set[:,0:1])
 
# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
 
n_future = 20  # Number of days you want to predict into the future
n_past = 60  # Number of past days you want to use to predict the future
 
for i in range(n_past, len(training_set_scaled) - n_future + 1):
    X_train.append(training_set_scaled[i - n_past:i, 0:5])
    y_train.append(training_set_scaled[i+n_future-1:i + n_future, 0])
 
X_train, y_train = np.array(X_train), np.array(y_train)
 
# Part 2 - Building the RNN
 
# Import Libraries and packages from Keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
 
# Initializing the RNN
regressor = Sequential()
 
# Adding fist LSTM layer and Drop out Regularization
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(n_past, 4)))
regressor.add(Dropout(.2))
 
# Part 3 - Adding more layers
 
# Adding 2nd layer with some drop out regularization
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(.2))
 
# Adding 3rd layer with some drop out regularization
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(.2))
 
# Adding 4th layer with some drop out regularization
regressor.add(LSTM(units=50, return_sequences=False))
regressor.add(Dropout(.2))
 
# Output layer
regressor.add(Dense(units=1, activation='sigmoid'))
 
# Compiling the RNN
regressor.compile(optimizer='adam', loss="binary_crossentropy")  # Can change loss to mean-squared-error if you require.
 
# Fitting RNN to training set using Keras Callbacks. Read Keras callbacks docs for more info.
 
es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
tb = TensorBoard('logs')
 
history = regressor.fit(X_train, y_train, shuffle=True, epochs=100,
                        callbacks=[es, rlr,mcp, tb], validation_split=0.2, verbose=1, batch_size=64)
 
 
#Predicting on the training set  and plotting the values. the test csv only has 20 values and
# "ideally" cannot be used since we use 60 timesteps here.
 
predictions = regressor.predict(X_train)
 
#predictions[0] is supposed to predict y_train[19] and so on.
predictions_plot = sc_predict.inverse_transform(predictions[0:-20])
actual_plot = sc_predict.inverse_transform(y_train[19:-1])
 
hfm, = plt.plot(predictions_plot, 'r', label='predicted_stock_price')
hfm2, = plt.plot(actual_plot,'b', label = 'actual_stock_price')
 
plt.legend(handles=[hfm,hfm2])
plt.title('Predictions vs Actual Price')
plt.xlabel('Sample index')
plt.ylabel('Stock Price')
plt.savefig('graph.png', bbox_inches='tight')
plt.show()
plt.close()
 
 
# For generating new predictions, create an X_test dataset just like the X_train data of (at-least) 80 days previous data
# Format it for RNN input and use regressor.predict(new_X_test) to get predictions of the new_x_test 
# starting with day 81 to day 100