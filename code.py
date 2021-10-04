# -*- coding: utf-8 -*-
"""
Author: Binh Tran
Description: Code file for AAPL Price Prediction notebook
"""
# Import required modules
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from dateutil.parser import parse
%matplotlib inline

from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.dates as mandates
from sklearn import linear_model
from keras.models import Sequential
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model
from keras.utils.vis_utils import plot_model

# Import the data
df = pd.read_csv('data/AAPL.csv')
df.head()
# Describe the data
df.describe()

# Information
df.info()

# Data types
df.dtypes

# Convert the Date column to datetime
df.Date = pd.to_datetime(df.Date, infer_datetime_format=True)

# Set the Date as index
df.set_index('Date', inplace=True)
df.index

# Get the last business day of each month
lbd = df.resample('M').last()
lbd

# Difference between the lastest day anf the oldest day in the dataset
df.index[0] - df.index[-1]

# The number of months in the data
lbd.shape

# Use seaborn style defaults and set the default figure size
sns.set(rc={'figure.figsize':(11, 4)})

# Plotting
fig, ax = plt.subplots()
fig.set_size_inches(14, 10)
ax = sns.lineplot(data=df[['Open', 'Low', 'High', 'Close']])

# Predicted variable: Adj Close
# Features: Open, Low, High, Volume

#Set Target Variable

output_var = pd.DataFrame(df['Adj Close'])

#Selecting the Features

features = ['Open', 'High','Low','Volume']

#Scaling

scaler = MinMaxScaler()

feature_transform = scaler.fit_transform(df[features])

feature_transform= pd.DataFrame(columns=features, data=feature_transform, index=df.index)

feature_transform.head()

#Splitting to Training set and Test set

timesplit = TimeSeriesSplit(n_splits=10)

for train_index, test_index in timesplit.split(feature_transform):

        X_train, X_test = feature_transform[:len(train_index)], feature_transform[len(train_index): (len(train_index)+len(test_index))]

        y_train, y_test = output_var[:len(train_index)].values.ravel(), output_var[len(train_index): (len(train_index)+len(test_index))].values.ravel()

#Process the data for LSTM

trainX = np.array(X_train)

testX = np.array(X_test)

X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])

X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])

#Building the LSTM Model

lstm = Sequential()

lstm.add(LSTM(32, input_shape=(1, trainX.shape[1]), activation='relu', return_sequences=False))

lstm.add(Dense(1))

lstm.compile(loss='mean_squared_error', optimizer='adam')

plot_model(lstm, show_shapes=True, show_layer_names=True)

#Model Training

history = lstm.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1, shuffle=False)

#LSTM Prediction

y_pred= lstm.predict(X_test)

#True vs Predicted Adj Close Value – LSTM

plt.plot(y_test, label='True Value')

plt.plot(y_pred, label='LSTM Value')

plt.title('Prediction by LSTM')

plt.xlabel('Time Scale')

plt.ylabel('Scaled USD')

plt.legend()

plt.show()