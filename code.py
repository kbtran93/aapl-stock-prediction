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
from Keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib. dates as mandates
from sklearn import linear_model
from Keras.Models import Sequential
from Keras.Layers import Dense
import Keras.Backend as K
from Keras.Callbacks import EarlyStopping
from Keras.Optimisers import Adam
from Keras.Models import load_model
from Keras.Layers import LSTM
from Keras.utils.vis_utils import plot_model

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
df.Date

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

features = ['Open', 'High,' 'Low','Volume']

#Scaling

scaler = MinMaxScaler()

feature_transform = scaler.fit_transform(df[features])

feature_transform= pd.DataFrame(columns=features, data=feature_transform, index=df.index)

feature_transform.head()