# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Import required modules
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from dateutil.parser import parse
%matplotlib inline

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

