# Apple Stock Prediction

![alt text](https://www.cubicpromote.com.au/wp/wp-content/uploads/2018/08/current-apple-logo.jpg)

### Table of Contents

- [Installation](#installation)
- [Targets](#questions)
- [Data](#data)

## Installation<a name="installation"></a>

This project is executed by Jupyter Notebook using Python 3 coding. These packages are required to run the notebook.
- Numpy
- Pandas
- matplotlib 
- Seaborn
- statsmodels
- pmdarima
- yfinance
- Scikit-Learn

***Note***: the pmdarima library does not compatible well with the latest release of statsmodels. It is better to uninstall the current version of statsmodels and use the statsmodels scripted to install alongside with pmdarima

## Targets<a name="questions"></a>

- Time series analysis of the stock data
- Forecast the closing price


## Data<a name="data"></a>

The historical stock data was extracted from Yahoo! Finance database using the yfinance library. Data from 2010 to May 2022 were taken into consideration.

