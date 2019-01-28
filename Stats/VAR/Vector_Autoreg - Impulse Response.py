# https://www.statsmodels.org/dev/vector_ar.html 
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VAR, DynamicVAR
import matplotlib.pyplot as plt

def check_unitroot(data):
    # check for stationarity
    p_values = list()
    p_values.append(adfuller(data['Index'], maxlag=12)[1])
    p_values.append(adfuller(data['FED_RATE'], maxlag=12)[1])
    p_values.append(adfuller(data['MORT_15'], maxlag=12)[1])
    p_values.append(adfuller(data['MORT_30'], maxlag=12)[1])
    return p_values

# input data
inputDF = pd.ExcelFile('DHI_Data.xls')             
tabnames = inputDF.sheet_names

input_X = inputDF.parse(tabnames[0]).iloc[:,:]
data = pd.concat([input_X.iloc[1:,0:1],input_X.iloc[:,1:].diff().dropna()], axis=1)# index is already %, convert rest to diff (as they are %)

# check p_values
p_values = check_unitroot(data)

# make a VAR model
model = VAR(data)

# use aic for model selection
model.select_order(15)

# choose number of lags
results = model.fit(1)   
#print (results.summary())

lag_order = results.k_ar # extract chosen lag order

## forecast
#inputd = data.values[-lag_order:]
#forecast = results.forecast(data.values[-lag_order:], 5) # h_step ahead forecast i.e. 5 using lagged inputs
#results.plot_forecast(10)

# impulse response
irf = results.irf(10)
irf.plot(impulse='MORT_30') # asymptotic SE are plotted @95% s.l.


