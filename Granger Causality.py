import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR

# input data
inputDF = pd.ExcelFile('RE_Int_Daily.xlsx')             
tabnames = inputDF.sheet_names

input_X = inputDF.parse(tabnames[0]).iloc[:,[1,0]]
input_X_stat = input_X.pct_change().dropna()            # transform to stationary
plt.plot(input_X_stat)

grancas = grangercausalitytests(input_X_stat, maxlag=6, verbose = True)

# After transforming to stationary there is little correlation
