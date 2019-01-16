import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR

# input data
inputDF = pd.ExcelFile('S&P500.xlsx')             #
tabnames = inputDF.sheet_names

X = inputDF.parse(tabnames[0]).iloc[:,0:1]

grancas = grangercausalitytests(X, maxlag=3, verbose = True) 
