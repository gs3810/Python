# https://www.statsmodels.org/dev/vector_ar.html 
import numpy as np
import pandas
import statsmodels.api as sm
from statsmodels.tsa.api import VAR, DynamicVAR
from statsmodels.tsa.base.datetools import dates_from_str

mdata = sm.datasets.macrodata.load_pandas().data

# prepare the dates index
dates = mdata[['year', 'quarter']].astype(int).astype(str)
quarterly = dates["year"] + "Q" + dates["quarter"]


quarterly = dates_from_str(quarterly)
mdata = mdata[['realgdp','realcons','realinv']]
mdata.index = pandas.DatetimeIndex(quarterly)
data = np.log(mdata).diff().dropna()   # difference the data

 # make a VAR model
model = VAR(data)

# use aic to check to choose most likely order for the model
model.select_order(15) # up to this number of orders

# choose number of lags
results = model.fit(3)   
print (results.summary())
#results.plot()

lag_order = results.k_ar # extract chosen lag order

# forecast
inputd = data.values[-lag_order:]
forecast = results.forecast(data.values[-lag_order:], 5) # h_step ahead forecast i.e. 5 using lagged inputs
results.plot_forecast(10)

# impulse response
irf = results.irf(10)
irf.plot(orth=False) #  asymptotic SE are plotted @95% s.l.
# irf.plot(impulse='realgdp') for variable of interest








