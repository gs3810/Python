import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# specify metrics
start_period = 100
split = 0.4      
n_lead = 3

# extract data
inputDF = pd.ExcelFile('NBER_data.xls')
tabnames = inputDF.sheet_names
df = inputDF.parse(tabnames[0]).iloc[start_period:,:].dropna()

## normalize features
#scaler = MinMaxScaler(feature_range=(0, 1))
#scaled = scaler.fit_transform(df)

# create variants of lags
y = df.iloc[:,[0]]
X = df.iloc[:,1:]
y_lead = y.shift(periods=-n_lead).rename(index=str, columns={"REC": "y_lead"})   # shifting and re-naming

dataset = pd.concat([y, y_lead, X], axis=1)

# difference the data (not pct)
y = dataset.iloc[1:,1:2]        # choose the y here
X = dataset.iloc[:,2:]
X = pd.DataFrame(np.diff(X, axis=0), index=y.index)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, shuffle=False)

# fit RF
rf = RandomForestRegressor(n_estimators=100, bootstrap='True', max_depth=10, max_features=3, random_state = None)
rf.fit(X_train, y_train)

yhatRF = rf.predict(X_test)
yhat_trainRF = rf.predict(X_train)

# train plot
plt.plot(y_train.values)
plt.plot(yhat_trainRF)
plt.show()

dsp = 200
# test plot
plt.plot(y_test.values[0:dsp])
plt.plot(yhatRF[0:dsp])
plt.show()