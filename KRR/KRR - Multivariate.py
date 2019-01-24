import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# specify metrics
start_period = 100
split = 0.4      

# extract data
inputDF = pd.ExcelFile('Data.xlsx')
tabnames = inputDF.sheet_names
df = inputDF.parse(tabnames[0]).iloc[start_period:,:]

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(df)
y = scaled[:,[0]]
X = scaled[:,1:]

# difference the data (not pct)
y = np.diff(y, axis=0)
X = np.diff(X, axis=0)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, shuffle=False)

""" 
- Examine the impact of alpha, very low values (with nonlinear kernel) overfits the data
- High values and it underfits
- Make gamma large; it massively overfits 
 
"""
clf = KernelRidge(alpha=1e-5, gamma=1e2, kernel='rbf', kernel_params=None)
clf.fit(X_train, y_train) 

# make trainng and test predictions
yhat = clf.predict(X_test)
yhat_train = clf.predict(X_train)

# plot training fit
fig = plt.figure(figsize=(8,6)) 
plt.figure(1)
ax = fig.add_subplot(3,1,1)
ax.set_title("Train/test plots")
plt.plot(yhat_train)
plt.plot(y_train)

# plot test fit
ax = fig.add_subplot(3,1,2)
plt.plot(yhat)
plt.plot(y_test)
plt.show()

# inverse and plot actual
inv_yhat = np.cumsum(yhat)
inv_y = np.cumsum(y_test)
plt.plot(inv_y)
plt.plot(inv_yhat)
plt.show()

