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

# extract data
inputDF = pd.ExcelFile('NBER_data.xls')
tabnames = inputDF.sheet_names
df = inputDF.parse(tabnames[0]).iloc[start_period:,:].dropna()

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(df)

y = scaled[:,[0]] # differencing results in 1 less
X = pd.DataFrame(scaled[:,1:])

# difference the data (not pct)
#y = np.diff(y, axis=0)
#X = np.diff(X, axis=0)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, shuffle=False)

# fit SVC
clf = SVC(gamma='auto')
clf.fit(X_train, y_train.ravel())

yhatSVC = clf.predict(X_test)
yhat_trainSVC = clf.predict(X_train)

# fit RF
rf = RandomForestRegressor(n_estimators=2000, bootstrap='True', max_depth=10,max_features=1, random_state = None)
rf.fit(X_train, y_train.ravel())

yhatRF = rf.predict(X_test)
yhat_trainRF = rf.predict(X_train)

plt.plot(y_test)
plt.plot(yhatRF)

    