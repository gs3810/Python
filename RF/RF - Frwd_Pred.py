import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def bin_classify(yhat_precla, threshold=0.8, rang=[1,0]):

    for i,samp in enumerate(yhat_precla):
        if samp>threshold:
            yhat_precla[i] = rang[0]
        else:
            yhat_precla[i] = rang[1]
    
    return yhat_precla

def bin_accuracy (yhat_inst, test_y, threshold):
    accuracy = []
    for i in range(len(yhat_inst)):
                
        if abs(yhat_inst[i] - test_y[i]) < threshold: 
            accuracy = np.append(accuracy,1)
            
        else:
            accuracy = np.append(accuracy,0)
            
    return np.mean(accuracy)*100

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

# create variants of lead/lags
y = df.iloc[:,[0]]
X = df.iloc[:,1:]
y_lead = y.shift(periods=-n_lead).rename(index=str, columns={"REC": "y_lead"})   # shifting and re-naming

# add the data lags
X_1lag = X.shift(periods=1)
X_2lag = X.shift(periods=1)
X_3lag = X.shift(periods=1)

dataset = pd.concat([y, y_lead, X, X_1lag, X_2lag, X_3lag], axis=1).dropna()

# difference the data (not pct)
y = dataset.iloc[1:,1:2]        # choose the y_lead here
X = dataset.iloc[:,2:]
X = pd.DataFrame(np.diff(X, axis=0), index=y.index)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, shuffle=False)

# fit RF
rf = RandomForestRegressor(n_estimators=500, bootstrap='True', max_depth=10, max_features=3, random_state = None)
rf.fit(X_train, y_train)

yhatRF = rf.predict(X_test)
yhat_trainRF = rf.predict(X_train)

dsp = 2000
# train plot
plt.plot(y_train.values)
plt.plot(yhat_trainRF)
plt.show()

yhatRF = bin_classify(yhatRF,0.3)

# test plot
plt.plot(y_test.values[0:dsp])
plt.plot(yhatRF[0:dsp])
plt.show()

print ("Accuracy of correctly predicting", n_lead,"month fwd = ",'%.1f' % bin_accuracy(yhatRF, y_test.values, 0.5))

