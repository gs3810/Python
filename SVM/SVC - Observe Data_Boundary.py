import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# specify metrics
start_period = 100
split = 0.4      
n_lead = 3

# extract data
inputDF = pd.ExcelFile('NBER_data.xls')
tabnames = inputDF.sheet_names
df = inputDF.parse(tabnames[0]).iloc[start_period:,:].dropna()

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(df)

# specify X, y
y = scaled[:,[0]]
X = scaled[:,[3,4]]     # try different versions of the data, [1,2], [1,3] , [2,3]]
#y_lead = y.shift(periods=-n_lead).rename(index=str, columns={"REC": "y_lead"})   # shifting and re-naming

## add the data lags
#X_1lag = X.shift(periods=1)
#X_2lag = X.shift(periods=1)
#X_3lag = X.shift(periods=1)
#
#dataset = pd.concat([y, y_lead, X, X_1lag, X_2lag, X_3lag], axis=1).dropna()
#
## difference the data (not pct)
#y = dataset.iloc[1:,1:2]        # choose the y_lead here
#X = dataset.iloc[:,2:]
#X = pd.DataFrame(np.diff(X, axis=0), index=y.index)
#
## split data
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, shuffle=False)

"""-----------------------------------------------------------------------"""

## we create two clusters of random points
#n_samples_1 = 1000
#n_samples_2 = 100
#centers = [[0.0, 0.0], [2.0, 2.0]]
#clusters_std = [1.5, 0.5]
#X, y = make_blobs(n_samples=[n_samples_1, n_samples_2], centers=centers, cluster_std=clusters_std, random_state=0, shuffle=False)

# fit the model and get the separating hyperplane
clf = svm.SVC(kernel='linear', C=1.0)
clf.fit(X, y)

# fit the model and get the separating hyperplane using weighted classes
wclf = svm.SVC(kernel='rbf', class_weight={1: 10})
wclf.fit(X, y)

## plot the samples
plt.scatter(X[:, 0], X[:, 1], c=y[:,0], cmap=plt.cm.Paired, edgecolors='k')

# plot the decision functions for both classifiers
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# create grid to evaluate model
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T

# get the separating hyperplane
Z = clf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins
a = ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5, linestyles=['-'])

# get the separating hyperplane for weighted classes
Z = wclf.decision_function(xy).reshape(XX.shape)

# plot decision boundary and margins for weighted classes
b = ax.contour(XX, YY, Z, colors='r', levels=[0], alpha=0.5, linestyles=['-'])

plt.legend([a.collections[0], b.collections[0]], ["non weighted", "weighted"],
           loc="upper right")
plt.show()