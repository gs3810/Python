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
X = scaled[:,[1,4]]     # try different versions of the data, [1,2], [1,3] , [2,3]]
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

from sklearn import datasets
#
#iris = datasets.load_iris()
#X = iris.data
#y = iris.target
#
#X = X[y != 0, :2]
#y = y[y != 0]

n_sample = len(X)

np.random.seed(0)
order = np.random.permutation(n_sample)
X = X[order]
y = y[order].astype(np.float)

X_train = X[:int(.9 * n_sample)]
y_train = y[:int(.9 * n_sample)]
X_test = X[int(.9 * n_sample):]
y_test = y[int(.9 * n_sample):]

# fit the model
for fig_num, kernel in enumerate(('rbf', 'linear')): # ('linear', 'rbf', 'poly') Becareful - the polynomial is dangerous!!!
    clf = svm.SVC(kernel=kernel, C=3, gamma=300)
    clf.fit(X_train, y_train)

    plt.figure(fig_num)
    plt.clf()
    plt.scatter(X[:, 0], X[:, 1], c=y[:,0], zorder=10, cmap=plt.cm.Paired, edgecolor='k', s=20)  # convert y to vector

    # Circle out the test data
    plt.scatter(X_test[:, 0], X_test[:, 1], s=80, facecolors='none',
                zorder=10, edgecolor='k')

    plt.axis('tight')
    x_min = X[:, 0].min()
    x_max = X[:, 0].max()
    y_min = X[:, 1].min()
    y_max = X[:, 1].max()

    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(XX.shape)
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'],
                linestyles=['--', '-', '--'], levels=[-.5, 0, .5])

    plt.title(kernel)
plt.show()