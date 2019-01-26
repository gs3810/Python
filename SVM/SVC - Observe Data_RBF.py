import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
scaler = StandardScaler()
scaled = scaler.fit_transform(df.iloc[:,1:])

# specify X, y
y = df.iloc[:,[0]].values
X = scaled[:,[1,4]]     # try different versions of the data, [1,2], [1,3] , [2,3]]

"""-----------------------------------------------------------------------"""
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
    clf = svm.SVC(kernel=kernel, C=50, gamma=50)
    clf.fit(X_train, y_train.ravel())

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