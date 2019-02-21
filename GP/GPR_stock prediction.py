import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcess
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

np.random.seed(1)

#def f(x):
#    """The function to predict."""
#    return x * np.sin(x)

samples = 300
inputDF = pd.ExcelFile('Stock_Prices.xlsx')
tabnames = inputDF.sheet_names  
all_stocks = inputDF.parse(tabnames[0])

# choose trainng sample for stocks
stock = all_stocks['F'].values
stock_train = stock[0:samples]

X = np.linspace(0.1, 9.9, samples)
X = np.atleast_2d(X).T

# choose test samples for stocks
test_siz = 350
test_stock = stock[0:test_siz]
X_full = np.linspace(0.1, test_siz/samples*9.9, test_siz)
X_full = np.atleast_2d(X_full).T

# Observations and noise
dy = 2 + 2* np.random.random(stock_train.shape)
noise = np.random.normal(0, dy)
y = stock_train + noise

# mesh the input space for evaluations of the real function
x = np.atleast_2d(np.linspace(0, 12, 1000)).T

# instanciate a Gaussian Process model
gp = GaussianProcess(corr='squared_exponential', theta0=1e-1, thetaL=1e-3, thetaU=1, 
                     nugget=(dy / y) ** 2, random_start=100)                        # regularization for ill-proposed problems 

# fit to data using MLE of the parameters
gp.fit(X, y)

# make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, MSE = gp.predict(x, eval_MSE=True)
sigma = np.sqrt(MSE)

# plot the function with error in std
fig = plt.figure(figsize=(8,5))
#plt.errorbar(X.ravel(), y, dy, fmt='r.', markersize=1, label=u'Observations')
plt.plot(X_full.ravel(), test_stock, color='black') # plot original data
plt.plot(x, y_pred, 'b-', label=u'Prediction')

for i in range(1,5):
    plt.fill(np.concatenate([x, x[::-1]]), 
            np.concatenate([y_pred - i/2 * sigma,                     # concat +/- std, reverse the array
                           (y_pred + i/2 * sigma)[::-1]]),
            alpha=i/20, fc='b', ec='None', label=str(i/2)+' st. deviation')

    
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.legend(loc='upper left')

plt.show()