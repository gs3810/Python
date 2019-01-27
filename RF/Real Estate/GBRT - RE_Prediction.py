import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
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

def class_accuracy(yhat_inst, test_y, threshold):
    accuracy = []
    
    for i in range(len(yhat_inst)):
        if test_y[i] == 1:
            if (test_y[i]-yhat_inst[i]) < threshold: 
                accuracy = np.append(accuracy,1)
            else:
                accuracy = np.append(accuracy,0)
        else:
            pass
    return np.mean(accuracy)*100

def createlaglead(X, y, n_lead, n_lags):
    # create lead variable
    y_lead = y.shift(periods=-n_lead).rename(index=str, columns={"Index": "Pred_Index"})   # shifting and re-naming
    
    # add required lags
    dataset = pd.concat([y, y_lead, X], axis=1).dropna()
    
    for i in range (n_lags):
        # perform concat
        dataset = pd.concat([dataset, X.shift(periods=i+1)], axis=1).dropna()
        # rename columns
        cols=pd.Series(dataset.columns)
        for dup in dataset.columns.get_duplicates(): cols[dataset.columns.get_loc(dup)]=[dup+'.'+str(d_idx+i) if d_idx!=0 else dup for d_idx in range(dataset.columns.get_loc(dup).sum())]
        dataset.columns=cols
    
    dataset = dataset.drop(['Index'], axis=1)
    return dataset     

# specify metrics
start_period = 0
split = 0.4       
n_lead = 1
n_lags = 4  

# extract data
inputDF = pd.ExcelFile('DHI_Data.xls')
tabnames = inputDF.sheet_names
df = inputDF.parse(tabnames[0]).iloc[start_period:,:].dropna()

## specify X and y and create lad and lead
#curr_y = df.iloc[:,[0]]
#X = df.iloc[:,1:]   # choose features
#featcols = X.columns
#dataset = createlaglead(X, curr_y, n_lead, n_lags)  
#
## difference the data (not pct)
#y = dataset.iloc[1:,[0]]        # reflect impact of differencing 
#X = dataset.iloc[:,1:]          # remove impacts of y
#cols = list(X.columns)
#X = pd.DataFrame(np.diff(X, axis=0), index=y.index)

# initially specify X and y 
curr_y = df.iloc[:,[0]]
X = df.iloc[:,:]   # HAS TO INCLUDE current_y 
cols = list(X.columns)

# difference the data (not pct)
y = curr_y[1:]        # reflect impact of differencing 
X = pd.DataFrame(np.diff(X.iloc[:,1:], axis=0), index=y.index)
X = pd.concat([curr_y, X], axis=1).dropna()
X.columns = cols

# create lag and lead and respecify
dataset = createlaglead(X, curr_y, n_lead, n_lags)
y = dataset.iloc[:,[0]]
X = dataset.iloc[:,1:]

y_curr_train, y_curr_test = train_test_split(curr_y, test_size=split, shuffle=False) # actual values

# split data and reinsert columns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, shuffle=False)
X_train = pd.DataFrame(X_train)
X_train.columns, X_test.columns, featcols = X.columns, X.columns, X.columns

# fit gbr
gbr = GradientBoostingRegressor(n_estimators=300, max_depth=7, max_features=3, learning_rate=0.1, random_state=None)
gbr.fit(X_train, y_train.values.ravel())

# extract feature importances
importances = pd.DataFrame(gbr.feature_importances_, index=featcols, columns=['Variables'])

# extract partial feature dependence 
features = list(range(0,len(featcols)))  #[1,2,3,4,5,6,7]
fig, axs = plot_partial_dependence(gbr, X, features, feature_names=featcols)
fig.set_size_inches(9, 10)
plt.show()

yhatgbr = gbr.predict(X_test)
yhat_traingbr = gbr.predict(X_train)
dsp=100

## train plot
#plt.figure(figsize=(10,5))
#plt.plot(y_train.index, y_train.values)
#plt.plot(y_train.index, yhat_traingbr)
#plt.show()

index = y_curr_test.index # save index
#yhatgbr = bin_classify(yhatgbr,0)
#y_curr_test = bin_classify(y_test.values,0)

# test plot
plt.figure(figsize=(10,5))
plt.plot(index[0:dsp], y_curr_test[0:dsp])   #y_curr_test
plt.plot(y_test.index[0:dsp], yhatgbr[0:dsp], alpha=0.8, color='red')
plt.show()

print ("Accuracy of correctly predicting", n_lead,"month fwd = ",'%.1f' % bin_accuracy(yhatgbr, y_test.values, 0.5),'%')
print ("Accuracy of recession predicting", n_lead,"month fwd = ",'%.1f' % class_accuracy(yhatgbr, y_test.values, 0.65),'%')
