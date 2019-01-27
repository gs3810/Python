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
    y_lead = y.shift(periods=-n_lead).rename(index=str, columns={"REC": "PRED_REC"})   # shifting and re-naming
    
    # add required lags
    dataset = pd.concat([y, y_lead, X], axis=1).dropna()
    
    for i in range (n_lags):
        # perform concat
        dataset = pd.concat([dataset, X.shift(periods=i+1)], axis=1).dropna()
        # rename columns
        cols=pd.Series(dataset.columns)
        for dup in dataset.columns.get_duplicates(): cols[dataset.columns.get_loc(dup)]=[dup+'.'+str(d_idx+i) if d_idx!=0 else dup for d_idx in range(dataset.columns.get_loc(dup).sum())]
        dataset.columns=cols
    
    dataset = dataset.drop(['REC'], axis=1)
    return dataset     

# specify metrics
start_period = 0
split = 0.4       
n_lead = 3
X_lags = 3  

# extract data
inputDF = pd.ExcelFile('NBER_data.xls')
tabnames = inputDF.sheet_names
df = inputDF.parse(tabnames[0]).iloc[start_period:,:].dropna()

## normalize features
#scaler = MinMaxScaler(feature_range=(0, 1))
#scaled = scaler.fit_transform(df)

# specify X and y and create lad and lead
curr_y = df.iloc[:,[0]]
X = df.iloc[:,[0,3,4,5,6,7,8,10]]   # choose features
featcols = X.columns
dataset = createlaglead(X, curr_y, n_lead, X_lags)  

# difference the data (not pct)
y = dataset.iloc[1:,[0]]        # reflect impact of differencing 
X = dataset.iloc[:,1:]          # remove impacts of y
cols = list(X.columns)
X = pd.DataFrame(np.diff(X, axis=0), index=y.index)

y_curr_train, y_curr_test = train_test_split(dataset.iloc[1:,[1]], test_size=split, shuffle=False) # actual values

# split data and reinsert columns
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split, shuffle=False)
X_train = pd.DataFrame(X_train)
X_train.columns = cols

# fit gbr
gbr = GradientBoostingRegressor(n_estimators=500, max_depth=3, max_features=3, learning_rate=0.1, random_state=None)
gbr.fit(X_train, y_train.values.ravel())

# data analysis
# extract feature importances
importances = pd.DataFrame(gbr.feature_importances_, index=cols, columns=['Variables'])

# extract partial feature dependence 
features = list(range(1,len(featcols)))  #[1,2,3,4,5,6,7]
fig, axs = plot_partial_dependence(gbr, X, features, feature_names=cols)
fig.set_size_inches(9, 10)
plt.show()

yhatgbr = gbr.predict(X_test)
yhat_traingbr = gbr.predict(X_train)
dsp = 250

## train plot
#plt.plot(y_train.index, y_train.values)
#plt.plot(y_train.index, yhat_traingbr)
#plt.show()

#yhatgbr = bin_classify(yhatgbr,0.3)

# test plot
plt.plot(y_curr_test.index[0:dsp], y_curr_test.values[0:dsp])
plt.plot(y_test.index[0:dsp], yhatgbr[0:dsp], alpha=0.8)
plt.show()

print ("Accuracy of correctly predicting", n_lead,"month fwd = ",'%.1f' % bin_accuracy(yhatgbr, y_test.values, 0.5),'%')
print ("Accuracy of recession predicting", n_lead,"month fwd = ",'%.1f' % class_accuracy(yhatgbr, y_test.values, 0.65),'%')
