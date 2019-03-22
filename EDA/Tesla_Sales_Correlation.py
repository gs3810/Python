import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.preprocessing import MinMaxScaler, scale
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binned_statistic
import warnings
warnings.filterwarnings("ignore")

# extract data
inputDF = pd.ExcelFile('TimeSeriesCleanCarSales.xlsx')
tabnames = inputDF.sheet_names
df = inputDF.parse(tabnames[1]).iloc[:,:]

# select models
data = df #.iloc[20:,[0,1,2,3,4,5,6,7,8,10,11,13,14,15,23,17]]

differenced =data.diff()  

data_diff= differenced.iloc[1:,:]

# scale datasets
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(data_diff)

cars = pd.DataFrame(scaled, columns=data_diff.columns)

# cols name
cols = list(data_diff.columns)

diff_selector = data_diff[['Prius','i3','i8','7-series','A5 / S5 / RS5','A8 / S8','Model S','5-series']]
diff_selector_30 = diff_selector.iloc[30:,:]
diff_selector.plot.line(figsize=(15,7))

cars.plot.line(figsize=(15,7))

plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()

#coef = np.corrcoef(data_diff.values)

# initiate empty dataframe
corr = pd.DataFrame()
for a in list(data_diff.columns):
    for b in list(data_diff.columns.values):
        corr.loc[a, b] = df.corr().loc[a, b]
        
sns.heatmap(corr,cmap="YlGnBu") 

sns.clustermap(corr,cmap="YlGnBu")
