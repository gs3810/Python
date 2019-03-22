import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.preprocessing import MinMaxScaler, scale
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binned_statistic
import warnings
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")
from pandas import ExcelWriter

def pca_results(data, pca):

    # Dimension indexing
    dimensions = ['Dimension {}'.format(i) for i in range(1,len(pca.components_)+1)]
    
    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4), columns = data.keys()) 
    components.index = dimensions

    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1) 
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns = ['Explained Variance']) 
    variance_ratios.index = dimensions

    # Create a bar plot visualization
    fig, ax = plt.subplots(figsize = (14,8))

    # Plot the feature weights as a function of the components
    components.plot(ax = ax, kind = 'bar')
    ax.set_ylabel("Feature Weights") 
    ax.set_xticklabels(dimensions, rotation=0)

    # Display the explained variance ratios# 
    for i, ev in enumerate(pca.explained_variance_ratio_): 
        ax.text(i-0.40, ax.get_ylim()[1] + 0.05, "Explained Variance\n %.4f"%(ev))

    # Return a concatenated DataFrame
    return pd.concat([variance_ratios, components], axis = 1)

data = pd.read_excel("X:\Training & Support\Quants Tech\Machine Learning\Project\Scraping and EDA\Tesla\Datasets\Backup\TimeSeriesCleanCarSales - Backup.xlsx")
data = data.iloc[10:,:]
cars = ['i3','Leaf','Oil Price'] # ['5-series','7-series','C-class','E-class','S-class','Model S'] ['5-series','7-series','A7 / S7','A8 / S8','S-class','Model S']
x = data.loc[:,cars]

# diff dataset
x = x.diff().dropna()

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(x)
scaled_df = pd.DataFrame(scaled, columns=x.columns)

cars_scaled = scaled_df.iloc[:,:-1]

# practice PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(cars_scaled)
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
expl_ratio=pca.explained_variance_ratio_ 

# Componeents explnation
pca_results = pca_results(cars_scaled, pca)
plt.show()

# charting PC_1
PCA_sum = pca_results.cumsum()

#loading oil dataset

#oil_price=pd.read_excel("X:\Training & Support\Quants Tech\Machine Learning\Project\Scraping and EDA\Tesla\Datasets\Oil_price_monthly.xlsx")
#oil_price=oil_price.set_index('Date')
#oil_price.plot.line()

# explained variance
fig, ax1 = plt.subplots(figsize = (12,8))
plt.plot(data.index[1:],principalComponents[:,[1]])
plt.plot(data.index[1:],scaled_df['Oil Price'])
plt.xlabel('number of components')
plt.ylabel('explained variance')
#plt.legend
color = 'tab:red'
#ax2=ax1.twinx()
#ax2.set_ylabel('OilPrice', color=color)  # we already handled the x-label with ax1
#ax2.plot(data.index[1:], oil_price[:,[1]], color=color)
#ax2.tick_params(axis='y', labelcolor=color)

coef_cor = np.correlate(principalComponents[:,[0]].ravel(),scaled_df['Model S'].values)

# plot TSLA
plt.show()

# scaled differences
fig, ax = plt.subplots(figsize = (12,8))
plt.plot(scaled[:,:-1])
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()

new_df = pd.DataFrame([principalComponents[:,[0]].ravel(),scaled_df['Model S']], columns=data.index[1:], index= ['PC_1','Model S']).T #, index=data.index[1:])

#writer = ExcelWriter('Sorted_DF_2018Feb.xlsx')
#new_df.to_excel(writer,'Sheet1')
#writer.save()




