import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, scale
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import binned_statistic
import warnings
from pandas import ExcelWriter
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")

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

data = pd.read_excel("Model_S_trends.xlsx")
data = data.iloc[:,:]
trends = ['Price of Tesla Model S','Tesla Model S Cost','model s tesla price','How much is a Model S','cost of tesla model s'] # ['5-series','7-series','C-class','E-class','S-class','Model S'] ['5-series','7-series','A7 / S7','A8 / S8','S-class','Model S']
x = data.loc[:,trends]
x = x.diff().dropna() # diff dataset# diff dataset

scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(x)
scaled_df = pd.DataFrame(scaled, columns=x.columns)

trends_scaled = scaled_df.iloc[:,:]

# PCA
pca = PCA(n_components=2)
pca_scaled = pca.fit_transform(trends_scaled)
print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
expl_ratio=pca.explained_variance_ratio_ 

# create new df with pc component
pc_ = pd.DataFrame(pca_scaled[:,[1]],index= data.index[1:], columns=['PC_1'])

# Isomap
embedding = Isomap(n_components=2)
iso_scaled = embedding.fit_transform(trends_scaled)

# create new df with pc component
iso_ = pd.DataFrame(iso_scaled[:,[1]],index= data.index[1:], columns=['ISO_1'])


fig, ax1 = plt.subplots(figsize = (12,8))
plt.plot(iso_)
plt.xlabel('number of components')
plt.ylabel('explained variance')
plt.legend

iso_.to_csv('ISO_out.csv')
pc_.to_csv('PCA_out.csv')