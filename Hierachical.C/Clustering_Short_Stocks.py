import numpy as np
import seaborn as sns
import pandas as pd
from scipy import stats
from sklearn import preprocessing
import scipy.cluster.hierarchy as hac
from scipy.cluster.hierarchy import fcluster
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from fastdtw import fastdtw

"""Using the fastdtw 0.3.2 from Kazuaki Tanida"""

def _dtw(x, y):
    distance, path = fastdtw(x, y, dist=euclidean)
    return distance 

def corrfunc(x, y, **kws):
    r, _ = stats.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r), xy=(.25, .7), xycoords=ax.transAxes, size=14)

def print_clusters(timeSeries, Z, k, plot=False):
    # k Number of clusters I'd like to extract
    results = fcluster(Z, k, criterion='maxclust')

    # check the results
    s = pd.Series(results)
    clusters = s.unique()

    for c in clusters:
        cluster_indeces = s[s==c].index
#        print(cluster_indeces)
        print("Cluster %d number of entries %d" % (c, len(cluster_indeces)))
#        print (timeSeries)
        if plot:
            timeSeries.T.iloc[:,cluster_indeces].plot()
            plt.show()

# extract data
inputDF = pd.ExcelFile('Test_DS_Project.xlsx')
tabnames = inputDF.sheet_names
inputdf = inputDF.parse(tabnames[0])
values = inputdf.astype('float32')  

# get two timseries seperated by weeks
timeSeries_df = inputdf.fillna(0)

#----------------------------

timeSerieschng_df = timeSeries_df.pct_change().dropna()*100

sns.set_context(font_scale=2)

g = sns.PairGrid(timeSerieschng_df.iloc[:,:])
g.map_upper(plt.scatter, s=9, color='red')
g.map_diag(sns.distplot, kde=True, color='red')
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_lower(corrfunc);

plt.show()

#---------------------------

timeSeries = timeSeries_df.values
labels = list(timeSeries_df.columns)

# normalize the data
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
transp = scaler.fit_transform(timeSeries)
transp = np.diff(transp, axis=0) 

# re-transpose 
inv_transp = np.transpose(transp) 

metric = 'correlation' # cust. distance function -> _dtw, euclidean

# Do the clustering
Z = hac.linkage(inv_transp, method='single', metric=metric)  

if metric=='euclidean' or 'correlation': 
    pass
else:
    Z[:,2] = np.round((Z[:,2]/timeSeries.shape[0]),decimals=3)*100  # calc similiarity 

# Plot dendogram
plt.figure(figsize=(8, 3))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Company')
plt.ylabel('Dissimilarity %')
hac.dendrogram(Z, leaf_rotation=90., leaf_font_size=8., labels=labels)  # rotates the x axis labels,  # font size for the x axis labels
plt.show()

print_clusters(timeSeries, Z, 5, plot=False)                            # use this to check clustering 
