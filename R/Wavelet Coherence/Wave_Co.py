import pandas as pd
import numpy as np
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

def find_nearest(a, a0):
    idx = np.abs(a - a0).argmin()
    return a.flat[idx]

def select_state(Z_st, thresh):
    for i in range(Z_st.shape[0]):
        if Z_st[i] > thresh:
            Z_st[i] = 1
        else:
            Z_st[i] = 0                                       
    return Z_st

def corr_indic(trend_period,period_axis,data,thresh):
    nearest = find_nearest(period_axis, trend_period)
    data_T = data.T
    freq_col = data_T.loc[data.T.iloc[:,[-1]].squeeze()==nearest].T.iloc[1:-1,:].values # index last col
    indic = select_state(np.copy(freq_col),thresh)
    return indic

def rang_indic(startper, endper, thresh, rsq):
    per_rang = rsq.iloc[:,startper:endper]
    thresh = np.ones((len(rsq.index),len(per_rang.columns)))*thresh                     # create np.ones for comparison
    thresh_check = np.greater(per_rang,thresh)
    thresh_check = thresh_check.any(axis=1).astype(int)                                 # convert to bool array containg any value greater
    return thresh_check
    
def day_rang_indic(short_days, long_days, rsq, threshold):
    # extract indicators for list of time trends
    cols = np.array(list(rsq.columns))                                                  # selecting days
    indic_short = rang_indic(np.where(cols>=short_days[0])[0][0],
                             np.where(cols>=short_days[1])[0][0], threshold, rsq)          # select the first instance of column element
    indic_long = rang_indic(np.where(cols>=long_days[0])[0][0], 
                            np.where(cols>=long_days[1])[0][0], threshold, rsq) 
    indic = list([np.array(indic_short), np.array(indic_long)])
    return indic

# extract data
data = pd.read_csv("rsqV_B.csv").T
rsq = data.iloc[1:-1,:]
period_axis = data.tail(1).values
period_axis_int= period_axis.astype(int)

inputDF = pd.ExcelFile('ValeBov.xlsx')
tabnames = inputDF.sheet_names
df = inputDF.parse(tabnames[0])

# reset index (and set format) and columns
date = df['DATE']
rsq.index=date
rsq.columns=list(period_axis_int)
rsq.index=rsq.index.strftime('%Y-%m')

# plot coherence map
fig = plt.figure(figsize=(12,12)) 
ax = fig.add_subplot(2,1,1)
sns.heatmap(rsq.T)
ax.set(xlabel='Time', ylabel="Correlated trends time scale")

# plot both stocks on a different plot
stocks = df.set_index('DATE').drop(['INDEX'],axis=1)
stock1 = stocks.iloc[:,0:1]
stock2 = stocks.iloc[:,1:2]

# choose the date ranges 
short_days = [20,50]
long_days = [70,135]
label_txt = ['Short term','Long term']
indic = day_rang_indic(short_days, long_days, rsq, threshold=0.8)

# plot the graphs for 
fig = plt.figure(figsize=(9.8,5)) 
ax1 = fig.add_subplot(1,1,1)
ax1.set_title("Periods of correlation") 

color=iter(cm.Oranges(np.linspace(0.4,0.7,len(indic))))                                 # choose start/end point for colour scheme
ax1.set_ylim([min(stock1.values)*0.9,max(stock1.values)*1.1])
for counter,inid in enumerate(indic):
    c=next(color)                                  
    ax1.fill_between(stock1.index,0,inid.ravel()*1e5, facecolor=c, alpha=0.5, label=label_txt[counter])
ax1.plot(stocks.index, stock1.values, color='black', label=stock1.columns[0])
_=plt.xticks(rotation=90)  
ax1.legend(loc=2)

ax2 = ax1.twinx()
ax2.plot(stock2, color='red', label=stock2.columns[0])
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax2.legend(loc=1)
plt.show()


