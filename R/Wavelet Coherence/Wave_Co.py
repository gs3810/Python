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
    freq_col = data_T.loc[data.T.iloc[:,[-1]].squeeze()==nearest].T.iloc[1:-1,:].values  # index last col
    indic = select_state(np.copy(freq_col),thresh)
    return indic
    
def indic_list(trend_per,period_axis,data,corr_thresh):
    indic = list()
    for per in trend_per:    
        indic.append(corr_indic(per,period_axis,data,corr_thresh))
    return indic

def rang_indic(startper, endper, thresh, rsq):
    per_rang = rsq.iloc[:,startper:endper]
    thresh = np.ones((len(rsq.index),len(per_rang.columns)))*thresh   # create np.ones for comparison
    thresh_check = np.greater(per_rang,thresh)
    thresh_check = thresh_check.any(axis=1).astype(int) # convert to bool array containg any value greater
    return thresh_check
    
# extract data
data = pd.read_csv("rsqK_AL.csv").T
rsq = data.iloc[1:-1,:]
period_axis = data.tail(1).values
period_axis_int= period_axis.astype(int)

inputDF = pd.ExcelFile('Alumin_stocksAL.xlsx')
tabnames = inputDF.sheet_names
df = inputDF.parse(tabnames[0])

# reset index (and set format) and columns
date = df['DATE']
rsq.index=date
rsq.columns=list(period_axis_int)
rsq.index= rsq.index.strftime('%Y-%m')

# plot coherence map
fig = plt.figure(figsize=(12,12)) 
ax = fig.add_subplot(2,1,1)
sns.heatmap(rsq.T)
ax.set(xlabel='Time', ylabel="Correlated trends time scale")
axt = ax.twinx()
axt.set

# plot both stocks on a different plot
stocks = df.set_index('DATE').drop(['INDEX'],axis=1)
stock1 = stocks.iloc[:,0:1]
stock2 = stocks.iloc[:,1:2]

# test rang indicators 
cols = np.array(rsq.columns)  # to help with selecting days
r_indic = rang_indic(26, 400, 0.8, rsq)

# extract indicators for list of time trends
trend_per = [20, 70, 100, 132]
indic = list([np.array(r_indic)]) # indic_list(trend_per,period_axis,data,0.7)

# plot the graphs for 
fig = plt.figure(figsize=(9.8,5)) 
ax1 = fig.add_subplot(1,1,1)
ax1.set_title("Period of correlation for "+str(trend_per)+' day trend') 

color=iter(cm.Oranges(np.linspace(0.3,0.8,len(indic))))       # choose start/end point for colour scheme
ax1.set_ylim([min(stock1.values)*0.9,max(stock1.values)*1.1])
for counter,inid in enumerate(indic):
    c=next(color)                                  
    ax1.fill_between(stock1.index,0,inid.ravel()*1e5, facecolor=c, alpha=0.5, label=trend_per[counter])
ax1.plot(stocks.index, stock1.values, color='black', label=stock1.columns[0])
_=plt.xticks(rotation=90)  
ax1.legend()

ax2 = ax1.twinx()
ax2.plot(stock2, color='red', label=stock2.columns[0])
ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
ax2.legend(loc=0)
plt.show()

# Another comparisoin graph 
fig= plt.figure(figsize=(9.8,5))
ax3 = fig.add_subplot(1,1,1) 
stock3 = stocks.iloc[:,2:3]
ax3.plot(stock3, color='Blue', label=stock3.columns[0])
ax3.legend(loc=0)
plt.show()


