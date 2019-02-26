import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
from scipy.stats import binned_statistic
import seaborn as sns
import rshmm
import warnings
warnings.filterwarnings("ignore")

def select_state(Z_st, state):
    for i in range(Z_st.shape[0]):
        if Z_st[i] == state:
            pass
        else:
            Z_st[i] = -1                                        # to allow distinguishment from 0
    return Z_st
    
def kde_cont_binning(bins, df, y_label, dens_label, label):    
    bin_means = binned_statistic(df[y_label].values, df[y_label].values, bins=bins)
    bin_matrix = pd.DataFrame(np.concatenate([df[y_label].values.reshape(-1,1),
                                              df[dens_label].values.reshape(-1,1),
                                              bin_means[2].reshape(-1,1)], axis=1), columns=[y_label, dens_label, 'BIN'])
    
    for i in range(2,bins+1):  # only for current Z_Sel
        sns.kdeplot(bin_matrix.ix[bin_matrix['BIN']==i, dens_label].values, label=label+' distribution', shade=True)
    pyplot.xlabel(label)    
    pyplot.show()
    
def convert_to_df(inp_df,Z_sel_state):    
    new_df = pd.concat([pd.DataFrame(inp_df), pd.DataFrame(Z_sel_state, index=inp_df.index, columns=["Z_SEL"])], axis=1)
    return new_df 
    
# input data
inputDF = pd.ExcelFile('Stock_Fundm.xlsx')
tabnames = inputDF.sheet_names
inputdf = inputDF.parse(tabnames[0]).dropna()                                   # df.dropna(subset=[1]) for specific col

input_PS = inputdf['P/S']
EBIT_marg = inputdf['EBIT_MARGIN']
Fund_PE = inputdf['Vehicle_sales']

# number of model states
states=4

# include the X matrix
#scaled_input = pd.concat([input_PS, EBIT_marg], axis=1)    
input_X = pd.concat([input_PS, EBIT_marg], axis=1)  # display purposes  

# scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_arr = scaler.fit_transform(input_X.values)
#scaled_input = scaler.fit_transform(scaled_input.values)

# F matrix of ones
unity = (np.ones(len(inputdf.index))).reshape(-1, 1)

# fit HMM model 
model = rshmm.HMMRS(n_components=states, verbose=True, n_iter=100)           # n_comp is the number of states
model.fit(scaled_arr, unity) 

Z_state, logl, viterbi_lattice = model.predict(scaled_arr, unity)
probs = model._compute_likelihood(scaled_arr, unity)

state_no = Z_state[-1] 
Z_sel_state = select_state(np.copy(Z_state),state_no)                        # select which state you want to examine closely
Z_sel_state = scaler.fit_transform(Z_sel_state.reshape(-1, 1))
       
# plot inputs and states
fig = pyplot.figure(figsize=(8,8)) 
pyplot.figure(1)
ax = fig.add_subplot(2,1,1)
ax.set_title("Periods sharing the current P/S multiple's state") 
pyplot.ylabel("P/S multiple")
thresh = 0.05                                                                # Threshhold for graph shading
ax.set_ylim([-thresh+min(input_X.values[:,0:1]),max(input_X.values[:,0:1])+thresh])                                   # rescaling for imaging, first number is range floor 
ax.fill_between(inputdf.index,0,Z_sel_state.ravel()*10, facecolor='orange', alpha=0.5)  # multiply by some no to cover 
pyplot.plot(inputdf.index, input_X.values[:,0:1], color = 'black')

ax = fig.add_subplot(2,1,2)
ax.set_title("EBIT margin during time regimes") 
pyplot.ylabel("Margin")
thresh=0.01
offset=0.5
ax.set_ylim([-thresh+min(EBIT_marg.values),max(EBIT_marg.values)+thresh])                                     # rescaling for imaging 
ax.fill_between(inputdf.index,-offset,(Z_sel_state.ravel()-offset)*max(EBIT_marg.values)*3, facecolor='orange', alpha=0.5)    
pyplot.plot(EBIT_marg.index, EBIT_marg.values, color = 'black')
pyplot.show()

# plot dist. for fundamentals
input_PS_out = convert_to_df(input_PS, Z_sel_state)
kde_cont_binning(2, input_PS_out, 'Z_SEL', 'P/S', label='P/S')

EBIT_marg_out = convert_to_df(EBIT_marg, Z_sel_state)
kde_cont_binning(2, EBIT_marg_out, 'Z_SEL', 'EBIT_MARGIN', label='EBIT margin')

PE_out = convert_to_df(Fund_PE, Z_sel_state)
kde_cont_binning(2, PE_out, 'Z_SEL', 'Vehicle_sales', label='Vehicle sales')

# model state metrics
startprob = model.startprob_
transmat = model.transmat_                                                                  

for i in range(model.n_components):
    print("{0}th hidden state".format(i))
    print("standard dev = ", np.sqrt(np.diag(model.covmat_[i])))           


