import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot
from scipy.stats import binned_statistic
import seaborn as sns
import rshmm

def select_state(Z_st, state):
    for i in range(Z_st.shape[0]):
        if Z_st[i] == state:
            pass
        else:
            Z_st[i] = -1                                        # to allow distinguishment from 0
    return Z_st
    
def kde_cont_binning(bins, df, y_label, dens_label):    
    bin_means = binned_statistic(df[y_label].values, df[y_label].values, bins=bins)
    bin_matrix = pd.DataFrame(np.concatenate([df[y_label].values.reshape(-1,1),
                                              df[dens_label].values.reshape(-1,1),
                                              bin_means[2].reshape(-1,1)], axis=1), columns=[y_label, dens_label, 'BIN'])
    
    for i in range(2,bins+1):  # only for current Z_Sel
        sns.kdeplot(bin_matrix.ix[bin_matrix['BIN']==i, dens_label].values, label=i, shade=True)
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
Fund_PE = inputdf['P/E']

# number of model states
states = 4

# include the X matrix
scaled_input = pd.concat([input_PS], axis=1)    
input_X = pd.concat([input_PS], axis=1)    

# scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_arr = scaler.fit_transform(input_X.values)
scaled_input = scaler.fit_transform(scaled_input.values)

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
fig = pyplot.figure(figsize=(8,4)) 
pyplot.figure(1)
ax = fig.add_subplot(1,1,1)
ax.set_title("Periods sharing the current P/S multiple's state") 
pyplot.ylabel("P/S multiple")
ax.set_ylim([1,max(input_X.values)+0.05])                                    # rescaling for imaging 
ax.fill_between(inputdf.index,0,Z_sel_state.ravel()*max(input_X.values), facecolor='orange', alpha=0.5)   # max is there to scale i back to normal 
pyplot.plot(inputdf.index, input_X.values[:,0:1], color = 'black')

# plot distributions
fig = pyplot.figure(figsize=(8,4))                                         
pyplot.figure(2)
ax = fig.add_subplot(1,1,1)
ax.set_title("Probability distribution of P/S multiples for current state")
pyplot.xlabel("Multiple (x)")
pyplot.ylabel("Probablity density (arbitary units)")
pyplot.plot(input_PS.values, probs[:,state_no:state_no+1], '.', color='black')
pyplot.show()

# plot dist.for fundamentals
EBIT_marg_out = convert_to_df(EBIT_marg, Z_sel_state)
kde_cont_binning(2, EBIT_marg_out, 'Z_SEL', 'EBIT_MARGIN')

PE_out = convert_to_df(Fund_PE, Z_sel_state)
kde_cont_binning(2, PE_out, 'Z_SEL', 'P/E')

# model state metrics
startprob = model.startprob_
transmat = model.transmat_                                                  # size n_states^2                 

for i in range(model.n_components):
    print("{0}th hidden state".format(i))
    print("standard dev = ", np.sqrt(np.diag(model.covmat_[i])))            # covmat matrix is n_feat^2. the diag is the var matrix for each feature



