import rshmm
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot

def select_state(Z_st, state):
    for i in range(Z_st.shape[0]):
        if Z_st[i] == state:
            pass
        else:
            Z_st[i] = 0
    return Z_st
    
# input data
inputDF = pd.ExcelFile('Stock_Fundm.xlsx')
tabnames = inputDF.sheet_names
inputdf = inputDF.parse(tabnames[0]).iloc[:,5:6]
inputdf = inputdf.dropna()                                      # df.dropna(subset=[1]) for specific col

# number of model states
states = 4
state_no = 1                                                    # examine state

# scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_arr = scaler.fit_transform(inputdf.values)

# F matrix of ones
unity = (np.ones(len(inputdf.index))).reshape(-1, 1)

# fit HMM model 
model = rshmm.HMMRS(n_components=states, verbose=True, n_iter=100)           # n_comp is the number of states
model.fit(scaled_arr, unity) 

Z_state, logl, viterbi_lattice = model.predict(scaled_arr, unity)
probs = model._compute_likelihood(scaled_arr, unity)

Z_sel_state = select_state(np.copy(Z_state),state_no)                               # select which state you want to examine closely
        
# plot inputs and states
fig = pyplot.figure(figsize=(10,8)) 
pyplot.figure(1)
ax = fig.add_subplot(3,1,1)
pyplot.plot(inputdf.index, Z_state)
ax = fig.add_subplot(3,1,2)
pyplot.plot(inputdf.index, Z_sel_state)
ax = fig.add_subplot(3,1,2)
pyplot.plot(inputdf.index, scaled_arr)

# plot distributions
fig = pyplot.figure(figsize=(10,6))                                         # fix to indicate states
pyplot.figure(2)
pyplot.plot(inputdf.values, probs[:,state_no:state_no+1], 'o')

# model state metrics
startprob = model.startprob_
transmat = model.transmat_                                                 # size n_states^2                 

for i in range(model.n_components):
    print("{0}th hidden state".format(i))
#    print("loading matrix = ", model.loadingmat_[i])
    print("standard dev = ", np.sqrt(np.diag(model.covmat_[i])))           # covmat matrix is n_feat^2. the diag is the var matrix for each feature



