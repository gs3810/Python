import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors.kde import KernelDensity
from matplotlib import pyplot
import rshmm

def select_state(Z_st, state):
    for i in range(Z_st.shape[0]):
        if Z_st[i] == state:
            pass
        else:
            Z_st[i] = -1                                        # to allow distinguishment from 0
    return Z_st
    
# input data
inputDF = pd.ExcelFile('Stock_Fundm.xlsx')
tabnames = inputDF.sheet_names
inputdf = inputDF.parse(tabnames[0])
inputdf = inputdf.dropna()                                      # df.dropna(subset=[1]) for specific col

input_PS = inputdf['P/S']
#gdp_level = inputdf['GDP_level']
gdp_growth = inputdf['GDP_growth']
auto_sales = inputdf['Vehicle_sales']

# carry out KDE
kde = KernelDensity(kernel='linear', bandwidth=2).fit(auto_sales.reshape(-1,1))
auto_sales_KD = pd.DataFrame(kde.score_samples(auto_sales.reshape(-1,1)), index=inputdf.index)

# number of model states
states = 4

# include the X matrix
scaled_input = pd.concat([input_PS, auto_sales_KD, gdp_growth], axis=1)    
input_X = pd.concat([input_PS], axis=1)    # , auto_sales

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
fig = pyplot.figure(figsize=(10,14)) 
pyplot.figure(1)
#ax = fig.add_subplot(4,1,1)
#pyplot.plot(inputdf.index, Z_state)
ax = fig.add_subplot(3,1,2)
ax.set_title("Periods sharing the current state") 
x_axis = np.arrange()
#ax.fill_between(,Z_sel_state,0)
pyplot.plot(inputdf.index(), Z_sel_state)

ax = fig.add_subplot(3,1,3)
ax.set_title("P/S multiple")
pyplot.plot(inputdf.index, scaled_input[:,0:1])
#ax = fig.add_subplot(3,1,4)
#ax.set_title("GDP growth (in %)")
#pyplot.plot(inputdf.index, gdp_growth*100)
#pyplot.subplots_adjust(hspace=0.4) 

# plot distributions
fig = pyplot.figure(figsize=(9,5))                                         # fix to indicate states
pyplot.figure(2)
ax = fig.add_subplot(1,1,1)
ax.set_title("Probability distribution of P/S multiples for current state")
pyplot.xlabel("Multiple (x)")
pyplot.ylabel("Probablity density (arbitary units)")
pyplot.plot(input_PS.values, probs[:,state_no:state_no+1], 'o')

# model state metrics
startprob = model.startprob_
transmat = model.transmat_                                                 # size n_states^2                 

for i in range(model.n_components):
    print("{0}th hidden state".format(i))
#    print("loading matrix = ", model.loadingmat_[i])
    print("standard dev = ", np.sqrt(np.diag(model.covmat_[i])))           # covmat matrix is n_feat^2. the diag is the var matrix for each feature



