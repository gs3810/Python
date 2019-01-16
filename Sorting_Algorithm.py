import pandas as pd
import xlwings as xw
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import warnings

def is_nan(x):
    return (x is np.nan or x != x)

def df_emp(text):
    df_empty = pd.DataFrame(index=text,columns=df_CFO.columns)
    return df_empty 

def convert_index_list(index):
    index = pd.DataFrame((index))
    index = index.dropna().iloc[:,0].tolist()
    return index

def train_model():
    train_df = inputDF.parse(tabnames[2])
    
    # X train
    input_docs = train_df .iloc[:,0:1]
    input_docs = list(input_docs["Input TEXT"])
    X_train = np.array(input_docs)
    
    # y train
    output_docs = train_df .iloc[:, 1:2]
    output_docs = list(output_docs["Output TEXT"]) 
    y_train_text = [[i] for i in output_docs]
     
    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(y_train_text)
    
    classifier = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
        ('clf', OneVsRestClassifier(LinearSVC()))])
    
    classifier.fit(X_train, Y)

    return classifier, mlb

def suggest_words(new_input): 

    X_test = np.array(new_input)
        
    classifier,mlb = train_model()
    predicted = classifier.predict(X_test)
    all_labels = mlb.inverse_transform(predicted)
    
    return all_labels

def condense_data(text_df, start_row=0, end_row=50, col=0):
    no_years = int(len(text_df.columns)/2)                                      # each year has two columns 
    df_list = list()
    df_list_reind, df_list_join, index_list = list(), list(), list()
     
    df = text_df.iloc[start_row:end_row, col:col+2].set_index("Statement_1")
    df_list.append(df)
    for i in range(1,no_years): 
        df = text_df.iloc[start_row:end_row, col+(i*2):col+(i*2+2)].set_index("Statement_"+str(i+1))
        df_list.append(df)
    
    df = df_list[0].reset_index().drop_duplicates(subset="Statement_1", keep='first').set_index("Statement_1")
    df_list_reind.append(df)
    for i in range(1,no_years): 
        df = df_list[i].reset_index().drop_duplicates(subset="Statement_"+str(i+1), keep='first').set_index("Statement_"+str(i+1))
        df_list_reind.append(df)
    
    df_list_join = df_list_reind[0].join(df_list_reind[1], how='outer')
    index_list   = df_list_reind[0].index.union(df_list_reind[1].index)
    for i in range(1,no_years-1): 
        df_list_join = df_list_join.join(df_list_reind[i+1], how='outer')
        index_list = index_list.union(df_list_reind[i+1].index)
    
    df_list_join = df_list_join.reset_index().drop_duplicates(subset='index', keep='first').set_index('index')
    df = df_list_join.reindex(index_list)
    
    # give the new places
    df_index = convert_index_list(df.index)
    new_df_index = suggest_words(df_index)
    
    new_df_index= pd.DataFrame(new_df_index).iloc[0:(df.shape[0]),:]
    new_df_index.columns = ['Suggestion']
    orig_df_index = pd.DataFrame(df_index).iloc[0:(df.shape[0]),:]
    orig_df_index.columns = ['Orig']
    
    label_col = pd.concat([orig_df_index,new_df_index], axis=1)
    label_col = label_col.set_index(label_col.columns[0])
    
    df = pd.concat([df, label_col], axis=1)
    df = df.reindex(index_list)
    
    return df

def insert_suggestion(df_full):
    dupl_df_list = pd.DataFrame()                                               # initialisation for the store of combined df
    index_dupl = list()
    
    df_fill = df_full.set_index('Suggestion')
    dupl_list = df_fill.index[df_fill.index.duplicated()].unique()
    
    for dupl in dupl_list:
        try:                                                                    # try statement to allow if it fails                
            if is_nan(dupl) is True or dupl == '-' or dupl== None:
                pass
                
            else:
                index_dupl.append(dupl)
                dupl_df = df_fill.loc[dupl,:]
                dupl_df = dupl_df.apply(pd.to_numeric, errors='ignore')
                dupl_df = dupl_df.groupby(dupl_df.index).sum().reset_index()    # groups by similar index
                dupl_df.index = [dupl]
                dupl_df_list = pd.concat([dupl_df_list, dupl_df], axis=0)
        except:
            pass
            
    df_dummy = df_full.reset_index().set_index('Suggestion')                    # create a dummy with the 'Suggestions' as the index
    df_dummy = df_dummy.drop(index_dupl).set_index('index')
    df_new = pd.concat([df_dummy, dupl_df_list], axis=0)

    return df_new 

def process_df(df_samp, start_row, end_row, replace= "Yes"):
    df_samp = condense_data(df_samp,start_row,end_row)
    
    if replace =="Yes":
        try:
            df_samp = insert_suggestion(df_samp)
        except:
            print("Couldn't rearrange") 
        
    df_samp = df_samp.fillna(0)

    return df_samp

# extract data
warnings.filterwarnings("ignore")
inputDF = pd.ExcelFile('Financials.xlsx')
tabnames = inputDF.sheet_names
text_df = inputDF.parse(tabnames[1])

# open input excel
input_wb = xw.Book('Financials.xlsx')
input_sht = input_wb.sheets[1]
ctrl_sht = input_wb.sheets[0]

ML_enable = ctrl_sht.range('E3').value

# create the df
df_CFO = process_df(text_df,1,100, ML_enable)
df_CFI = process_df(text_df,100,200, ML_enable)
df_CFF = process_df(text_df,200,300, ML_enable)
df_empty = pd.DataFrame(index=['nan'],columns=df_CFO.columns)
df_full = pd.concat([df_emp([input_sht.range('A2').value]), df_CFO, df_emp([' ',input_sht.range('A102').value]), df_CFI, df_emp([' ',input_sht.range('A202').value]), df_CFF], axis=0)

# take the control input
periods = int(ctrl_sht.range('A22').value)
df_suggest = df_full['Suggestion']
df_full = df_full.iloc[:, 0:periods]

# df range manipulation
if ctrl_sht.range('E8').value =="Yes":
    cols = list(df_full.columns)
    try: 
        cols = list(cols.reverse())
    except:
        pass
    df_full = df_full[cols]
    
df_full = pd.concat([df_full,df_suggest], axis=1)      

# open output excel
wb = xw.Book('Output_Financials.xlsx')
sht = wb.sheets[0]

sht.range('A1').value = pd.DataFrame(index=np.full(1000, np.nan), columns=np.full(30, np.nan))
sht.range('A1').value = df_full
