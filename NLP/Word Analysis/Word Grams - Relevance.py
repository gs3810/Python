"""https://medium.com/analytics-vidhya/automated-keyword-extraction-from-articles-using-nlp-bfd864f41b34"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import warnings
import xlwings as xw
warnings.filterwarnings("ignore")

def clean_text(df, nlp):
    for i in range(df.shape[0]):
        doc = nlp(df['Headline'][i])
        
        # Remove punctuation, digit and lemmatize... # Lower case
        text = []
        for w in doc:
            if not (w.is_stop | w.is_punct):
                if str(w).isupper() and str(w) not in keywords:
                    text.append(w.lemma_.strip().capitalize())
                else:
                    text.append(w.lemma_.strip())
        
        text = " ".join(text)
        
        df['Headline'][i] = text
    return df

def get_top_n2_words(corpus, n_w, n=None):
    vec1 = CountVectorizer(ngram_range=(n_w,n_w),  
            max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]

def create_relv(thematic,string_df,name):
    relv_df = pd.DataFrame() 
    for i in range(0,string_df.shape[0]):
        relv_df = relv_df.append([nlp(string_df['Theme'][i]).similarity(nlp(thematic))])
    
    relv_df = relv_df.reset_index(drop=True)
    relv_df.columns = [name]
    return relv_df

def sortby_intrelv(sort_df, col_no, thresh=0.6):
    econ_df= sort_df.iloc[:,col_no:col_no+2]
    col = econ_df.columns[1]
    econ_df = econ_df[econ_df[col]>thresh]
    
    # simliarty heatmap
    str_sim_df = pd.DataFrame([0])
    sim_map_df = pd.DataFrame()
    
    for i in range(0,econ_df.shape[0]):
        for j in range(0,econ_df.shape[0]):
            str_sim_df = str_sim_df.append([nlp(econ_df['Theme'][i]).similarity(nlp(econ_df['Theme'][j]))])
            
        str_sim_df = str_sim_df.reset_index(drop=True)    
        sim_map_df = pd.concat([sim_map_df,str_sim_df], axis=1)        
        str_sim_df = pd.DataFrame([0])
    
    sim_map_df = sim_map_df.iloc[1:,:] 
    sim_map_df.columns = [econ_df['Theme']]         
    sim_map_df.index = [econ_df['Theme']]
    
    rel_mean_df = pd.DataFrame(sim_map_df.mean(axis=1), columns=['Sim']).sort_values(by=['Sim'], ascending=False)
    econ_df = pd.concat([rel_mean_df,econ_df.set_index('Theme')], axis=1)
    return econ_df

inputDF = pd.ExcelFile("News_dataset.xlsx")
tabnames = inputDF.sheet_names
news = inputDF.parse(tabnames[0])
news = news.drop_duplicates(subset=['Headline']).reset_index(drop=True)     # drop duplicate headlines

# word tokenize
nlp = spacy.load('en_core_web_lg')      # change threshold with this...
keywords = ['NATO','U.S.','U.K.','BRL','EM']
news_df = clean_text(news.iloc[0:100,:], nlp)
string_df = pd.DataFrame(news_df['Headline'])
string_df.columns = ['Theme']

# build a similarity index
#sim_df = pd.DataFrame([0])
#for i in range(0,top_df.shape[0]-1):
#    sim_df = sim_df.append([nlp(top_df['Headline'][i]).similarity(nlp(top_df['Headline'][i+1]))])
#sim_df = sim_df.reset_index(drop=True)
#sim_df.columns = ['Seq. Similarity'] 
#string_df = pd.concat([top_df, sim_df], axis=1)
#string_df.columns = ['Theme','Seq. Similarity']

## perform string union...
#threshold = 0.95
#if top_df['Seq. Similarity'][1] < threshold:                  # if first element is sim. to rest 
#    string_df = pd.DataFrame([[top_df['Headline'][0]]])
#else:
#    string_df = pd.DataFrame()
#string_first = ""
#
#for i in range(1,top_df.shape[0]-1):
#  
#    if top_df['Seq. Similarity'][i] >= threshold:             # check for sequential similarity
#
#        if string_first=="":                                  # initialize the memory string
#            string_first=top_df['Headline'][i-1]
#        string_first = string_firstion(string_first, top_df['Headline'][i])
#
#        if top_df['Seq. Similarity'][i+1] < threshold:        # store to string df
#            string_df = string_df.append(pd.DataFrame([[string_first]]))
#            string_first = ""
#string_df = string_df.reset_index(drop=True)
#string_df.columns = ['Theme']
#
## simliarty heatmap
#str_sim_df = pd.DataFrame([0])
#freq_cnt_df = pd.DataFrame([0])
#
#sim_map_df = pd.DataFrame()
#freq_map_df = pd.DataFrame()
# 
#for i in range(0,string_df.shape[0]):
#    for j in range(0,string_df.shape[0]):
#        str_sim_df = str_sim_df.append([nlp(string_df['Theme'][i]).similarity(nlp(string_df['Theme'][j]))])
#        freq_cnt_df = freq_cnt_df.append([string_df['Freq'][i]+string_df['Freq'][j]])
#        
#    str_sim_df = str_sim_df.reset_index(drop=True)
#    freq_cnt_df = freq_cnt_df.reset_index(drop=True)
#    
#    sim_map_df = pd.concat([sim_map_df,str_sim_df], axis=1)    
#    freq_map_df = pd.concat([freq_map_df,freq_cnt_df], axis=1)
#    
#    str_sim_df = pd.DataFrame([0])
#    freq_cnt_df = pd.DataFrame([0])
#
#sim_map_df = sim_map_df.iloc[1:,:] 
#sim_map_df.columns = [string_df['Theme']]         
#sim_map_df.index = [string_df['Theme']]
#
#freq_map_df = freq_map_df.iloc[1:,:] 
#freq_map_df.columns = [string_df['Theme']]         
#freq_map_df.index = [string_df['Theme']]
#
## remove co-variance
#sim_map = sim_map_df.values
#sim_map[np.where(sim_map==np.max(sim_map))] = 0
#sim_map= np.array(sim_map)
##sim_map = np.triu(sim_map, k=1)     # upper triangular
#sim_map_df = pd.DataFrame(sim_map, columns=sim_map_df.columns, index=sim_map_df.index)
#
## graph of themes
#sns.set(rc={'figure.figsize':(14,6)})
#g = sns.barplot(x="Theme", y="Freq", data=string_df)
#g.set_xticklabels(g.get_xticklabels(), rotation=80)
#
## build max relevance engine
#sim_map_df = sim_map_df.reset_index()
#freq_map_df = freq_map_df.reset_index()
#key_thm_df = pd.DataFrame()
#key_freq_df = pd.DataFrame()
#cols = list(sim_map_df.columns)
#
#n = 3
#for it in range(1,sim_map_df.shape[1]):
#    themes = sim_map_df.nlargest(3,[cols[it]])['Theme'].reset_index(drop=True)
#    freq = string_df.set_index('Theme').loc[themes,['Freq']].T
#    key_thm_df = key_thm_df.append([themes])
#    freq.columns = [0,1,2]
#    key_freq_df = key_freq_df.append([freq], ignore_index = True)
#    
#key_thm_df.index = sim_map_df['Theme']
#key_thm_df = key_thm_df.reset_index()
#
#key_freq_df.index = string_df['Freq']
#key_freq_df = key_freq_df.reset_index()

theme_words = ['trade, jobs, economy, fiscal','election, ballot','inflation, repo, interest rates']
theme_headers = ['Economy', 'Election','Rates']

relv_df = pd.concat([string_df['Theme'], create_relv(theme_words[0],string_df,theme_headers[0])], axis=1)
for i in range(1,len(theme_words)):
    relv_df = pd.concat([relv_df, create_relv(theme_words[i],string_df,theme_headers[i])], axis=1)

sort_df = pd.concat([string_df['Theme'], create_relv(theme_words[0],string_df,theme_headers[0])], axis=1)
sort_df  = sort_df.sort_values(by=theme_headers[0], ascending=False).reset_index(drop=True)
for i in range(1,len(theme_words)):
    desc_df  = pd.concat([string_df['Theme'], create_relv(theme_words[i],string_df,theme_headers[i])], axis=1).sort_values(by=theme_headers[i], ascending=False).reset_index(drop=True)
    sort_df  = pd.concat([sort_df, desc_df], axis=1)

# open output excel
wb = xw.Book('Relevance.xlsx')
sht = wb.sheets[0]
sht.range('A1').value = pd.DataFrame(index=np.full(1000, np.nan), columns=np.full(30, np.nan))
sht.range('A1').value = relv_df

sht = wb.sheets[1]
sht.range('A1').value = pd.DataFrame(index=np.full(1000, np.nan), columns=np.full(30, np.nan))
sht.range('A1').value = sort_df

sortby_df = sortby_intrelv(sort_df, col_no=0, thresh=0.6) # change col number in factors of 2

