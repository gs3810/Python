import pandas as pd
import numpy as np
import seaborn as sns
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import warnings
import xlwings as xw
warnings.filterwarnings("ignore")

def clean_text(df, nlp):
    for i in range(df.shape[0]):
        doc = nlp(df['Theme'][i])
        
        # Remove punctuation, digit and lemmatize... # Lower case
        text = []
        for w in doc:
            if not (w.is_stop | w.is_punct):
                if str(w).isupper() and str(w) not in keywords:
                    text.append(w.lemma_.strip().capitalize())
                else:
                    text.append(w.lemma_.strip())
        
        text = " ".join(text)
        
        df['Theme'][i] = text
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

def create_relv(thematic,sentences_df,name):
    relv_df = pd.DataFrame() 
    for i in range(0,sentences_df.shape[0]):
        relv_df = relv_df.append([nlp(sentences_df['Theme'][i]).similarity(nlp(thematic))])
    
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

inputDF = pd.ExcelFile("EC_Text test.xlsx")
tabnames = inputDF.sheet_names
doc = inputDF.parse(tabnames[0])
full_txt = doc.iloc[0,0]

# word tokenize
nlp = spacy.load('en_core_web_lg')
doc_sen = nlp(full_txt)
sentences = [sent.string.strip() for sent in doc_sen.sents]
sentences_df = pd.DataFrame(sentences, columns=['Theme'])

# cleaning sentence
keywords = []
#sentences_df = clean_text(sentences_df.iloc[0:,:], nlp)

theme_words = ['Revenue, sales, volume','operating expenses, costs']
theme_headers = ['Revenue','Opex']

relv_df = pd.concat([sentences_df['Theme'], create_relv(theme_words[0],sentences_df,theme_headers[0])], axis=1)
for i in range(1,len(theme_words)):
    relv_df = pd.concat([relv_df, create_relv(theme_words[i],sentences_df,theme_headers[i])], axis=1)

"""  
sort_df = pd.concat([sentences_df['Theme'], create_relv(theme_words[0],sentences_df,theme_headers[0])], axis=1)
sort_df  = sort_df.sort_values(by=theme_headers[0], ascending=False).reset_index(drop=True)
for i in range(1,len(theme_words)):
    desc_df  = pd.concat([sentences_df['Theme'], create_relv(theme_words[i],sentences_df,theme_headers[i])], axis=1).sort_values(by=theme_headers[i], ascending=False).reset_index(drop=True)
    sort_df  = pd.concat([sort_df, desc_df], axis=1)

# open output excel
wb = xw.Book('Lira_Relevance.xlsx')
sht = wb.sheets[0]
sht.range('A1').value = pd.DataFrame(index=np.full(1000, np.nan), columns=np.full(30, np.nan))
sht.range('A1').value = relv_df

sht = wb.sheets[1]
sht.range('A1').value = pd.DataFrame(index=np.full(1000, np.nan), columns=np.full(30, np.nan))
sht.range('A1').value = sort_df

sortby_df = sortby_intrelv(sort_df, col_no=0, thresh=0.6) # change col number in factors of 2
"""
