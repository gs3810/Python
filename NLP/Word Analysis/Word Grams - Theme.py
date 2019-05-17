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
        
        # Make Lower case remove stop word, punctuation, digit and lemmatize
        text = [w.lemma_.lower().strip() for w in doc 
               if not (w.is_stop |
                    w.is_punct |
                    w.is_digit)
               ]
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

def string_firstion(string1,string2):
    last_word = string2.split()[-1]
    string = string1 + " " + last_word
    return string

inputDF = pd.ExcelFile("News_dataset.xlsx")
tabnames = inputDF.sheet_names
news = inputDF.parse(tabnames[0])

# Word tokenize
nlp = spacy.load('en_core_web_sm')
news_df = clean_text(news.iloc[0:400,:], nlp)
corpus = news_df['Headline']

stop_words = ["say","is", "to", "result", "large", "also", "iv", "one", "two", "new", "previously", "shown"]

# Vectorize the words 
cv = CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=10000, ngram_range=(1,3))
X = cv.fit_transform(corpus)

top_words = get_top_n2_words(corpus, n_w=5, n=100)
top_df = pd.DataFrame(top_words)
top_df.columns=["Bi-gram", "Freq"]
top_df['Bi-gram'] = top_df['Bi-gram'].str.title()

# build a similarity index
sim_df = pd.DataFrame([0])

for i in range(0,top_df.shape[0]-1):
    sim_df = sim_df.append([nlp(top_df['Bi-gram'][i]).similarity(nlp(top_df['Bi-gram'][i+1]))])
sim_df = sim_df.reset_index(drop=True)
sim_df.columns = ['Seq. Similarity'] 
top_df = pd.concat([top_df, sim_df], axis=1)

# perform string union...
string_df = pd.DataFrame([[top_df['Bi-gram'][0],top_df['Freq'][0]]])
string_first = ""
threshold = 0.9

for i in range(1,top_df.shape[0]-1):
  
    if top_df['Seq. Similarity'][i] >= threshold:             # check for sequential similarity

        if string_first=="":                                  # initialize the memory string
            string_first=top_df['Bi-gram'][i-1]
        string_first = string_firstion(string_first, top_df['Bi-gram'][i])

        if top_df['Seq. Similarity'][i+1] < threshold:        # store to string df
            string_df = string_df.append(pd.DataFrame([[string_first,top_df['Freq'][i]]]))
            string_first = ""
string_df = string_df.reset_index(drop=True)
string_df.columns = ['Theme','Freq']

# simliarty heatmap
str_sim_df = pd.DataFrame([0])
freq_cnt_df = pd.DataFrame([0])

sim_map_df = pd.DataFrame()
freq_map_df = pd.DataFrame()
 
for i in range(0,string_df.shape[0]):
    for j in range(0,string_df.shape[0]):
        str_sim_df = str_sim_df.append([nlp(string_df['Theme'][i]).similarity(nlp(string_df['Theme'][j]))])
        freq_cnt_df = freq_cnt_df.append([string_df['Freq'][i]+string_df['Freq'][j]])

    str_sim_df = str_sim_df.reset_index(drop=True)
    freq_cnt_df = freq_cnt_df.reset_index(drop=True)
    
    sim_map_df = pd.concat([sim_map_df,str_sim_df], axis=1)    
    freq_map_df = pd.concat([freq_map_df,freq_cnt_df], axis=1)
    
    str_sim_df = pd.DataFrame([0])
    freq_cnt_df = pd.DataFrame([0])

sim_map_df = sim_map_df.iloc[1:,:] 
sim_map_df.columns = [string_df['Theme']]         
sim_map_df.index = [string_df['Theme']]

freq_map_df = freq_map_df.iloc[1:,:] 
freq_map_df.columns = [string_df['Theme']]         
freq_map_df.index = [string_df['Theme']]

# remove co-variance
sim_map = sim_map_df.values
sim_map[np.where(sim_map==np.max(sim_map))] = 0
sim_map= np.array(sim_map)
sim_map_df = pd.DataFrame(sim_map, columns=sim_map_df.columns, index=sim_map_df.index)

# graph of themes
sns.set(rc={'figure.figsize':(14,6)})
g = sns.barplot(x="Theme", y="Freq", data=string_df)
g.set_xticklabels(g.get_xticklabels(), rotation=80)

# open output excel
wb = xw.Book('Output.xlsx')
sht = wb.sheets[0]
sht.range('A1').value = pd.DataFrame(index=np.full(1000, np.nan), columns=np.full(30, np.nan))
sht.range('A1').value = sim_map_df

sht = wb.sheets[1]
sht.range('A1').value = pd.DataFrame(index=np.full(1000, np.nan), columns=np.full(30, np.nan))
sht.range('A1').value = freq_map_df

# build max relvance engine
sim_map_df = sim_map_df.reset_index()
cols = list(sim_map_df.columns)
indx = int(sim_map_df[[cols[2]]].idxmax().values)
theme = sim_map_df['Theme'].iloc[indx:indx+1,:]
print (theme)





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
        
        # Make Lower case remove stop word, punctuation, digit and lemmatize
        text = [w.lemma_.lower().strip() for w in doc 
               if not (w.is_stop |
                    w.is_punct |
                    w.is_digit)
               ]
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

def string_firstion(string1,string2):
    last_word = string2.split()[-1]
    string = string1 + " " + last_word
    return string

inputDF = pd.ExcelFile("News_dataset.xlsx")
tabnames = inputDF.sheet_names
news = inputDF.parse(tabnames[0])

# Word tokenize
nlp = spacy.load('en_core_web_sm')
news_df = clean_text(news.iloc[0:400,:], nlp)
corpus = news_df['Headline']

stop_words = ["say","is", "to", "result", "large", "also", "iv", "one", "two", "new", "previously", "shown"]

# Vectorize the words 
cv = CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=10000, ngram_range=(1,3))
X = cv.fit_transform(corpus)

top_words = get_top_n2_words(corpus, n_w=5, n=100)
top_df = pd.DataFrame(top_words)
top_df.columns=["Bi-gram", "Freq"]
top_df['Bi-gram'] = top_df['Bi-gram'].str.title()

# build a similarity index
sim_df = pd.DataFrame([0])

for i in range(0,top_df.shape[0]-1):
    sim_df = sim_df.append([nlp(top_df['Bi-gram'][i]).similarity(nlp(top_df['Bi-gram'][i+1]))])
sim_df = sim_df.reset_index(drop=True)
sim_df.columns = ['Seq. Similarity'] 
top_df = pd.concat([top_df, sim_df], axis=1)

# perform string union...
string_df = pd.DataFrame([[top_df['Bi-gram'][0],top_df['Freq'][0]]])
string_first = ""
threshold = 0.9

for i in range(1,top_df.shape[0]-1):
  
    if top_df['Seq. Similarity'][i] >= threshold:             # check for sequential similarity

        if string_first=="":                                  # initialize the memory string
            string_first=top_df['Bi-gram'][i-1]
        string_first = string_firstion(string_first, top_df['Bi-gram'][i])

        if top_df['Seq. Similarity'][i+1] < threshold:        # store to string df
            string_df = string_df.append(pd.DataFrame([[string_first,top_df['Freq'][i]]]))
            string_first = ""
string_df = string_df.reset_index(drop=True)
string_df.columns = ['Theme','Freq']

# simliarty heatmap
str_sim_df = pd.DataFrame([0])
freq_cnt_df = pd.DataFrame([0])

sim_map_df = pd.DataFrame()
freq_map_df = pd.DataFrame()
 
for i in range(0,string_df.shape[0]):
    for j in range(0,string_df.shape[0]):
        str_sim_df = str_sim_df.append([nlp(string_df['Theme'][i]).similarity(nlp(string_df['Theme'][j]))])
        freq_cnt_df = freq_cnt_df.append([string_df['Freq'][i]+string_df['Freq'][j]])

    str_sim_df = str_sim_df.reset_index(drop=True)
    freq_cnt_df = freq_cnt_df.reset_index(drop=True)
    
    sim_map_df = pd.concat([sim_map_df,str_sim_df], axis=1)    
    freq_map_df = pd.concat([freq_map_df,freq_cnt_df], axis=1)
    
    str_sim_df = pd.DataFrame([0])
    freq_cnt_df = pd.DataFrame([0])

sim_map_df = sim_map_df.iloc[1:,:] 
sim_map_df.columns = [string_df['Theme']]         
sim_map_df.index = [string_df['Theme']]

freq_map_df = freq_map_df.iloc[1:,:] 
freq_map_df.columns = [string_df['Theme']]         
freq_map_df.index = [string_df['Theme']]

# remove co-variance
sim_map = sim_map_df.values
sim_map[np.where(sim_map==np.max(sim_map))] = 0
sim_map= np.array(sim_map)
#sim_map = np.triu(sim_map, k=1)     # upper triangular
sim_map_df = pd.DataFrame(sim_map, columns=sim_map_df.columns, index=sim_map_df.index)

# graph of themes
sns.set(rc={'figure.figsize':(14,6)})
g = sns.barplot(x="Theme", y="Freq", data=string_df)
g.set_xticklabels(g.get_xticklabels(), rotation=80)

# open output excel
wb = xw.Book('Output.xlsx')
sht = wb.sheets[0]
sht.range('A1').value = pd.DataFrame(index=np.full(1000, np.nan), columns=np.full(30, np.nan))
sht.range('A1').value = sim_map_df

sht = wb.sheets[1]
sht.range('A1').value = pd.DataFrame(index=np.full(1000, np.nan), columns=np.full(30, np.nan))
sht.range('A1').value = freq_map_df

# build max relvance engine
sim_map_df = sim_map_df.reset_index()
key_thm_df = pd.DataFrame()
cols = list(sim_map_df.columns)

#for it in range(1,sim_map_df.shape[1]):
#    indx = int(sim_map_df[[cols[it]]].idxmax().values)
#    key_thm_df = key_thm_df.append([sim_map_df['Theme'].iloc[indx:indx+1,:]])
#
#indx = sim_map_df.nlargest(3,[cols[3]])['Theme']

for it in range(1,sim_map_df.shape[1]):
    themes = sim_map_df.nlargest(3,[cols[it]])['Theme']
    key_thm_df = key_thm_df.append([themes.T])







