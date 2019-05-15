"""https://medium.com/analytics-vidhya/automated-keyword-extraction-from-articles-using-nlp-bfd864f41b34"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from sklearn.feature_extraction.text import CountVectorizer
import re

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

# choose a column with certian number of words...
text = news['Headline'].values

# Word tokenize
nlp = spacy.load('en_core_web_sm')
doc = nlp(text[0])                                          # select first element

df = pd.DataFrame(
{
    'token': [w.text for w in doc],
    'lemma':[w.lemma_ for w in doc],
    'POS': [w.pos_ for w in doc],
    'TAG': [w.tag_ for w in doc],
    'DEP': [w.dep_ for w in doc],
    'is_stopword': [w.is_stop for w in doc],
    'is_punctuation': [w.is_punct for w in doc],
    'is_digit': [w.is_digit for w in doc],
})

news_df = clean_text(news.iloc[0:400,:], nlp)
corpus = news_df['Headline']

stop_words = ["say","is", "to", "result", "large", "also", "iv", "one", "two", "new", "previously", "shown"]

# Vectorize the words 
cv = CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=10000, ngram_range=(1,3))
X = cv.fit_transform(corpus)

top_words = get_top_n2_words(corpus, n_w=5, n=40)
top_df = pd.DataFrame(top_words)
top_df.columns=["Bi-gram", "Freq"]
top_df['Bi-gram'] = top_df['Bi-gram'].str.title()
print(top_df)

sns.set(rc={'figure.figsize':(17,8)})
g = sns.barplot(x="Bi-gram", y="Freq", data=top_df)
g.set_xticklabels(g.get_xticklabels(), rotation=80)

# build a similarity index
sim_df = pd.DataFrame([0])

for i in range(0,top_df.shape[0]-1):
    sim_df = sim_df.append([nlp(top_df['Bi-gram'][i]).similarity(nlp(top_df['Bi-gram'][i+1]))])
sim_df = sim_df.reset_index(drop=True)
sim_df.columns = ['Seq. Similarity'] 
top_df = pd.concat([top_df, sim_df], axis=1)

# perform string union...
string_df = pd.DataFrame([top_df['Bi-gram'][0]])
string_first = ""

for i in range(1,top_df.shape[0]-1):
  
    if top_df['Seq. Similarity'][i] >= 0.9:             # check for sequential similarity

        if string_first=="":                            # initialize the memory string
            string_first=top_df['Bi-gram'][i-1]
        string_first = string_firstion(string_first, top_df['Bi-gram'][i])

        if top_df['Seq. Similarity'][i+1] < 0.9:        # store to string df
            string_df = string_df.append([string_first])
            string_first = ""





    
    
