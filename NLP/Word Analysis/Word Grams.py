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
        
        # Make Lower case
        # Remove Stop word, punctuation, digit and lemmatize
        text = [w.lemma_.lower().strip() for w in doc 
               if not (w.is_stop |
                    w.is_punct |
                    w.is_digit)
               ]
        text = " ".join(text)
        
        df['Headline'][i] = text
    return df

def get_top_n2_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(2,2),  
            max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]

inputDF = pd.ExcelFile("News_dataset_test.xlsx")
tabnames = inputDF.sheet_names
news = inputDF.parse(tabnames[0])

# choose a column with certian number of words...
news['num_words'] = news['Headline'].apply(lambda x: len(str(x).split()))
text = news[news['num_words'] == 14]['Headline'].values

# Word tokenize
nlp = spacy.load('en_core_web_sm')
doc = nlp(text[0])                  # select first element

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

news_df = clean_text(news.iloc[0:,:], nlp)

cv=CountVectorizer(max_df=0.8,stop_words=stop_words, max_features=10000, ngram_range=(1,3))
X=cv.fit_transform(corpus)
# check stopwords...

top2_words = get_top_n2_words(text, n=20)
top2_df = pd.DataFrame(top2_words)
top2_df.columns=["Bi-gram", "Freq"]

print(top2_df)
