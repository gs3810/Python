"""https://www.kaggle.com/sudhirnl7/news-Headline-eda-spacy/notebook"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import warnings
import xlwings as xw

warnings.filterwarnings("ignore")
seed = 100

def clean_text(df, nlp):
    """
    Tokenize, make lower case, remove Stop words, punctuation, digit, lemmatize
    """
    for i in range(df.shape[0]):
        doc = nlp(df['Headline'][i])
        # Word Tokenize
        #token = [w.text for w in doc]
        
        # Make Lower case
        # Remove Stop word, punctuation, digit and lemmatize
        text = [w.lemma_.lower().strip() for w in doc 
               if not (w.is_stop |
                    w.is_punct |
                    w.is_digit)
               ]
        text = " ".join(text)
        
#        if i <5: print('Sentence:',i,text)
        df['Headline'][i] = text
    return df

def create_relv(thematic,string_df,name):
    relv_df = pd.DataFrame() 
    for i in range(0,string_df.shape[0]):
        relv_df = relv_df.append([nlp(string_df['Theme'][i]).similarity(nlp(thematic))])
    
    relv_df = relv_df.reset_index(drop=True)
    relv_df.columns = [name]
    return relv_df  

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

news_df = clean_text(news.iloc[0:100,:], nlp)
string_df = pd.DataFrame(news_df['Headline'])
string_df.columns = ['Theme']

# drop duplicates
news_df = news_df.drop_duplicates(subset='Headline', keep ='first') 

theme_words = ['trade, jobs, economy, fiscal','election, ballot','inflation, repo, interest rates']
theme_headers = ['Economy', 'Election','Rates']

relv_df = pd.concat([string_df['Theme'], create_relv(theme_words[0],string_df,theme_headers[0])], axis=1)
for i in range(1,len(theme_words)):
    relv_df = pd.concat([relv_df, create_relv(theme_words[i],string_df,theme_headers[i])], axis=1)

sns.countplot(news_df['Relevance']) 

tf = TfidfVectorizer(analyzer='word',ngram_range=(1,3),max_features=50)
X = tf.fit_transform(news_df['Headline'])

"""Have to figure outputput of tf"""

y = news_df['Relevance']
X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.3, random_state=seed)

nb = BernoulliNB()
nb.fit(X_train,y_train)
pred = nb.predict(X_valid)

y_valid = y_valid.values

print('Confusion matrix\n',confusion_matrix(y_valid,pred))
print('Classification_report\n',classification_report(y_valid,pred))




