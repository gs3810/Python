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

#inputDF = pd.ExcelFile('Sarcasm_Headlines_Dataset.xlsx')
#tabnames = inputDF.sheet_names
#news = inputDF.parse(tabnames[0])

inputDF = pd.ExcelFile(r"X:\Training & Support\Quants Tech\Machine Learning\Project\NLP\News_Headlines.xlsx")
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

# drop duplicates
news_df = news_df.drop_duplicates(subset='Headline', keep ='first') 

sns.countplot(news_df['Relevance']) 

tf = TfidfVectorizer(analyzer='word',ngram_range=(1,3),max_features=50)
X = tf.fit_transform(news_df['Headline'])

y = news_df['Relevance']
X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.3, random_state=seed)

nb = BernoulliNB()
nb.fit(X_train,y_train)
pred = nb.predict(X_valid)

y_valid = y_valid.values

print('Confusion matrix\n',confusion_matrix(y_valid,pred))
print('Classification_report\n',classification_report(y_valid,pred))




