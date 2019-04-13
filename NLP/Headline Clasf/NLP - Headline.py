"""https://www.kaggle.com/sudhirnl7/news-Headline-eda-spacy/notebook"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
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

inputDF = pd.ExcelFile('News_dataset.xlsx')
tabnames = inputDF.sheet_names
news = inputDF.parse(tabnames[0])

# choose a column with certian number of words...
news['num_words'] = news['Headline'].apply(lambda x: len(str(x).split()))
text = news[news['num_words'] == 2]['Headline'].values

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

news_df = clean_text(news, nlp)

sns.countplot(news_df['Relevance']) 

tf = TfidfVectorizer(analyzer='word',ngram_range=(1,3),max_features=60) # NB Bern takes only few features
X = tf.fit_transform(news_df['Headline'])

y = news_df['Relevance']
X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.3, random_state=seed)

# NB Bern
nb = BernoulliNB()
nb.fit(X_train,y_train)
y_predNB = nb.predict(X_valid)

# GBRT
gbr = GradientBoostingClassifier(n_estimators=100, max_depth=20, max_features=30, random_state=None)
gbr.fit(X_train, y_train.values.ravel())
y_predGBRT = gbr.predict(X_valid)

# SVM
svmc = svm.SVC(kernel='rbf', C=100, gamma=0.25)
svmc.fit(X_train, y_train.values.ravel())
y_predSVM = svmc.predict(X_valid)

y_valid = y_valid.values

print('Confusion matrix\n',confusion_matrix(y_valid, y_predNB))
print('Classification_report\n',classification_report(y_valid, y_predNB))





