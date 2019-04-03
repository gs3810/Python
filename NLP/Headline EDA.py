"""https://www.kaggle.com/sudhirnl7/news-headline-eda-spacy/notebook"""
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

#news = pd.read_json('Sarcasm_Headlines_Dataset.json',lines=True)
#news.to_csv('Sarcasm_Headlines_Dataset.csv',index=False)

inputDF = pd.ExcelFile('Sarcasm_Headlines_Dataset.xlsx')
tabnames = inputDF.sheet_names
news = inputDF.parse(tabnames[0])

# Add column with max word no
news['num_words'] = news['headline'].apply(lambda x: len(str(x).split()))
#print('Maximum number of word',news['num_words'].max())

# Choose the longest heading
#print('\nSentence:\n',news[news['num_words'] == 39]['headline'].values)
text = news[news['num_words'] == 39]['headline'].values

# Word tokenize
nlp = spacy.load('en_core_web_sm')
doc = nlp(text[0])                  # select first element

# List compresion method to get tokens
token = [w.text for w in doc ]
print(token)

#print('Quotes:',spacy.lang.punctuation.LIST_QUOTES)
#print('\nPunctuations:',spacy.lang.punctuation.LIST_PUNCT)
#print('\n Currency:',spacy.lang.punctuation.LIST_CURRENCY)

# list of punctuation contains most of punctuation, we will use only that for our analysis
punc = [w.text for w in doc  if  w.is_punct ]

# identify stop words
stopwords = list(spacy.lang.en.stop_words.STOP_WORDS)
stop = [w.text for w in doc if w.is_stop]

# identify if there are any digits
digit = [w.text for w in doc if w.is_digit]

# identify lemmas
lemma = [w.lemma_ for w in doc]



