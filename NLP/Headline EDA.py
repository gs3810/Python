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

#path = 'dataset/'
path = '../input/'
news = pd.read_json(path+ 'Sarcasm_Headlines_Dataset.json',lines=True)
news.head()

news['num_words'] = news['headline'].apply(lambda x: len(str(x).split()))
print('Maximum number of word',news['num_words'].max())

print('\nSentence:\n',news[news['num_words'] == 39]['headline'].values)
text = news[news['num_words'] == 39]['headline'].values

# Word tokenize
nlp = spacy.load('en')
doc = nlp(text[0])

# List compresion method to get tokens
token = [w.text for w in doc ]
print(token)

"""https://www.kaggle.com/sudhirnl7/news-headline-eda-spacy/notebook"""