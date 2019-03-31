import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn import metrics

news = pd.read_csv("uci-news-aggregator.csv", error_bad_lines=False)

categories = news['CATEGORY']
titles = news['TITLE']
N = len(titles)
#print('Number of news',N)

encoder = LabelEncoder()
y = encoder.fit_transform(news['CATEGORY']) # encode categ - m,e,t,b..

labels = list(set(categories))
print('possible categories',labels)

encoder = LabelEncoder()
ncategories = encoder.fit_transform(categories)

Ntrain = int(N * 0.7)
titles, ncategories = shuffle(titles, ncategories, random_state=0)

X_train = titles[:Ntrain]
y_train = ncategories[:Ntrain]
X_test = titles[Ntrain:]
y_test = ncategories[Ntrain:]

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
                     ])  # sticking multiple processes into a single scikit-learn estimator

text_clf = text_clf.fit(X_train, y_train)
predicted = text_clf.predict(X_test)

print('accuracy_score', metrics.accuracy_score(y_test,predicted))
print(metrics.classification_report(y_test, predicted, target_names=labels))

