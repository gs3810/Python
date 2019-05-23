import pandas as pd
import numpy as np
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
    
    for i in range(df.shape[0]):
        doc = nlp(df['Headline'][i])
        #token = [w.text for w in doc]
        
        text = [w.lemma_.lower().strip() for w in doc 
               if not (w.is_punct | w.is_digit) # w.is_stop
               ]
        text = " ".join(text)
        
        df['Headline'][i] = text
    return df

def create_relv(thematic,string_df,name):
    relv_df = pd.DataFrame() 
    for i in range(0,string_df.shape[0]):
        relv_df = relv_df.append([nlp(string_df['Theme'][i]).similarity(nlp(thematic))])
    
    relv_df = relv_df.reset_index(drop=True)
    relv_df.columns = [name]
    return relv_df  

inputDF = pd.ExcelFile("News_Headlines.xlsx")
tabnames = inputDF.sheet_names
news = inputDF.parse(tabnames[0])

## Word tokenize
nlp = spacy.load('en_core_web_lg')
news_df = clean_text(news.iloc[0:,:], nlp)
news_df = news_df.drop_duplicates(subset='Headline', keep ='first').reset_index(drop=True)   # drop duplicates

# perform random undersampling
df_relv_0 = news_df[news_df['Relevance'] == 0]
df_relv_1 = news_df[news_df['Relevance'] == 1]
df_relv_0_undsamp = df_relv_0.sample(int(df_relv_1.shape[0]*1.5),random_state=1)

news_df = pd.concat([df_relv_0_undsamp,df_relv_1], axis=0, ignore_index=True)

# create similarity
string_df = pd.DataFrame(news_df['Headline'])
string_df.columns = ['Theme']

theme_words = ['trade, jobs, economy, fiscal budget surplus, pension, ministry PEMX','election, vote, sanctions, foreign minister','banks, issues ratings, inflation, repo, interest rates']
theme_headers = ['Economy', 'Election','Rates']

relv_df = pd.concat([string_df['Theme'], create_relv(theme_words[0],string_df,theme_headers[0])], axis=1)
for i in range(1,len(theme_words)):
    relv_df = pd.concat([relv_df, create_relv(theme_words[i],string_df,theme_headers[i])], axis=1)

sns.countplot(news_df['Relevance']) 

tf = TfidfVectorizer(analyzer='word',ngram_range=(1,3),max_features=100)
X = tf.fit_transform(news_df['Headline'])

# convert to array
X = pd.DataFrame(X.toarray())
X_add = relv_df.iloc[:,1:4]
X = pd.concat([X, X_add],axis=1).values

y = news_df['Relevance']
X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.3, random_state=seed)

nb = BernoulliNB()
nb.fit(X_train,y_train)
pred = nb.predict(X_valid)

y_valid = y_valid.values

print('Confusion matrix\n',confusion_matrix(y_valid,pred))
print('Classification_report\n',classification_report(y_valid,pred))

comp_out = pd.DataFrame([pred, y_valid]).T
comp_out = pd.concat([relv_df.iloc[X_train.shape[0]:,0:4].reset_index(drop=True), comp_out], axis=1, ignore_index= True)
comp_out.columns = ['News','Econ','Elc','Rates','Pred','Actual']

# open output excel
wb = xw.Book('Output.xlsx')
sht = wb.sheets[0]
sht.range('A1').value = pd.DataFrame(index=np.full(1000, np.nan), columns=np.full(30, np.nan))
sht.range('A1').value = comp_out


