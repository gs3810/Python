import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
seed = 100

def clean_text(df, nlp):
    """
    Tokenize, make lower case, remove Stop words, punctuation, digit, lemmatize
    """
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

inputDF = pd.ExcelFile("News_dataset.xlsx")
tabnames = inputDF.sheet_names
news = inputDF.parse(tabnames[0])

# Word tokenize
nlp = spacy.load('en_core_web_sm')

# initialize df
datatxt_df = pd.DataFrame()
txt = news['Headline'][0].capitalize()
txt_corp = nlp(txt)
    
for ent in txt_corp.ents:
    datatxt_df = datatxt_df.append([[ent.text, ent.start_char, ent.end_char, ent.label_]])

for i in range(1,news.shape[0]):
    txt = news['Headline'][i]
    txt_corp = nlp(txt)
    
    for ent in txt_corp.ents:
        datatxt_df = datatxt_df.append([[ent.text, ent.start_char, ent.end_char, ent.label_]])

datatxt_df.columns = list(['Entity','Start','End','Type'])
datatxt_df = datatxt_df.reset_index(drop=True)

# extract data types
people = datatxt_df.loc[datatxt_df['Type']=='PERSON','Entity']
org = datatxt_df.loc[datatxt_df['Type']=='ORG','Entity']

# draw freq. plots
order=pd.value_counts(people).iloc[:10].index
fig = plt.figure(figsize=(12,6)) 
plt.figure(1)
ax = sns.countplot(people,order=pd.value_counts(people).iloc[:10].index)
        
# draw freq. plots
order=pd.value_counts(people).iloc[:10].index
fig = plt.figure(figsize=(12,6)) 
plt.figure(2)
ax = sns.countplot(org,order=pd.value_counts(org).iloc[:10].index)
### drop all orgs that overlap with people