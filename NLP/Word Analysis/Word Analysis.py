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

inputDF = pd.ExcelFile("News_dataset.xlsx")
tabnames = inputDF.sheet_names
news = inputDF.parse(tabnames[0])

# Word tokenize
nlp = spacy.load('en_core_web_sm')

# initialize df
datatxt_df = pd.DataFrame()
txt = news['Headline'][0]
txt_corp = nlp(txt)
    
for ent in txt_corp.ents:
    datatxt_df = datatxt_df.append([[ent.text, ent.start_char, ent.end_char, ent.label_]])
#    print(ent.text, ent.start_char, ent.end_char, ent.label_)

for i in range(1,500):
    txt = news['Headline'][i]
    txt_corp = nlp(txt)
    
    for ent in txt_corp.ents:
        datatxt_df = datatxt_df.append([[ent.text, ent.start_char, ent.end_char, ent.label_]])
#        print(ent.text, ent.start_char, ent.end_char, ent.label_)

datatxt_df.columns = list(['entity','strs','end','type'])
datatxt_df = datatxt_df.reset_index(drop=True)

# extract data types
people = datatxt_df.loc[datatxt_df['type']=='PERSON','entity']
org = datatxt_df.loc[datatxt_df['type']=='ORG','entity']

