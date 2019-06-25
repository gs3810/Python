import pandas as pd
import numpy as np
from flair.data import Sentence, Corpus
from flair.models import SequenceTagger, TextClassifier
from flair.datasets import ColumnCorpus, ClassificationCorpus, TREC_6, NEWSGROUPS
from flair.data_fetcher import NLPTaskDataFetcher
from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings, DocumentRNNEmbeddings, DocumentLSTMEmbeddings
from flair.models import TextClassifier
from flair.trainers import ModelTrainer
from pathlib import Path

inputDF = pd.ExcelFile("News_dataset.xlsx")
tabnames = inputDF.sheet_names
data = inputDF.parse(tabnames[0])

# randomnly balance dataset
df_relv_0 = data[data['Relevance'] == 'Irrelevant']
df_relv_1 = data[data['Relevance'] == 'Relevant']
df_relv_0_undsamp = df_relv_0.sample(int(df_relv_1.shape[0]*1.2),random_state=1)
data = pd.concat([df_relv_0_undsamp,df_relv_1], axis=0, ignore_index=True)
data = data.sample(frac=1)

data = data[['Relevance', 'Headline']].rename(columns={"Relevance":"label", "Headline":"text"})
 
# convert to fastext format
data = '__label__' + data['label'].astype(str) +' ' + data['text']

data.iloc[0:int(len(data)*0.8)].to_csv('Data/train.csv', index = False, header = False)
data.iloc[int(len(data)*0.8):int(len(data)*0.9)].to_csv('Data/test.csv', index = False, header = False)
data.iloc[int(len(data)*0.9):].to_csv('Data/dev.csv', index = False, header = False)

corpus = NLPTaskDataFetcher.load_classification_corpus(Path('Data/'), 
                                                       test_file='test.csv',
                                                       dev_file='dev.csv',
                                                       train_file='train.csv')
# convert to Flair dictionary 
label_dict = corpus.make_label_dictionary()

word_embeddings = [WordEmbeddings('glove'),
                   FlairEmbeddings('news-forward-fast'),
                   FlairEmbeddings('news-backward-fast')]

document_embeddings: DocumentLSTMEmbeddings = DocumentLSTMEmbeddings(word_embeddings,
                                                                     hidden_size=512,
                                                                     reproject_words=True,
                                                                     reproject_words_dimension=256,
                                                                     )

classifier = TextClassifier(document_embeddings, label_dictionary=label_dict)

trainer = ModelTrainer(classifier, corpus)

"""Train this section separately"""
trainer.train('Model/',
              learning_rate=0.5,
              mini_batch_size=16,
              max_epochs=15)


