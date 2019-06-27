from flair.models import TextClassifier
from flair.data import Sentence
import pandas as pd

# load model
classifier = TextClassifier.load('Model/Best model/best-model_larger set.pt')

data = pd.read_csv('Data/test.csv')
data.columns = ['Text']

# get label
label_list = list()
for text in list(data['Text']):
    label_list.append(' '.join(text.split()[0:1]))

label = pd.DataFrame.from_records([label_list]).T
label.columns = ['Label']

# strip the fasttext label
strip_list = list()
for text in list(data['Text']):
    strip_list.append(' '.join(text.split()[1:]))

data = pd.DataFrame.from_records([strip_list]).T
data.columns = ['Text']

data_list = list()

for labl in data['Text']:
    sentence = Sentence(labl)
    classifier.predict(sentence)
    data_list.append(list(sentence.labels))

pred_df = pd.DataFrame.from_records(data_list)

# get output matrix
output_df = pd.concat([pred_df,label],axis=1)
output_df.to_csv('Flair_output.csv')
