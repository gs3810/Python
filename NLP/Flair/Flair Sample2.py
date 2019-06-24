import flair.datasets
from flair.data_fetcher import NLPTaskDataFetcher
from flair.data import Corpus
from flair.datasets import ClassificationCorpus
from pathlib import Path

# this is the folder in which train, test and dev files reside
#data_folder = path('/data')

# load corpus containing training, test and dev data
corpus: Corpus = ClassificationCorpus(Path('./data'),
                                      test_file='test.txt',
                                      dev_file='dev.txt',
                                      train_file='train.txt')

corpus = NLPTaskDataFetcher.load_classification_corpus(Path('./data'), 
                                                       test_file='test.txt',
                                                       dev_file='dev.txt',
                                                       train_file='train.txt')

print(corpus.make_label_dictionary())
df = str(list(corpus.train)[0:1])

label_dict = corpus.make_label_dictionary()
print(label_dict)










