import nltk
import os
import pickle
import yaml

from cleaner import Cleaner
from sequences_generator import SequencesGenerator

from ml_preprocessing import STOPWORDS

# document = "/Users/fjmoronreyes/text_generation/repositories/datasets/jrr_tolkien-la-ultima-cancion-de-bilbo.txt"
# f = open(document, "r")
# document = f.read()
# a_list = nltk.tokenize.sent_tokenize(document)
# cleaner_dataset = Cleaner()
# print(cleaner_dataset.make_clean(a_list, stopwords=False))

path = "/Users/fjmoronreyes/text_generation/repositories/datasets/"
sequences = SequencesGenerator(path, stopwords=False)
sequences.create_sentences_files()
testing = sequences.get_sentences()
print(testing)
