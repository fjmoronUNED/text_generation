import nltk
import os
import pickle
import yaml

from cleaner import Cleaner
from sequences_generator import SequencesGenerator

from ml_preprocessing import es_stopwords


path = "/Users/fjmoronreyes/text_generation/repositories/datasets/"
sequences = SequencesGenerator(path, stopwords=False)
sequences.create_sentences_files()
testing = sequences.get_sequences()
print(testing)
