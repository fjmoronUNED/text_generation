import nltk
import os
import pickle
import yaml

from cleaner import Cleaner
from sequences_generator import SequencesGenerator

from ml_preprocessing import es_stopwords


################
# test cleaner #
################
"""
cleaner = Cleaner()
with open(
    "/Users/fjmoronreyes/text_generation/repositories/datasets/raw_files/jrr_tolkien-la-ultima-cancion-de-bilbo.txt"
) as f:
    content = f.read()

# file_sentences = nltk.tokenize.sent_tokenize(content)
file_cleaned = cleaner.make_clean(content, stopwords=False)
print(file_cleaned)
"""

############################
# test sequences_generator #
############################

path = "/Users/fjmoronreyes/text_generation/repositories/datasets/"
sequences = SequencesGenerator(path, stopwords=False)
sequences.create_sentences_files()
testing = sequences.get_sequences()
print(testing)
