import os
from ml_training import BILSTM_FILES
import pickle as pkl
from ml_preprocessing import SENTENCES_FILES
from ml_preprocessing import SEQUENCES_FILES

complete_dataset = []
for file in os.listdir(SEQUENCES_FILES):
    if file.endswith(".pkl"):
        complete_dataset.append(file)
        #print(file)
        #with open(file, "rb") as f:
        #    sentences = pkl.load(f)
        #    for sentence in sentences:
        #        print(sentence)

if 'Las-Dos-Torres-J-R-R-Tolkien_sequences.pkl' in complete_dataset:
    print('hola')