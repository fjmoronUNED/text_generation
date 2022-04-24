import nltk
import os
import pickle

from cleaner import Cleaner

# from ml_preprocessing import STOPWORDS


# document = "/Users/fjmoronreyes/text_generation/repositories/datasets/jrr_tolkien-la-ultima-cancion-de-bilbo.txt"

# f = open(document, "r")
# document = f.read()
#
# a_list = nltk.tokenize.sent_tokenize(document)
#
# cleaner_dataset = Cleaner()
# print(cleaner_dataset.make_clean(a_list, stopwords=False))

cleaner_dataset = Cleaner()

path = "/Users/fjmoronreyes/text_generation/repositories/datasets/raw_files/"
sentences_file_path = (
    "/Users/fjmoronreyes/text_generation/repositories/datasets/sentences_file/"
)
os.chdir(path)


def read_text_files(file_path):
    with open(file_path, "r") as f:
        return f.read()


for file in os.listdir():
    if file.endswith(".txt"):
        file_name = file.replace(".txt", "")
        file_path = f"{path}/{file}"
        file_content = read_text_files(file_path)
        file_sentences = nltk.tokenize.sent_tokenize(file_content)
        file_cleaned = cleaner_dataset.make_clean(file_sentences, stopwords=False)
        with open(sentences_file_path + file_name + "_sentences.pkl", "wb") as f:
            pickle.dump(file_cleaned, f)

# a_list = nltk.tokenize.sent_tokenize(document)
# cleaner_dataset = Cleaner()
# print(cleaner_dataset.make_clean(a_list, stopwords=False))
