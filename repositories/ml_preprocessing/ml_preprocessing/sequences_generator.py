import os
from cleaner import Cleaner
import pickle as pkl
import nltk


class SequencesGenerator:
    def __init__(self, files_path, stopwords=False):

        self.files_path = files_path
        self.raw_path = files_path + "raw_files/"
        self.stopwords = stopwords

        if not os.path.isdir(self.raw_path):
            print("Error, se necesita un directorio para generar los archivos")
        if not os.path.isdir(self.files_path + "sentences_files"):
            os.makedirs(self.files_path + "sentences_files")
            print("CREATING SENTENCES FILES FOLDER")
        if not os.path.isdir(self.files_path + "sequences_files"):
            os.makedirs(self.files_path + "sequences_files")
            print("CREATING SEQUENCES FILES FOLDER")

        self.sentences_files_path = files_path + "sentences_files/"  # hard coded
        self.sequences_files_path = files_path + "sequences_files/"  # hard coded

    def create_sentences_files(self):
        """
        Function to get sentences from the document
        """

        def read_text_files(file_path):
            with open(file_path, "r") as f:
                return f.read()

        cleaner_dataset = Cleaner()
        os.chdir(self.raw_path)

        for file in os.listdir():
            if file.endswith(".txt"):
                file_name = file.replace(".txt", "")
                file_path = f"{self.raw_path}/{file}"
                file_content = read_text_files(file_path)
                file_sentences = nltk.tokenize.sent_tokenize(file_content)
                if not self.stopwords:
                    file_cleaned = cleaner_dataset.make_clean(
                        file_sentences, stopwords=False
                    )
                else:
                    file_cleaned = cleaner_dataset.make_clean(
                        file_sentences, stopwords=True
                    )
                with open(
                    self.sentences_files_path + file_name + "_sentences.pkl", "wb"
                ) as f:
                    pkl.dump(file_cleaned, f)
        return file_cleaned

    def get_sentences(self):
        os.chdir(self.sentences_files_path)

        complete_dataset = []
        for file in os.listdir():
            if file.endswith(".pkl"):
                with open(file, "rb") as f:
                    sentences = pkl.load(f)
                    for sentence in sentences:
                        complete_dataset.append(sentence)
        return complete_dataset
