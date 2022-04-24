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

    def sentences_to_sequences(self, sentences, train_len=8, min_sentence_size=2):
        """
        create corpus sequence
        """
        final_seqs = []
        for token_group in sentences:
            for pos_init in range(0, len(token_group) - 1):
                pos_end_max = len(token_group) - pos_init
                for pos_end in range(1, min(pos_end_max, train_len)):
                    seq = token_group[pos_init : (pos_init + pos_end + 1)]
                    if len(seq) >= min_sentence_size:
                        final_seqs.append(seq)

        counter = 1
        final_dataset = []

        for i in final_seqs:
            for counter in range(0, len(i)):
                final_seq = len(i) - 1
                init_seq = final_seq - counter
                grammar_sequence = i[init_seq:final_seq] + [i[final_seq]]
                if len(grammar_sequence) > 1:
                    final_dataset.append(grammar_sequence)

        return final_dataset

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
                # file_sentences = nltk.tokenize.sent_tokenize(file_content)
                if not self.stopwords:
                    file_cleaned = cleaner_dataset.make_clean(
                        file_content, stopwords=False
                    )
                else:
                    file_cleaned = cleaner_dataset.make_clean(
                        file_content, stopwords=True
                    )
                with open(
                    self.sentences_files_path + file_name + "_sentences.pkl", "wb"
                ) as f:
                    pkl.dump(file_cleaned, f)
                file_sequences = self.sentences_to_sequences(file_cleaned)
                with open(
                    self.sequences_files_path + file_name + "_sequences.pkl", "wb"
                ) as f:
                    pkl.dump(file_sequences, f)
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

    def get_sequences(self):
        os.chdir(self.sequences_files_path)

        complete_dataset = []
        for file in os.listdir():
            if file.endswith(".pkl"):
                with open(file, "rb") as f:
                    sentences = pkl.load(f)
                    for sentence in sentences:
                        complete_dataset.append(sentence)
        return complete_dataset
