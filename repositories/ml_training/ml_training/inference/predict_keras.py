import json
import pickle as pkl
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

class PredictKeras:
    def __init__(self, model_name):

        self.model_name = model_name
        self.padding = 'pre'

    def load_model(self):
        if self.model_name == 'lstm':
            self.model = load_model("./lstm_keras/model.h5")
            max_len_file = open("./lstm_keras/max_len.txt", "r")
            value = max_len_file.read()
            int_max_len = int(value)
            self.max_len = int_max_len - 1
            with open("./lstm_keras/tokenizer.pkl", "rb") as tok_file:
                self.tokenizer = pkl.load(tok_file)
        elif self.model_name == 'bilstm':
            self.model = load_model("./bilstm_keras/model.h5")
            max_len_file = open("./bilstm_keras/max_len.txt", "r")
            value = max_len_file.read()
            int_max_len = int(value)
            self.max_len = int_max_len - 1
            with open("./bilstm_keras/tokenizer.pkl", "rb") as tok_file:
                self.tokenizer = pkl.load(tok_file)
        elif self.model_name == 'gru':
            self.model = load_model("./gru_keras/model.h5")
            max_len_file = open("./gru_keras/max_len.txt", "r")
            value = max_len_file.read()
            int_max_len = int(value)
            self.max_len = int_max_len - 1
            with open("./gru_keras/tokenizer.pkl", "rb") as tok_file:
                self.tokenizer = pkl.load(tok_file)
        else:
            print('Try with one of the following models: lstm, bilstm, gru')

    def get_tokenizer(self):
        print(self.tokenizer.word_index)

    def predict(self, seed_text):
        self.load_model()
        inverse_tokenizer = {v: k for k, v in self.tokenizer.word_index.items()}
        predictions = 10

        text_generation = []
        for i in range(0, predictions):
            seed_seq = self.tokenizer.texts_to_sequences([seed_text])[0]
            seed_pad_seq = pad_sequences(
                [seed_seq], maxlen=self.max_len, padding=self.padding
            )
            predicted_proba = self.model.predict(seed_pad_seq, verbose=0)
            indices = (predicted_proba).argsort()[0][0]
            word_predicted = inverse_tokenizer[indices]
            seed_text = seed_text + ' ' + word_predicted
            if i == predictions - 1:
                print(seed_text)
                return seed_text


predicter = PredictKeras('lstm')
#print()
#predicter.load_model()
seed_text = 'frodo y sam'
predicter.predict(seed_text)