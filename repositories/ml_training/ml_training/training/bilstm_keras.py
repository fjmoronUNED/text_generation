import re
import os
from shlex import join
import numpy as np
import yaml
import pickle as pkl
from ml_preprocessing.cleaner import Cleaner

import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


class BiLstmKerasTrainer:
    def __init__(self, stopwords=False):

        self.sequences_path = (
           "/Users/fjmoronreyes/text_generation/repositories/datasets/sequences_files"
        )
        self.sequences_path = "../../../datasets/sequences_files"
        self.config_path = "./config/bilstm_keras_config.yaml"
        self.model_files = "/Users/fjmoronreyes/text_generation/repositories/ml_training/ml_training/inference/bilstm_keras"

        with open(self.config_path) as config_file:
            bilstm_config = yaml.load(config_file, Loader=yaml.FullLoader)

        self.embedding_dim = bilstm_config["embedding_dim"]
        self.dropout = bilstm_config["dropout"]
        self.bilstm_dim = bilstm_config["lstm_dim"]
        self.secbilstm_dim = bilstm_config["2lstm_dim"]
        self.learning_rate = bilstm_config["learning_rate"]
        self.loss_function = bilstm_config["loss_function"]
        self.metrics = bilstm_config["metrics"]
        self.padding = bilstm_config["padding"]
        self.training_portion = bilstm_config["training_portion"]
        self.epochs = bilstm_config["epochs"]

    def get_rolling_window_sequence(self):
        os.chdir(self.sequences_path)

        complete_dataset = []
        for file in os.listdir():
            if file.endswith(".pkl"):
                with open(file, "rb") as f:
                    sentences = pkl.load(f)
                    for sentence in sentences:
                        complete_dataset.append(sentence)
        return complete_dataset

    def keras_embeddings(self, sentences):
        """
        Convert the input word sequences with keras texts_to_sequences
        """
        train_size = int(len(sentences) * self.training_portion)

        train_sentences = sentences[:train_size]
        validation_sentences = sentences[train_size:]

        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(train_sentences)
        train_sequences = tokenizer.texts_to_sequences(train_sentences)
        validation_sequences = tokenizer.texts_to_sequences(validation_sentences)

        self.max_sequence_len = max([len(x) for x in train_sequences])
        self.total_words = len(tokenizer.word_index)

        encoded_train = np.array(
            pad_sequences(
                train_sequences, maxlen=self.max_sequence_len, padding=self.padding
            )
        )
        encoded_val = np.array(
            pad_sequences(
                validation_sequences, maxlen=self.max_sequence_len, padding=self.padding
            )
        )

        with open(self.model_files + "/tokenizer.pkl", "wb") as handle:
            pkl.dump(tokenizer, handle, protocol=pkl.HIGHEST_PROTOCOL)
        with open(self.model_files + "/max_len.txt", "w") as txt_f:
            txt_f.write(str(self.max_sequence_len))

        return encoded_train, encoded_val

    def train_test_sequences(self, encoded_train, encoded_val):
        """
        Define X, y, val_X and val_y values
        """
        X = encoded_train[:, :-1]
        y = encoded_train[:, -1:]

        val_X = encoded_val[:, :-1]
        val_y = encoded_val[:, -1:]

        print("X size: {}".format(len(X)))
        print("y size: {}".format(len(y)))
        print("val_X size: {}".format(len(val_X)))
        print("val_y size: {}".format(len(val_y)))

        print("X shape: {}".format(X.shape))
        print("y shape: {}".format(y.shape))
        print("val_X shape: {}".format(val_X.shape))
        print("val_y shape: {}".format(val_y.shape))
        return X, y, val_X, val_y

    def bilstm_keras(self, X, y, val_X, val_y):
        """
        Create the neural network model for document wordpredict
        """

        print("Training LSTM KERAS model using Keras embeddings (text to sequences)...")
        model = Sequential()
        model.add(
            Embedding(
                self.total_words + 1,
                self.embedding_dim,
                input_length=self.max_sequence_len - 1,
            )
        )
        model.add(Dropout(self.dropout))
        model.add(Bidirectional(LSTM(self.bilstm_dim, return_sequences = True)))
        model.add(Dropout(self.dropout))
        model.add(Bidirectional(LSTM(self.secbilstm_dim)))
        model.add(Dense(self.total_words + 1, activation="softmax"))
        opt = Adam(learning_rate=self.learning_rate)
        model.summary()
        model.compile(loss=self.loss_function, optimizer=opt, metrics=[self.metrics])
        model.fit(X, y, epochs=self.epochs, validation_data=(val_X, val_y), verbose=1)
        model.save(self.model_files + "/model.h5")
        return model

    def train(self):
        sentences = self.get_rolling_window_sequence()
        encoded_train, encoded_val = self.keras_embeddings(sentences)
        X, y, val_X, val_y = self.train_test_sequences(encoded_train, encoded_val)
        model = self.bilstm_keras(X, y, val_X, val_y)
        return model


trainer = BiLstmKerasTrainer()
trainer.train()
