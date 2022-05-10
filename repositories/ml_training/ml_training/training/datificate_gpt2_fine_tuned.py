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
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from transformers import TFAutoModel, AutoTokenizer, AutoConfig
import tensorflow as tf


class LstmKerasTrainer:
    def __init__(self, stopwords=False):

        self.sequences_path = (
           "/Users/fjmoronreyes/text_generation/repositories/datasets/sequences_files"
        )
        self.sequences_path = "../../../datasets/sequences_files"
        self.config_path = "./config/lstm_keras_config.yaml"
        self.model_files = "/Users/fjmoronreyes/text_generation/repositories/ml_training/ml_training/inference/lstm_keras"

        with open(self.config_path) as config_file:
            lstm_config = yaml.load(config_file, Loader=yaml.FullLoader)

        self.model_name = lstm_config['model_name']
        self.max_length = lstm_config['max_length']
        self.padding = lstm_config["padding"]
        self.truncation = lstm_config['truncation']
        self.tensors = lstm_config['tensors']

        self.embedding_dim = lstm_config["embedding_dim"]
        self.lstm_dim = lstm_config["lstm_dim"]
        self.learning_rate = lstm_config["learning_rate"]
        self.loss_function = lstm_config["loss_function"]
        self.metrics = lstm_config["metrics"]
        self.training_portion = lstm_config["training_portion"]
        self.epochs = lstm_config["epochs"]

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

        return train_sentences, validation_sentences

    def datificate_gpt2_embeddings(self, sentences):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = TFAutoModel.from_pretrained(self.model_name)

        encoded_input = tokenizer(sentences,
                                  max_length=self.max_length,
                                  padding=self.padding,
                                  truncation=self.truncation,
                                  return_tensors=self.tensors)

        output = model(encoded_input)
        return output['last_hidden_state']

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

    def lstm_keras(self, X, y, val_X, val_y):
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
        model.add(LSTM(self.lstm_dim))
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
        model = self.lstm_keras(X, y, val_X, val_y)
        return model


trainer = LstmKerasTrainer()
#sentences = trainer.get_rolling_window_sequence()
#trainer.keras_embeddings(sentences)
trainer.train()
