import re
import os
import numpy as np
import yaml

import tensorflow as tf

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

with open("./config/lstm_keras_config.yaml") as file:
    fruits_list = yaml.load(file, Loader=yaml.FullLoader)
    print(fruits_list)
