import os
import logging


logger = logging.getLogger("ml_training")

FILE_PATH = os.path.dirname(__file__)

BILSTM_FILES = os.environ.get("BILSTM_FILES", f"{FILE_PATH}/inference/bilstm_keras")
GRU_FILES = os.environ.get("GRU_FILES", f"{FILE_PATH}/inference/gru_keras")
LSTM_FILES = os.environ.get("LSTM_FILES", f"{FILE_PATH}/inference/lstm_keras")
GPT2_HF = os.environ.get("LSTM_FILES", f"{FILE_PATH}/inference/gpt2_hf")
GPT2_FINE_TUNING = os.environ.get("GPT2_FINE_TUNING", f"{FILE_PATH}/inference/gpt2_fine_tuned")
CONFIG = os.environ.get("CONFIG", f"{FILE_PATH}/training/config")
