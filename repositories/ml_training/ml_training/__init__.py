import os
import logging


logger = logging.getLogger("ml_training")

FILE_PATH = os.path.dirname(__file__)

BILSTM_FILES = os.environ.get("BILSTM_FILES", f"{FILE_PATH}/inference/bilstm_keras")
GRU_FILES = os.environ.get("GRU_FILES", f"{FILE_PATH}/inference/gru_keras")
LSTM_FILES = os.environ.get("LSTM_FILES", f"{FILE_PATH}/inference/lstm_keras")
GPT2_HF_DATIFICATE = os.environ.get("GPT2_HF_DATIFICATE", f"{FILE_PATH}/inference/gpt2_fine_tuned_datificate")
GPT2_HF_DEEPESP = os.environ.get("GPT2_HF_DEEPESP", f"{FILE_PATH}/inference/gpt2_fine_tuned_deepesp")
GPT2_DATIFICATE = os.environ.get("GPT2_DATIFICATE", f"{FILE_PATH}/inference/gpt2_datificate")
GPT2_DEEPESP = os.environ.get("GPT2_DEEPESP", f"{FILE_PATH}/inference/gpt2_deepesp")
CONFIG = os.environ.get("CONFIG", f"{FILE_PATH}/training/config")
