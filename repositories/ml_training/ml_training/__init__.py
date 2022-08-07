import os
import logging


logger = logging.getLogger("ml_training")

FILE_PATH = os.path.dirname(__file__)

BILSTM_FILES = os.environ.get("BILSTM_FILES", f"{FILE_PATH}/inference/bilstm_keras")
GRU_FILES = os.environ.get("GRU_FILES", f"{FILE_PATH}/inference/gru_keras")
LSTM_FILES = os.environ.get("LSTM_FILES", f"{FILE_PATH}/inference/lstm_keras")
GPT2_HF = os.environ.get("LSTM_FILES", f"{FILE_PATH}/inference/gpt2_hf")
CONFIG = os.environ.get("CONFIG", f"{FILE_PATH}/training/config")
HF_DATIFICATE_MODEL = os.environ.get("HF_DATIFICATE_MODEL", f"{FILE_PATH}/../../../hf_models/gpt2_datificate_transformers")