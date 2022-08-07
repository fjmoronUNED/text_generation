import os
import logging

logger = logging.getLogger("ml_preprocessing")

FILE_PATH = os.path.dirname(__file__)

STOPWORDS = os.environ.get("STOPWORDS", f"{FILE_PATH}/stopwords")

DATASETS = os.environ.get("DATASETS", f"{FILE_PATH}/../../datasets/raw_files")
SENTENCES_FILES = os.environ.get("DATASETS", f"{FILE_PATH}/../../datasets/sentences_files")
SEQUENCES_FILES = os.environ.get("DATASETS", f"{FILE_PATH}/../../datasets/sequences_files")