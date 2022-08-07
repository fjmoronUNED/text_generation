import os
import logging

logger = logging.getLogger("ml_preprocessing")

FILE_PATH = os.path.dirname(__file__)

STOPWORDS = os.environ.get("STOPWORDS", f"{FILE_PATH}/stopwords")

DATASETS = os.environ.get("DATASETS", f"{FILE_PATH}/../../datasets/raw_files")
HF_FILES = os.environ.get("HF_FILES", f"{FILE_PATH}/../../datasets/hf_files")
SENTENCES_FILES = os.environ.get("SENTENCES_FILES", f"{FILE_PATH}/../../datasets/sentences_files")
SEQUENCES_FILES = os.environ.get("SEQUENCES_FILES", f"{FILE_PATH}/../../datasets/sequences_files")
PARAGRAPHS_FILES = os.environ.get("PARAGRAPHS_FILES", f"{FILE_PATH}/../../datasets/paragraphs_files")