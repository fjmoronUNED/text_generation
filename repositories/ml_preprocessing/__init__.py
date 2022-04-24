import os
import logging

logger = logging.getLogger("ml_preprocessing")

FILE_PATH = os.path.dirname(__file__)

STOPWORDS = os.environ.get("STOPWORDS", f"{FILE_PATH}/ml_preprocessing/stopwords")
