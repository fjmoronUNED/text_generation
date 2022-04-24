import os
import logging

logger = logging.getLogger("ml_preprocessing")

FILE_PATH = os.path.dirname(__file__)

STOPWORDS = os.environ.get("STOPWORDS", f"{os.getcwd()}/ml_preprocessing/stopwords")
