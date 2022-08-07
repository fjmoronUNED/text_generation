from transformers import TFAutoModel, AutoTokenizer, AutoConfig
import tensorflow as tf

from transformers import pipeline

chef = pipeline(
    "text-generation",
    model="/Users/fjmoronreyes/text_generation/repositories/ml_training/ml_training/inference/gpt2_datificate_transformers",
    tokenizer="datificate/gpt2-small-spanish",
    # config={"max_length": 800},
)

print(chef("los hobbits se dirigían"))
