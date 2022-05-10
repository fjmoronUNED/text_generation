from transformers import TFAutoModel, AutoTokenizer, AutoConfig
import tensorflow as tf

model_name = "datificate/gpt2-small-spanish"

#tokenizer = AutoTokenizer.from_pretrained('./transformer_models/gpt2-small-spanish/tokenizer')
#config = AutoConfig.from_pretrained('./transformer_models/gpt2-small-spanish/config.json')
#model = TFAutoModel.from_pretrained('./transformer_models/gpt2-small-spanish/tf_model.h5', config=config)

tokenizer = AutoTokenizer.from_pretrained(model_name)
#tokenizer.save_pretrained('./transformer_models/gpt2-small-spanish/tokenizer')
model = TFAutoModel.from_pretrained(model_name)
text = 'hola a todos'

inputs = tokenizer(text, return_tensors='tf')
output = model(inputs)

print(output['last_hidden_state'])