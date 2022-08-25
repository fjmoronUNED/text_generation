import pickle as pkl
from ml_preprocessing import PARAGRAPHS_FILES

with open(PARAGRAPHS_FILES + '/perplexity_per_model.pkl', 'rb') as f:
    data = pkl.load(f)

print(data)