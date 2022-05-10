import math
import torch
from flair.embeddings import FlairEmbeddings, StackedEmbeddings

# get language model
model = FlairEmbeddings('es-forward').lm

# example text
#text = 'Estaba paseando por mi casa cuando alguien intentó atacarme'
text = 'Estaba paseando meteorito casa teja azulejo grindear hama volcan'

# input ids
input = torch.tensor([model.dictionary.get_idx_for_item(char) for char in text[:-1]]).unsqueeze(1)

# push list of character IDs through model
hidden = model.init_hidden(1)
prediction, _, hidden = model.forward(input, hidden)

# the target is always the next character
targets = torch.tensor([model.dictionary.get_idx_for_item(char) for char in text[1:]])

# use cross entropy loss to compare output of forward pass with targets
cross_entroy_loss = torch.nn.CrossEntropyLoss()
loss = cross_entroy_loss(prediction.view(-1, len(model.dictionary)), targets).item()

# exponentiate cross-entropy loss to calculate perplexity
perplexity = math.exp(loss)

print(perplexity)