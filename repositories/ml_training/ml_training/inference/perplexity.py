import math
import torch
from flair.embeddings import FlairEmbeddings, StackedEmbeddings
import pickle as pkl
from ml_preprocessing import PARAGRAPHS_FILES

from ml_training.inference import predict_keras
from ml_training.inference import predict_hf


def calculate_perplexity(sentence):
    model = FlairEmbeddings('es-forward').lm
    input = torch.tensor([model.dictionary.get_idx_for_item(char) for char in sentence[:-1]]).unsqueeze(1)

    hidden = model.init_hidden(1)
    prediction, _, hidden = model.forward(input, hidden)

    targets = torch.tensor([model.dictionary.get_idx_for_item(char) for char in sentence[1:]])
    cross_entroy_loss = torch.nn.CrossEntropyLoss()
    loss = cross_entroy_loss(prediction.view(-1, len(model.dictionary)), targets).item()

    perplexity = math.exp(loss)
    return perplexity

def average_perplexity(perplexity_list):
    total_perplexity = 0
    n_examples = len(perplexity_list)

    for i in perplexity_list:
        total_perplexity += i

    return total_perplexity / n_examples

def get_tolkien_paragraphs():
    with open(PARAGRAPHS_FILES + '/tolkien_paragraphs.pkl', 'rb') as f:
        data = pkl.load(f)

    return data

def get_seed_tolkien_paragraphs():
    with open(PARAGRAPHS_FILES + '/seed_text_paragraphs.pkl', 'rb') as f:
        data = pkl.load(f)

    return data


lstm_keras_generated_list = []
bilstm_keras_generated_list = []
gru_keras_generated_list = []
datificate_transformers_generated_list = []
deepesp_transformers_generated_list = []
fine_tuned_transformers_generated_list = []
counter = 0

predicter_lstm = predict_keras.PredictKeras('lstm')
predicter_bilstm = predict_keras.PredictKeras('bilstm')
predicter_gru = predict_keras.PredictKeras('gru')
predicter_dat = predict_hf.PredictGPT2('datificate')
predicter_deep = predict_hf.PredictGPT2('deepesp')
predicter_fine_tuned = predict_hf.PredictGPT2('datificate')

seed_texts = get_tolkien_paragraphs()
seed_texts = seed_texts[:500]

for sentence in seed_texts:
    try:
        counter += 1
        print(str(counter) + '/' + str(len(seed_texts)))
        lstm_keras_generated_list.append(predicter_lstm.predict(sentence))
        bilstm_keras_generated_list.append(predicter_bilstm.predict(sentence))
        gru_keras_generated_list.append(predicter_gru.predict(sentence))
        datificate_transformers_generated_list.append(predicter_dat.predict(sentence))
        deepesp_transformers_generated_list.append(predicter_deep.predict(sentence))
        fine_tuned_transformers_generated_list.append(predicter_fine_tuned.predict(sentence))
    except Exception as e:
        print(e)

rrn_dictionary_title = ['ORIGINAL', 'LSTM', 'BILSTM', 'GRU', 'DATIFICATE', 'DEEPESP', 'FINE_TUNE']
perplexity_average_list = []

#ORIGINAL
tolkien_paragraphs = get_tolkien_paragraphs()
tolkien_paragraphs = tolkien_paragraphs[:500]
original_perplexity = [calculate_perplexity(sentence) for sentence in tolkien_paragraphs]
perplexity_average_list.append(average_perplexity(original_perplexity))
print('ORIGINAL perplexity {}'.format(average_perplexity(original_perplexity)))

#LSTM
lstm_keras_perplexity_list = [calculate_perplexity(sentence) for sentence in lstm_keras_generated_list]
perplexity_average_list.append(average_perplexity(lstm_keras_perplexity_list))
print('LSTM perplexity {}'.format(average_perplexity(lstm_keras_perplexity_list)))
#BILSTM
bilstm_keras_perplexity_list = [calculate_perplexity(sentence) for sentence in bilstm_keras_generated_list]
perplexity_average_list.append(average_perplexity(bilstm_keras_perplexity_list))
print('BILSTM perplexity {}'.format(average_perplexity(bilstm_keras_perplexity_list)))
#GRU
gru_keras_perplexity_list = [calculate_perplexity(sentence) for sentence in gru_keras_generated_list]
perplexity_average_list.append(average_perplexity(gru_keras_perplexity_list))
print('GRU perplexity {}'.format(average_perplexity(gru_keras_perplexity_list)))
#DATIFICATE
datificate_perplexity_list = [calculate_perplexity(sentence) for sentence in datificate_transformers_generated_list]
perplexity_average_list.append(average_perplexity(datificate_perplexity_list))
print('GPT2-Datificate perplexity {}'.format(average_perplexity(datificate_perplexity_list)))
#DEEPESP
deepesp_perplexity_list = [calculate_perplexity(sentence) for sentence in deepesp_transformers_generated_list]
perplexity_average_list.append(average_perplexity(deepesp_perplexity_list))
print('DEEPESP perplexity {}'.format(average_perplexity(deepesp_perplexity_list)))
#FINE TUNE
fine_tuned_perplexity_list = [calculate_perplexity(sentence) for sentence in fine_tuned_transformers_generated_list]
perplexity_average_list.append(average_perplexity(fine_tuned_perplexity_list))
print('FINE TUNED GPT2 perplexity {}'.format(average_perplexity(fine_tuned_perplexity_list)))

res = {}
for key in rrn_dictionary_title:
    for value in perplexity_average_list:
        res[key] = value
        perplexity_average_list.remove(value)
        break

print(res)
with open(PARAGRAPHS_FILES + '/perplexity_per_model.pkl', 'wb') as handle:
    pkl.dump(res, handle, protocol=pkl.HIGHEST_PROTOCOL)