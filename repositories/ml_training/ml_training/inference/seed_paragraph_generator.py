import pickle as pkl
import os
from ml_preprocessing import PARAGRAPHS_FILES
from nltk import word_tokenize

def get_paragraphs():
    with open(PARAGRAPHS_FILES + '/tolkien_paragraphs_cuentos_perdidos.pkl', 'rb') as f:
        data = pkl.load(f)
    print('Total useful paragraphs: {}'.format(len(data)))

    total_tokens = 0

    for sentence in data:
        total_tokens += len(word_tokenize(sentence))

    print('Total tokens: {}'.format(total_tokens))

    average_percent = total_tokens / len(data)

    print('Token average x paragraph: {}'.format(int(average_percent)))

    info_list = [str('Total useful paragraphs: {}'.format(len(data))),
                str(('Total tokens: {}'.format(total_tokens))),
                str('Token average x paragraph: {}'.format(int(average_percent)))]

    with open(PARAGRAPHS_FILES + '/info_paragraphs.txt', 'w') as f:
        for line in info_list:
            f.write(line)
            f.write('\n')

    return data, average_percent


def save_tokenized_paragraphs():
    data, average_percent = get_paragraphs()

    tokenized_list = []

    for sentence in data:
        tokenized_list.append(word_tokenize(sentence))

    with open(PARAGRAPHS_FILES + '/tokenized_tolkien_paragraphs_cuentos_perdidos.pkl', "wb") as f:
        pkl.dump(tokenized_list, f)

def save_seed_texts():
    with open(PARAGRAPHS_FILES + '/tokenized_tolkien_paragraphs_cuentos_perdidos.pkl', 'rb') as f:
        data = pkl.load(f)

    seed_text_list = []

    for i in data:
        seed_text_list.append(' '.join(i[:100]))

    with open(PARAGRAPHS_FILES + '/seed_text_paragraphs_cuentos_perdidos.pkl', "wb") as f:
        pkl.dump(seed_text_list, f)

get_paragraphs()
save_tokenized_paragraphs
save_seed_texts()