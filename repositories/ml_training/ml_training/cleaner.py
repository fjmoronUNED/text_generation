import re
from nltk import word_tokenize
import yaml


class Cleaner:
    def __init__(self, text_list=True):

        if type(text_list) is str:
            print("CLASE PENSADA PARA UNA LISTA")
        if type(text_list) is list:
            print("LIMPIADO LISTA")

    def get_stopwords(self):
        with open(r"./es_stopwords.yaml") as file:
            self.es_stopwords = yaml.load(file, Loader=yaml.FullLoader)

    def remove_accents(self, token):
        token = str(token)
        token = re.sub("[àáâãäå]", "a", token)
        token = re.sub("[èéêë]", "e", token)
        token = re.sub("[ìíîï]", "i", token)
        token = re.sub("[òóôõö]", "o", token)
        token = re.sub("[ùúûü]", "u", token)
        return token

    def remove_punctuations(self, token):
        """Removes common punctuation characters."""
        token = str(token)
        token = re.sub("[.,:;…]", "", token)
        token = re.sub("[”“''" "«»<>]", "", token)
        token = re.sub("[¿?()!¡]", "", token)
        token = re.sub("[~–|-]", "", token)
        token = re.sub("[\\\\/]", "", token)
        return token

    def remove_symbols(self, token):
        """Removes common symbol characters."""
        token = str(token)
        token = re.sub("[@•�]", "", token)
        token = re.sub("[%#*]", "", token)
        token = re.sub("[¢€$]", "", token)
        token = re.sub("[-=+÷∞]", "", token)
        token = re.sub("[¬]", "", token)
        return token

    def remove_numbers(self, token):
        """Removes unique numbers."""
        token = str(token)
        token = re.sub("[0-9]{1}", "", token)
        return token

    def clean_dataset(self, text_list):
        new_text = []
        for sentence in text_list:
            clean_text = sentence.replace("\n", "")
            new_text.append(clean_text)
        print("DATASET LIMPIADO")
        return new_text

    def tokenize_dataset(self, new_text):
        """
        Tokenize every sentence
        """
        tokenized_sentences = []
        for sentence in new_text:
            sentence = str(sentence)
            tokens = word_tokenize(sentence)
            tokenized_sentences.append(tokens)

        print("DATASET TOKENIZADO")
        return tokenized_sentences

    def clean_tok_dataset(self, tokenized_sentences):
        """
        Clean token by token in a tokenized list
        """
        clean_sentences = []
        for sentence in tokenized_sentences:
            sentences = []
            for token in sentence:
                token = str(token)
                token = token.strip()
                token = token.lower()
                token = self.remove_accents(token)
                token = self.remove_symbols(token)
                token = self.remove_punctuations(token)
                token = self.remove_numbers(token)
                if len(token) > 2:
                    sentences.append(token)
            clean_sentences.append(sentences)

        print("TOKENS LIMPIADOS")
        return clean_sentences

    def remove_stopwords(self, tokenized_sentences):
        """
        Only Model_search. Remove stopwords from the sentences
        """
        clean_sentences = []
        for sentence in tokenized_sentences:
            sentences = []
            for token in sentence:
                if token not in self.es_stopwords:
                    sentences.append(token)
            clean_sentences.append(sentences)
        print("STOPWORDS LIMPIADAS")
        return clean_sentences

    def get_only_useful_sentences(self, sentences, min_sentence_size=2):
        """
        Keep sentences wiht a determinate size
        """
        useful_sentences = []
        for sentence in sentences:
            sentence_size = len(sentence)
            if sentence_size >= min_sentence_size:
                useful_sentences.append(sentence)

        print(
            "ELIMINADAS AQUELLAS FRASES CUYA LONGITUD ES INFERIOR A {}".format(
                min_sentence_size
            )
        )
        print("NÚMERO TOTAL DE FRASES UTILES: {}".format(len(useful_sentences)))
        return useful_sentences

    def make_clean(self, text, stopwords=False):
        new_dataset = self.clean_dataset(text)
        tok_dataset = self.tokenize_dataset(new_dataset)
        clean_dataset = self.clean_tok_dataset(tok_dataset)
        if stopwords:
            clean_dataset = self.remove_stopwords(clean_dataset)
        def_dataset = self.get_only_useful_sentences(clean_dataset)
        return def_dataset
