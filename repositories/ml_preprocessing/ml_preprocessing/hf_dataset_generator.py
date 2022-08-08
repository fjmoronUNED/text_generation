import os
import pickle as pkl
import re

from ml_preprocessing import HF_FILES
from ml_preprocessing import DATASETS

class HfDatasetGenerator:
    def __init__(self):

        self.raw_path = DATASETS
        self.hf_files = HF_FILES

        if not os.path.isdir(self.raw_path):
            print("Error, se necesita un directorio para generar los archivos")
        if not os.path.isdir(self.hf_files):
            print("Error, se necesita un directorio para generar los archivos de frases")

    def create_raw_text(self):
        '''
        crate raw text file
        '''
        def read_text_files(file_path):
            with open(file_path, "r") as f:
                return f.read()

        for file in os.listdir(self.raw_path):
            raw_file = open(self.raw_path + '/data.txt', 'a')
            data = ''
            if not file.endswith("ta.txt"):
                file_path = f"{DATASETS}/{file}"
                file_content = read_text_files(file_path)
                raw_content = re.sub(r'\n', ' ', file_content)
                raw_content = re.sub(r'\x0c', ' ', raw_content)
                with open(HF_FILES + '/data.txt', 'a') as final_file:
                    final_file.write(raw_content)

    def get_raw_text(self):
        '''
        get raw text
        '''
        f = open(self.hf_files + '/data.txt', 'r')
        return f.read()

    def train_test_split_dataset(self):
        raw_text = self.get_raw_text()

        total_text_length = len(raw_text)
        test_size = 0.15
        splitter = total_text_length - int(total_text_length * test_size)
        splitter = splitter

        train = raw_text[:splitter]
        test = raw_text[splitter:]

        with open(self.hf_files + '/train.txt', 'a') as train_file:
            train_file.write(train)

        with open(self.hf_files + '/test.txt', 'a') as test_file:
            test_file.write(test)