import os
import pickle as pkl

from ml_preprocessing import DATASETS
from ml_preprocessing import PARAGRAPHS_FILES

class ParagraphsGenerator:
    def __init__(self):

        self.raw_path = DATASETS
        self.paragraphs_files = PARAGRAPHS_FILES

        if not os.path.isdir(self.raw_path):
            print("Error, se necesita un directorio para generar los archivos")
        if not os.path.isdir(self.paragraphs_files):
            print("Error, se necesita un directorio para generar los archivos de frases")

    def create_paragraphs_files(self, min_paragraph_len=30):
        """
        Function to get sentences from the document
        """

        def read_text_files(file_path):
            with open(file_path, "r") as f:
                return f.read()

        paragraph_list = []

        for file in os.listdir(self.raw_path):
            if file.endswith("Numenor-y-la-Tier-J-R-R-Tolkien.txt"):
                file_path = f"{self.raw_path}/{file}"
                file_content = read_text_files(file_path)
                paragraph = file_content.split('\n\n')
                if len(paragraph) > min_paragraph_len:
                    paragraph_list.append(paragraph)

        flat_paragraph_list = [item for sublist in paragraph_list for item in sublist]
        print('Número de párrafos totales {}'.format(len(flat_paragraph_list)))

        with open(self.paragraphs_files + '/tolkien_paragraphs_numenor.pkl', "wb") as f:
            pkl.dump(flat_paragraph_list, f)

creator = ParagraphsGenerator()
creator.create_paragraphs_files()