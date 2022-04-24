import re
import os
import numpy as np


class SequencesGenerator:
    def __init__(self, document):
        self.document = document

        if os.path.isfile(document):
            print("CARGANDO UN ÚNICO DOCUMENTO")
        if os.path.isdir(document):
            print("CARGANDO GRUPO DE DOCUMENTOS")

    def get_sentences(self):
        """
        Function to get sentences from the document
        """
        return True
