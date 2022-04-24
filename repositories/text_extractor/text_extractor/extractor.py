import PyPDF2
import textract
from PyPDF2 import PdfFileReader


class TextExtractor:
    def __init__(self, document):

        self.document = document

        file_obj = open(document, "rb")
        self.pdf = PyPDF2.PdfFileReader(file_obj)
        self.info = self.pdf.getDocumentInfo()
        self.num_pages = self.pdf.getNumPages()

        self.creator = self.info.creator
        self.author = self.info.author
        self.producer = self.info.producer
        self.subject = self.info.subject
        self.title = self.info.title

    def get_text(self, count=0):
        text = ""
        while count < self.num_pages:
            page = self.pdf.getPage(count)
            count += 1
            text += page.extractText()

        return text

    def get_text_from_img(self):
        return "txt_from_img"

    def get_formula_from_paper(self):
        return "form_from_pp"

    def run_extraction(self):
        return True
