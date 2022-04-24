from extractor import TextExtractor
import PyPDF2

path = "/Users/fjmoronreyes/text_generation/repositories/datasets/jrr_tolkien-la-ultima-cancion-de-bilbo.pdf"
document = TextExtractor(path)
print(document.title)
