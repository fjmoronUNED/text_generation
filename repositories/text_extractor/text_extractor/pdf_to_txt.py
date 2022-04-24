from extractor import TextExtractor

path = "/Users/fjmoronreyes/sp3ctrumV/backend/libraries/text_extractor/text_extractor/example.pdf"
# path = "./example.pdf"

# class TxtGenerator:
#    def __init__(self):

document = TextExtractor(path)
document.run_extraction()

# with open("./roberto.txt", "w") as text_file:
#    text_file.write(document.get_text())
