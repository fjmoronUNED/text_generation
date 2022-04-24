import nltk

from cleaner import Cleaner


document = "/Users/fjmoronreyes/text_generation/repositories/datasets/jrr_tolkien-la-ultima-cancion-de-bilbo.txt"
f = open(document, "r")
document = f.read()

a_list = nltk.tokenize.sent_tokenize(document)

cleaner_dataset = Cleaner()
print(cleaner_dataset.make_clean(a_list))
