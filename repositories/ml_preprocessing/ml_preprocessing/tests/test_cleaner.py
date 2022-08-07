from ml_preprocessing.cleaner import Cleaner
from ml_preprocessing import DATASETS


def test_ml_preprocessing_cleaner():
    cleaner = Cleaner()
    with open(DATASETS + '/El-Hobbit-J-R-R-Tolkien.txt', 'r') as f:
        content = f.read()

    file_cleaned = cleaner.make_clean(content, stopwords=False)
    return file_cleaned

print(test_ml_preprocessing_cleaner())