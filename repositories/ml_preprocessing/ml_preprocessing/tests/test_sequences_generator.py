from ml_preprocessing.sequences_generator import SequencesGenerator

sequences = SequencesGenerator(stopwords=False)
sequences.create_sentences_files()