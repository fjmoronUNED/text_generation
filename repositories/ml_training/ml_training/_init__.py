import os
import logging
import pickle as pkl

logger = logging.getLogger("ml_training")

FILES_PATH = "/Users/fjmoronreyes/text_generation/repositories/datasets/"
FILES_PATH_RAW = "/Users/fjmoronreyes/text_generation/repositories/datasets/raw_files"
FILES_PATH_SENTENCES = (
    "/Users/fjmoronreyes/text_generation/repositories/datasets/sentences_files"
)
FILES_PATH_SEQUENCES = (
    "/Users/fjmoronreyes/text_generation/repositories/datasets/sequences_files"
)


def get_rolling_window_sequence(files_path_sequences):
    os.chdir(files_path_sequences)

    complete_dataset = []
    for file in os.listdir():
        if file.endswith(".pkl"):
            with open(file, "rb") as f:
                sentences = pkl.load(f)
                for sentence in sentences:
                    complete_dataset.append(sentence)
    return complete_dataset
