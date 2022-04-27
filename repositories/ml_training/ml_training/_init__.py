import os
import logging
import pickle as pkl

logger = logging.getLogger("ml_training")

FILES_PATH = "../../../datasets"
FILES_PATH_RAW = "../../../datasets/raw_files"
FILES_PATH_SENTENCES = "../../../datasets/sentences_files"
FILES_PATH_SEQUENCES = "../../../datasets/sequences_files"


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
