from ml_preprocessing.hf_dataset_generator import HfDatasetGenerator

datasets = HfDatasetGenerator()
datasets.create_raw_text()
datasets.train_test_split_dataset()