import os

FILE_PATH = os.path.dirname(__file__)
testing_dir = os.environ.get(
    "TESTING", f"{FILE_PATH}/../inference"
)
print(testing_dir)