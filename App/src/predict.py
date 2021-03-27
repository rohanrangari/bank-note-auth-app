"""Module for defining the functions related to prediction"""
# Standard libraries
import pickle
from pathlib import Path

# User defined libraries
from App.src.constants import CLF_PATH

# Constants/ Globals
CLF_PATH = Path(CLF_PATH)


def load_model():
    """
    loads the saved model
    - args:
        - input: None
        - output: Classifer
    """
    pickle_in = open(CLF_PATH, "rb")
    print(f"[INFO]: Model-Name:{CLF_PATH.stem}")
    clf = pickle.load(pickle_in)
    print(f"[INFO]: {CLF_PATH.stem} Loaded")
    return clf


def process_result(prediction):
    """
    Process the prediction from the model
    -args:
        - input: raw prediction/result
        - output: Processed prediction/result
    """
    print(prediction, type(prediction))
    if prediction[0] == 0:
        print("[INFO]: Bank Note is not Authenticated")
        return "Bank Note is not Authenticated"
    if prediction[0] == 1:
        print("[INFO]: Bank Note is Authenticated")
        return "Bank Note is Authenticated"


def main():
    load_model()


if __name__ == "__main__":
    main()
