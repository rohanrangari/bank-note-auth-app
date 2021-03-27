"""Module for Training the model"""
# Standard libraries
import pickle
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# User defines Modules
from src.constants import MODEL_SAVE_PATH
from src.constants import DATASET_FOLDER, DATASET_FILENAME

# Constants / Globals
DATASET_PATH = Path(DATASET_FOLDER) / DATASET_FILENAME


def load_dataset():
    """
    To load input dataset
    args:
        input : None / global Variable DATASET_PATH
        output : dataframe
    """
    print("[INFO]: loading dataset")
    df = pd.read_csv(DATASET_PATH)
    print(df.head())
    print("[INFO]: loaded dataset")
    return df, list(df.columns), df.shape


def check_balance(df):
    """
    To check if the dataset is balanced or not.
    - args:
        - input : dataframe/ dataset
        - output : Distribution of Target Class
    """
    print("[INFO]: Checking balance if the dataset is balanced or not.")
    print(df["class"].value_counts())


def create_x_y(df):
    """
    To Create Input-features/X  and Output Features/y
    - args:
        - input: dataframe/ dataset
        - output: X, y
    """
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y


def create_train_test_sets(X, y):
    """
    For creating train and test sets
    - args:
        - input: X, y
        - output: Training set and test set
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=0,
    )
    return X_train, X_test, y_train, y_test


def model(X_train, X_test, y_train):
    """
    Defining the model and Training on it
    - args:
        - input: Training set and test set
        - output: trained model
    """
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    model_path = save_model(clf)
    return y_pred, model_path


def classification_metrics(y_pred, y_test):
    """
    Performance Metrics which can be used for classification problems
    - args:
        - input: y_pred, y_test
        - output: Metrics result
    """
    acc_score = accuracy_score(y_pred, y_test)
    return acc_score


def save_model(clf):
    """
    Save the trained model
    """
    model_path = Path(MODEL_SAVE_PATH) / "clf.pkl"
    pickle_out = open(model_path, "wb")
    pickle.dump(clf, pickle_out)
    pickle_out.close()
    print(f"[INFO]: Model Saved at {model_path}")
    return model_path


def main():
    """ Main """
    df, cols_name, df_shape = load_dataset()
    rows, cols = df_shape
    print(f"[INFO]: Column Names: {cols_name}")
    print(f"[INFO]: No of rows, columns in the dataset: {rows}, {cols}")
    check_balance(df)
    X, y = create_x_y(df)
    X_train, X_test, y_train, y_test = create_train_test_sets(X, y)
    y_pred, model_path = model(X_train, X_test, y_train)
    acc_score = classification_metrics(y_pred, y_test)
    print(f"[INFO]: Accuracy of Model is :{acc_score}")


if __name__ == "__main__":
    main()
