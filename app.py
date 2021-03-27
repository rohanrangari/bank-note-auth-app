"""Module for hosting the app using fastapi and defining routes"""
# Standard libraries
import uvicorn
from fastapi import FastAPI

# User defined libraries
from App.src.predict import load_model, process_result
from App.BankNotes import BankNote

# Constants / Globals
app = FastAPI()


@app.get("/")
def index():
    """
    Home Route for the BanK Bank Note Authentication App.
    - args:
        - input: None
        - output: Welcome Msg
    """
    return "Welcome To Bank Note Authentication App"


@app.post("/predict")
def predict_bank_note(data: BankNote):
    """
    API Route for predicting the Bank Note Authentication
    - args:
        - input : Variance, skewness, curtosis, Entropy
        - output : Bank Note is Authenticated or Not
    """
    clf = load_model()
    data = data.dict()
    print(data)
    variance = data.get("variance")
    skewness = data.get("skewness")
    curtosis = data.get("curtosis")
    entropy = data.get("entropy")
    prediction = clf.predict([[variance, skewness, curtosis, entropy]])
    print(prediction)
    result = process_result(prediction)
    print(result)
    return {"Prediction": result}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)