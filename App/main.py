from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
from src.predict import load_model, process_result

app = Flask(__name__)


@app.route("/home")
@app.route("/")
def index():
    return "<h1>Welcome To Bank Note Authentication App</h1>"


@app.route("/predict", methods=["POST"])
def predict_note():
    clf = load_model()
    variance = request.args.get("variance")
    skewness = request.args.get("skewness")
    curtosis = request.args.get("curtosis")
    entropy = request.args.get("entropy")
    prediction = clf.predict([[variance, skewness, curtosis, entropy]])
    result = process_result(prediction)
    print(result)
    return "<h1>The prediction is:{}</h1>".format(result)


if __name__ == "__main__":
    app.run(debug=True)