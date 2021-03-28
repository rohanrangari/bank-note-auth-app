import pandas as pd
import streamlit as st
from App.src.predict import load_model, process_result
from App.BankNotes import BankNote

st.write(
    """
# BankNote Authenticator App
This app predicts the ** Bank Note Fake or Not** type
"""
)

st.sidebar.header("User Input Parameters")


def user_input_features():
    variance = st.sidebar.slider("variance", 0.0, 15.0, 5.4)
    skewness = st.sidebar.slider("skewness", 0.0, 15.0, 5.4)
    kurtosis = st.sidebar.slider("kurtosis", 0.0, 15.0, 5.4)
    entropy = st.sidebar.slider("entropy", 0.0, 15.0, 5.4)
    data = {
        "variance": variance,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "entropy": entropy,
    }
    features = pd.DataFrame(data, index=[0])
    return features


df = user_input_features()

st.header("User Input Parameters")
st.write(df)


clf = load_model()
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

print(prediction)
result = process_result(prediction)

st.subheader("Prediction")
st.write(result)

st.subheader("Prediction Probability")
st.write(prediction_proba)
