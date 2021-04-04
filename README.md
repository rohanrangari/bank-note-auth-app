# bank-note-auth-app

Bank Note Authentication

## Data Source

https://www.kaggle.com/ritesaluja/bank-note-authentication-uci-data

## Descrption

Data were extracted from images that were taken from genuine and forged banknote-like specimens. For digitization, an industrial camera usually used for print inspection was used. The final images have 400x 400 pixels. Due to the object lens and distance to the investigated object gray-scale pictures with a resolution of about 660 dpi were gained. Wavelet Transform tool were used to extract features from images.

## Project Structure

- Src : Contains all the codes
- Models : Contains all the trained & stored Models
- Data : Contains all the data for training and testing
- app.py : Host routes & runs the app
- .gitingore: Contains all the extensions to be ignored by git
- Readme.md : Contains all the describtion for the app
- requirements.txt : Contains all the dependencies for the app
  ![alt text](Screenshots/Project-Structure.PNG "App Structure")

## Running the Application Via FastAPI

**unicorn app:app --reload**
http://127.0.0.1:8000/docs

![alt text](Screenshots/Docs-Apis.PNG "docs")
![alt text](Screenshots/predict-route.PNG "Features")

## Running the Application Via Streamlit

**streamlit run bank-ml.py**
![alt text](Screenshots/streamlit-app.PNG "Streamlit")

# Bank Note Classifier web app deployed on Heroku

The deployed web app is live at https://bank-notes-app.herokuapp.com/

This web app predicts the bank note as fake or not as a function of their input parameters (variance, skewness, kurtosis, entropy).

The web app was built in Python using the following libraries:

- streamlit
- pandas
- numpy
- scikit-learn
- pickle

![alt text](Screenshots/heroku-bank-ml.PNG "Heroku-App")
