import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from streamlit.uploaded_file_manager import UploadedFile

st.write("""
# Penguin Prediction App
This app predicts the *Palmer Penguin* species!
Data obtained from the [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.
""")

st.sidebar.header('User Input Features')

st.sidebar.markdown("""[Example CSV input file](https://raw.githubusercontent.com/Hongyanlee0614/Streamlit/main/Penguin%20Classification/penguins_example.csv)""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader(
    "Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
    input_len = len(input_df)
else:
    def user_input_features():
        island = st.sidebar.selectbox(
            'Island', ('Biscoe', 'Dream', 'Torgersen'))
        sex = st.sidebar.selectbox('Sex', ('male', 'female'))
        bill_length_mm = st.sidebar.slider(
            'Bill length (mm)', 32.1, 59.6, 43.9)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
        flipper_length_mm = st.sidebar.slider(
            'Flipper length (mm)', 172.0, 231.0, 201.0)
        body_mass_g = st.sidebar.slider(
            'Body mass (g)', 2700.0, 6300.0, 4207.0)
        data = {'island': island,
                'bill_length_mm': bill_length_mm,
                'bill_depth_mm': bill_depth_mm,
                'flipper_length_mm': flipper_length_mm,
                'body_mass_g': body_mass_g,
                'sex': sex}
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase -> make sure the input shape is correct, or else when we run pd.get_dummies, there will be no all three dummy variables but only one because we only use one input

# this file is solely for prediction purpose together with input data/uploaded file to ensure the pd.get_dummies work as expected
penguins_raw = pd.read_csv('penguins_cleaned.csv')
penguins = penguins_raw.drop(columns=['species'])
df = pd.concat([input_df, penguins], axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering

encode = ['sex', 'island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]


# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    df = df[0:input_len]
    st.write(df)
else:
    df = df[:1]
    st.write(
        'Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

penguins_species = np.array(['Adelie', 'Chinstrap', 'Gentoo'])
st.subheader('Reference for Index')
st.write(penguins_species)


st.subheader('Prediction Result')

prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)
st.write(penguins_species[prediction])
st.subheader('Prediction Probability Result')
st.write(prediction_proba)