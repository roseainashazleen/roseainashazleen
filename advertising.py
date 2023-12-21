import streamlit as st
import pandas as pd
import seaborn as sns
import pickle

st.write("# Sales Prediction based on TV, Radio and Newspaper")
st.write("This app predicts the **Sales** !")

st.sidebar.header('User Input Parameters')

def user_input_features():
    Tv = st.sidebar.slider('Tv', 0.7, 296.4, 5.4)
    Radio = st.sidebar.slider('Radio', 0.0, 49.6, 3.4)
    Newspaper = st.sidebar.slider('Newspaper', 0.3, 114.0, 1.3)
    data = {'Tv': Tv,
            'Radio': Radio,
            'Newspaper': Newspaper}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

loaded_model = pickle.load(open("AdvertisingLinearRegression.h5", "rb"))

SalesPrediction=loaded_model.predict(df)
st.subheader('User Input parameters')
st.write(SalesPrediction)



