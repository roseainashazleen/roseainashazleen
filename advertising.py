import streamlit as st
import pandas as pd
import seaborn as sns

st.write("# Sales Prediction based on TV, Radio and Newspaper")
st.write("This app predicts the **Sales** !")

st.sidebar.header('User Input Parameters')

def user_input_features():
    Tv_Sales = st.sidebar.slider('Tv Sales', 0.7, 296.4, 5.4)
    Radio_Sales = st.sidebar.slider('Radio Sales', 0.0, 49.6, 3.4)
    Newspaper_Sales = st.sidebar.slider('Newspaper Sales', 0.3, 114.0, 1.3)
    data = {'Tv': Tv_Sales,
            'Radio': Radio_Sales,
            'Newspaper': Newspaper_Sales}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

loaded_model = pickle.load(open("AdvertisingLinearRegression.h5", "rb"))

SalesPrediction=loaded_model.predict(df)
st.subheader('User Input parameters')
st.write(SalesPrediction)



