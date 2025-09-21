import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load(r"C:\DS Projects\House price prediction\model.pkl")


st.title("House Price Prediction App")

st.divider()

st.write("This app uses machine learning to predict house price based on given features. Enter the inputs below and click Predict.")

st.divider()

# Input fields
bedrooms = st.number_input("Number of bedrooms", min_value=0, value=0)
bathrooms = st.number_input("Number of bathrooms", min_value=0, value=0)
livingarea = st.number_input("Living area (sq ft)", min_value=200, value=2000)
condition = st.number_input("Condition (1-5)", min_value=0, value=3)
numberofschools = st.number_input("Number of schools nearby", min_value=0, value=0)

st.divider()

# Features for prediction
X = [[bedrooms, bathrooms, livingarea, condition, numberofschools]]

# Prediction button
predictbutton = st.button("Predict")

if predictbutton:
    st.balloons()

    X_array = np.array(X).reshape(1, -1)
    prediction = model.predict(X_array)

    st.success(f"🏠 Estimated House Price: ${prediction[0]:,.2f}")
else:
    st.info("Please enter values and click Predict")
