import streamlit as st
import joblib
import numpy as np
import sys
import os
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

label_mappings = {}
with open("ml/saved_model/label_mappings.json", "r") as f:
    label_mappings = json.load(f)

label_mapping = label_mappings["label"]
output = {v: k for k, v in label_mapping.items()}


model = joblib.load("ml/saved_model/NaiveBayes.pkl")
st.title("Crop Recommendation System")
N_input = st.text_input("Nitrogen (N)")
P_input = st.text_input("Phosphorous (P)")
K_input = st.text_input("Potassium (K)")
temperature_input = st.text_input("Temperature (°C)")
humidity_input = st.text_input("Humidity (%)")
ph_input = st.text_input("pH Level")
rainfall_input = st.text_input("Rainfall (mm)")
if st.button("Predict Crop"):
    if (
        N_input
        and P_input
        and K_input
        and temperature_input
        and humidity_input
        and ph_input
        and rainfall_input
    ):
        try:
            N = int(N_input)
            P = int(P_input)
            K = int(K_input)
            temperature = int(temperature_input)
            humidity = int(humidity_input)
            ph = int(ph_input)
            rainfall = int(rainfall_input)

            features = np.array([N, P, K, temperature, humidity, ph, rainfall]).reshape(
                1, -1
            )
            prediction = model.predict(features)
            st.success(f"Recommended Crop: {output[prediction[0]]}")
        except ValueError:
            st.error("Please enter valid numeric values for all fields.")
    else:
        st.warning("Please fill in all the fields.")
