import streamlit as st
import pickle
import numpy as np
import pandas as pd
import subprocess
import sys

# Function to install a package
def install_package(package):
    try:
        __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install Streamlit if not already installed
install_package("streamlit")

# Load the saved objects
try:
    with open('standard_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('polynomial_features.pkl', 'rb') as f:
        poly = pickle.load(f)
    with open('polynomial_regression_model.pkl', 'rb') as f:
        model_poly = pickle.load(f)
except FileNotFoundError:
    st.error("Error: One or more pickle files not found. Make sure 'standard_scaler.pkl', 'polynomial_features.pkl', and 'polynomial_regression_model.pkl' are in the same directory as this script.")
    st.stop() # Stop the app if files are not found

st.set_page_config(page_title="Food Delivery Time Predictor", layout="centered")

st.title("🍔 Food Delivery Time Prediction")
st.write("Enter the details below to predict the delivery time.")

# Input fields for the features, matching the order used in training:
# ['Distance_km', 'Preparation_Time_min', 'Weather_Snowy', 'Traffic_Level', 'Courier_Experience_yrs', 'Weather_Rainy']

distance_km = st.slider("Distance (km)", min_value=0.1, max_value=20.0, value=10.0, step=0.1)
preparation_time_min = st.slider("Preparation Time (min)", min_value=5, max_value=30, value=15, step=1)
courier_experience_yrs = st.slider("Courier Experience (years)", min_value=0.0, max_value=9.0, value=2.0, step=0.1)

weather_snowy = st.checkbox("Is it Snowy Weather?")
weather_rainy = st.checkbox("Is it Rainy Weather?")

# Traffic Level mapping based on observed values from `df.head()` and `df_dummies.head()` in the notebook
# Assuming: Heavy -> 0, Low -> 1, Medium -> 2, High -> 3
traffic_level_options = {'Heavy': 0, 'Low': 1, 'Medium': 2, 'High': 3}
traffic_level_input_str = st.selectbox("Traffic Level", list(traffic_level_options.keys()), index=1) # Default to 'Low'
encoded_traffic_level = traffic_level_options[traffic_level_input_str]

# Prepare input data for prediction
if st.button("Predict Delivery Time"):
    # Create a DataFrame for the input features in the exact order the scaler was fitted on
    input_features_for_scaler = pd.DataFrame([[
        distance_km,
        preparation_time_min,
        1 if weather_snowy else 0, # Weather_Snowy (binary)
        encoded_traffic_level,      # Traffic_Level (label encoded integer)
        courier_experience_yrs,
        1 if weather_rainy else 0   # Weather_Rainy (binary)
    ]], columns=['Distance_km', 'Preparation_Time_min', 'Weather_Snowy', 'Traffic_Level', 'Courier_Experience_yrs', 'Weather_Rainy'])

    # Scale the input features
    scaled_input = scaler.transform(input_features_for_scaler)

    # Apply polynomial features
    poly_input = poly.transform(scaled_input)

    # Make prediction
    prediction = model_poly.predict(poly_input)[0]

    st.success(f"**Predicted Delivery Time: {prediction:.2f} minutes**")

    