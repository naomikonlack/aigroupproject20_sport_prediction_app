import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the trained XGBoost model and the scaler
loaded_model = joblib.load("xgb_model.joblib")
scaler = joblib.load("C:\\Users\\Lucie Konlack\\OneDrive - Ashesi University\\2ND YEAR SEMESTER 2 COURSES\\INTRODUCTION TO AI\\Group20_Project\\scaler.joblib")


st.title("Player Rating Predictor using XGBoost")

# Define input fields for the features
movement_reactions = st.number_input("Movement Reactions", value=50)
mentality_composure = st.number_input("Mentality Composure")
passing = st.number_input("Passing")
potential = st.number_input("Potential")
release_clause_eur = st.number_input("Release Clause (EUR)")
dribbling = st.number_input("Dribbling")
wage_eur = st.number_input("Wage (EUR)")
power_shot_power = st.number_input("Power Shot Power")
value_eur = st.number_input("Value (EUR)")
mentality_vision = st.number_input("Mentality Vision")
attacking_short_passing = st.number_input("Attacking Short Passing")

# Get all inputs in an array
features = [
    movement_reactions, 
    mentality_composure, 
    passing, 
    potential,
    release_clause_eur, 
    dribbling, 
    wage_eur, 
    power_shot_power, 
    value_eur, 
    mentality_vision, 
    attacking_short_passing
]

# Button to make predictions
if st.button("Predict"):
    # Scale the features
    scaled_features = scaler.transform([features])
    
    # Make predictions
    prediction = loaded_model.predict(scaled_features)
    st.write(f"Predicted Rating: {prediction[0]}")

import streamlit as st
import joblib
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the trained XGBoost model and the scaler
loaded_model = joblib.load("xgb_model.joblib")
scaler = joblib.load("C:\\Users\\Lucie Konlack\\OneDrive - Ashesi University\\2ND YEAR SEMESTER 2 COURSES\\INTRODUCTION TO AI\\Group20_Project\\scaler.joblib")


st.title("Player Rating Predictor using XGBoost")

# Define input fields for the features
movement_reactions = st.number_input("Movement Reactions", value=50)
mentality_composure = st.number_input("Mentality Composure")
passing = st.number_input("Passing")
potential = st.number_input("Potential")
release_clause_eur = st.number_input("Release Clause (EUR)")
dribbling = st.number_input("Dribbling")
wage_eur = st.number_input("Wage (EUR)")
power_shot_power = st.number_input("Power Shot Power")
value_eur = st.number_input("Value (EUR)")
mentality_vision = st.number_input("Mentality Vision")
attacking_short_passing = st.number_input("Attacking Short Passing")

# Get all inputs in an array
features = [
    movement_reactions, 
    mentality_composure, 
    passing, 
    potential,
    release_clause_eur, 
    dribbling, 
    wage_eur, 
    power_shot_power, 
    value_eur, 
    mentality_vision, 
    attacking_short_passing
]

# Button to make predictions
if st.button("Predict"):
    # Scale the features
    scaled_features = scaler.transform([features])
    
    # Make predictions
    prediction = loaded_model.predict(scaled_features)
    st.write(f"Predicted Rating: {prediction[0]}")

