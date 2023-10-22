import streamlit as st
import joblib
import pandas as pd
import os

# Define the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Define the paths to the model and scaler
model_path = os.path.join(current_directory, "xgb_model.joblib")
scaler_path = os.path.join(current_directory, "scaler.joblib")

# Load the trained XGBoost model and the scaler
loaded_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Define the Streamlit app title
st.title("Player Rating Predictor using XGBoost")

# Create input fields for user input
st.header("Enter Player Attributes")

movement_reactions = st.slider("Movement Reactions", min_value=0, max_value=100, value=50)
mentality_composure = st.slider("Mentality Composure", min_value=0, max_value=100)
passing = st.slider("Passing", min_value=0, max_value=100)
potential = st.slider("Potential", min_value=0, max_value=100)
release_clause_eur = st.slider("Release Clause (EUR)", min_value=0, max_value=100000000, step=1000000)
dribbling = st.slider("Dribbling", min_value=0, max_value=100)
wage_eur = st.slider("Wage (EUR)", min_value=0, max_value=1000000, step=10000)
power_shot_power = st.slider("Power Shot Power", min_value=0, max_value=100)
value_eur = st.slider("Value (EUR)", min_value=0, max_value=100000000, step=1000000)
mentality_vision = st.slider("Mentality Vision", min_value=0, max_value=100)
attacking_short_passing = st.slider("Attacking Short Passing", min_value=0, max_value=100)

# Store user inputs in a dictionary
user_inputs = {
    "movement_reactions": movement_reactions,
    "mentality_composure": mentality_composure,
    "passing": passing,
    "potential": potential,
    "release_clause_eur": release_clause_eur,
    "dribbling": dribbling,
    "wage_eur": wage_eur,
    "power_shot_power": power_shot_power,
    "value_eur": value_eur,
    "mentality_vision": mentality_vision,
    "attacking_short_passing": attacking_short_passing
}

# Create a button to make predictions
if st.button("Predict"):
    # Convert user inputs to a DataFrame
    input_data = pd.DataFrame(user_inputs, index=[0])
    
    # Scale the input features using the loaded scaler
    scaled_features = scaler.transform(input_data)
    
    # Make predictions using the loaded model
    prediction = loaded_model.predict(scaled_features)
    
    # Display the predicted rating
    st.subheader("Predicted Rating:")
    st.write(prediction[0])

# Add a section for displaying sample data or instructions
st.sidebar.header("Instructions")
st.sidebar.markdown("Adjust the sliders to enter player attributes, then click the 'Predict' button to see the predicted rating.")

# You can also add any additional content or explanations here
