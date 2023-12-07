import streamlit as st
import pandas as pd
import joblib

# Load the pre-trained model
model = joblib.load('wine_quality_model.joblib')
import pandas as pd
data=pd.read_csv('winequality-red.csv')
def predict_wine_quality(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol):
    input_data = [[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol]]
    prediction = model.predict(input_data)
    return prediction[0]
st.title('Wine Quality Prediction App')
fixed_acidity = st.slider('Fixed Acidity', float(4.0), float(16.0), float(9.0))
volatile_acidity = st.slider('Volatile Acidity', float(0.0), float(2.0), float(0.5))
citric_acid = st.slider('Citric Acid', float(0.0), float(2.0), float(0.5))
residual_sugar = st.slider('Residual Sugar', float(0.0), float(30.0), float(15.0))
chlorides = st.slider('Chlorides', float(0.0), float(1.0), float(0.5))
free_sulfur_dioxide = st.slider('Free Sulfur Dioxide', float(0.0), float(100.0), float(50.0))
total_sulfur_dioxide = st.slider('Total Sulfur Dioxide', float(0.0), float(300.0), float(150.0))
density = st.slider('Density', float(0.990), float(1.005), float(0.995))
pH = st.slider('pH', float(2.0), float(4.0), float(3.0))
sulphates = st.slider('Sulphates', float(0.0), float(2.0), float(1.0))
alcohol = st.slider('Alcohol', float(8.0), float(16.0), float(12.0))

# Use text input for features with non-numeric values

# Create a DataFrame with user input
user_data = pd.DataFrame({
    'fixed acidity': [fixed_acidity ],
    'volatile acidity': [volatile_acidity],
    'citric acid': [citric_acid],
    'residual sugar': [residual_sugar],
    'chlorides': [chlorides],
    'free sulfur dioxide': [free_sulfur_dioxide],
    'total sulfur dioxide': [total_sulfur_dioxide],
    'density': [density],
    'pH': [pH],
    'sulphates': [sulphates],
    'alcohol': [alcohol]
})

# Make a prediction based on user input
prediction = model.predict(user_data)

# Display the predicted wine quality
st.subheader('Predicted Wine Quality:')
st.write(prediction[0])