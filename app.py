import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_breast_cancer

# Load breast cancer dataset
cancer = load_breast_cancer()

# Load the trained model
with open('svm_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Main Streamlit app
st.write('# Breast Cancer Classification App')

# Sidebar for user input
st.sidebar.write('## User Input Features')

# Function to get user inputs
def get_user_input():
    input_features = {}
    for feature in cancer['feature_names']:
        # Assuming numeric input features
        input_features[feature] = st.sidebar.slider(f'Enter {feature}', float(df_cancer[feature].min()), float(df_cancer[feature].max()), float(df_cancer[feature].mean()))
    return pd.DataFrame([input_features], columns=cancer['feature_names'])

# Create a DataFrame for demonstration (replace it with your actual data)
df_cancer = pd.DataFrame(np.c_[cancer['data'], cancer['target']], columns=np.append(cancer['feature_names'], ['target']))

# Get user input
user_input_df = get_user_input()

# Display user input
st.subheader('User Input:')
st.write(user_input_df)

# Add a submit button
if st.sidebar.button('Submit'):
    # Preprocess user input
    scaler = MinMaxScaler()
    user_input_scaled = scaler.fit_transform(user_input_df)

    # Make predictions with feature names
    prediction = model.predict(user_input_scaled)

    # Display prediction
    st.subheader('Prediction:')
    if prediction[0] == 0:
        st.write('The model predicts: Benign')
    else:
        st.write('The model predicts: Malignant')

