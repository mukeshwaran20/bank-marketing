# app.py

import streamlit as st
import pickle
import numpy as np

# Load the trained model
with open('decision_tree_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Define input features manually
feature_names = [
    'age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan',
    'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome'
]

# Some features are categorical and need selection
categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

# Categorical options (you need to match these manually from training data)
job_options = ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired',
               'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown']

marital_options = ['divorced', 'married', 'single', 'unknown']
education_options = ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown']
default_options = ['no', 'yes', 'unknown']
housing_options = ['no', 'yes', 'unknown']
loan_options = ['no', 'yes', 'unknown']
contact_options = ['cellular', 'telephone']
month_options = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
poutcome_options = ['failure', 'nonexistent', 'success']

options_map = {
    'job': job_options,
    'marital': marital_options,
    'education': education_options,
    'default': default_options,
    'housing': housing_options,
    'loan': loan_options,
    'contact': contact_options,
    'month': month_options,
    'poutcome': poutcome_options
}

st.title(' Bank Term Deposit Prediction App')

st.markdown('### Enter Customer Details:')

# Create user input fields
user_input = {}

for feature in feature_names:
    if feature in categorical_features:
        user_input[feature] = st.selectbox(f"{feature.capitalize()}:", options_map[feature])
    else:
        user_input[feature] = st.number_input(f"{feature.capitalize()}:", value=0)

# Predict button
if st.button('Predict'):
    # Manual encoding (convert categories to simple numbers)
    input_data = []

    for feature in feature_names:
        if feature in categorical_features:
            input_data.append(options_map[feature].index(user_input[feature]))
        else:
            input_data.append(user_input[feature])

    input_array = np.array(input_data).reshape(1, -1)

    # Prediction
    prediction = model.predict(input_array)

    if prediction[0] == 1:
        st.success(' The customer is likely to SUBSCRIBE to a term deposit!')
    else:
        st.error(' The customer is NOT likely to subscribe.')
