# app.py

import streamlit as st
import pickle
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder

# Load the trained model
with open('decision_tree_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Fetch the bank marketing dataset again for reference
bank_marketing = fetch_ucirepo(id=222)
X_sample = bank_marketing.data.features.copy()

# Label encode features the same way as during training
label_encoders = {}
for column in X_sample.columns:
    if X_sample[column].dtype == 'object':
        le = LabelEncoder()
        X_sample[column] = le.fit_transform(X_sample[column])
        label_encoders[column] = le

st.title(' Bank Term Deposit Prediction App')

st.markdown('### Enter Customer Details:')

# Create input fields dynamically
user_input = {}
for column in X_sample.columns:
    if column in label_encoders:
        options = list(label_encoders[column].classes_)
        user_input[column] = st.selectbox(f"{column.capitalize()}:", options)
    else:
        user_input[column] = st.number_input(f"{column.capitalize()}:", value=0)

# Predict button
if st.button('Predict'):
    # Prepare input data
    input_data = []

    for column in X_sample.columns:
        if column in label_encoders:
            value = label_encoders[column].transform([user_input[column]])[0]
        else:
            value = user_input[column]
        input_data.append(value)

    input_array = np.array(input_data).reshape(1, -1)

    # Predict
    prediction = model.predict(input_array)

    if prediction[0] == 1:
        st.success(' The customer is likely to SUBSCRIBE to a term deposit!')
    else:
        st.error(' The customer is NOT likely to subscribe.')

