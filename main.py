import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the model and scaler (assuming both burnout and anxiety use the same model)
with open('logistic_regression_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


st.title("Burnout/Anxiety Prediction Web App")

# Input fields for the features
age = st.number_input("Age", min_value=18, max_value=100, value=30)
work_location = st.selectbox("Work Location", ["Remote", "Hybrid", "Onsite"])
hours_worked_per_week = st.number_input("Hours Worked Per Week", min_value=10, max_value=80, value=40)
work_life_balance_rating = st.slider("Work-Life Balance Rating (1-5)", min_value=1, max_value=5, value=3)
virtual_meetings = st.number_input("Number of Virtual Meetings per Week", min_value=0, max_value=50, value=5)
productivity_change = st.selectbox("Productivity Change", ["Increase", "Decrease", "No Change"])
social_isolation_rating = st.slider("Social Isolation Rating (1-5)", min_value=1, max_value=5, value=3)
satisfaction_with_remote_work = st.selectbox("Satisfaction with Remote Work", ["Satisfied", "Unsatisfied"])
company_support = st.slider("Company Support for Remote Work (1-5)", min_value=1, max_value=5, value=3)
physical_activity = st.selectbox("Physical Activity", ["Weekly", "None"])
sleep_quality = st.selectbox("Sleep Quality", ["Good", "Average", "Poor"])
region = st.selectbox("Region", ["North America", "Europe", "Asia", "South America"])

# Encode categorical variables the same way they were encoded during training
work_location_encoded = {"Remote": 0, "Hybrid": 1, "Onsite": 2}[work_location]
productivity_change_encoded = {"Increase": 0, "Decrease": 1, "No Change": 2}[productivity_change]
satisfaction_with_remote_work_encoded = {"Satisfied": 0, "Unsatisfied": 1}[satisfaction_with_remote_work]
physical_activity_encoded = {"Weekly": 0, "None": 1}[physical_activity]
sleep_quality_encoded = {"Good": 0, "Average": 1, "Poor": 2}[sleep_quality]
region_encoded = {"North America": 0, "Europe": 1, "Asia": 2, "South America": 3}[region]

# Define feature names (these must match the features used when the scaler was fitted)
feature_names = ['Age', 'Work_Location', 'Hours_Worked_Per_Week', 'Work_Life_Balance_Rating',
                 'Number_of_Virtual_Meetings', 'Productivity_Change', 'Social_Isolation_Rating',
                 'Satisfaction_with_Remote_Work', 'Company_Support_for_Remote_Work', 'Physical_Activity',
                 'Sleep_Quality', 'Region']

# Convert inputs to a DataFrame with the correct feature names
input_data = pd.DataFrame([[age, work_location_encoded, hours_worked_per_week, work_life_balance_rating, 
                            virtual_meetings, productivity_change_encoded, social_isolation_rating, 
                            satisfaction_with_remote_work_encoded, company_support, physical_activity_encoded, 
                            sleep_quality_encoded, region_encoded]], 
                           columns=feature_names)

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Two buttons: one for Burnout prediction and one for Anxiety prediction
if st.button("Predict Burnout"):
    burnout_prediction = model.predict(input_data_scaled)  
    if burnout_prediction == 1:
        st.write("The model predicts that the individual is at risk of Burnout.")
    else:
        st.write("The model predicts that the individual is not at risk of Burnout.")

if st.button("Predict Anxiety"):
    anxiety_prediction = model.predict(input_data_scaled) 
    if anxiety_prediction == 1:
        st.write("The model predicts that the individual is at risk of Anxiety.")
    else:
        st.write("The model predicts that the individual is not at risk of Anxiety.")
