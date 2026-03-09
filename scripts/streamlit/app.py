import streamlit as st
import pandas as pd
import pickle

# Load model and scaler
model = pickle.load(open("../models/rf_model.pkl", "rb"))
scaler = pickle.load(open("../models/scaler.pkl", "rb"))

st.title("Customer Churn Prediction")

# Input fields for all features
customerID = st.text_input("Customer ID")
gender = st.selectbox("Gender", [0, 1])  # 0: Female, 1: Male
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", [0, 1])
Dependents = st.selectbox("Dependents", [0, 1])
tenure = st.number_input("Tenure", 0, 100)
PhoneService = st.selectbox("Phone Service", [0, 1])
MultipleLines = st.selectbox("Multiple Lines", [0, 1, 2])
InternetService = st.selectbox("Internet Service", [0, 1, 2])
OnlineSecurity = st.selectbox("Online Security", [0, 1, 2])
OnlineBackup = st.selectbox("Online Backup", [0, 1, 2])
DeviceProtection = st.selectbox("Device Protection", [0, 1, 2])
TechSupport = st.selectbox("Tech Support", [0, 1, 2])
StreamingTV = st.selectbox("Streaming TV", [0, 1, 2])
StreamingMovies = st.selectbox("Streaming Movies", [0, 1, 2])
Contract = st.selectbox("Contract", [0, 1, 2])
PaperlessBilling = st.selectbox("Paperless Billing", [0, 1])
PaymentMethod = st.selectbox("Payment Method", [0, 1, 2, 3])
MonthlyCharges = st.number_input("Monthly Charges", 0.0, 1000.0)
TotalCharges = st.number_input("Total Charges", 0.0, 10000.0)

if st.button("Predict"):
    # Create DataFrame from inputs
    input_data = pd.DataFrame([[
        customerID, gender, SeniorCitizen, Partner, Dependents, tenure,
        PhoneService, MultipleLines, InternetService, OnlineSecurity,
        OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
        StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
        MonthlyCharges, TotalCharges
    ]], columns=[
        "customerID","gender","SeniorCitizen","Partner","Dependents","tenure",
        "PhoneService","MultipleLines","InternetService","OnlineSecurity",
        "OnlineBackup","DeviceProtection","TechSupport","StreamingTV",
        "StreamingMovies","Contract","PaperlessBilling","PaymentMethod",
        "MonthlyCharges","TotalCharges"
    ])
    
    # Drop customerID if not used in model
    input_data = input_data.drop(columns=["customerID"])
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)
    
    if prediction[0] == 1:
        st.error("Customer is likely to churn")
    else:
        st.success("Customer is not likely to churn")
