# Customer Churn Prediction

## Overview
The Customer Churn Prediction System is a machine learning project that predicts whether a customer is likely to leave a service (churn) based on historical data.
Goals:
-	Identify potential churners before they leave.
-	Allow businesses to take proactive measures (offers, engagement campaigns).
-	Provide an interactive Streamlit frontend for user input.
-	Provide a FastAPI backend for serving predictions via API.

## Dataset
- Telco Customer Churn (IBM / Kaggle)

## Features
- Tenure, Services, Billing info, Demographics

## Models
- Logistic Regression, Random Forest, XGBoost

## Scripts
ProjectNotebook contains all the Data Preprocessing, training, model developement, and  saving the models scripts.
## Result Exectuion
To execute the code, go to Scripts folder , choose either deployment method Streamlit or FastAPI and run the pythonscripts.

## Installation
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
pip install -r requirements.txt
