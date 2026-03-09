from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

model = pickle.load(open("../../models/rf_model.pkl","rb"))
scaler = pickle.load(open("../../models/scaler.pkl","rb"))

@app.get("/")
def home():
    return {"message": "Churn Prediction API"}

@app.post("/predict")
def predict(data: dict):

    df = pd.DataFrame([data])
    df_scaled = scaler.transform(df)

    prediction = model.predict(df_scaled)

    return {"prediction": int(prediction[0])}
