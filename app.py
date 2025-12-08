# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# --- 1. Load Model ---
MODEL_FILE = 'banana_nasik_rf_model.pkl'
try:
    # This is the function that successfully reads the binary file!
    model = joblib.load(MODEL_FILE)
    app = FastAPI(title="Agriyukt Price Prediction API")
    print("Model loaded successfully.")
except FileNotFoundError:
    raise RuntimeError(f"Model file {MODEL_FILE} not found. Ensure it is in the project folder.")

# --- 2. Define Input Schema ---
# The mobile app will send the last 30 days of prices and the date for prediction.
class PriceData(BaseModel):
    last_30_prices: list[float] # List of 30 historical prices (most recent last)
    target_date: str = datetime.now().strftime("%Y-%m-%d")

# --- 3. Feature Generation Function ---
# This re-implements the feature engineering logic for a single data point
def generate_single_prediction_features(last_30_prices: list[float], target_date_str: str):
    if len(last_30_prices) < 30:
        raise ValueError("Must provide at least 30 days of historical prices for feature calculation.")

    # Lag and Rolling Features Calculation
    target_price_series = pd.Series(last_30_prices)
    lag_7 = target_price_series.iloc[-7] 
    lag_30 = target_price_series.iloc[0] 
    rolling_mean_7 = target_price_series.iloc[-7:].mean()
    rolling_std_30 = target_price_series.std()
    
    # Cyclical Features Calculation
    target_date = pd.to_datetime(target_date_str)
    month = target_date.month
    day_of_year = target_date.day_of_year
    
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    day_of_year_sin = np.sin(2 * np.pi * day_of_year / 365.25)
    day_of_year_cos = np.cos(2 * np.pi * day_of_year / 365.25)
    
    # Feature vector (Order MUST match the training columns!)
    feature_vector = [
        month_sin, month_cos, day_of_year_sin, day_of_year_cos,
        lag_7, lag_30, rolling_mean_7, rolling_std_30
    ]
    
    return pd.DataFrame([feature_vector], columns=[
        'Month_Sin', 'Month_Cos', 'DayOfYear_Sin', 'DayOfYear_Cos',
        'Lag_7', 'Lag_30', 'Rolling_Mean_7', 'Rolling_Std_30'
    ])

# --- 4. Prediction Endpoint ---
@app.post("/predict")
async def predict_price(data: PriceData):
    """
    Predicts the Modal Price for Banana (Khandesh) in Nasik APMC 
    based on the last 30 days of price history provided in the request body.
    """
    try:
        features_df = generate_single_prediction_features(data.last_30_prices, data.target_date)
        
        # Use the loaded model to make the prediction
        prediction = model.predict(features_df)[0]
        
        return {
            "commodity": "Banana (Khandesh)",
            "market": "Nasik APMC",
            "prediction_date": data.target_date,
            "predicted_modal_price_rs_quintal": round(float(prediction), 2)
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

@app.get("/")
def read_root():
    return {"message": "Agriyukt Price Prediction API is operational."}