from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import numpy as np
import logging
import os
from dotenv import load_dotenv

load_dotenv()

# Настройка логов
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Credit Scoring API", version="1.0.0")

# Загрузка модели
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pkl")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")
FEATURES_PATH = os.getenv("FEATURES_PATH", "models/feature_names.pkl")

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    logger.info(f"Model loaded: {MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None
    scaler = None
    feature_names = None

class CreditInput(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float = Field(..., ge=0, le=100)
    age: int = Field(..., ge=18, le=120)
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(..., ge=0)
    DebtRatio: float = Field(..., ge=0)
    MonthlyIncome: float = Field(..., ge=0)
    NumberOfOpenCreditLinesAndLoans: int = Field(..., ge=0)
    NumberOfTimes90DaysLate: int = Field(..., ge=0)
    NumberRealEstateLoansOrLines: int = Field(..., ge=0)
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(..., ge=0)
    NumberOfDependents: int = Field(..., ge=0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "RevolvingUtilizationOfUnsecuredLines": 0.5,
                "age": 45,
                "NumberOfTime30_59DaysPastDueNotWorse": 0,
                "DebtRatio": 0.3,
                "MonthlyIncome": 5000,
                "NumberOfOpenCreditLinesAndLoans": 5,
                "NumberOfTimes90DaysLate": 0,
                "NumberRealEstateLoansOrLines": 1,
                "NumberOfTime60_89DaysPastDueNotWorse": 0,
                "NumberOfDependents": 2
            }
        }

class CreditOutput(BaseModel):
    probability_default: float
    predicted_class: int
    is_default: bool

@app.get("/")
async def root():
    return {"message": "Credit Scoring API", "status": "running"}

@app.get("/health")
async def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=CreditOutput)
async def predict(input_data: CreditInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        input_dict = input_data.model_dump()
        df = pd.DataFrame([input_dict])
        df = df[feature_names]
        scaled = scaler.transform(df)
        proba = model.predict_proba(scaled)[0, 1]
        pred_class = int(proba >= 0.5)
        
        logger.info(f"Prediction: prob={proba:.4f}, class={pred_class}")
        
        return CreditOutput(
            probability_default=float(proba),
            predicted_class=pred_class,
            is_default=bool(pred_class)
        )
    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)