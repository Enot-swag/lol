from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from typing import List, Dict
import joblib
import numpy as np
import pandas as pd
import logging
import json
import time
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from fastapi.responses import Response
import os
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format='{"time": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}')
logger = logging.getLogger(__name__)

app = FastAPI(title="Credit Scoring API")

# Метрики Prometheus
REQUESTS = Counter("http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"])
LATENCY = Histogram("http_request_duration_seconds", "Request latency", ["method", "endpoint"])

# Загрузка модели
MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pkl")
model = joblib.load(MODEL_PATH)
scaler = joblib.load("models/scaler.pkl")  # нужно сохранить из train_models
feature_names = joblib.load("models/feature_names.pkl")

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
        schema_extra = {
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

@app.middleware("http")
async def log_and_metrics(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    REQUESTS.labels(method=request.method, endpoint=request.url.path, status=response.status_code).inc()
    LATENCY.labels(method=request.method, endpoint=request.url.path).observe(duration)
    logger.info(json.dumps({
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "duration": duration,
        "client": request.client.host if request.client else None
    }))
    return response

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict")
async def predict(input_data: CreditInput):
    try:
        df = pd.DataFrame([input_data.dict()])
        df = df[feature_names]  # гарантируем порядок
        scaled = scaler.transform(df)
        proba = model.predict_proba(scaled)[0, 1]
        pred_class = int(proba >= 0.5)
        return {"probability_default": float(proba), "predicted_class": pred_class}
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))