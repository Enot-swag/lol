import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_valid():
    payload = {
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
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "probability_default" in response.json()
    assert "predicted_class" in response.json()