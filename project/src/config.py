import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Пути к данным и моделям
    DATA_PATH = os.getenv("DATA_PATH", "data/cs-training.csv")
    MODEL_PATH = os.getenv("MODEL_PATH", "models/best_model.pkl")
    SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.pkl")
    FEATURES_PATH = os.getenv("FEATURES_PATH", "models/feature_names.pkl")
    
    # Параметры MLflow
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    
    # Параметры данных
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    RANDOM_STATE = 42
    TARGET_COL = 'SeriousDlqin2yrs'
    
    # Параметры API
    PORT = int(os.getenv("PORT", 8000))
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")