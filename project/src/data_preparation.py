import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import Config

def load_data():
    """Загрузка данных"""
    print(f"   Loading data from {Config.DATA_PATH}")
    df = pd.read_csv(Config.DATA_PATH)
    
    # Удаляем столбец индекса если есть
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    
    print(f"   Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def handle_missing(df):
    """Обработка пропусков"""
    missing_before = df.isnull().sum()
    if missing_before.sum() > 0:
        print(f"   Missing values: {missing_before[missing_before > 0].to_dict()}")
    
    # Заполняем пропуски медианой
    for col in ['MonthlyIncome', 'NumberOfDependents']:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)
    
    return df

def remove_outliers_iqr(df, columns, factor=1.5):
    """Обработка выбросов методом IQR"""
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - factor * IQR
        upper = Q3 + factor * IQR
        df[col] = df[col].clip(lower, upper)
    return df

def preprocess_pipeline(df, target_col=Config.TARGET_COL):
    """Основной пайплайн предобработки"""
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X = handle_missing(X)
    
    num_cols = X.select_dtypes(include=[np.number]).columns
    X = remove_outliers_iqr(X, num_cols)
    
    return X, y

def split_data(X, y):
    """Разделение на train/val/test"""
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, 
        random_state=Config.RANDOM_STATE, stratify=y
    )
    val_relative = Config.VAL_SIZE / (1 - Config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative,
        random_state=Config.RANDOM_STATE, stratify=y_temp
    )
    print(f"   Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_features(X_train, X_val, X_test):
    """Масштабирование признаков"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler