import joblib
import pandas as pd
from src.data_preparation import load_data, preprocess_pipeline, split_data, scale_features
from src.config import Config
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings('ignore')

def main():
    print("="*60)
    print("CREDIT SCORING - MODEL TRAINING")
    print("="*60)
    
    # 1. Загрузка данных
    print("\n[1/6] Loading data...")
    try:
        df = load_data()
        print(f"   Success! Shape: {df.shape}")
    except Exception as e:
        print(f"   Error loading data: {e}")
        return
    
    # 2. Предобработка
    print("\n[2/6] Preprocessing...")
    try:
        X, y = preprocess_pipeline(df)
        print(f"   Features shape: {X.shape}")
        print(f"   Target shape: {y.shape}")
        print(f"   Features: {list(X.columns)}")
    except Exception as e:
        print(f"   Error in preprocessing: {e}")
        return
    
    # 3. Разделение данных
    print("\n[3/6] Splitting data...")
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
        print(f"   Train: {X_train.shape}")
        print(f"   Val: {X_val.shape}")
        print(f"   Test: {X_test.shape}")
    except Exception as e:
        print(f"   Error in splitting: {e}")
        return
    
    # 4. Масштабирование
    print("\n[4/6] Scaling features...")
    try:
        X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test)
        
        # Создаем папку models если её нет
        import os
        os.makedirs("models", exist_ok=True)
        
        # Сохраняем scaler и feature names
        joblib.dump(scaler, Config.SCALER_PATH)
        joblib.dump(list(X_train.columns), Config.FEATURES_PATH)
        print(f"   Scaler saved to {Config.SCALER_PATH}")
        print(f"   Features saved to {Config.FEATURES_PATH}")
    except Exception as e:
        print(f"   Error in scaling: {e}")
        return
    
    # 5. Обучение моделей
    print("\n[5/6] Training models...")
    results = {}
    
    # Logistic Regression
    print("\n   → Training Logistic Regression...")
    try:
        lr = LogisticRegression(max_iter=1000, random_state=Config.RANDOM_STATE)
        lr.fit(X_train_scaled, y_train)
        y_pred_lr = lr.predict_proba(X_val_scaled)[:, 1]
        results['Logistic Regression'] = {
            'ROC-AUC': roc_auc_score(y_val, y_pred_lr),
            'Avg Precision': average_precision_score(y_val, y_pred_lr)
        }
        print(f"      ROC-AUC: {results['Logistic Regression']['ROC-AUC']:.4f}")
        print(f"      Avg Precision: {results['Logistic Regression']['Avg Precision']:.4f}")
    except Exception as e:
        print(f"      Error: {e}")
    
    # Random Forest
    print("\n   → Training Random Forest...")
    try:
        rf = RandomForestClassifier(n_estimators=100, random_state=Config.RANDOM_STATE, n_jobs=-1)
        rf.fit(X_train_scaled, y_train)
        y_pred_rf = rf.predict_proba(X_val_scaled)[:, 1]
        results['Random Forest'] = {
            'ROC-AUC': roc_auc_score(y_val, y_pred_rf),
            'Avg Precision': average_precision_score(y_val, y_pred_rf)
        }
        print(f"      ROC-AUC: {results['Random Forest']['ROC-AUC']:.4f}")
        print(f"      Avg Precision: {results['Random Forest']['Avg Precision']:.4f}")
    except Exception as e:
        print(f"      Error: {e}")
    
    # Выбор лучшей модели
    if results:
        best_model_name = max(results, key=lambda x: results[x]['ROC-AUC'])
        best_model = rf if best_model_name == 'Random Forest' else lr
        
        # Сохраняем лучшую модель
        joblib.dump(best_model, Config.MODEL_PATH)
        print(f"\n   ✅ Best model: {best_model_name}")
        print(f"      Saved to {Config.MODEL_PATH}")
    else:
        print("\n   ❌ No models were trained successfully!")
        return
    
    # 6. Результаты
    print("\n[6/6] Final Results:")
    print("-" * 40)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric_name, value in metrics.items():
            print(f"   {metric_name}: {value:.4f}")
    
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)

if __name__ == "__main__":
    main()