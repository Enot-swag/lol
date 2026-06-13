import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from src.data_preparation import load_data, preprocess_pipeline, split_data, scale_features
from src.config import Config
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

def main():
    # Создаём папку models
    os.makedirs("models", exist_ok=True)
    
    print("="*60)
    print("CREDIT SCORING - MODEL TRAINING")
    print("="*60)
    
    # 1. Загрузка
    print("\n[1/6] Loading data...")
    df = load_data()
    
    # 2. Предобработка
    print("\n[2/6] Preprocessing...")
    X, y = preprocess_pipeline(df)
    
    # 3. Разделение
    print("\n[3/6] Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    
    # 4. Масштабирование
    print("\n[4/6] Scaling features...")
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test)
    
    # Сохраняем scaler
    joblib.dump(scaler, Config.SCALER_PATH)
    joblib.dump(list(X_train.columns), Config.FEATURES_PATH)
    print(f"   Scaler: {Config.SCALER_PATH}")
    print(f"   Features: {Config.FEATURES_PATH}")
    
    # 5. Обучение
    print("\n[5/6] Training models...")
    results = {}
    
    # Logistic Regression
    print("\n   → Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=Config.RANDOM_STATE)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict_proba(X_val_scaled)[:, 1]
    results['Logistic Regression'] = {
        'ROC-AUC': roc_auc_score(y_val, y_pred_lr),
        'Avg Precision': average_precision_score(y_val, y_pred_lr)
    }
    print(f"      ROC-AUC: {results['Logistic Regression']['ROC-AUC']:.4f}")
    
    # Random Forest
    print("\n   → Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=Config.RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict_proba(X_val_scaled)[:, 1]
    results['Random Forest'] = {
        'ROC-AUC': roc_auc_score(y_val, y_pred_rf),
        'Avg Precision': average_precision_score(y_val, y_pred_rf)
    }
    print(f"      ROC-AUC: {results['Random Forest']['ROC-AUC']:.4f}")
    
    # Выбор лучшей модели
    best_name = max(results, key=lambda x: results[x]['ROC-AUC'])
    best_model = rf if best_name == 'Random Forest' else lr
    joblib.dump(best_model, Config.MODEL_PATH)
    
    # 6. Результаты
    print("\n[6/6] Results:")
    print("-" * 40)
    for name, metrics in results.items():
        print(f"\n{name}:")
        for metric, value in metrics.items():
            print(f"   {metric}: {value:.4f}")
    
    print(f"\n✅ Best: {best_name} (ROC-AUC: {results[best_name]['ROC-AUC']:.4f})")
    print(f"   Saved to: {Config.MODEL_PATH}")
    print("="*60)

if __name__ == "__main__":
    main()