import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.model_selection import cross_val_score
import joblib
from src.config import Config

def train_and_log_models(X_train, y_train, X_val, y_val):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=Config.RANDOM_STATE),
        "LightGBM": lgb.LGBMClassifier(random_state=Config.RANDOM_STATE),
        "XGBoost": xgb.XGBClassifier(random_state=Config.RANDOM_STATE, eval_metric='logloss')
    }
    
    mlflow.set_tracking_uri(Config.MLFLOW_URI)
    best_auc = 0
    best_model = None
    
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            # Кросс-валидация на train
            cv_auc = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc').mean()
            
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            val_auc = roc_auc_score(y_val, y_pred_proba)
            ap = average_precision_score(y_val, y_pred_proba)
            
            mlflow.log_params(model.get_params())
            mlflow.log_metrics({"cv_roc_auc": cv_auc, "val_roc_auc": val_auc, "val_avg_precision": ap})
            
            if val_auc > best_auc:
                best_auc = val_auc
                best_model = model
                joblib.dump(best_model, Config.MODEL_PATH)
                mlflow.log_artifact(Config.MODEL_PATH)
    print(f"Best model saved with val_auc={best_auc:.4f}")
    return best_model