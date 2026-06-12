import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import mlflow
from sklearn.metrics import roc_auc_score
from src.config import Config

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.BatchNorm1d(hdim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

def train_nn(X_train, y_train, X_val, y_val):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_t = torch.tensor(X_train.astype(np.float32))
    y_train_t = torch.tensor(y_train.values if hasattr(y_train, 'values') else y_train, dtype=torch.float32).view(-1,1)
    X_val_t = torch.tensor(X_val.astype(np.float32))
    y_val_t = torch.tensor(y_val.values if hasattr(y_val, 'values') else y_val, dtype=torch.float32).view(-1,1)
    
    train_ds = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    val_ds = TensorDataset(X_val_t, y_val_t)
    val_loader = DataLoader(val_ds, batch_size=256)
    
    model = MLP(input_dim=X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    with mlflow.start_run(run_name="MLP_Baseline"):
        for epoch in range(50):
            model.train()
            for Xb, yb in train_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                optimizer.zero_grad()
                preds = model(Xb)
                loss = criterion(preds, yb)
                loss.backward()
                optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_preds = torch.cat([model(Xb.to(device)) for Xb, _ in val_loader]).cpu().numpy()
                val_auc = roc_auc_score(y_val, val_preds)
            print(f"Epoch {epoch+1}, Val AUC: {val_auc:.4f}")
            mlflow.log_metric("val_auc", val_auc, step=epoch)
        
        torch.save(model.state_dict(), "models/nn_model.pth")
        mlflow.log_artifact("models/nn_model.pth")
    return model