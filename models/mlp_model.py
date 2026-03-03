"""
MLP Neural Network model trainer with purged walk-forward CV.
Uses PyTorch with early stopping and cosine LR scheduling.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from cv import PurgedWalkForwardCV
from features import MAX_WEIGHT


class StockMLP(nn.Module):
    """
    Simple 2-hidden-layer feed-forward network for stock move prediction.

    Architecture chosen for small-sample regime (10-200 observations):
    - Narrow layers (32→16) to prevent overfitting
    - Dropout for regularization
    - BatchNorm for training stability
    """
    def __init__(self, n_features: int, hidden1: int = 32, hidden2: int = 16, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp_model(
    X: np.ndarray,
    y: np.ndarray,
    dates: list[str],
    factor_names: list[str],
    purge_days: int = 2,
    embargo_days: int = 1,
    epochs: int = 200,
    lr: float = 0.005,
    batch_size: int = 32,
    patience: int = 20,
) -> dict:
    """
    Train a feed-forward MLP with purged walk-forward CV.
    Returns model metrics and gradient-based feature importance as weights.
    """
    n_splits = min(5, max(2, len(X) // 15))
    cv = PurgedWalkForwardCV(n_splits=n_splits, purge_days=purge_days, embargo_days=embargo_days)

    cv_scores = []
    cv_correlations = []
    best_model_state = None
    best_scaler = None
    best_mse = float("inf")

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, dates=dates)):
        scaler = StandardScaler()
        X_train = torch.FloatTensor(scaler.fit_transform(X[train_idx]))
        y_train = torch.FloatTensor(y[train_idx])
        X_val = torch.FloatTensor(scaler.transform(X[val_idx]))
        y_val = torch.FloatTensor(y[val_idx])

        model = StockMLP(n_features=X.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        train_ds = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_ds, batch_size=min(batch_size, len(X_train)), shuffle=True)

        model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(X_val)
                val_loss = criterion(val_pred, y_val).item()
            model.train()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state:
            model.load_state_dict(best_state)

        model.eval()
        with torch.no_grad():
            preds = model(X_val).numpy()

        mse = float(mean_squared_error(y[val_idx], preds))
        cv_scores.append(mse)

        if len(preds) > 2:
            corr = float(np.corrcoef(preds, y[val_idx])[0, 1])
            if not np.isnan(corr):
                cv_correlations.append(corr)

        if mse < best_mse:
            best_mse = mse
            best_model_state = best_state
            best_scaler = scaler

    if best_model_state is None:
        return {"status": "insufficient_data", "reason": "no_valid_cv_folds"}

    final_model = StockMLP(n_features=X.shape[1])
    final_model.load_state_dict(best_model_state)
    final_model.eval()

    X_full_scaled = torch.FloatTensor(best_scaler.transform(X))
    with torch.no_grad():
        full_preds = final_model(X_full_scaled).numpy()

    correlation = float(np.corrcoef(full_preds, y)[0, 1])
    if np.isnan(correlation):
        correlation = 0.0

    direction_correct = np.sum(np.sign(full_preds) == np.sign(y))
    direction_accuracy = float(direction_correct / len(y)) if len(y) > 0 else 0

    # Gradient-based feature importance
    X_importance = torch.FloatTensor(best_scaler.transform(X))
    X_importance.requires_grad_(True)
    out = final_model(X_importance)
    out.sum().backward()
    grad_importance = X_importance.grad.abs().mean(dim=0).numpy()

    total_imp = grad_importance.sum()
    importance = {}
    for i, fname in enumerate(factor_names):
        importance[fname] = round(float(grad_importance[i] / total_imp * 100) if total_imp > 0 else 0, 2)
    importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    weights = {}
    for i, fname in enumerate(factor_names):
        normed_imp = grad_importance[i] / total_imp if total_imp > 0 else 0
        raw_w = 1.0 + normed_imp * (MAX_WEIGHT - 1.0) * 2
        weights[fname] = round(float(np.clip(raw_w, -MAX_WEIGHT, MAX_WEIGHT)), 4)

    return {
        "status": "trained",
        "model_type": "mlp",
        "correlation": round(correlation, 4),
        "direction_accuracy": round(direction_accuracy, 4),
        "cv_mse": [round(s, 6) for s in cv_scores],
        "cv_correlations": [round(c, 4) for c in cv_correlations],
        "weights": weights,
        "importance": importance,
        "observations": len(X),
        "factors": factor_names,
        "architecture": "32→16→1 (dropout=0.3, BatchNorm, AdamW)",
        "purged_samples": cv.purged_samples,
    }

