"""
models/lstm_model.py

LSTMs are a type of Recurrent Neural Network (RNN) designed to learn from
sequences. Unlike RF/XGBoost which treat each day independently, LSTMs
process a window of past days and maintain hidden state across time steps.

Why LSTM for finance:
- Captures temporal dependencies that tabular models miss
- Can learn complex nonlinear patterns across multiple time steps
- Handles the sequential nature of market data naturally

Architecture:
  Input sequence [t-seq_len, ..., t-1] → LSTM layers → Dense → P(up/down)

Key concepts:
- seq_len: number of past days to "look at" (e.g., 20 = one trading month)
- hidden_size: width of the LSTM hidden state
- num_layers: stacked LSTM cells
- Dropout: regularization to prevent overfitting
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import StandardScaler
import joblib



# PyTorch Dataset
class SequenceDataset(Dataset):
    """
    Converts flat (n_samples, n_features) arrays into sliding windows for sequence models

    Each sample is (X[t-seq_len:t], y[t]): a window of past features and the target for the next day
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor((y + 1) / 2)  # Remap -1/+1 → 0/1
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x_seq = self.X[idx : idx + self.seq_len]   # (seq_len, n_features)
        y_val = self.y[idx + self.seq_len]           # scalar target
        return x_seq, y_val



# LSTM Network Definition
class LSTMNet(nn.Module):
    """
    Stacked LSTM with dropout and a binary classification head

    Architecture:
        LSTM(input_size, hidden_size, num_layers, dropout)
            ↓  (take last hidden state)
        LayerNorm
            ↓
        Dropout
            ↓
        Linear(hidden_size, 64) → ReLU
            ↓
        Linear(64, 1) → Sigmoid
            ↓
        P(up) ∈ [0, 1]
    """

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,    # Input shape: (batch, seq_len, features)
            bidirectional=False, # Unidirectional: no future data leakage
        )
        self.norm    = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)           # (batch, seq_len, hidden_size)
        last_hidden  = lstm_out[:, -1, :]    # Take LAST time step
        normed       = self.norm(last_hidden)
        dropped      = self.dropout(normed)
        out          = self.fc(dropped)      # (batch, 1)
        return out.squeeze(1)                # (batch,)



# LSTM Model Wrapper
# Full training/inference wrapper around LSTMNet
class LSTMModel:
    """
    Usage:
        model = LSTMModel(seq_len=20, hidden_size=128)
        model.train(X_train, y_train)
        signal = model.get_signal_strength(X_test)
    """

    def __init__( self, seq_len: int = 20, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.3, lr: float = 1e-3, batch_size: int = 64, epochs: int = 50, patience: int = 10 ):
        self.seq_len     = seq_len
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.dropout     = dropout
        self.lr          = lr
        self.batch_size  = batch_size
        self.epochs      = epochs
        self.patience    = patience

        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler  = StandardScaler()
        self.net     = None

        print(f"[LSTM] Using device: {self.device}")


    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Training loop with:
        - StandardScaler normalization (LSTM is sensitive to input scale)
        - BCELoss (Binary Cross Entropy) as objective
        - Adam optimizer
        - ReduceLROnPlateau scheduler (halves LR when val loss plateaus)
        - Early stopping (stops when val loss doesn't improve for `patience` epochs)
        """
        # Normalize features
        X_scaled = self.scaler.fit_transform(X.values)
        y_vals   = y.values

        # Train/val split (temporal)
        val_size = max(int(len(X) * 0.15), self.seq_len + 1)
        X_tr, X_val = X_scaled[:-val_size], X_scaled[-val_size:]
        y_tr, y_val = y_vals[:-val_size], y_vals[-val_size:]

        train_ds = SequenceDataset(X_tr, y_tr, self.seq_len)
        val_ds   = SequenceDataset(X_val, y_val, self.seq_len)

        train_dl = TorchDataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_dl   = TorchDataLoader(val_ds,   batch_size=self.batch_size, shuffle=False)

        # Build model
        n_features = X_scaled.shape[1]
        self.net   = LSTMNet(n_features, self.hidden_size, self.num_layers, self.dropout)
        self.net.to(self.device)

        # Optimizer & scheduler
        optimizer  = torch.optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler  = ReduceLROnPlateau(optimizer, factor=0.5, patience=5, verbose=True)
        criterion  = nn.BCELoss()

        # Training loop
        best_val_loss = float("inf")
        best_weights  = None
        patience_cnt  = 0

        print(f"[LSTM] Training for up to {self.epochs} epochs...")
        for epoch in range(self.epochs):
            # Train
            self.net.train()
            train_loss = 0.0
            for X_batch, y_batch in train_dl:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                preds = self.net(X_batch)
                loss  = criterion(preds, y_batch)
                loss.backward()

                # Gradient clipping: prevents exploding gradients in RNNs
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()

            # Validate
            self.net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_batch, y_batch in val_dl:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    preds    = self.net(X_batch)
                    val_loss += criterion(preds, y_batch).item()

            train_loss /= len(train_dl)
            val_loss   /= len(val_dl)
            scheduler.step(val_loss)

            if (epoch + 1) % 5 == 0:
                print(f"[LSTM] Epoch {epoch+1:3d}/{self.epochs}  "
                      f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights  = {k: v.clone() for k, v in self.net.state_dict().items()}
                patience_cnt  = 0
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience:
                    print(f"[LSTM] Early stopping at epoch {epoch+1}")
                    break

        # Restore best weights
        self.net.load_state_dict(best_weights)
        print(f"[LSTM] Best val loss: {best_val_loss:.4f}")
        return self


    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        """
        Run inference on X. Returns P(up) for each sample
        First seq_len rows will be dropped (need history for first prediction)
        """
        if self.net is None:
            raise RuntimeError("Model not trained.")
        X_scaled = self.scaler.transform(X.values)
        y_dummy  = np.zeros(len(X_scaled))

        ds = SequenceDataset(X_scaled, y_dummy, self.seq_len)
        dl = TorchDataLoader(ds, batch_size=256, shuffle=False)

        self.net.eval()
        probs = []
        with torch.no_grad():
            for X_batch, _ in dl:
                p = self.net(X_batch.to(self.device)).cpu().numpy()
                probs.extend(p.tolist())
        return np.array(probs)


    def get_signal_strength(self, X: pd.DataFrame) -> pd.Series:
        """
        P(up) → signal in [-1, +1]
        First seq_len rows are NaN (model needs history to start)
        """
        probs  = self.predict_proba_raw(X)
        signal = 2 * probs - 1

        # Align back to original index (shift by seq_len)
        full_signal = np.full(len(X), np.nan)
        full_signal[self.seq_len:] = signal
        return pd.Series(full_signal, index=X.index, name="lstm_signal")

    def save(self, path: str):
        torch.save({"net_state": self.net.state_dict(), "wrapper": self}, path)
        print(f"[LSTM] Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "LSTMModel":
        checkpoint = torch.load(path, map_location="cpu")
        wrapper = checkpoint["wrapper"]
        wrapper.net.load_state_dict(checkpoint["net_state"])
        return wrapper



if __name__ == "__main__":
    import sys; sys.path.insert(0, "..")
    from data.data_loader import DataLoader
    from features.feature_engineer import FeatureEngineer

    loader = DataLoader()
    df = loader.fetch("AAPL", "2018-01-01", "2024-01-01")
    fe = FeatureEngineer()
    X, y = fe.build(df)

    split = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    lstm = LSTMModel(seq_len=20, hidden_size=64, epochs=20, patience=5)
    lstm.train(X_train, y_train)
    sig = lstm.get_signal_strength(X_test)
    print(f"\nSignal stats:\n{sig.dropna().describe()}")