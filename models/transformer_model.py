"""
models/transformer_model.py

Transformer Models  use Self-Attention instead of recurrence. Each time step can directly "attend" to any other time step,
capturing long-range dependencies more efficiently than LSTMs.

Why Transformers for finance:
- Self-attention can spot patterns across arbitrary time lags (not just recent)
- Positional encoding preserves temporal order without recurrence
- Parallel training: much faster than LSTMs
- State-of-the-art on many sequence tasks

Architecture:
  Input window → Projection → Positional Encoding
     → N x (Multi-Head Self-Attention + FFN) → CLS token → Dense → P(up)

Key concepts:
- Multi-Head Attention: multiple "views" of the sequence simultaneously
- Positional Encoding: Transformers have no recurrence, so inject position information via sinusoidal embeddings
- CLS token: a learnable "summary" token prepended to the sequence,whose final representation is used for classification
"""

import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.preprocessing import StandardScaler


# Positional Encoding
class PositionalEncoding(nn.Module):
    """
    Adds sinusoidal positional encoding to token embeddings

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """

    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Build PE matrix once, register as buffer (not a parameter)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)



# Transformer Network
class TransformerNet(nn.Module):
    """
    Encoder-only Transformer for sequence classification

    Architecture:
        Input features → Linear projection → [CLS] token prepend
            → Positional Encoding
            → N x TransformerEncoderLayer (self-attention + FFN)
            → CLS token output → Classification head
    """

    def __init__( self, input_size: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 256, dropout: float = 0.1 ):
        super().__init__()

        # Project raw features to d_model dimensions
        self.input_proj = nn.Linear(input_size, d_model)

        # Learnable [CLS] token (BERT-style)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,  # (batch, seq, features)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Classification head
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        batch_size = x.size(0)

        # Project to d_model
        x = self.input_proj(x)                           # (batch, seq, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        x   = torch.cat([cls, x], dim=1)                 # (batch, 1+seq, d_model)

        # Add positional encoding
        x = self.pos_enc(x)                              # (batch, 1+seq, d_model)

        # Transformer encoder
        x = self.transformer(x)                          # (batch, 1+seq, d_model)

        # Use CLS token (position 0) for classification
        cls_out = x[:, 0, :]                             # (batch, d_model)
        return self.head(cls_out).squeeze(1)             # (batch,)



# Dataset (same structure as LSTM)
class SequenceDataset(Dataset):
    def __init__(self, X, y, seq_len):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor((y + 1) / 2)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        return self.X[idx:idx + self.seq_len], self.y[idx + self.seq_len]



# Transformer Model Wrapper
class TransformerModel:
    """
    Full training/inference wrapper for the Transformer

    Usage:
        model = TransformerModel(seq_len=30, d_model=64)
        model.train(X_train, y_train)
        signal = model.get_signal_strength(X_test)
    """

    def __init__( self, seq_len: int = 30, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 256, dropout: float = 0.1, lr: float = 5e-4, batch_size: int = 64, epochs: int = 50, patience: int = 10 ):
        self.seq_len       = seq_len
        self.d_model       = d_model
        self.nhead         = nhead
        self.num_layers    = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout       = dropout
        self.lr            = lr
        self.batch_size    = batch_size
        self.epochs        = epochs
        self.patience      = patience

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.net    = None

        print(f"[Transformer] Using device: {self.device}")

    # Train the Transformer with cosine annealing learning rate schedule
    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Cosine annealing starts at lr_max and smoothly decays to lr_min,
        which gives better generalization than a fixed learning rate
        """
        X_scaled = self.scaler.fit_transform(X.values)
        y_vals   = y.values

        val_size = max(int(len(X) * 0.15), self.seq_len + 1)
        X_tr, X_val = X_scaled[:-val_size], X_scaled[-val_size:]
        y_tr, y_val = y_vals[:-val_size], y_vals[-val_size:]

        train_dl = TorchDataLoader(
            SequenceDataset(X_tr, y_tr, self.seq_len),
            batch_size=self.batch_size, shuffle=True,
        )
        val_dl = TorchDataLoader(
            SequenceDataset(X_val, y_val, self.seq_len),
            batch_size=self.batch_size, shuffle=False,
        )

        n_features = X_scaled.shape[1]
        self.net   = TransformerNet(
            input_size=n_features,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=1e-4
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=1e-6)
        criterion = nn.BCELoss()

        best_val_loss = float("inf")
        best_weights  = None
        patience_cnt  = 0

        print(f"[Transformer] Training for up to {self.epochs} epochs...")
        for epoch in range(self.epochs):
            # Train
            self.net.train()
            train_loss = 0.0
            for X_b, y_b in train_dl:
                X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.net(X_b), y_b)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            scheduler.step()

            # Validate 
            self.net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_b, y_b in val_dl:
                    X_b, y_b = X_b.to(self.device), y_b.to(self.device)
                    val_loss += criterion(self.net(X_b), y_b).item()

            train_loss /= len(train_dl)
            val_loss   /= len(val_dl)

            if (epoch + 1) % 5 == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"[Transformer] Epoch {epoch+1:3d}/{self.epochs}  "
                      f"train={train_loss:.4f}  val={val_loss:.4f}  lr={lr_now:.2e}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights  = {k: v.clone() for k, v in self.net.state_dict().items()}
                patience_cnt  = 0
            else:
                patience_cnt += 1
                if patience_cnt >= self.patience:
                    print(f"[Transformer] Early stopping at epoch {epoch+1}")
                    break

        self.net.load_state_dict(best_weights)
        print(f"[Transformer] Best val loss: {best_val_loss:.4f}")
        return self

    def predict_proba_raw(self, X: pd.DataFrame) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("Train the model first.")
        X_scaled = self.scaler.transform(X.values)
        y_dummy  = np.zeros(len(X_scaled))
        ds = SequenceDataset(X_scaled, y_dummy, self.seq_len)
        dl = TorchDataLoader(ds, batch_size=256, shuffle=False)

        self.net.eval()
        probs = []
        with torch.no_grad():
            for X_b, _ in dl:
                p = self.net(X_b.to(self.device)).cpu().numpy()
                probs.extend(p.tolist())
        return np.array(probs)

    def get_signal_strength(self, X: pd.DataFrame) -> pd.Series:
        probs  = self.predict_proba_raw(X)
        signal = 2 * probs - 1
        full   = np.full(len(X), np.nan)
        full[self.seq_len:] = signal
        return pd.Series(full, index=X.index, name="transformer_signal")



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

    tf = TransformerModel(seq_len=20, epochs=15, patience=5)
    tf.train(X_train, y_train)
    sig = tf.get_signal_strength(X_test)
    print(f"\nSignal stats:\n{sig.dropna().describe()}")