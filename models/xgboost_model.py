"""
models/xgboost_model.py

XGBoost (eXtreme Gradient Boosting) is often the strongest performer on tabular financial data. 
Key advantages over Random Forest:
1. Gradient boosting: each tree corrects errors of the previous ones
2. Built-in regularization (L1 + L2) → less overfitting
3. Handles missing values natively
4. Typically higher accuracy than RF on same data
5. scale_pos_weight handles class imbalance perfectly

How gradient boosting differs from random forests:
- RF: N independent trees, vote equally
- XGB: N sequential trees, each learning from the residuals of the ensemble so far
  → More powerful, but requires more careful tuning to avoid overfitting
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report
import joblib

# XGBoost classifier for directional prediction
class XGBoostModel:
    """
    Usage:
        model = XGBoostModel()
        model.train(X_train, y_train)
        signal = model.get_signal_strength(X_test)
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,       # L1 regularization
        reg_lambda: float = 1.0,      # L2 regularization
        random_state: int = 42,
    ):
        """
        Key hyperparameters:
        - n_estimators:     Number of boosting rounds. More = slower but better
        - max_depth:        Depth of each tree
        - learning_rate:    Step size shrinkage
        - subsample:        Fraction of rows per tree
        - colsample_bytree: Fraction of features per tree
        - reg_alpha/lambda: Regularization
        """
        self.params = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
            objective="binary:logistic",
        )
        self.model = None
        self.feature_names = None

    # Train XGBoost with early stopping to prevent overfitting
    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Split off a validation set (last 20% of training data, in time order) and stop training when validation loss stops improving

        y must be 0/1 for XGBoost binary classification, so remap the {-1, +1} targets used by the rest of the pipeline
        """
        self.feature_names = list(X.columns)

        # Remap -1/+1 → 0/1 for binary classification
        y_binary = ((y + 1) / 2).astype(int)

        # Split for early stopping (purely temporal)
        val_size = max(int(len(X) * 0.15), 50)
        X_tr, X_val = X.iloc[:-val_size], X.iloc[-val_size:]
        y_tr, y_val = y_binary.iloc[:-val_size], y_binary.iloc[-val_size:]

        # Handle class imbalance
        n_pos = y_tr.sum()
        n_neg = len(y_tr) - n_pos
        scale_pos = n_neg / (n_pos + 1e-9)

        print(f"[XGB] Training XGBoost (scale_pos_weight={scale_pos:.2f})...")
        self.model = xgb.XGBClassifier(
            **self.params,
            scale_pos_weight=scale_pos,
            early_stopping_rounds=30,
        )
        self.model.fit(
            X_tr.values, y_tr.values,
            eval_set=[(X_val.values, y_val.values)],
            verbose=False,
        )

        best_iter = self.model.best_iteration
        print(f"[XGB] Best iteration: {best_iter}")
        print(f"[XGB] Top 10 features:")
        self._print_top_features(10)

        return self
    
    # Walk-forward cross-validation
    # Each fold trains on past and tests on future
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
        y_binary = ((y + 1) / 2).astype(int)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
            y_tr, y_te = y_binary.iloc[train_idx], y_binary.iloc[test_idx]

            # Scale pos weight per fold
            scale_pos = (y_tr == 0).sum() / (y_tr == 1).sum()
            m = xgb.XGBClassifier(**self.params, scale_pos_weight=scale_pos)
            m.fit(X_tr.values, y_tr.values, verbose=False)

            preds = m.predict(X_te.values)
            acc = accuracy_score(y_te.values, preds)
            scores.append(acc)
            print(f"[XGB] Fold {fold+1}: accuracy = {acc:.4f}")

        result = {"mean": np.mean(scores), "std": np.std(scores), "folds": scores}
        print(f"[XGB] CV Accuracy: {result['mean']:.4f} ± {result['std']:.4f}")
        return result

    # Return predictions in {-1, +1} space (matches pipeline convention)
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Train the model first.")
        raw = self.model.predict(X.values)
        return np.where(raw == 1, 1.0, -1.0)

    # Return P(up) for each sample
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Shape: (n_samples,) — probability of the next day being positive.
        """
        if self.model is None:
            raise RuntimeError("Train the model first.")
        proba = self.model.predict_proba(X.values)  # shape (n, 2)
        return proba  # [:, 1] = P(up)

    def get_signal_strength(self, X: pd.DataFrame) -> pd.Series:
        """
        Convert P(up) → signal in [-1, +1].
        signal = 2 * P(up) - 1
        - P(up) = 0.5 → signal = 0 (uncertain)
        - P(up) = 1.0 → signal = +1 (strong buy)
        - P(up) = 0.0 → signal = -1 (strong sell)
        """
        proba = self.predict_proba(X)[:, 1]
        signal = 2 * proba - 1
        return pd.Series(signal, index=X.index, name="xgb_signal")

    # Print classification metrics in original {-1, +1} space
    def evaluate(self, X: pd.DataFrame, y: pd.Series):
        preds = self.predict(X)
        print(f"[XGB] Test Accuracy: {accuracy_score(y, preds):.4f}")
        print(classification_report(y, preds, target_names=["Down", "Up"]))

    # Feature importances from XGBoost's gain metric
    def feature_importance(self) -> pd.DataFrame:
        if self.model is None:
            raise RuntimeError("Train the model first.")
        scores = self.model.feature_importances_
        return pd.DataFrame({
            "feature":    self.feature_names,
            "importance": scores,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    def save(self, path: str):
        joblib.dump(self, path)
        print(f"[XGB] Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "XGBoostModel":
        return joblib.load(path)

    def _print_top_features(self, n: int = 10):
        imp = self.feature_importance().head(n)
        for _, row in imp.iterrows():
            bar = "█" * int(row["importance"] * 500)
            print(f"  {row['feature']:<30} {row['importance']:.4f}  {bar}")



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

    xgb_model = XGBoostModel()
    xgb_model.train(X_train, y_train)
    xgb_model.evaluate(X_test, y_test)
    sig = xgb_model.get_signal_strength(X_test)
    print(f"\nSignal preview:\n{sig.describe()}")