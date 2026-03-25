"""
models/random_forest_model.py


Random Forest Models are an excellent baseline for financial ML because:
1. No feature scaling required (tree-based)
2. Built-in feature importance
3. Robust to outliers and irrelevant features
4. No overfitting risk from feature scaling choices
5. Fast to train and explain

How it works:
- Trains N decision trees, each on a random subset of data and features
- Prediction = majority vote across all trees
- Probability = fraction of trees voting for each class

Key hyperparameters:
- n_estimators: more trees → more stable, slower
- max_depth: shallow = underfit, deep = overfit
- min_samples_leaf: minimum samples in leaf → controls regularization
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
import joblib
from pathlib import Path

# Random Forest classifier for next-day return direction prediction
class RandomForestModel:
    """
    Usage:
        model = RandomForestModel()
        model.train(X_train, y_train)
        probs = model.predict_proba(X_test)
    """

    def __init__( self, n_estimators: int = 500, max_depth: int = 6, min_samples_leaf: int = 50, n_jobs: int = -1, random_state: int = 42 ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",     # Handles class imbalance
            n_jobs=n_jobs,
            random_state=random_state,
            oob_score=True,              # Out-of-bag error estimate (free validation)
        )
        self.feature_names = None

    # Train the Random Forest
    def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Use TimeSeriesSplit for cross-validation, NOT standard k-fold
        Standard k-fold shuffles data randomly, which leaks future information into training in time series
        TimeSeriesSplit always trains on past, tests on future
        """
        self.feature_names = list(X.columns)

        print("[RF] Training Random Forest...")
        self.model.fit(X.values, y.values)

        print(f"[RF] OOB Score (out-of-bag accuracy): {self.model.oob_score_:.4f}")
        print(f"[RF] Top 10 features by importance:")
        self._print_top_features(10)

        return self

    # Walk-forward cross-validation using TimeSeriesSplit
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> dict:
        """
        This mimics live trading: each fold trains on all past data and tests on the next unseen period
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = cross_val_score(
            self.model, X.values, y.values,
            cv=tscv, scoring="accuracy", n_jobs=-1
        )
        result = {"mean": scores.mean(), "std": scores.std(), "folds": scores.tolist()}
        print(f"[RF] CV Accuracy: {result['mean']:.4f} ± {result['std']:.4f}")
        return result

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Return class predictions: +1 or -1
        return self.model.predict(X.values)

    # Return probability for each class
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Shape: (n_samples, 2) → [:, 1] = P(up), [:, 0] = P(down)

        High confidence predictions get larger position sizes in the backtest
        """
        return self.model.predict_proba(X.values)

    def get_signal_strength(self, X: pd.DataFrame) -> pd.Series:
        """
        Convert class probabilities → scalar signal in [-1, +1]

        signal = P(up) - P(down)
        - +1.0 = very confident upward
        - -1.0 = very confident downward
        - ~0.0 = uncertain, stay flat
        """
        proba = self.predict_proba(X)
        classes = list(self.model.classes_)
        up_idx   = classes.index(1.0) if 1.0 in classes else 1
        down_idx = classes.index(-1.0) if -1.0 in classes else 0
        signal = proba[:, up_idx] - proba[:, down_idx]
        return pd.Series(signal, index=X.index, name="rf_signal")

    # Print a full classification report
    def evaluate(self, X: pd.DataFrame, y: pd.Series):
        preds = self.predict(X)
        print(f"[RF] Test Accuracy: {accuracy_score(y, preds):.4f}")
        print(classification_report(y, preds, target_names=["Down", "Up"]))

    # Return feature importances as a sorted DataFrame
    def feature_importance(self) -> pd.DataFrame:
        if self.feature_names is None:
            raise RuntimeError("Model not trained yet.")
        imp = pd.DataFrame({
            "feature":   self.feature_names,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return imp

    # Persist model to disk
    def save(self, path: str):
        joblib.dump(self, path)
        print(f"[RF] Model saved to {path}")

    @classmethod
    def load(cls, path: str) -> "RandomForestModel":
        return joblib.load(path)

    def _print_top_features(self, n: int = 10):
        imp = self.feature_importance().head(n)
        for _, row in imp.iterrows():
            bar = "█" * int(row["importance"] * 200)
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

    rf = RandomForestModel()
    rf.train(X_train, y_train)
    rf.evaluate(X_test, y_test)

    sig = rf.get_signal_strength(X_test)
    print(f"\nSignal preview:\n{sig.describe()}")