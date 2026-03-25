"""
signals/signal_generator.py

This module combines the outputs of all four models into a single unified "alpha signal": 
a scalar between -1 and +1 that tells the backtest how strongly to be long or short

Ensemble methods work because:
- Each model captures different aspects of the data (RF=tabular, LSTM=sequential)
- Errors tend to cancel out across models
- Diversification reduces overfitting risk

Signal combination methods:
1. Equal-weight average (simple, robust baseline)
2. Weighted average (weight by recent model accuracy)
3. Confidence-thresholded: only trade when models agree strongly

Alpha signals → Trading signals:
- |signal| > high_threshold  → full position
- |signal| > low_threshold   → half position
- |signal| < low_threshold   → flat (no trade)
"""

import numpy as np
import pandas as pd
from typing import Optional


class SignalGenerator:
    """
    Combines multiple model signals into a clean trading signal

    Usage:
        sg = SignalGenerator(threshold=0.1)
        sg.add_signal("rf",   rf_signal)
        sg.add_signal("xgb",  xgb_signal)
        sg.add_signal("lstm", lstm_signal)
        sg.add_signal("trf",  trf_signal)

        composite = sg.composite_signal(method="weighted", weights=[0.3, 0.4, 0.15, 0.15])
        positions = sg.threshold_signal(composite, low=0.1, high=0.3)
    """

    def __init__(self, threshold: float = 0.1):
        self.threshold = threshold
        self.signals   = {}
    
    # Register a model's signal. Signal must be in [-1, +1]
    def add_signal(self, name: str, signal: pd.Series):
        self.signals[name] = signal.rename(name)

    # Combine all registered signals into one composite alpha
    def composite_signal( self, method: str = "equal", weights: Optional[list] = None ) -> pd.Series:
        """
        Parameters:
        method : str
            "equal"    → simple average
            "weighted" → weighted average (weights must sum to 1)
            "vote"     → sign of majority vote (each model votes ±1)
        weights : list, optional
            Weights for "weighted" method, aligned to order of add_signal() calls.

        Returns:
        pd.Series
            Composite signal in [-1, +1]
        """
        if not self.signals:
            raise RuntimeError("No signals added yet.")

        # Align all signals on common index, forward-fill NaNs from warm-up
        df = pd.DataFrame(self.signals)
        df.ffill(inplace=True)

        if method == "equal":
            composite = df.mean(axis=1)

        elif method == "weighted":
            if weights is None:
                weights = [1.0 / len(df.columns)] * len(df.columns)
            assert len(weights) == len(df.columns), "weights length must match number of signals"
            w = np.array(weights) / np.sum(weights)  # Normalize
            composite = df.values @ w
            composite = pd.Series(composite, index=df.index)

        elif method == "vote":
            # Each model votes +1 (bullish) or -1 (bearish)
            votes = np.sign(df)
            composite = np.sign(votes.sum(axis=1))  # Majority vote
            composite = pd.Series(composite.values, index=df.index)

        else:
            raise ValueError(f"Unknown method: {method}. Use 'equal', 'weighted', or 'vote'.")

        composite.name = "composite_signal"
        return composite

    #  Convert continuous signal → discrete positions with thresholds
    def threshold_signal( self, signal: pd.Series, low: float = 0.1, high: float = 0.3 ) -> pd.Series:
        """
        Logic:
          |signal| >= high  → full position  (±1.0)
          |signal| >= low   → half position  (±0.5)
          |signal| <  low   → flat           (0.0)

        This prevents trading on low-confidence signals, which are more likely to be noise than genuine alpha.

        Parameters:
        signal : pd.Series
            Raw signal in [-1, +1]
        low : float
            Below this → flat
        high : float
            Above this → full position

        Returns:
        pd.Series
            Discrete position in {-1.0, -0.5, 0.0, +0.5, +1.0}
        """
        pos = pd.Series(0.0, index=signal.index, name="position")

        strong_long  = signal >=  high
        mid_long     = (signal >=  low) & (signal <  high)
        mid_short    = (signal <= -low) & (signal > -high)
        strong_short = signal <= -high

        pos[strong_long]  =  1.0
        pos[mid_long]     =  0.5
        pos[mid_short]    = -0.5
        pos[strong_short] = -1.0

        return pos

    def signal_summary(self, signals: dict = None) -> pd.DataFrame:
        """
        Print correlation matrix of model signals and summary stats
        High correlation → models are redundant; Low → good diversification
        """
        sigs = signals or self.signals
        df   = pd.DataFrame(sigs).dropna()

        print("\n Signal Correlations ")
        print(df.corr().round(3).to_string())

        print("\n Signal Statistics ")
        print(df.describe().round(4).to_string())

        # Agreement rate: fraction of days all models agree on direction
        signs = np.sign(df)
        agreement = (signs == signs.iloc[:, 0].values.reshape(-1, 1)).all(axis=1).mean()
        print(f"\nAll-models agreement rate: {agreement:.2%}")

        return df.corr()

    # Information Coefficient (IC) = rank correlation between signal and actual forward returns
    def information_coefficient( self, signal: pd.Series, forward_returns: pd.Series ) -> float:
        """
        IC > 0    → signal has positive predictive power
        IC > 0.05 → useful signal (rare in practice)
        IC > 0.1  → excellent signal

        Uses Spearman rank correlation (robust to outliers)
        """
        from scipy.stats import spearmanr
        aligned = pd.DataFrame({"signal": signal, "fwd": forward_returns}).dropna()
        ic, pval = spearmanr(aligned["signal"], aligned["fwd"])
        print(f"IC = {ic:.4f}  (p={pval:.4f})")
        return ic

    # Rolling IC over a window (e.g. 63 = 1 quarter)
    def rolling_ic( self, signal: pd.Series, forward_returns: pd.Series, window: int = 63 ) -> pd.Series:
        """
        Detects when a signal is becoming stale or regime-dependent.
        """
        from scipy.stats import spearmanr

        aligned = pd.DataFrame({"signal": signal, "fwd": forward_returns}).dropna()
        ic_series = []
        dates = []

        for i in range(window, len(aligned)):
            chunk = aligned.iloc[i - window : i]
            ic, _ = spearmanr(chunk["signal"], chunk["fwd"])
            ic_series.append(ic)
            dates.append(aligned.index[i])

        return pd.Series(ic_series, index=dates, name="rolling_ic")



if __name__ == "__main__":
    import sys; sys.path.insert(0, "..")

    # Simulate two mock signals for demonstration
    np.random.seed(42)
    idx     = pd.date_range("2022-01-01", periods=252)
    sig_a   = pd.Series(np.random.randn(252) * 0.3, index=idx, name="model_a")
    sig_b   = pd.Series(np.random.randn(252) * 0.3, index=idx, name="model_b")
    fwd_ret = pd.Series(np.random.randn(252) * 0.01, index=idx)

    sg = SignalGenerator()
    sg.add_signal("model_a", sig_a)
    sg.add_signal("model_b", sig_b)

    composite = sg.composite_signal(method="equal")
    positions = sg.threshold_signal(composite, low=0.1, high=0.25)
    sg.signal_summary()
    sg.information_coefficient(composite, fwd_ret)

    print(f"\nPosition distribution:\n{positions.value_counts()}")