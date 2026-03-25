"""
features/feature_engineer.py

ML models need *structured* inputs that capture market
microstructure, momentum, mean-reversion and regime information.

Feature Categories:
1. Price-based: Moving averages, Bollinger Bands, ATR
2. Momentum: RSI, MACD, Rate-of-Change
3. Volume: OBV, Volume-price trends
4. Volatility: Rolling std, realized vol, vol of vol
5. Calendar: Day-of-week effects
6. Lag features: Autoregressive structure in returns
7. Cross-sectional: Z-scores to normalize across time
"""

import pandas as pd
import numpy as np

# Transforms raw OHLCV to ML ready feature matrix
class FeatureEngineer:
    """
    Usage:
        fe = FeatureEngineer(lags=[1, 2, 3, 5, 10])
        X, y = fe.build(df, forward_periods=1)
    """

    def __init__(self, lags: list = None):
        # Lag periods for autoregressive return features
        self.lags = lags or [1, 2, 3, 5, 10, 21]

    
    # Public API
    # Build feature matrix X and target vector y
    def build(self, df: pd.DataFrame, forward_periods: int = 1) -> tuple:
        """
        Parameters:
        df : pd.DataFrame
            Raw OHLCV + returns DataFrame from DataLoader
        forward_periods : int
            How many days ahead to predict. 1 = next-day return

        Returns:
        X : pd.DataFrame
            Feature matrix (rows = trading days, cols = features)
        y : pd.Series
            Target = forward log return, classified as +1 (up) or -1 (down)
        """
        feat = pd.DataFrame(index=df.index)

        # Build each feature group
        self._price_features(df, feat)
        self._momentum_features(df, feat)
        self._volume_features(df, feat)
        self._volatility_features(df, feat)
        self._calendar_features(df, feat)
        self._lag_features(df, feat)
        self._zscore_features(df, feat)

        # Target: next-day return direction
        future_return = df["returns"].shift(-forward_periods)
        y = np.sign(future_return).rename("target")

        # Drop where target is exactly 0 
        mask = y != 0

        # Final cleanup 
        X = feat.loc[mask]
        y = y.loc[mask]

        # Drop rows with any NaN
        valid = X.notna().all(axis=1)
        X = X.loc[valid]
        y = y.loc[valid]

        # Clip extreme values
        X = X.clip(-10, 10)

        print(f"[FeatureEngineer] Built {X.shape[1]} features × {X.shape[0]} samples")
        return X, y

    # Feature groups (private methods)
    def _price_features(self, df: pd.DataFrame, feat: pd.DataFrame):
        """
        Price-relative features capture where price is relative to its history.

        Moving average crossovers are classic trend signals: when fast MA > slow MA,
        the asset is in an uptrend. Bollinger Bands add volatility context.
        """
        close = df["Close"]

        # Moving average ratios
        for w in [5, 10, 20, 50, 200]:
            ma = close.rolling(w).mean()
            feat[f"price_to_ma{w}"] = (close / ma - 1).shift(1)

        # MA crossovers
        feat["ma5_to_ma20"]  = (close.rolling(5).mean() / close.rolling(20).mean() - 1).shift(1)
        feat["ma10_to_ma50"] = (close.rolling(10).mean() / close.rolling(50).mean() - 1).shift(1)
        feat["ma20_to_ma200"]= (close.rolling(20).mean() / close.rolling(200).mean() - 1).shift(1)

        # Bollinger Bands: price position within its volatility envelope
        # BB_position = (price - lower) / (upper - lower), ranges [0,1]
        for w in [20]:
            ma  = close.rolling(w).mean()
            std = close.rolling(w).std()
            upper = ma + 2 * std
            lower = ma - 2 * std
            feat[f"bb_position_{w}"] = ((close - lower) / (upper - lower + 1e-9)).shift(1)
            feat[f"bb_width_{w}"]    = ((upper - lower) / ma).shift(1)  # Relative band width

        # Average True Range (ATR): normalized volatility
        high, low = df["High"], df["Low"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low  - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        feat["atr_pct"] = (tr.rolling(14).mean() / close).shift(1)

        # High-Low range as fraction of close
        feat["hl_range"] = ((high - low) / close).shift(1)

        # Gap: today's open vs yesterday's close
        feat["overnight_gap"] = (df["Open"] / close.shift(1) - 1).shift(1)

    def _momentum_features(self, df: pd.DataFrame, feat: pd.DataFrame):
        """
        Momentum features capture trend persistence.

        Key insight: price momentum (past N-day return) is one of the most
        well-documented anomalies in finance (Jegadeesh & Titman, 1993).
        RSI and MACD are technical indicators that quantify momentum.
        """
        close = df["Close"]
        ret   = df["returns"]

        # Rate of Change (ROC)
        for w in [5, 10, 21, 63, 126, 252]:
            feat[f"roc_{w}d"] = close.pct_change(w).shift(1)

        # RSI: Relative Strength Index (0-100)
        # RSI > 70 = overbought, RSI < 30 = oversold
        for w in [7, 14, 21]:
            feat[f"rsi_{w}"] = self._rsi(close, w).shift(1)

        # MACD: Moving Average Convergence/Divergence
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        macd_line   = ema12 - ema26
        signal_line = macd_line.ewm(span=9).mean()
        feat["macd_line"]      = (macd_line   / close).shift(1)
        feat["macd_signal"]    = (signal_line / close).shift(1)
        feat["macd_histogram"] = ((macd_line - signal_line) / close).shift(1)

        # Cumulative return over various windows (skip 1 day to reduce microstructure noise)
        for w in [5, 21, 63]:
            feat[f"cumul_ret_{w}d"] = ret.rolling(w).sum().shift(1)

    # Volume features reveal conviction behind price moves
    def _volume_features(self, df: pd.DataFrame, feat: pd.DataFrame):
        """
        High volume on up days = institutional buying
        High volume on down days = institutional selling

        On-Balance Volume (OBV) accumulates volume in the direction of price
        """
        close  = df["Close"]
        volume = df["Volume"]
        ret    = df["returns"]

        # Volume relative to its moving average
        for w in [5, 20, 50]:
            feat[f"vol_ratio_{w}d"] = (volume / volume.rolling(w).mean()).shift(1)

        # Log volume level (normalized)
        feat["log_volume"] = np.log1p(volume).shift(1)

        # On-Balance Volume (OBV): cumulative directional volume
        obv = (np.sign(ret) * volume).cumsum()
        feat["obv_ma_ratio"] = (obv / obv.rolling(20).mean() - 1).shift(1)

        # Volume-price trend: Are price moves accompanied by volume?
        feat["price_volume_corr"] = (
            pd.Series(ret).rolling(20)
            .corr(pd.Series(np.log1p(volume)))
        ).shift(1)

        # Volume surge: big volume spike (often precedes breakouts)
        feat["vol_surge"] = (volume / volume.rolling(20).mean()).shift(1)

    # Volatility features capture risk regime
    def _volatility_features(self, df: pd.DataFrame, feat: pd.DataFrame):
        """
        Key insight: volatility clusters (high-vol periods follow high-vol periods)
        """
        ret = df["returns"]

        # Rolling realized volatility (annualized)
        for w in [5, 10, 21, 63]:
            feat[f"rvol_{w}d"] = (ret.rolling(w).std() * np.sqrt(252)).shift(1)

        # Volatility of volatility (Vol-of-vol)
        vol21 = ret.rolling(21).std()
        feat["vov_21d"] = vol21.rolling(21).std().shift(1)

        # Vol ratio: short-term vs long-term vol
        # > 1 means volatility is elevated
        feat["vol_ratio_5_21"] = (ret.rolling(5).std() / (ret.rolling(21).std() + 1e-9)).shift(1)

        # Skewness and Kurtosis of recent returns
        feat["skew_21d"] = ret.rolling(21).skew().shift(1)
        feat["kurt_21d"] = ret.rolling(21).kurt().shift(1)


    def _calendar_features(self, df: pd.DataFrame, feat: pd.DataFrame):
        """
        Calendar effects are among the most studied anomalies in finance:
        - Monday effect: stocks tend to underperform on Mondays
        - January effect: small caps outperform in January
        - Turn-of-month: buying pressure at month end/start
        """
        idx = df.index
        feat["day_of_week"]    = idx.dayofweek         # 0=Mon, 4=Fri
        feat["month"]          = idx.month
        feat["is_monday"]      = (idx.dayofweek == 0).astype(float)
        feat["is_friday"]      = (idx.dayofweek == 4).astype(float)
        feat["is_january"]     = (idx.month == 1).astype(float)
        feat["is_month_start"] = idx.is_month_start.astype(float)
        feat["is_month_end"]   = idx.is_month_end.astype(float)
        feat["quarter"]        = idx.quarter

    def _lag_features(self, df: pd.DataFrame, feat: pd.DataFrame):
        """
        Lag features give the model autoregressive (AR) structure to essentially
        telling the model: "here's what happened yesterday, 2 days ago, etc."

        This captures mean-reversion (short lags) and momentum (longer lags)
        """
        ret = df["returns"]
        for lag in self.lags:
            feat[f"ret_lag_{lag}"] = ret.shift(lag)

        # Consecutive up/down days (streak)
        direction = np.sign(ret)
        feat["up_streak"]   = direction.rolling(5).sum().shift(1)   # Net up days in last 5
        feat["down_streak"]  = (-direction).clip(lower=0).rolling(5).sum().shift(1)

    def _zscore_features(self, df: pd.DataFrame, feat: pd.DataFrame):
        """
        Z-scoring normalizes price levels to be stationary (mean-reverting)

        A z-score of +2 means the price is 2 standard deviations above its
        rolling mean → potential mean-reversion opportunity
        """
        close = df["Close"]
        ret   = df["returns"]

        for w in [20, 63]:
            mu  = close.rolling(w).mean()
            std = close.rolling(w).std()
            feat[f"zscore_{w}d"] = ((close - mu) / (std + 1e-9)).shift(1)

        # Return z-score
        for w in [21, 63]:
            mu  = ret.rolling(w).mean()
            std = ret.rolling(w).std()
            feat[f"ret_zscore_{w}d"] = ((ret - mu) / (std + 1e-9)).shift(1)


    # Helpers
    @staticmethod
    def _rsi(series: pd.Series, window: int) -> pd.Series:
        """
        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss over `window` days.
        """
        delta = series.diff()
        gain  = delta.clip(lower=0)
        loss  = (-delta).clip(lower=0)
        avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
        avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()
        rs  = avg_gain / (avg_loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi / 100.0  # Normalize to [0, 1]



if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from data.data_loader import DataLoader

    loader = DataLoader()
    df = loader.fetch("AAPL", "2018-01-01", "2024-01-01")

    fe = FeatureEngineer()
    X, y = fe.build(df)

    print(f"\nFeature matrix: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    print(f"\nSample features:\n{X.head(3).to_string()}")