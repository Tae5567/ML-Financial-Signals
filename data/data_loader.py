"""
data/data_loader.py

This module handles fetching and caching historical OHLCV (Open, High, Low, Close, Volume)
price data from Yahoo Finance via yfinance.
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from pathlib import Path


# Fetches and caches OHLCV price data
class DataLoader:
    """
    Usage:
        loader = DataLoader(cache_dir="./cache")
        df = loader.fetch("AAPL", start="2018-01-01", end="2024-01-01")
    """

    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # Fetch OHLCV data for a ticker between start and end dates
    """
        Parameters:
        ticker : str
            Stock symbol, e.g. "AAPL", "SPY", "BTC-USD"
        start : str
            ISO date string, e.g. "2018-01-01"
        end : str
            ISO date string, e.g. "2024-01-01"
        force_reload : bool
            If True, skip cache and re-fetch from Yahoo Finance

        Returns:
        pd.DataFrame
            Columns: Open, High, Low, Close, Volume, returns
            Index: DatetimeIndex (trading days only)
    """
    def fetch(self, ticker: str, start: str, end: str, force_reload: bool = False) -> pd.DataFrame:
       
        cache_file = self.cache_dir / f"{ticker}_{start}_{end}.parquet"

        # Load from cache if available 
        if cache_file.exists() and not force_reload:
            print(f"[DataLoader] Loading {ticker} from cache...")
            df = pd.read_parquet(cache_file)
            return df

        # Fetch from Yahoo Finance
        print(f"[DataLoader] Fetching {ticker} from Yahoo Finance ({start} → {end})...")
        raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)

        if raw.empty:
            raise ValueError(f"No data returned for {ticker}. Check the ticker and date range.")

        # Flatten MultiIndex columns if present 
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        # Clean and normalize 
        df = raw[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.index.name = "Date"
        df.sort_index(inplace=True)

        # Forward-fill gaps (e.g., holidays where some exchanges trade, others don't)
        df.ffill(inplace=True)
        df.dropna(inplace=True)

        # Core derived column: log returns 
        # Log returns are preferred over simple returns in finance because:
        # 1. They're additive over time: log(P_t/P_0) = sum of daily log returns
        # 2. They're more normally distributed (better for ML)
        # 3. They're symmetric: +10% and -10% are equidistant
        df["returns"] = np.log(df["Close"] / df["Close"].shift(1))
        df.dropna(inplace=True)

        # Cache to disk 
        df.to_parquet(cache_file)
        print(f"[DataLoader] Saved {len(df)} rows to cache: {cache_file.name}")

        return df

    # Fetch data for multiple tickers
    def fetch_multiple(self, tickers: list, start: str, end: str) -> dict:
        """
        Returns a dict: {ticker: DataFrame}
        Useful for universe-level signal generation or building features from
        correlated assets (e.g., sector ETFs, futures).
        """
        return {ticker: self.fetch(ticker, start, end) for ticker in tickers}



if __name__ == "__main__":
    loader = DataLoader()
    df = loader.fetch("AAPL", "2020-01-01", "2024-01-01")
    print(df.tail())
    print(f"\nShape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df.index[0].date()} → {df.index[-1].date()}")