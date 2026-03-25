"""
main.py

Full Pipeline Runner
This script ties all components together and runs the complete ML signal generation pipeline end-to-end.

Pipeline flow:
  Data → Features → Train Models → Generate Signals → Backtest → Report

Walk-forward training:
  Instead of training once and testing on a held-out period, we simulate
  what a practitioner would actually do: periodically re-train the model
  as new data arrives. This is called "walk-forward" or "rolling window" analysis.

  [─────Train─────][Test]
           [─────Train─────][Test]
                    [─────Train─────][Test]
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

# Path setup 
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from data.data_loader import DataLoader
from features.feature_engineer import FeatureEngineer
from models.random_forest_model import RandomForestModel
from models.xgboost_model import XGBoostModel
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from signals.signal_generator import SignalGenerator
from backtest.backtester import Backtester, BacktestConfig
from utils.metrics import PerformanceAnalyzer



# Configuration

CONFIG = {
    # Data
    "ticker":     "SPY",         # S&P 500 ETF: liquid, well-behaved
    "start_date": "2015-01-01",
    "end_date":   "2024-12-31",

    # Train/test split
    "train_ratio":  0.70,        # 70% train, 30% test
    "forward_days": 1,           # Predict 1-day ahead

    # Models to run (set False to skip slow deep learning models)
    "use_rf":          True,
    "use_xgb":         True,
    "use_lstm":        True,     # Set False if no GPU
    "use_transformer": True,     # Set False if no GPU

    # Signal generation
    "signal_low_threshold":  0.10,  # Below this → flat
    "signal_high_threshold": 0.25,  # Above this → full size

    # Ensemble weights [RF, XGB, LSTM, Transformer]
    "ensemble_weights": [0.30, 0.35, 0.20, 0.15],

    # Backtest
    "initial_capital":    100_000,
    "commission_pct":     0.001,   # 0.1% per trade
    "slippage_pct":       0.0005,  # 0.05% slippage
    "max_drawdown_limit": 0.25,    # Stop if down 25%
}



# Main Pipeline
def main():
    print("\n" + "═" * 60)
    print("  ML FINANCIAL SIGNAL GENERATION PIPELINE")
    print("═" * 60)

    # STEP 1: Fetch Data 
    print("\n[Step 1/6] Fetching market data...")
    loader = DataLoader(cache_dir=str(ROOT / "cache"))
    df = loader.fetch(CONFIG["ticker"], CONFIG["start_date"], CONFIG["end_date"])
    print(f"  Loaded {len(df)} trading days for {CONFIG['ticker']}")
    print(f"  Date range: {df.index[0].date()} → {df.index[-1].date()}")

    # STEP 2: Feature Engineering 
    print("\n[Step 2/6] Engineering features...")
    fe = FeatureEngineer(lags=[1, 2, 3, 5, 10, 21])
    X, y = fe.build(df, forward_periods=CONFIG["forward_days"])
    print(f"  Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")

    # STEP 3: Train/Test Split
    print("\n[Step 3/6] Splitting data (temporal — no shuffling)...")
    split_idx = int(len(X) * CONFIG["train_ratio"])
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"  Train: {X_train.index[0].date()} → {X_train.index[-1].date()} ({len(X_train)} days)")
    print(f"  Test:  {X_test.index[0].date()}  → {X_test.index[-1].date()}  ({len(X_test)} days)")
    print(f"  Train target distribution: {y_train.value_counts().to_dict()}")

    # STEP 4: Train Models
    print("\n[Step 4/6] Training models...")
    signals = {}

    # 4a. Random Forest
    if CONFIG["use_rf"]:
        print("\n  ─── Random Forest ───")
        rf = RandomForestModel(n_estimators=300, max_depth=6, min_samples_leaf=40)
        rf.train(X_train, y_train)
        rf_sig = rf.get_signal_strength(X_test)
        signals["rf"] = rf_sig
        print(f"  RF Signal: mean={rf_sig.mean():.4f}  std={rf_sig.std():.4f}")

    # 4b. XGBoost
    if CONFIG["use_xgb"]:
        print("\n  ─── XGBoost ───")
        xgb = XGBoostModel(n_estimators=400, max_depth=4, learning_rate=0.05)
        xgb.train(X_train, y_train)
        xgb_sig = xgb.get_signal_strength(X_test)
        signals["xgb"] = xgb_sig
        print(f"  XGB Signal: mean={xgb_sig.mean():.4f}  std={xgb_sig.std():.4f}")

    # 4c. LSTM 
    if CONFIG["use_lstm"]:
        print("\n  ─── LSTM ───")
        lstm = LSTMModel(
            seq_len=20, hidden_size=128, num_layers=2,
            dropout=0.3, lr=1e-3, batch_size=64,
            epochs=60, patience=10,
        )
        lstm.train(X_train, y_train)
        lstm_sig = lstm.get_signal_strength(X_test)
        signals["lstm"] = lstm_sig
        print(f"  LSTM Signal: mean={lstm_sig.dropna().mean():.4f}  std={lstm_sig.dropna().std():.4f}")

    # 4d. Transformer 
    if CONFIG["use_transformer"]:
        print("\n  ─── Transformer ───")
        trf = TransformerModel(
            seq_len=30, d_model=64, nhead=4, num_layers=2,
            dropout=0.1, lr=5e-4, batch_size=64,
            epochs=60, patience=10,
        )
        trf.train(X_train, y_train)
        trf_sig = trf.get_signal_strength(X_test)
        signals["transformer"] = trf_sig
        print(f"  TRF Signal: mean={trf_sig.dropna().mean():.4f}  std={trf_sig.dropna().std():.4f}")

    # STEP 5: Generate Ensemble Signal
    print("\n[Step 5/6] Generating ensemble signal...")
    sg = SignalGenerator()

    active_models = list(signals.keys())
    for name, sig in signals.items():
        sg.add_signal(name, sig)

    # Compute ensemble weights based on which models are active
    all_weights  = {"rf": 0.30, "xgb": 0.35, "lstm": 0.20, "transformer": 0.15}
    used_weights = [all_weights[m] for m in active_models]
    total = sum(used_weights)
    used_weights = [w / total for w in used_weights]  # Renormalize

    composite = sg.composite_signal(method="weighted", weights=used_weights)
    positions  = sg.threshold_signal(
        composite,
        low=CONFIG["signal_low_threshold"],
        high=CONFIG["signal_high_threshold"],
    )

    print(f"\n  Ensemble weights: {dict(zip(active_models, [f'{w:.2f}' for w in used_weights]))}")
    print(f"  Composite signal: mean={composite.mean():.4f}  std={composite.std():.4f}")
    print(f"  Position distribution:\n{positions.value_counts().sort_index().to_string()}")

    # IC Analysis 
    forward_ret = df["returns"].shift(-1).loc[X_test.index]
    print(f"\n  Information Coefficient (IC):")
    sg.information_coefficient(composite, forward_ret)

    # STEP 6: Backtest
    print("\n[Step 6/6] Running backtest...")
    bt_config = BacktestConfig(
        initial_capital=CONFIG["initial_capital"],
        commission_pct=CONFIG["commission_pct"],
        slippage_pct=CONFIG["slippage_pct"],
        max_drawdown_limit=CONFIG["max_drawdown_limit"],
        signal_lag=1,
    )
    bt       = Backtester(bt_config)
    test_prices = df.loc[X_test.index]
    portfolio   = bt.run(test_prices, positions)

    # Print summary 
    metrics = bt.print_summary(portfolio)

    # Full analytics report 
    bench_ret = df["returns"].loc[X_test.index]
    analyzer  = PerformanceAnalyzer(portfolio, bench_ret)
    analyzer.full_report()

    # Generate charts 
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    chart_path = str(output_dir / f"{CONFIG['ticker']}_performance.png")
    analyzer.plot_all(save_path=chart_path)

    print(f"\n Pipeline complete! Results saved to: {output_dir}")
    print(f"   Performance chart: {chart_path}")

    return portfolio, metrics



# Walk-forward Backtesting 
def walk_forward_backtest(
    ticker: str = "SPY",
    start: str  = "2015-01-01",
    end: str    = "2024-12-31",
    train_window: int = 504,   # 2 years of training data
    test_window:  int = 63,    # 3 months of testing
    step:         int = 63,    # Retrain every 3 months
):
    """
    Walk-forward analysis: retrain models periodically and stitch together
    the out-of-sample predictions to form a full backtest.

    This is the gold standard for evaluating ML trading strategies because:
    1. Models adapt to regime changes (retrained on recent data)
    2. True out-of-sample: each day's signal is from a model that hadn't seen that day's data
    3. Better reflects live trading conditions

    Parameters:
    train_window : int
        Number of days in each training window
    test_window : int
        Number of days in each test window (before retraining)
    step : int
        Days between retraining events
    """
    print(f"\n{'═'*60}")
    print(f"  WALK-FORWARD ANALYSIS: {ticker}")
    print(f"  Train={train_window}d, Test={test_window}d, Step={step}d")
    print(f"{'═'*60}")

    loader = DataLoader(cache_dir="./cache")
    df     = loader.fetch(ticker, start, end)
    fe     = FeatureEngineer()
    X, y   = fe.build(df)

    all_signals = pd.Series(dtype=float)

    # Rolling windows
    i = train_window
    fold = 0
    while i + test_window <= len(X):
        fold += 1
        train_slice = slice(i - train_window, i)
        test_slice  = slice(i, i + test_window)

        X_tr = X.iloc[train_slice]
        y_tr = y.iloc[train_slice]
        X_te = X.iloc[test_slice]

        print(f"\n── Fold {fold}: Train [{X_tr.index[0].date()}→{X_tr.index[-1].date()}] "
              f"Test [{X_te.index[0].date()}→{X_te.index[-1].date()}]")

        # Train quick models (RF + XGB, skip deep learning for speed)
        rf = RandomForestModel(n_estimators=200, max_depth=5)
        rf.train(X_tr, y_tr)
        rf_sig = rf.get_signal_strength(X_te)

        xgb_m = XGBoostModel(n_estimators=200, max_depth=4)
        xgb_m.train(X_tr, y_tr)
        xgb_sig = xgb_m.get_signal_strength(X_te)

        # Ensemble
        sg = SignalGenerator()
        sg.add_signal("rf",  rf_sig)
        sg.add_signal("xgb", xgb_sig)
        composite = sg.composite_signal(method="equal")
        all_signals = pd.concat([all_signals, composite])

        i += step  # Move window forward

    # Backtest stitched signals
    if len(all_signals) == 0:
        print("No signals generated. Increase date range or decrease window sizes.")
        return None

    positions = SignalGenerator().threshold_signal(all_signals, low=0.1, high=0.25)
    test_prices = df.loc[all_signals.index]

    bt     = Backtester(BacktestConfig(initial_capital=100_000))
    port   = bt.run(test_prices, positions)
    metrics = bt.print_summary(port)

    bench = df["returns"].loc[all_signals.index]
    analyzer = PerformanceAnalyzer(port, bench)
    analyzer.full_report()
    analyzer.plot_all(save_path=f"outputs/{ticker}_wf_performance.png")

    return port, metrics



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ML Financial Signal Pipeline")
    parser.add_argument("--mode", choices=["full", "walkforward"], default="full",
                        help="'full' = single train/test split; 'walkforward' = rolling windows")
    parser.add_argument("--ticker", default=CONFIG["ticker"])
    parser.add_argument("--start",  default=CONFIG["start_date"])
    parser.add_argument("--end",    default=CONFIG["end_date"])
    args = parser.parse_args()

    CONFIG["ticker"]     = args.ticker
    CONFIG["start_date"] = args.start
    CONFIG["end_date"]   = args.end

    if args.mode == "full":
        main()
    elif args.mode == "walkforward":
        walk_forward_backtest(ticker=args.ticker, start=args.start, end=args.end)