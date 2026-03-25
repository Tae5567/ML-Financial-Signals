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

from data.data_loader import DataLoader
from features.feature_engineer import FeatureEngineer
from models.random_forest_model import RandomForestModel
from models.xgboost_model import XGBoostModel
from models.lstm_model import LSTMModel
from models.transformer_model import TransformerModel
from signals.signal_generator import SignalGenerator
from backtest.backtester import Backtester, BacktestConfig
from utils.metrics import PerformanceAnalyzer


warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


CONFIG = {
    "ticker":     "SPY",
    "start_date": "2015-01-01",
    "end_date":   "2024-12-31",
    "train_ratio":  0.70,
    "forward_days": 1,

    # Set use_lstm and use_transformer to False for a fast run (RF + XGB only, ~1-2 min)
    # Set to True for the full run (~15-25 min on CPU)
    "use_rf":          True,
    "use_xgb":         True,
    "use_lstm":        False,
    "use_transformer": False,

    "signal_low_threshold":  0.10,
    "signal_high_threshold": 0.25,
    "ensemble_weights": [0.30, 0.35, 0.20, 0.15],

    "initial_capital":    100_000,
    "commission_pct":     0.001,
    "slippage_pct":       0.0005,
    "max_drawdown_limit": 0.25,
}


def main():
    print("\n" + "=" * 60)
    print("  ML FINANCIAL SIGNAL GENERATION PIPELINE")
    print("=" * 60)

    # STEP 1: Fetch Data 
    print("\n[Step 1/6] Fetching market data...")
    loader = DataLoader(cache_dir=str(ROOT / "cache"))
    df = loader.fetch(CONFIG["ticker"], CONFIG["start_date"], CONFIG["end_date"])
    print(f"  Loaded {len(df)} trading days for {CONFIG['ticker']}")
    print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")

    # STEP 2: Feature Engineering
    print("\n[Step 2/6] Engineering features...")
    fe = FeatureEngineer(lags=[1, 2, 3, 5, 10, 21])
    X, y = fe.build(df, forward_periods=CONFIG["forward_days"])
    print(f"  Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")

    # STEP 3: Train/Test Split
    print("\n[Step 3/6] Splitting data (no shuffling, time order preserved)...")
    split_idx = int(len(X) * CONFIG["train_ratio"])
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"  Train: {X_train.index[0].date()} to {X_train.index[-1].date()} ({len(X_train)} days)")
    print(f"  Test:  {X_test.index[0].date()} to {X_test.index[-1].date()} ({len(X_test)} days)")

    # STEP 4: Train Models
    print("\n[Step 4/6] Training models...")
    signals = {}

    if CONFIG["use_rf"]:
        print("\n  Random Forest ")
        rf = RandomForestModel(n_estimators=300, max_depth=6, min_samples_leaf=40)
        rf.train(X_train, y_train)
        rf_sig = rf.get_signal_strength(X_test)
        signals["rf"] = rf_sig
        print(f"  RF Signal: mean={rf_sig.mean():.4f}  std={rf_sig.std():.4f}")

    if CONFIG["use_xgb"]:
        print("\n  XGBoost ")
        xgb = XGBoostModel(n_estimators=400, max_depth=4, learning_rate=0.05)
        xgb.train(X_train, y_train)
        xgb_sig = xgb.get_signal_strength(X_test)
        signals["xgb"] = xgb_sig
        print(f"  XGB Signal: mean={xgb_sig.mean():.4f}  std={xgb_sig.std():.4f}")

    if CONFIG["use_lstm"]:
        print("\n  LSTM (slow on CPU, ~5-10 min, prints progress every 5 epochs) ")
        lstm = LSTMModel(
            seq_len=20, hidden_size=64, num_layers=2,
            dropout=0.3, lr=1e-3, batch_size=64,
            epochs=30, patience=7,
        )
        lstm.train(X_train, y_train)
        lstm_sig = lstm.get_signal_strength(X_test)
        signals["lstm"] = lstm_sig
        print(f"  LSTM done. Signal: mean={lstm_sig.dropna().mean():.4f}  std={lstm_sig.dropna().std():.4f}")

    if CONFIG["use_transformer"]:
        print("\n  Transformer (slow on CPU, ~5 min, prints progress every 5 epochs) ")
        trf = TransformerModel(
            seq_len=30, d_model=64, nhead=4, num_layers=2,
            dropout=0.1, lr=5e-4, batch_size=64,
            epochs=30, patience=7,
        )
        trf.train(X_train, y_train)
        trf_sig = trf.get_signal_strength(X_test)
        signals["transformer"] = trf_sig
        print(f"  Transformer done. Signal: mean={trf_sig.dropna().mean():.4f}  std={trf_sig.dropna().std():.4f}")

    # STEP 5: Generate Ensemble Signal
    print("\n[Step 5/6] Generating ensemble signal...")
    sg = SignalGenerator()

    active_models = list(signals.keys())
    for name, sig in signals.items():
        sg.add_signal(name, sig)

    all_weights  = {"rf": 0.30, "xgb": 0.35, "lstm": 0.20, "transformer": 0.15}
    used_weights = [all_weights[m] for m in active_models]
    total = sum(used_weights)
    used_weights = [w / total for w in used_weights]

    composite = sg.composite_signal(method="weighted", weights=used_weights)
    positions  = sg.threshold_signal(
        composite,
        low=CONFIG["signal_low_threshold"],
        high=CONFIG["signal_high_threshold"],
    )

    print(f"\n  Ensemble weights: {dict(zip(active_models, [f'{w:.2f}' for w in used_weights]))}")
    print(f"  Composite signal: mean={composite.mean():.4f}  std={composite.std():.4f}")
    print(f"  Position distribution:\n{positions.value_counts().sort_index().to_string()}")

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
    bt          = Backtester(bt_config)
    test_prices = df.loc[X_test.index]
    portfolio   = bt.run(test_prices, positions)

    metrics = bt.print_summary(portfolio)

    bench_ret = df["returns"].loc[X_test.index]
    analyzer  = PerformanceAnalyzer(portfolio, bench_ret)
    analyzer.full_report()

    # Save outputs
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)

    # Performance chart
    chart_path = str(output_dir / f"{CONFIG['ticker']}_performance.png")
    analyzer.plot_all(save_path=chart_path)

    # Signal debug CSV (open in Excel/Sheets to inspect day-by-day)
    signal_df = pd.DataFrame({
        "close":     df["Close"].loc[X_test.index],
        "composite": composite,
        "position":  positions,
        "daily_ret": portfolio["net_return"],
        "portfolio": portfolio["portfolio_value"],
    })
    for name, sig in signals.items():
        signal_df[f"{name}_signal"] = sig

    csv_path = str(output_dir / f"{CONFIG['ticker']}_signals_debug.csv")
    signal_df.to_csv(csv_path)

    print(f"\n{'=' * 60}")
    print(f"  DONE. Outputs saved to: {output_dir}/")
    print(f"  - {CONFIG['ticker']}_performance.png  (charts)")
    print(f"  - {CONFIG['ticker']}_signals_debug.csv (day-by-day signal data)")
    print(f"{'=' * 60}")

    return portfolio, metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ML Financial Signal Pipeline")
    parser.add_argument("--ticker", default=CONFIG["ticker"])
    parser.add_argument("--start",  default=CONFIG["start_date"])
    parser.add_argument("--end",    default=CONFIG["end_date"])
    parser.add_argument("--fast",   action="store_true",
                        help="Skip LSTM and Transformer for a quick ~2 min run")
    args = parser.parse_args()

    CONFIG["ticker"]     = args.ticker
    CONFIG["start_date"] = args.start
    CONFIG["end_date"]   = args.end

    if args.fast:
        CONFIG["use_lstm"]        = False
        CONFIG["use_transformer"] = False
        print("  [Fast mode] Skipping LSTM and Transformer")

    main()