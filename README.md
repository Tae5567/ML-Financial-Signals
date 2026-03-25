## Financial Signal Geeration with Machine Learning 

Predicts whether a stock will go up or down the next day using machine learning, then simulates trading on those predictions to measure performance.

## How it works

1. Downloads historical stock price/volume data (Yahoo Finance)
2. Extracts ~60 indicators from the raw data (momentum, volatility, volume trends, etc.)
3. Trains 4 ML models on those indicators: Random Forest, XGBoost, LSTM, Transformer
4. Combines all 4 predictions into one trading signal
5. Backtests the signal against historical data with realistic costs (fees, slippage)
6. Reports performance: Sharpe ratio, max drawdown, win rate, annualized return

## Stack

Python, scikit-learn, XGBoost, PyTorch, pandas, yfinance

## Run

```bash
pip install -r requirements.txt
python main.py
```

Results and charts are saved to `/outputs`.