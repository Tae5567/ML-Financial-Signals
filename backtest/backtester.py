"""
backtest/backtester.py

A backtest simulates what would have happened if you had traded using
your ML signals historically. The key word is "REALISTIC" — naive backtests
overestimate performance by ignoring real-world frictions.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass 
class BacktestConfig:
    # All parameters for a backtest run
    initial_capital: float = 100_000.0   # Starting portfolio value
    commission_pct: float  = 0.001       # 0.1% per trade (both sides)
    slippage_pct: float    = 0.0005      # 0.05% execution slippage
    max_position_size: float = 1.0       # Max fraction of capital in one position
    max_drawdown_limit: float = 0.20     # Stop trading if DD exceeds 20%
    signal_lag: int = 1                  # Days delay: signal at t → trade at t+1
    use_log_returns: bool = True         # Use log returns (more accurate for compounding)


# Event driven vectorized backtester
class Backtester:
    """
    Usage:
        bt = Backtester(config)
        results = bt.run(prices_df, positions_series)
        bt.print_summary(results)
    """

    def __init__(self, config: BacktestConfig = None):
        self.config = config or BacktestConfig()

    # Run the backtest
    """
        Parameters:
        prices : pd.DataFrame
            OHLCV DataFrame (must include price_col and 'Open')
        positions : pd.Series
            Target positions in {-1, -0.5, 0, 0.5, 1}
            Index must align with prices.
        price_col : str
            Column to use for mark-to-market P&L (typically 'Close')

        Returns:
        pd.DataFrame
            Daily portfolio state:
            - position, returns, gross_returns, costs, portfolio_value, drawdown
    """
    def run( self, prices: pd.DataFrame, positions: pd.Series, price_col: str = "Close" ) -> pd.DataFrame:

        cfg = self.config

        # Align data
        common_idx = prices.index.intersection(positions.index)
        prices_    = prices.loc[common_idx].copy()
        positions_ = positions.loc[common_idx].copy()

        # Signal lag: shift positions forward by signal_lag days
        positions_ = positions_.shift(cfg.signal_lag).fillna(0)

        # Execution price: use Open of next day (not Close that generated signal)
        exec_prices = prices_["Open"].shift(-1).ffill()

        # Calculate daily returns on the underlying
        raw_returns = prices_[price_col].pct_change().fillna(0)

        # Portfolio simulation
        portfolio = pd.DataFrame(index=common_idx)
        portfolio["position"]      = positions_
        portfolio["price"]         = prices_[price_col]
        portfolio["raw_return"]    = raw_returns

        # Position changes (used to calculate transaction costs)
        portfolio["pos_change"] = portfolio["position"].diff().abs().fillna(0)

        # Transaction costs
        # Cost applied when position changes (entering, exiting or sizing)
        # Total cost = commission + slippage
        total_cost_pct = cfg.commission_pct + cfg.slippage_pct
        portfolio["cost_pct"] = portfolio["pos_change"] * total_cost_pct

        # Strategy returns
        # Gross return = position × underlying return
        # Net return   = gross return − transaction costs
        portfolio["gross_return"] = portfolio["position"] * portfolio["raw_return"]
        portfolio["net_return"]   = portfolio["gross_return"] - portfolio["cost_pct"]

        # Portfolio value (compounded) 
        portfolio["portfolio_value"] = cfg.initial_capital * (
            1 + portfolio["net_return"]
        ).cumprod()

        # Drawdown
        # Drawdown = (current_value - peak_value) / peak_value
        rolling_max = portfolio["portfolio_value"].cummax()
        portfolio["drawdown"] = (portfolio["portfolio_value"] - rolling_max) / rolling_max

        # Max drawdown circuit breaker
        # If drawdown exceeds limit, force positions to zero
        dd_breach = portfolio["drawdown"] < -cfg.max_drawdown_limit
        if dd_breach.any():
            first_breach = dd_breach.idxmax()
            portfolio.loc[first_breach:, "position"] = 0
            print(f"[Backtest] Max drawdown limit hit on {first_breach.date()}")

        # Benchmark (buy and hold)
        portfolio["bnh_value"] = cfg.initial_capital * (1 + raw_returns).cumprod()

        # Trade log
        portfolio["is_trade"] = portfolio["pos_change"] > 0

        self.portfolio_ = portfolio
        return portfolio

    # Print a comprehensive performance summary
    def print_summary(self, portfolio: pd.DataFrame = None):
        p = portfolio if portfolio is not None else self.portfolio_
        if p is None:
            raise RuntimeError("Run backtest first.")

        metrics = self._compute_metrics(p)

        print("\n" + "═" * 55)
        print("  BACKTEST RESULTS SUMMARY")
        print("═" * 55)
        print(f"  Period:              {p.index[0].date()} → {p.index[-1].date()}")
        print(f"  Trading Days:        {len(p):,}")
        print(f"  Initial Capital:     ${self.config.initial_capital:>12,.2f}")
        print(f"  Final Capital:       ${p['portfolio_value'].iloc[-1]:>12,.2f}")
        print(f"  B&H Final Capital:   ${p['bnh_value'].iloc[-1]:>12,.2f}")
        print("─" * 55)
        print(f"  Total Return:        {metrics['total_return']:>10.2%}")
        print(f"  B&H Return:          {metrics['bnh_return']:>10.2%}")
        print(f"  Annualized Return:   {metrics['ann_return']:>10.2%}")
        print(f"  Annualized Vol:      {metrics['ann_vol']:>10.2%}")
        print(f"  Sharpe Ratio:        {metrics['sharpe']:>10.4f}")
        print(f"  Sortino Ratio:       {metrics['sortino']:>10.4f}")
        print(f"  Calmar Ratio:        {metrics['calmar']:>10.4f}")
        print(f"  Max Drawdown:        {metrics['max_drawdown']:>10.2%}")
        print(f"  Max DD Duration:     {metrics['max_dd_days']} days")
        print("─" * 55)
        print(f"  Win Rate:            {metrics['win_rate']:>10.2%}")
        print(f"  Profit Factor:       {metrics['profit_factor']:>10.4f}")
        print(f"  Avg Win:             {metrics['avg_win']:>10.4%}")
        print(f"  Avg Loss:            {metrics['avg_loss']:>10.4%}")
        print(f"  Total Trades:        {metrics['n_trades']:>10,}")
        print(f"  Avg Holding Period:  {metrics['avg_hold']:.1f} days")
        print("═" * 55)

        return metrics

    # Compute all performance metrics from the portfolio DataFrame
    def _compute_metrics(self, p: pd.DataFrame) -> dict:
    
        ret = p["net_return"]
        trading_days = 252

        # Return metrics 
        total_return = p["portfolio_value"].iloc[-1] / self.config.initial_capital - 1
        bnh_return   = p["bnh_value"].iloc[-1] / self.config.initial_capital - 1
        n_years      = len(p) / trading_days
        ann_return   = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else 0

        # Risk metrics 
        ann_vol  = ret.std() * np.sqrt(trading_days)
        max_dd   = p["drawdown"].min()

        # Max drawdown duration
        in_drawdown = p["drawdown"] < 0
        if in_drawdown.any():
            dd_periods = in_drawdown.astype(int)
            max_dd_days = int(dd_periods.groupby((dd_periods != dd_periods.shift()).cumsum()).sum().max())
        else:
            max_dd_days = 0

        # Risk-adjusted returns 
        risk_free = 0.04 / trading_days  # ~4% annual risk-free rate, daily
        excess    = ret - risk_free

        sharpe  = excess.mean() / (excess.std() + 1e-9) * np.sqrt(trading_days)

        # Sortino: penalizes only downside volatility
        downside_std = ret[ret < 0].std()
        sortino = (ret.mean() - risk_free) / (downside_std + 1e-9) * np.sqrt(trading_days)

        # Calmar: annualized return / max drawdown (higher = better)
        calmar = ann_return / (abs(max_dd) + 1e-9)

        # Trade statistics
        active = p[p["position"] != 0]["gross_return"]
        win_rate = (active > 0).mean() if len(active) > 0 else 0

        wins   = active[active > 0]
        losses = active[active < 0]
        profit_factor = wins.sum() / (abs(losses.sum()) + 1e-9)
        avg_win  = wins.mean()  if len(wins)   > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0

        n_trades = int(p["is_trade"].sum())

        # Average holding period: number of consecutive non-zero positions
        pos_nonzero = (p["position"] != 0).astype(int)
        streaks = pos_nonzero.groupby((pos_nonzero != pos_nonzero.shift()).cumsum()).sum()
        avg_hold = streaks[streaks > 0].mean() if len(streaks[streaks > 0]) > 0 else 0

        return {
            "total_return":  total_return,
            "bnh_return":    bnh_return,
            "ann_return":    ann_return,
            "ann_vol":       ann_vol,
            "sharpe":        sharpe,
            "sortino":       sortino,
            "calmar":        calmar,
            "max_drawdown":  max_dd,
            "max_dd_days":   max_dd_days,
            "win_rate":      win_rate,
            "profit_factor": profit_factor,
            "avg_win":       avg_win,
            "avg_loss":      avg_loss,
            "n_trades":      n_trades,
            "avg_hold":      avg_hold,
        }



if __name__ == "__main__":
    import sys; sys.path.insert(0, "..")
    from data.data_loader import DataLoader

    loader = DataLoader()
    prices = loader.fetch("AAPL", "2020-01-01", "2024-01-01")

    # Simulate random positions for testing
    np.random.seed(42)
    positions = pd.Series(
        np.random.choice([-1, 0, 1], size=len(prices), p=[0.3, 0.4, 0.3]),
        index=prices.index,
        name="position",
    ).astype(float)

    bt = Backtester(BacktestConfig(initial_capital=100_000, commission_pct=0.001))
    portfolio = bt.run(prices, positions)
    bt.print_summary(portfolio)