"""
utils/metrics.py

Quantitative finance has a rich vocabulary for measuring strategy quality
This module provides all the standard metrics plus visualization

Key metrics explained:
- Sharpe Ratio: excess return per unit of total risk (higher = better, >1 is good)
- Sortino Ratio: like Sharpe but only penalizes downside volatility
- Calmar Ratio: annualized return / max drawdown
- Information Ratio: active return / tracking error (vs benchmark)
- Max Drawdown: worst peak-to-trough decline (lower = better)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats

# Comprehensive performance analytics for backtested strategies
class PerformanceAnalyzer:
    """
    Usage:
        analyzer = PerformanceAnalyzer(portfolio, benchmark_returns)
        analyzer.plot_all("results.png")
    """

    def __init__( self, portfolio: pd.DataFrame, benchmark_returns: pd.Series = None, risk_free_rate: float = 0.04,trading_days: int = 252 ):
        """
        Parameters:
        portfolio : pd.DataFrame
            Output from Backtester.run()
        benchmark_returns : pd.Series
            Buy-and-hold daily returns for comparison
        risk_free_rate : float
            Annual risk-free rate (US T-bill)
        trading_days : int
            Trading days per year (252 for equities)
        """
        self.portfolio  = portfolio
        self.bench_ret  = benchmark_returns
        self.rfr        = risk_free_rate / trading_days  # Daily risk-free
        self.td         = trading_days

        self.ret = portfolio["net_return"]


    # Metrics
    # Sharpe ratio
    def sharpe_ratio(self, ret: pd.Series = None) -> float:
        """
        Sharpe = (mean_return - risk_free) / std_return x √252

        Interpretation:
          < 0    → strategy loses vs risk-free
          0-1    → mediocre, possible with high transaction costs
          1-2    → good
          > 2    → excellent (institutional hedge funds target 1-3)
        """
        r = ret if ret is not None else self.ret
        excess = r - self.rfr
        return excess.mean() / (excess.std() + 1e-9) * np.sqrt(self.td)

    # Sortino ratio
    def sortino_ratio(self, ret: pd.Series = None) -> float:
        """
        Sortino = (mean_return - risk_free) / downside_std x √252

        Better than Sharpe for asymmetric return distributions because it doesn't penalize upside volatility
        """
        r = ret if ret is not None else self.ret
        excess = r - self.rfr
        downside = r[r < 0]
        return excess.mean() / (downside.std() + 1e-9) * np.sqrt(self.td)

    # Max Drawdown = min((value - rolling_max) / rolling_max)
    def max_drawdown(self, values: pd.Series = None) -> float:
        """
        Represents the worst loss an investor would have experienced
        A drawdown of -0.20 means the strategy fell 20% from its peak
        """
        v = values if values is not None else self.portfolio["portfolio_value"]
        rolling_max = v.cummax()
        drawdown = (v - rolling_max) / rolling_max
        return drawdown.min()
    

    # Calmar = Annualized Return / |Max Drawdown|; Higher = better
    def calmar_ratio(self) -> float:
        ann_ret = self.annualized_return()
        mdd     = abs(self.max_drawdown()) + 1e-9
        return ann_ret / mdd

    # Annualized return via compounding
    def annualized_return(self, ret: pd.Series = None) -> float:
        r = ret if ret is not None else self.ret
        total   = (1 + r).prod()
        n_years = len(r) / self.td
        return total ** (1 / n_years) - 1 if n_years > 0 else 0

    # IR = (strategy_return - benchmark_return) / tracking_error
    def information_ratio(self) -> float:
        """
        Measures how much alpha the strategy generates per unit of active risk taken relative to the benchmark
        """
        if self.bench_ret is None:
            return np.nan
        aligned = pd.DataFrame({
            "strat": self.ret,
            "bench": self.bench_ret,
        }).dropna()
        active = aligned["strat"] - aligned["bench"]
        return active.mean() / (active.std() + 1e-9) * np.sqrt(self.td)

    # Value at Risk (VaR) and Conditional VaR (CVaR/Expected Shortfall)
    def var_cvar(self, confidence: float = 0.95) -> tuple:
        """
        VaR(95%) = worst daily loss we'd expect in 95% of days
        CVaR(95%) = average loss on the worst 5% of days
        """
        var  = np.percentile(self.ret, (1 - confidence) * 100)
        cvar = self.ret[self.ret <= var].mean()
        return var, cvar

    # CAPM Beta and Jensen's Alpha.
    def beta_alpha(self) -> tuple:
        """
        Beta:  sensitivity to market moves (1.0 = moves with market)
        Alpha: excess return unexplained by market exposure
        """
        if self.bench_ret is None:
            return np.nan, np.nan
        aligned = pd.DataFrame({
            "strat": self.ret,
            "bench": self.bench_ret,
        }).dropna()
        slope, intercept, r, p, se = stats.linregress(
            aligned["bench"], aligned["strat"]
        )
        beta  = slope
        alpha = intercept * self.td  # Annualized
        return beta, alpha

    # Compute and print all metrics
    def full_report(self) -> dict:

        var95, cvar95 = self.var_cvar(0.95)
        beta, alpha   = self.beta_alpha()
        ir            = self.information_ratio()

        metrics = {
            "Sharpe Ratio":       self.sharpe_ratio(),
            "Sortino Ratio":      self.sortino_ratio(),
            "Calmar Ratio":       self.calmar_ratio(),
            "Annualized Return":  self.annualized_return(),
            "Max Drawdown":       self.max_drawdown(),
            "Information Ratio":  ir,
            "Beta":               beta,
            "Alpha (ann)":        alpha,
            "VaR (95%)":          var95,
            "CVaR (95%)":         cvar95,
        }

        print("\n Full Performance Report ")
        for name, val in metrics.items():
            if isinstance(val, float) and not np.isnan(val):
                if "Return" in name or "Drawdown" in name or "VaR" in name:
                    print(f"  {name:<25} {val:>10.2%}")
                else:
                    print(f"  {name:<25} {val:>10.4f}")
            else:
                print(f"  {name:<25} {'N/A':>10}")
        return metrics


    # Visualization

    def plot_all(self, save_path: str = None, figsize=(18, 22)):
        """
        Generate a comprehensive 6-panel performance report:
        1. Cumulative return vs benchmark
        2. Drawdown over time
        3. Return distribution
        4. Rolling Sharpe ratio
        5. Monthly return heatmap
        6. Position distribution
        """
        fig = plt.figure(figsize=figsize)
        fig.patch.set_facecolor("#0d1117")

        # Color scheme
        STRAT  = "#00d4aa"
        BNH    = "#ff6b6b"
        ACCENT = "#ffd700"
        GRAY   = "#555555"
        TEXT   = "#c9d1d9"

        plt.rcParams.update({
            "text.color":       TEXT,
            "axes.labelcolor":  TEXT,
            "xtick.color":      TEXT,
            "ytick.color":      TEXT,
            "axes.facecolor":   "#161b22",
            "figure.facecolor": "#0d1117",
            "grid.color":       "#30363d",
        })

        # 1. Cumulative Returns
        ax1 = fig.add_subplot(4, 2, (1, 2))
        port_val = self.portfolio["portfolio_value"]
        bnh_val  = self.portfolio["bnh_value"]
        init     = port_val.iloc[0]

        ax1.fill_between(port_val.index,
                         port_val / init * 100,
                         100,
                         alpha=0.15, color=STRAT)
        ax1.plot(port_val.index, port_val / init * 100,
                 color=STRAT, linewidth=2, label="ML Strategy")
        ax1.plot(bnh_val.index,  bnh_val  / init * 100,
                 color=BNH, linewidth=1.5, alpha=0.8, linestyle="--", label="Buy & Hold")

        ax1.axhline(100, color=GRAY, linestyle=":", linewidth=1)
        ax1.set_title("Cumulative Return (Base = 100)", color=TEXT, fontsize=14, pad=12)
        ax1.set_ylabel("Portfolio Value (Base 100)", color=TEXT)
        ax1.legend(loc="upper left", framealpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax1.grid(alpha=0.3)

        # Annotate final values
        final_strat = port_val.iloc[-1] / init * 100
        final_bnh   = bnh_val.iloc[-1] / init * 100
        ax1.annotate(f"{final_strat:.1f}", xy=(port_val.index[-1], final_strat),
                     xytext=(10, 0), textcoords="offset points", color=STRAT, fontsize=11)
        ax1.annotate(f"{final_bnh:.1f}", xy=(bnh_val.index[-1], final_bnh),
                     xytext=(10, 0), textcoords="offset points", color=BNH, fontsize=11)

        # 2. Drawdown
        ax2 = fig.add_subplot(4, 2, (3, 4))
        dd = self.portfolio["drawdown"]
        ax2.fill_between(dd.index, dd * 100, 0,
                         alpha=0.6, color="#ff4444", label="Strategy Drawdown")
        ax2.set_title("Drawdown (%)", color=TEXT, fontsize=14, pad=12)
        ax2.set_ylabel("Drawdown (%)", color=TEXT)
        ax2.set_ylim(min(dd.min() * 100 * 1.2, -1), 2)
        ax2.axhline(0, color=GRAY, linewidth=0.8)
        ax2.grid(alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

        mdd_date = dd.idxmin()
        ax2.annotate(f"Max DD: {dd.min():.1%}",
                     xy=(mdd_date, dd.min() * 100),
                     xytext=(0, -20), textcoords="offset points",
                     color=ACCENT, fontsize=10, ha="center")

        # 3. Return Distribution 
        ax3 = fig.add_subplot(4, 2, 5)
        ret_pct = self.ret * 100
        ax3.hist(ret_pct, bins=60, color=STRAT, alpha=0.7, edgecolor="none",
                 label="Strategy")
        if self.bench_ret is not None:
            ax3.hist(self.bench_ret * 100, bins=60, color=BNH, alpha=0.5,
                     edgecolor="none", label="Benchmark")
        ax3.axvline(ret_pct.mean(), color=ACCENT, linewidth=2, label=f"Mean: {ret_pct.mean():.3f}%")
        ax3.set_title("Daily Return Distribution", color=TEXT, fontsize=13, pad=10)
        ax3.set_xlabel("Daily Return (%)", color=TEXT)
        ax3.legend(framealpha=0.3)
        ax3.grid(alpha=0.3, axis="y")

        # 4. Rolling Sharpe (63-day) 
        ax4 = fig.add_subplot(4, 2, 6)
        window = 63
        rolling_sharpe = (
            (self.ret.rolling(window).mean() - self.rfr)
            / self.ret.rolling(window).std()
            * np.sqrt(self.td)
        )
        ax4.plot(rolling_sharpe.index, rolling_sharpe, color=ACCENT, linewidth=1.5)
        ax4.fill_between(rolling_sharpe.index, rolling_sharpe, 0,
                         where=rolling_sharpe > 0, alpha=0.2, color=STRAT)
        ax4.fill_between(rolling_sharpe.index, rolling_sharpe, 0,
                         where=rolling_sharpe < 0, alpha=0.2, color=BNH)
        ax4.axhline(0, color=GRAY, linewidth=1)
        ax4.axhline(1, color=STRAT, linewidth=0.8, linestyle="--", alpha=0.5, label="Sharpe=1")
        ax4.set_title("Rolling 63-Day Sharpe Ratio", color=TEXT, fontsize=13, pad=10)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax4.legend(framealpha=0.3)
        ax4.grid(alpha=0.3)

        # 5. Monthly Return Heatmap 
        ax5 = fig.add_subplot(4, 2, (7, 8))
        monthly = self.ret.resample("ME").sum()
        monthly_pivot = pd.DataFrame({
            "year":  monthly.index.year,
            "month": monthly.index.month,
            "ret":   monthly.values,
        }).pivot(index="year", columns="month", values="ret")

        month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"]

        sns.heatmap(
            monthly_pivot * 100,
            annot=True, fmt=".1f",
            cmap="RdYlGn", center=0,
            ax=ax5,
            linewidths=0.5,
            cbar_kws={"label": "Return (%)"},
            xticklabels=month_labels[:monthly_pivot.shape[1]],
        )
        ax5.set_title("Monthly Returns Heatmap (%)", color=TEXT, fontsize=13, pad=10)
        ax5.set_xlabel("Month", color=TEXT)
        ax5.set_ylabel("Year", color=TEXT)

        plt.suptitle("ML Trading Strategy — Performance Report",
                     fontsize=16, color=TEXT, y=1.01)
        plt.tight_layout(pad=2.5)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"[Metrics] Chart saved to: {save_path}")
        else:
            plt.show()

        plt.close()
        return fig



if __name__ == "__main__":
    import sys; sys.path.insert(0, "..")
    from data.data_loader import DataLoader
    from backtest.backtester import Backtester, BacktestConfig

    loader = DataLoader()
    prices = loader.fetch("AAPL", "2020-01-01", "2024-01-01")

    np.random.seed(0)
    positions = pd.Series(
        np.random.choice([-1, 0, 1], size=len(prices), p=[0.25, 0.5, 0.25]),
        index=prices.index,
    ).astype(float)

    bt = Backtester(BacktestConfig())
    portfolio = bt.run(prices, positions)
    bench_ret = prices["Close"].pct_change().fillna(0)

    analyzer = PerformanceAnalyzer(portfolio, bench_ret)
    analyzer.full_report()
    analyzer.plot_all("test_output.png")