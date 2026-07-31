#!/usr/bin/env python3
"""
Quantitative Mean-Reversion Backtesting Pipeline
Target Asset: BTC/USD 5-minute High-Frequency Data
Mathematical Models: Ornstein-Uhlenbeck Half-Life, Kalman Filter, EWMA Volatility
"""

import warnings
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore', category=RuntimeWarning)
sns.set_theme(style="darkgrid")

@dataclass
class Config:
    data_path: str = 'BTCUSDFeaturesdata_5m.csv'
    initial_capital: float = 15000.0
    maker_fee: float = 0.0001
    taker_fee: float = 0.0004
    slippage_bps: float = 0.00015
    
    # Risk Management & Sizing
    base_risk: float = 0.015
    max_leverage: float = 3.0
    max_drawdown: float = 0.20
    
    # Mathematical Model Parameters
    ou_window: int = 40
    vol_span: int = 20
    kalman_delta: float = 0.0001  # Tighter process noise for fast mean reversion
    
    # Edge Thresholds
    z_entry: float = 2.0          # Normalized entry cutoff
    z_exit: float = 0.1           # Target mean-reversion exit boundary
    min_half_life: float = 2.0    # Lower bound for half-life (bars)
    max_half_life: float = 30.0   # Upper bound for half-life (bars)
    
    # Monte Carlo Settings
    mc_simulations: int = 1000
    mc_horizon_trades: int = 100

    # In-Sample / Out-of-Sample split (chronological, most recent OOS)
    insample_years: float = 4.0
    outsample_years: float = 1.0
    bar_minutes: int = 5


# ==========================================
# 1. DATA LOADER
# ==========================================
class DataLoader:
    """Loads, cleans, and structures CSV time series data."""
    def __init__(self, c: Config):
        self.c = c

    def load(self) -> pd.DataFrame:
        print("[1/3] Loading CSV Data...")
        try:
            df = pd.read_csv(self.c.data_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"CRITICAL ERROR: Data file '{self.c.data_path}' not found.")
        
        required_cols = ['close']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"CRITICAL ERROR: Required column '{col}' missing from {self.c.data_path}")

        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.sort_values('timestamp', inplace=True)
            
        df.ffill(inplace=True)
        return df.reset_index(drop=True)

    def split_is_oos(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Chronological walk-forward split:
          In-Sample  : preceding N years (default 4)
          Out-of-Sample : most recent M years (default 1)
        """
        is_years = self.c.insample_years
        oos_years = self.c.outsample_years
        total_years = is_years + oos_years

        if "timestamp" in df.columns:
            end = df["timestamp"].max()
            oos_start = end - pd.DateOffset(years=oos_years)
            is_start = oos_start - pd.DateOffset(years=is_years)

            is_df = df[(df["timestamp"] >= is_start) & (df["timestamp"] < oos_start)].copy()
            oos_df = df[df["timestamp"] >= oos_start].copy()
        else:
            bars_per_year = int(365.25 * 24 * 60 / self.c.bar_minutes)
            is_bars = int(bars_per_year * is_years)
            oos_bars = int(bars_per_year * oos_years)
            min_bars = is_bars + oos_bars

            if len(df) < min_bars:
                raise ValueError(
                    f"CRITICAL ERROR: Need >= {total_years:.1f} years of data "
                    f"({min_bars:,} bars @ {self.c.bar_minutes}m), got {len(df):,}."
                )

            tail = df.iloc[-min_bars:].reset_index(drop=True)
            is_df = tail.iloc[:is_bars].copy()
            oos_df = tail.iloc[is_bars:].copy()

        if is_df.empty or oos_df.empty:
            raise ValueError(
                "CRITICAL ERROR: IS/OOS split produced an empty partition. "
                "Check timestamp coverage or extend history."
            )

        is_df.reset_index(drop=True, inplace=True)
        oos_df.reset_index(drop=True, inplace=True)
        return is_df, oos_df


# ==========================================
# 2. MATHEMATICAL STATISTICAL ENGINE
# ==========================================
class StatisticalModels:
    """Calculates Kalman Filter Residuals, OU Decay Half-Life, and EWMA Volatility."""
    def __init__(self, c: Config):
        self.c = c

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        print("      Computing continuous mathematical signals...")
        df = df.copy()
        p = df['close'].values.astype(float)
        
        self._log_returns(df, p)
        self._volatility(df)
        self._kalman_filter(df, p)
        self._ornstein_uhlenbeck(df)
        
        return df

    def _log_returns(self, df: pd.DataFrame, p: np.ndarray):
        lr = np.full(len(p), np.nan)
        lr[1:] = np.log(p[1:] / p[:-1])
        df['log_ret'] = lr

    def _volatility(self, df: pd.DataFrame):
        lr = df['log_ret'].values
        span = self.c.vol_span
        vol = pd.Series(lr).ewm(span=span, min_periods=span).std().values
        df['volatility'] = vol

    def _kalman_filter(self, df: pd.DataFrame, p: np.ndarray):
        """1D Online State Estimator to extract non-lagging smooth trend line."""
        n = len(p)
        d = self.c.kalman_delta
        xe, pe = p[0], 1.0
        dev = np.zeros(n)
        
        obs_noise = 0.05
        
        for i in range(1, n):
            xp, pp = xe, pe + d
            k = pp / (pp + obs_noise)
            xe = xp + k * (p[i] - xp)
            pe = (1 - k) * pp
            dev[i] = p[i] - xe
            
        df['kalman_mean'] = xe
        df['kalman_dev'] = dev

    def _ornstein_uhlenbeck(self, df: pd.DataFrame):
        """
        Estimates the mean-reverting speed (theta) and half-life (tau) 
        via discrete AR(1) OLS on Kalman residuals: dX_t = lambda * X_{t-1} + e_t
        """
        dev = df['kalman_dev'].values
        n = len(dev)
        w = self.c.ou_window
        
        ou_z = np.full(n, np.nan)
        half_life = np.full(n, np.nan)
        
        for i in range(w, n):
            s = dev[i-w+1:i+1]
            x_t = s[:-1]
            x_tp1 = s[1:]
            
            cov_m = np.cov(x_t, x_tp1)
            var_x = cov_m[0, 0]
            
            if var_x > 1e-12:
                b = cov_m[0, 1] / var_x
                if 0 < b < 1.0:
                    lmbda = b - 1.0
                    hl = -np.log(2) / lmbda
                    half_life[i] = hl
            
            std = np.std(s, ddof=1)
            if std > 1e-12:
                ou_z[i] = dev[i] / std
                
        df['ou_zscore'] = ou_z
        df['half_life'] = half_life


# ==========================================
# 3. SIGNAL GENERATOR (ANALYTICAL EDGE)
# ==========================================
class SignalEngine:
    """Generates signals strictly based on stationarity and extreme deviations."""
    def __init__(self, c: Config):
        self.c = c

    def generate(self, df: pd.DataFrame) -> np.ndarray:
        print("      Filtering signals via stationarity criteria...")
        n = len(df)
        sig = np.zeros(n, dtype=np.int8)
        
        z = df['ou_zscore'].values
        hl = df['half_life'].values
        
        for i in range(n):
            if np.isnan(z[i]) or np.isnan(hl[i]):
                continue
                
            is_mean_reverting = self.c.min_half_life <= hl[i] <= self.c.max_half_life
            
            if is_mean_reverting:
                if z[i] <= -self.c.z_entry:
                    sig[i] = 1   # Long Entry
                elif z[i] >= self.c.z_entry:
                    sig[i] = -1  # Short Entry
                    
        return sig


# ==========================================
# 4. VOLATILITY RISK MANAGER
# ==========================================
class RiskManager:
    """Handles Portfolio Risk, Drawdown Stops, and Dynamic Volatility Sizing."""
    def __init__(self, c: Config):
        self.c = c
        self.capital = c.initial_capital
        self.peak_capital = c.initial_capital

    def update_pnl(self, pnl: float):
        self.capital += pnl
        self.peak_capital = max(self.peak_capital, self.capital)

    def is_trading_allowed(self) -> bool:
        drawdown = (self.peak_capital - self.capital) / self.peak_capital
        return drawdown < self.c.max_drawdown

    def get_position_size(self, price: float, vol: float, z_score: float) -> tuple[float, float]:
        if not self.is_trading_allowed() or np.isnan(vol) or vol <= 0:
            return 0.0, 1.0
            
        edge = min(abs(z_score) / self.c.z_entry, 2.0)
        
        target_vol = 0.002
        vol_scalar = target_vol / max(vol, 0.0005)
        
        risk_fraction = self.c.base_risk * edge * vol_scalar
        risk_fraction = max(0.005, min(risk_fraction, 0.05))
        
        pos_value = self.capital * risk_fraction * self.c.max_leverage
        leverage = min(pos_value / self.capital, self.c.max_leverage)
        btc_size = pos_value / price
        
        return btc_size, leverage


# ==========================================
# 5. BACKTEST ENGINE
# ==========================================
class Backtester:
    """Executes trades with friction, slippage, and statistical exit logic."""
    def __init__(self, c: Config, df: pd.DataFrame, signals: np.ndarray, rm: RiskManager):
        self.c = c
        self.df = df
        self.signals = signals
        self.rm = rm

    def run(self) -> tuple[pd.Series, pd.DataFrame]:
        print("      Executing trade simulation...")
        trades = []
        equity_curve = [self.rm.capital]
        
        pos_size = 0.0
        entry_price = 0.0
        pos_dir = 0
        entry_idx = 0
        lev = 1.0
        
        p = self.df['close'].values
        z = self.df['ou_zscore'].values
        v = self.df['volatility'].values
        
        for i in range(1, len(self.df)):
            sig = self.signals[i]
            price = p[i]
            z_val = z[i]
            
            should_exit = False
            if pos_dir == 1 and (z_val >= -self.c.z_exit or sig == -1):
                should_exit = True
            elif pos_dir == -1 and (z_val <= self.c.z_exit or sig == 1):
                should_exit = True

            # Exit Order
            if pos_size != 0 and should_exit:
                exit_price = price - (price * self.c.slippage_bps * pos_dir)
                pnl = pos_size * (exit_price - entry_price) if pos_dir == 1 else pos_size * (entry_price - exit_price)
                fee = (pos_size * exit_price) * self.c.taker_fee
                net_pnl = pnl - fee
                
                self.rm.update_pnl(net_pnl)
                trades.append({
                    'pnl': net_pnl,
                    'hold_bars': i - entry_idx,
                    'leverage': lev
                })
                pos_size, entry_price, pos_dir = 0.0, 0.0, 0
                
            # Entry Order
            if sig != 0 and pos_size == 0:
                sz, l = self.rm.get_position_size(price, v[i], z_val)
                if sz > 0:
                    entry_price = price + (price * self.c.slippage_bps * sig)
                    fee = (sz * entry_price) * self.c.taker_fee
                    self.rm.update_pnl(-fee)
                    
                    pos_size, pos_dir, lev, entry_idx = sz, sig, l, i
                    
            equity_curve.append(self.rm.capital)
            
        trades_df = pd.DataFrame(trades) if len(trades) > 0 else pd.DataFrame(columns=['pnl', 'hold_bars', 'leverage'])
        return pd.Series(equity_curve), trades_df


# ==========================================
# 6. VISUAL METRICS & REPORTING
# ==========================================
class PerformanceReport:
    """Computes mathematical analytics and draws performance dashboards."""
    def __init__(
        self,
        c: Config,
        equity: pd.Series,
        trades_df: pd.DataFrame,
        label: str = "Full Sample",
        df: pd.DataFrame | None = None,
    ):
        self.c = c
        self.eq = equity
        self.trades_df = trades_df
        self.label = label
        self.df = df

    def compute_metrics(self) -> dict | None:
        if len(self.trades_df) == 0:
            return None

        initial = self.eq.iloc[0]
        final = self.eq.iloc[-1]
        ret_pct = ((final - initial) / initial) * 100

        trades_pnl = self.trades_df["pnl"].values
        total_trades = len(trades_pnl)
        wins = trades_pnl[trades_pnl > 0]
        losses = trades_pnl[trades_pnl <= 0]

        winning_trades = len(wins)
        losing_trades = len(losses)
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

        gross_profit = np.sum(wins)
        gross_loss = np.abs(np.sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_trade = np.mean(trades_pnl) if total_trades > 0 else 0
        avg_win = np.mean(wins) if winning_trades > 0 else 0
        avg_loss = np.mean(losses) if losing_trades > 0 else 0

        largest_win = np.max(trades_pnl) if total_trades > 0 else 0
        largest_loss = np.min(trades_pnl) if total_trades > 0 else 0

        avg_hold = self.trades_df["hold_bars"].mean() * self.c.bar_minutes
        avg_leverage = self.trades_df["leverage"].mean()
        max_leverage = self.trades_df["leverage"].max()

        rm = np.maximum.accumulate(self.eq)
        drawdowns = (rm - self.eq) / rm
        max_dd = np.max(drawdowns) * 100

        period_start = period_end = "N/A"
        if self.df is not None and "timestamp" in self.df.columns:
            period_start = str(self.df["timestamp"].iloc[0].date())
            period_end = str(self.df["timestamp"].iloc[-1].date())

        return {
            "label": self.label,
            "period_start": period_start,
            "period_end": period_end,
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "avg_trade": avg_trade,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "largest_win": largest_win,
            "largest_loss": largest_loss,
            "profit_factor": profit_factor,
            "avg_hold": avg_hold,
            "avg_leverage": avg_leverage,
            "max_leverage": max_leverage,
            "final_equity": final,
            "total_return": ret_pct,
            "max_drawdown": max_dd,
            "drawdowns": drawdowns,
        }

    def generate(self, plot: bool = True) -> dict | None:
        print(f"[Report] {self.label} performance summary...")
        metrics = self.compute_metrics()

        if metrics is None:
            print(f"  Result ({self.label}): No trades executed.")
            return None

        print("=" * 50)
        print(f"   {self.label.upper()} PERFORMANCE")
        print("=" * 50)
        print(f"Period            : {metrics['period_start']} -> {metrics['period_end']}")
        print(f"Total Trades      : {metrics['total_trades']}")
        print(f"Winning Trades    : {metrics['winning_trades']}")
        print(f"Losing Trades     : {metrics['losing_trades']}")
        print(f"Win Rate          : {metrics['win_rate']:.2f}%")
        print(f"Average Trade     : ${metrics['avg_trade']:,.2f}")
        print(f"Average Winner    : ${metrics['avg_win']:,.2f}")
        print(f"Average Loser     : ${metrics['avg_loss']:,.2f}")
        print(f"Best Trade        : ${metrics['largest_win']:,.2f}")
        print(f"Worst Trade       : ${metrics['largest_loss']:,.2f}")
        print(f"Profit Factor     : {metrics['profit_factor']:.2f}")
        print(f"Average Hold      : {metrics['avg_hold']:.1f} min")
        print(f"Average Leverage  : {metrics['avg_leverage']:.2f}x")
        print(f"Maximum Leverage  : {metrics['max_leverage']:.2f}x")
        print(f"Final Equity      : ${metrics['final_equity']:,.2f}")
        print(f"Total Return      : {metrics['total_return']:.2f}%")
        print(f"Max Drawdown      : {metrics['max_drawdown']:.2f}%")
        print("=" * 50)

        if plot:
            self.plot_dashboard(metrics["drawdowns"])

        return metrics

    def plot_dashboard(self, drawdowns: pd.Series):
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=False)
        fig.suptitle(
            f"Mean-Reversion Dashboard — {self.label}",
            fontsize=16,
            fontweight="bold",
            y=0.98,
        )

        axes[0].plot(self.eq.values, label="Equity ($)", color="#1f77b4", linewidth=1.8)
        axes[0].axhline(
            y=self.c.initial_capital,
            color="gray",
            linestyle="--",
            alpha=0.7,
            label="Initial Capital",
        )
        axes[0].set_title("1. Strategy Equity Growth", fontsize=12, fontweight="semibold")
        axes[0].set_ylabel("Capital ($ USD)")
        axes[0].legend(loc="upper left")

        sim_horizon = self.c.mc_horizon_trades
        n_sims = self.c.mc_simulations
        trades_pnl = self.trades_df["pnl"].values

        if len(trades_pnl) > 0:
            sim_paths = np.zeros((n_sims, sim_horizon))
            sim_paths[:, 0] = self.eq.iloc[-1]

            for s in range(n_sims):
                sampled_pnl = np.random.choice(trades_pnl, size=sim_horizon - 1, replace=True)
                sim_paths[s, 1:] = sim_paths[s, 0] + np.cumsum(sampled_pnl)

            for s in range(min(n_sims, 100)):
                axes[1].plot(sim_paths[s, :], color="#17becf", alpha=0.08, linewidth=0.8)

            p5 = np.percentile(sim_paths, 5, axis=0)
            p50 = np.percentile(sim_paths, 50, axis=0)
            p95 = np.percentile(sim_paths, 95, axis=0)

            axes[1].plot(p50, color="#d62728", linewidth=2, label="Median (50th %ile)")
            axes[1].fill_between(
                range(sim_horizon),
                p5,
                p95,
                color="#17becf",
                alpha=0.2,
                label="95% Confidence Interval",
            )

        axes[1].set_title(
            f"2. Monte Carlo Forward Simulation ({n_sims} Paths)",
            fontsize=12,
            fontweight="semibold",
        )
        axes[1].set_xlabel("Forward Trades")
        axes[1].set_ylabel("Projected Capital ($)")
        axes[1].legend(loc="upper left")

        axes[2].plot(drawdowns.values * -100, color="#e377c2", linewidth=1.2, label="Drawdown %")
        axes[2].fill_between(
            range(len(drawdowns)),
            drawdowns.values * -100,
            0,
            color="#e377c2",
            alpha=0.3,
        )
        axes[2].axhline(
            y=-self.c.max_drawdown * 100,
            color="red",
            linestyle="--",
            label=f"Max Allowed DD (-{int(self.c.max_drawdown * 100)}%)",
        )
        axes[2].set_title("3. Underwater Drawdown Profile", fontsize=12, fontweight="semibold")
        axes[2].set_xlabel(f"Time Steps ({self.c.bar_minutes}m Bars)")
        axes[2].set_ylabel("Drawdown (%)")
        axes[2].legend(loc="lower left")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


class ISOSComparisonReport:
    """Side-by-side In-Sample vs Out-of-Sample summary and combined equity view."""

    def __init__(self, c: Config, is_metrics: dict, oos_metrics: dict):
        self.c = c
        self.is_metrics = is_metrics
        self.oos_metrics = oos_metrics

    def generate(
        self,
        is_equity: pd.Series,
        oos_equity: pd.Series,
        is_drawdowns: pd.Series,
        oos_drawdowns: pd.Series,
    ) -> None:
        print("\n" + "=" * 62)
        print("        IN-SAMPLE vs OUT-OF-SAMPLE COMPARISON")
        print("=" * 62)
        rows = [
            ("Period", f"{self.is_metrics['period_start']} -> {self.is_metrics['period_end']}",
             f"{self.oos_metrics['period_start']} -> {self.oos_metrics['period_end']}"),
            ("Total Trades", self.is_metrics["total_trades"], self.oos_metrics["total_trades"]),
            ("Win Rate (%)", f"{self.is_metrics['win_rate']:.2f}", f"{self.oos_metrics['win_rate']:.2f}"),
            ("Profit Factor", f"{self.is_metrics['profit_factor']:.2f}", f"{self.oos_metrics['profit_factor']:.2f}"),
            ("Total Return (%)", f"{self.is_metrics['total_return']:.2f}", f"{self.oos_metrics['total_return']:.2f}"),
            ("Max Drawdown (%)", f"{self.is_metrics['max_drawdown']:.2f}", f"{self.oos_metrics['max_drawdown']:.2f}"),
            ("Avg Trade ($)", f"{self.is_metrics['avg_trade']:,.2f}", f"{self.oos_metrics['avg_trade']:,.2f}"),
            ("Final Equity ($)", f"{self.is_metrics['final_equity']:,.2f}", f"{self.oos_metrics['final_equity']:,.2f}"),
        ]
        print(f"{'Metric':<22}{'In-Sample (4Y)':>20}{'Out-of-Sample (1Y)':>20}")
        print("-" * 62)
        for name, is_val, oos_val in rows:
            print(f"{name:<22}{str(is_val):>20}{str(oos_val):>20}")
        print("=" * 62)

        # Degradation diagnostics (OOS / IS) — key robustness check
        if self.is_metrics["total_return"] != 0:
            ret_ratio = self.oos_metrics["total_return"] / self.is_metrics["total_return"]
            print(f"OOS/IS Return Ratio : {ret_ratio:.2f}x  (closer to 1.0 = better generalization)")

        print("[7/7] Plotting IS/OOS comparison dashboard...")
        self._plot_comparison(is_equity, oos_equity, is_drawdowns, oos_drawdowns)

    def _plot_comparison(
        self,
        is_equity: pd.Series,
        oos_equity: pd.Series,
        is_drawdowns: pd.Series,
        oos_drawdowns: pd.Series,
    ) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            "Walk-Forward Backtest: In-Sample (4Y) vs Out-of-Sample (1Y)",
            fontsize=15,
            fontweight="bold",
        )

        axes[0, 0].plot(is_equity.values, color="#1f77b4", linewidth=1.6, label="IS Equity")
        axes[0, 0].axhline(self.c.initial_capital, color="gray", linestyle="--", alpha=0.6)
        axes[0, 0].set_title("In-Sample Equity (4 Years)")
        axes[0, 0].set_ylabel("Capital ($)")
        axes[0, 0].legend()

        axes[0, 1].plot(oos_equity.values, color="#ff7f0e", linewidth=1.6, label="OOS Equity")
        axes[0, 1].axhline(self.c.initial_capital, color="gray", linestyle="--", alpha=0.6)
        axes[0, 1].set_title("Out-of-Sample Equity (1 Year)")
        axes[0, 1].set_ylabel("Capital ($)")
        axes[0, 1].legend()

        axes[1, 0].plot(is_drawdowns.values * -100, color="#1f77b4", linewidth=1.2)
        axes[1, 0].fill_between(range(len(is_drawdowns)), is_drawdowns.values * -100, 0, alpha=0.25)
        axes[1, 0].set_title("In-Sample Drawdown (%)")
        axes[1, 0].set_xlabel(f"Bars ({self.c.bar_minutes}m)")
        axes[1, 0].set_ylabel("Drawdown (%)")

        axes[1, 1].plot(oos_drawdowns.values * -100, color="#ff7f0e", linewidth=1.2)
        axes[1, 1].fill_between(range(len(oos_drawdowns)), oos_drawdowns.values * -100, 0, alpha=0.25)
        axes[1, 1].set_title("Out-of-Sample Drawdown (%)")
        axes[1, 1].set_xlabel(f"Bars ({self.c.bar_minutes}m)")
        axes[1, 1].set_ylabel("Drawdown (%)")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


# ==========================================
# PIPELINE RUNNER
# ==========================================
def run_period_backtest(c: Config, df: pd.DataFrame, label: str) -> tuple[pd.Series, pd.DataFrame, dict | None]:
    """End-to-end signal generation and simulation for one chronological partition."""
    df = StatisticalModels(c).compute(df)
    signals = SignalEngine(c).generate(df)
    equity, trades = Backtester(c, df, signals, RiskManager(c)).run()
    metrics = PerformanceReport(c, equity, trades, label=label, df=df).generate(plot=False)
    return equity, trades, metrics


# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================
def main():
    c = Config()

    loader = DataLoader(c)
    full_df = loader.load()

    # Use chronological split; only backtest the most recent OOS window
    _, oos_df = loader.split_is_oos(full_df)

    print("[2/3] Running Out-of-Sample backtest (1 year)...")
    oos_df = StatisticalModels(c).compute(oos_df)
    signals = SignalEngine(c).generate(oos_df)
    oos_equity, oos_trades = Backtester(c, oos_df, signals, RiskManager(c)).run()

    print("[3/3] Out-of-Sample results...")
    PerformanceReport(
        c, oos_equity, oos_trades, label="Out-of-Sample (1Y)", df=oos_df
    ).generate(plot=True)

if __name__ == "__main__":
    main()