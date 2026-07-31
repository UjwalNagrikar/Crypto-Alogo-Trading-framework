#!/usr/bin/env python3
import itertools
import random
import time
import warnings
from dataclasses import dataclass, replace

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

    # Bars-per-year for annualizing Sharpe (5m bars, BTC trades 24/7)
    bars_per_year: int = 105_120


# ==========================================
# 1. DATA LOADER
# ==========================================
class DataLoader:
    """Loads, cleans, and structures CSV time series data."""
    def __init__(self, c: Config):
        self.c = c

    def load(self) -> pd.DataFrame:
        print("[1/6] Loading CSV Data...")
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


# ==========================================
# 2. MATHEMATICAL STATISTICAL ENGINE
# ==========================================
class StatisticalModels:
    """Calculates Kalman Filter Residuals, OU Decay Half-Life, and EWMA Volatility."""
    def __init__(self, c: Config):
        self.c = c

    def compute(self, df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
        if verbose:
            print("[2/6] Computing Continuous Mathematical Signals...")
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
        mean_est = np.zeros(n)
        mean_est[0] = xe

        obs_noise = 0.05

        for i in range(1, n):
            xp, pp = xe, pe + d
            k = pp / (pp + obs_noise)
            xe = xp + k * (p[i] - xp)
            pe = (1 - k) * pp
            dev[i] = p[i] - xe
            mean_est[i] = xe  # FIX: store per-bar estimate, not just the final scalar

        df['kalman_mean'] = mean_est
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
            s = dev[i - w + 1:i + 1]
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

    def generate(self, df: pd.DataFrame, verbose: bool = True) -> np.ndarray:
        if verbose:
            print("[3/6] Filtering Signals via Stationarity Criteria...")
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

    def run(self, verbose: bool = True) -> tuple[pd.Series, pd.DataFrame]:
        if verbose:
            print("[4/6] Executing Trade Simulation...")
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
# 6. METRICS (pure function — reused by optimizer + final report)
# ==========================================
def compute_metrics(equity: pd.Series, trades_df: pd.DataFrame, bars_per_year: int = 105_120) -> dict:
    if len(trades_df) == 0:
        return {
            'num_trades': 0, 'win_rate': 0.0, 'profit_factor': 0.0,
            'total_return_pct': 0.0, 'sharpe': 0.0, 'max_drawdown_pct': 0.0,
            'avg_trade': 0.0,
        }

    initial, final = equity.iloc[0], equity.iloc[-1]
    total_return_pct = ((final - initial) / initial) * 100

    rets = equity.pct_change().dropna()
    sharpe = (rets.mean() / rets.std()) * np.sqrt(bars_per_year) if rets.std() > 0 else 0.0

    running_max = np.maximum.accumulate(equity)
    drawdowns = (running_max - equity) / running_max
    max_dd = float(np.max(drawdowns)) * 100

    pnl = trades_df['pnl'].values
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    win_rate = (len(wins) / len(pnl)) * 100 if len(pnl) > 0 else 0.0
    gross_profit = np.sum(wins)
    gross_loss = np.abs(np.sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    return {
        'num_trades': len(pnl),
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'total_return_pct': total_return_pct,
        'sharpe': float(sharpe),
        'max_drawdown_pct': max_dd,
        'avg_trade': float(np.mean(pnl)),
    }


# ==========================================
# 7. VISUAL METRICS & REPORTING
# ==========================================
class PerformanceReport:
    """Computes mathematical analytics and draws performance dashboards."""
    def __init__(self, c: Config, equity: pd.Series, trades_df: pd.DataFrame):
        self.c = c
        self.eq = equity
        self.trades_df = trades_df

    def generate(self, plot: bool = True):
        print("[5/6] Summarizing Mathematical Performance...")

        if len(self.trades_df) == 0:
            print("  Result: No trades executed. Check window and threshold settings.")
            return

        m = compute_metrics(self.eq, self.trades_df, self.c.bars_per_year)

        trades_pnl = self.trades_df['pnl'].values
        wins = trades_pnl[trades_pnl > 0]
        losses = trades_pnl[trades_pnl <= 0]
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        largest_win = np.max(trades_pnl) if len(trades_pnl) > 0 else 0
        largest_loss = np.min(trades_pnl) if len(trades_pnl) > 0 else 0
        avg_hold = self.trades_df['hold_bars'].mean() * 5.0
        avg_leverage = self.trades_df['leverage'].mean()
        max_leverage = self.trades_df['leverage'].max()

        running_max = np.maximum.accumulate(self.eq)
        drawdowns = (running_max - self.eq) / running_max

        print("=" * 50)
        print("      PIPELINE PERFORMANCE SUMMARY")
        print("=" * 50)
        print(f"Total Trades      : {m['num_trades']}")
        print(f"Win Rate          : {m['win_rate']:.2f}%")
        print(f"Average Trade     : ${m['avg_trade']:,.2f}")
        print(f"Average Winner    : ${avg_win:,.2f}")
        print(f"Average Loser     : ${avg_loss:,.2f}")
        print(f"Best Trade        : ${largest_win:,.2f}")
        print(f"Worst Trade       : ${largest_loss:,.2f}")
        print(f"Profit Factor     : {m['profit_factor']:.2f}")
        print(f"Average Hold      : {avg_hold:.1f} min")
        print(f"Average Leverage  : {avg_leverage:.2f}x")
        print(f"Maximum Leverage  : {max_leverage:.2f}x")
        print(f"Final Equity      : ${self.eq.iloc[-1]:,.2f}")
        print(f"Total Return      : {m['total_return_pct']:.2f}%")
        print(f"Sharpe (annualized): {m['sharpe']:.2f}")
        print(f"Max Drawdown      : {m['max_drawdown_pct']:.2f}%")
        print("=" * 50)

        if plot:
            print("[6/6] Plotting Dashboards...")
            self.plot_dashboard(drawdowns)

    def plot_dashboard(self, drawdowns: pd.Series):
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=False)
        fig.suptitle('Quantitative Mean-Reversion Risk Dashboard — Out-of-Sample (Year 5)',
                      fontsize=16, fontweight='bold', y=0.98)

        axes[0].plot(self.eq.values, label='Equity ($)', color='#1f77b4', linewidth=1.8)
        axes[0].axhline(y=self.c.initial_capital, color='gray', linestyle='--', alpha=0.7, label='Initial Capital')
        axes[0].set_title('1. Strategy Equity Growth', fontsize=12, fontweight='semibold')
        axes[0].set_ylabel('Capital ($ USD)')
        axes[0].legend(loc='upper left')

        sim_horizon = self.c.mc_horizon_trades
        n_sims = self.c.mc_simulations
        trades_pnl = self.trades_df['pnl'].values

        if len(trades_pnl) > 0:
            sim_paths = np.zeros((n_sims, sim_horizon))
            sim_paths[:, 0] = self.eq.iloc[-1]

            for s in range(n_sims):
                sampled_pnl = np.random.choice(trades_pnl, size=sim_horizon - 1, replace=True)
                sim_paths[s, 1:] = sim_paths[s, 0] + np.cumsum(sampled_pnl)

            for s in range(min(n_sims, 100)):
                axes[1].plot(sim_paths[s, :], color='#17becf', alpha=0.08, linewidth=0.8)

            p5 = np.percentile(sim_paths, 5, axis=0)
            p50 = np.percentile(sim_paths, 50, axis=0)
            p95 = np.percentile(sim_paths, 95, axis=0)

            axes[1].plot(p50, color='#d62728', linewidth=2, label='Median (50th %ile)')
            axes[1].fill_between(range(sim_horizon), p5, p95, color='#17becf', alpha=0.2, label='95% Confidence Interval')

        axes[1].set_title(f'2. Monte Carlo Forward Simulation ({n_sims} Paths)', fontsize=12, fontweight='semibold')
        axes[1].set_xlabel('Forward Trades')
        axes[1].set_ylabel('Projected Capital ($)')
        axes[1].legend(loc='upper left')

        axes[2].plot(drawdowns.values * -100, color='#e377c2', linewidth=1.2, label='Drawdown %')
        axes[2].fill_between(range(len(drawdowns)), drawdowns.values * -100, 0, color='#e377c2', alpha=0.3)
        axes[2].axhline(y=-self.c.max_drawdown * 100, color='red', linestyle='--',
                         label=f'Max Allowed DD (-{int(self.c.max_drawdown * 100)}%)')
        axes[2].set_title('3. Underwater Drawdown Profile', fontsize=12, fontweight='semibold')
        axes[2].set_xlabel('Time Steps (5m Bars)')
        axes[2].set_ylabel('Drawdown (%)')
        axes[2].legend(loc='lower left')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('performance_dashboard.png', dpi=120)
        plt.show()


# ==========================================
# 8. DATA SPLITTER — chronological train / validation / test blocks
# ==========================================
class DataSplitter:
    """
    Splits raw price data into disjoint chronological blocks.
    Prefers calendar-year boundaries (matches how you described the split);
    falls back to row-count percentages if there's no 'timestamp' column.
    """
    def __init__(self, c: Config):
        self.c = c

    def split_by_years(self, df: pd.DataFrame, train_years: int = 3,
                        val_years: int = 1, test_years: int = 1):
        if 'timestamp' not in df.columns:
            print("  WARNING: no 'timestamp' column — falling back to row-count percentage split.")
            total = train_years + val_years + test_years
            return self.split_by_pct(df, train_years / total, val_years / total)

        start_year = df['timestamp'].dt.year.min()
        train_end = start_year + train_years
        val_end = train_end + val_years
        test_end = val_end + test_years

        years = df['timestamp'].dt.year
        train_df = df[years < train_end].reset_index(drop=True)
        val_df = df[(years >= train_end) & (years < val_end)].reset_index(drop=True)
        test_df = df[(years >= val_end) & (years < test_end)].reset_index(drop=True)

        return train_df, val_df, test_df

    def split_by_pct(self, df: pd.DataFrame, train_pct: float = 0.6, val_pct: float = 0.2):
        n = len(df)
        train_end = int(n * train_pct)
        val_end = train_end + int(n * val_pct)
        return (df.iloc[:train_end].reset_index(drop=True),
                df.iloc[train_end:val_end].reset_index(drop=True),
                df.iloc[val_end:].reset_index(drop=True))


# ==========================================
# 9. WALK-FORWARD OPTIMIZER
# ==========================================
# Search space — edit freely. Keep it small at first; the signal/OU loops
# are pure Python, so each config evaluation costs O(n_bars). A 3^6 full
# grid (729 configs) over 3 years of 5m bars (~315k rows) will be slow —
# use mode='random' with a bounded n_iter, or port the hot loops
# (_kalman_filter, _ornstein_uhlenbeck) to Numba as you've done elsewhere
# before scaling this up.
PARAM_GRID = {
    'z_entry':        [1.5, 2.0, 2.5],
    'z_exit':         [0.0, 0.1, 0.2],
    'ou_window':       [30, 40, 50],
    'kalman_delta':   [0.00005, 0.0001, 0.0002],
    'max_half_life':  [20.0, 30.0, 40.0],
    'base_risk':      [0.010, 0.015, 0.020],
}


class WalkForwardOptimizer:
    def __init__(self, base_config: Config, param_grid: dict):
        self.base_config = base_config
        self.param_grid = param_grid

    def _sample_configs(self, n_iter: int, mode: str, seed: int):
        keys = list(self.param_grid.keys())
        all_combos = list(itertools.product(*[self.param_grid[k] for k in keys]))
        if mode == 'grid' or n_iter >= len(all_combos):
            chosen = all_combos
        else:
            chosen = random.Random(seed).sample(all_combos, n_iter)
        return [replace(self.base_config, **dict(zip(keys, combo))) for combo in chosen]

    def _evaluate(self, raw_df: pd.DataFrame, cfg: Config, min_trades: int) -> dict:
        df = StatisticalModels(cfg).compute(raw_df.copy(), verbose=False)
        signals = SignalEngine(cfg).generate(df, verbose=False)
        rm = RiskManager(cfg)
        equity, trades_df = Backtester(cfg, df, signals, rm).run(verbose=False)
        metrics = compute_metrics(equity, trades_df, cfg.bars_per_year)

        eligible = metrics['num_trades'] >= min_trades and metrics['max_drawdown_pct'] <= cfg.max_drawdown * 100
        metrics['eligible'] = eligible
        metrics['score'] = metrics['sharpe'] if eligible else -np.inf
        return metrics

    def optimize(self, train_df: pd.DataFrame, n_iter: int = 30, mode: str = 'random',
                 min_trades: int = 30, seed: int = 42, verbose: bool = True):
        configs = self._sample_configs(n_iter, mode, seed)
        results = []
        t0 = time.time()
        for i, cfg in enumerate(configs):
            metrics = self._evaluate(train_df, cfg, min_trades)
            results.append((cfg, metrics))
            if verbose:
                flag = 'OK' if metrics['eligible'] else 'rejected'
                print(f"  [{i + 1}/{len(configs)}] sharpe={metrics['sharpe']:.2f} "
                      f"trades={metrics['num_trades']} dd={metrics['max_drawdown_pct']:.1f}%  ({flag})")
        results.sort(key=lambda r: r[1]['score'], reverse=True)
        if verbose:
            print(f"  Optimization finished in {time.time() - t0:.1f}s over {len(configs)} configs.")
        return results

    def validate(self, candidates: list, val_df: pd.DataFrame, top_k: int = 10,
                 min_trades: int = 10, verbose: bool = True):
        top_candidates = candidates[:top_k]
        val_results = []
        for cfg, train_metrics in top_candidates:
            val_metrics = self._evaluate(val_df, cfg, min_trades)
            val_results.append((cfg, train_metrics, val_metrics))
            if verbose:
                flag = 'OK' if val_metrics['eligible'] else 'rejected'
                print(f"  train_sharpe={train_metrics['sharpe']:5.2f}  ->  "
                      f"val_sharpe={val_metrics['sharpe']:5.2f}  (trades={val_metrics['num_trades']})  ({flag})")
        val_results.sort(key=lambda r: r[2]['score'], reverse=True)
        return val_results


# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================
def main():
    base_cfg = Config()

    df_raw = DataLoader(base_cfg).load()

    splitter = DataSplitter(base_cfg)
    train_df, val_df, test_df = splitter.split_by_years(df_raw, train_years=3, val_years=1, test_years=1)
    print(f"Train: {len(train_df):,} bars | Validation: {len(val_df):,} bars | Test: {len(test_df):,} bars")

    optimizer = WalkForwardOptimizer(base_cfg, PARAM_GRID)

    print("\n=== STAGE 1: Optimize on Training Data (Years 1-3) ===")
    train_results = optimizer.optimize(train_df, n_iter=30, mode='random', min_trades=30)

    print("\n=== STAGE 2: Validate Top Candidates on Year 4 ===")
    val_results = optimizer.validate(train_results, val_df, top_k=10, min_trades=10)

    if not val_results or val_results[0][2]['score'] == -np.inf:
        print("\nNo candidate survived validation. Widen PARAM_GRID, lower min_trades, or gather more data.")
        return

    best_cfg, best_train_metrics, best_val_metrics = val_results[0]
    print("\nSelected configuration (best validation Sharpe):")
    for k in PARAM_GRID:
        print(f"  {k:14s} = {getattr(best_cfg, k)}")
    print(f"  Train Sharpe: {best_train_metrics['sharpe']:.2f}  |  Validation Sharpe: {best_val_metrics['sharpe']:.2f}")
    if best_train_metrics['sharpe'] > 2 * max(best_val_metrics['sharpe'], 0.01):
        print("  NOTE: train Sharpe is much higher than validation Sharpe — sign of overfitting to Years 1-3.")

    print("\n=== STAGE 3: Final Out-of-Sample Test on Year 5 (evaluated ONCE) ===")
    test_df_ind = StatisticalModels(best_cfg).compute(test_df.copy())
    test_signals = SignalEngine(best_cfg).generate(test_df_ind)
    rm = RiskManager(best_cfg)
    equity, trades_df = Backtester(best_cfg, test_df_ind, test_signals, rm).run()
    PerformanceReport(best_cfg, equity, trades_df).generate()

    print("\nThis Year-5 result is your unbiased performance estimate.")
    print("Re-tuning parameters after seeing it would invalidate that — treat it as final.")


if __name__ == "__main__":
    main()