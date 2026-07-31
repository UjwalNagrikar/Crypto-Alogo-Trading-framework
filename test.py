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

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
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
        
        # Adaptive observation noise based on price variability
        obs_noise = 0.05
        
        for i in range(1, n):
            xp, pp = xe, pe + d
            k = pp / (pp + obs_noise)
            xe = xp + k * (p[i] - xp)
            pe = (1 - k) * pp
            dev[i] = p[i] - xe  # Raw distance from true dynamic mean
            
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
            
            # Linear Regression for discrete OU process
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
                # Volatility-normalized Z-score around Kalman state
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
        print("[3/6] Filtering Signals via Stationarity Criteria...")
        n = len(df)
        sig = np.zeros(n, dtype=np.int8)
        
        z = df['ou_zscore'].values
        hl = df['half_life'].values
        
        for i in range(n):
            if np.isnan(z[i]) or np.isnan(hl[i]):
                continue
                
            # Edge condition: Strong mean reversion speed (valid half-life)
            is_mean_reverting = self.c.min_half_life <= hl[i] <= self.c.max_half_life
            
            if is_mean_reverting:
                if z[i] <= -self.c.z_entry:
                    sig[i] = 1   # Long Entry (Undervalued)
                elif z[i] >= self.c.z_entry:
                    sig[i] = -1  # Short Entry (Overvalued)
                    
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
            
        # Mathematical Edge: Scale position with normalized statistical magnitude
        edge = min(abs(z_score) / self.c.z_entry, 2.0)
        
        # Volatility Sizing: Target constant volatility contribution per trade
        target_vol = 0.002
        vol_scalar = target_vol / max(vol, 0.0005)
        
        risk_fraction = self.c.base_risk * edge * vol_scalar
        risk_fraction = max(0.005, min(risk_fraction, 0.05))  # Clamp between 0.5% and 5% capital
        
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

    def run(self) -> tuple[pd.Series, list]:
        print("[4/6] Executing Trade Simulation...")
        trades = []
        equity_curve = [self.rm.capital]
        
        pos_size = 0.0
        entry_price = 0.0
        pos_dir = 0
        lev = 1.0
        
        p = self.df['close'].values
        z = self.df['ou_zscore'].values
        v = self.df['volatility'].values
        
        for i in range(1, len(self.df)):
            sig = self.signals[i]
            price = p[i]
            z_val = z[i]
            
            # Reversion Exit Condition: Position closes when price crosses standard mean (z ~ 0)
            should_exit = False
            if pos_dir == 1 and (z_val >= -self.c.z_exit or sig == -1):
                should_exit = True
            elif pos_dir == -1 and (z_val <= self.c.z_exit or sig == 1):
                should_exit = True

            # Process Exit Order
            if pos_size != 0 and should_exit:
                exit_price = price - (price * self.c.slippage_bps * pos_dir)
                pnl = pos_size * (exit_price - entry_price) if pos_dir == 1 else pos_size * (entry_price - exit_price)
                fee = (pos_size * exit_price) * self.c.taker_fee
                net_pnl = pnl - fee
                
                self.rm.update_pnl(net_pnl)
                trades.append(net_pnl)
                pos_size, entry_price, pos_dir = 0.0, 0.0, 0
                
            # Process Entry Order
            if sig != 0 and pos_size == 0:
                sz, l = self.rm.get_position_size(price, v[i], z_val)
                if sz > 0:
                    entry_price = price + (price * self.c.slippage_bps * sig)
                    fee = (sz * entry_price) * self.c.taker_fee
                    self.rm.update_pnl(-fee)  # Deduct fee on entry
                    
                    pos_size, pos_dir, lev = sz, sig, l
                    
            equity_curve.append(self.rm.capital)
            
        return pd.Series(equity_curve), trades


# ==========================================
# 6. VISUAL METRICS & REPORTING
# ==========================================
class PerformanceReport:
    """Computes mathematical analytics and draws performance dashboards."""
    def __init__(self, c: Config, equity: pd.Series, trades: list):
        self.c = c
        self.eq = equity
        self.trades = np.array(trades)

    def generate(self):
        print("[5/6] Summarizing Mathematical Performance...")
        
        if len(self.trades) == 0:
            print("  Result: No trades executed. Check window and threshold settings.")
            return
            
        initial = self.eq.iloc[0]
        final = self.eq.iloc[-1]
        ret_pct = ((final - initial) / initial) * 100
        
        wins = self.trades[self.trades > 0]
        losses = self.trades[self.trades <= 0]
        win_rate = (len(wins) / len(self.trades)) * 100 if len(self.trades) > 0 else 0
        
        gross_profit = np.sum(wins)
        gross_loss = np.abs(np.sum(losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        rm = np.maximum.accumulate(self.eq)
        drawdowns = (rm - self.eq) / rm
        max_dd = np.max(drawdowns) * 100
        
        # print("="*40)
        # print("  PIPELINE PERFORMANCE SUMMARY")
        # print("="*40)
        # print(f"  Total Trades:      {len(self.trades)}")
        # print(f"  Avg Tarde          {}")
        # print(f"  Final Equity:      ${final:,.2f}")
        # print(f"  Total Return:      {ret_pct:.2f}%")
        # print(f"  Win Rate:          {win_rate:.1f}%")
        # print(f"  Profit Factor:     {profit_factor:.2f}")
        # print(f"  Max Drawdown:      {max_dd:.2f}%")
        # print("="*40)

        # print("="*50)
        print("      PIPELINE PERFORMANCE SUMMARY")
        print("="*50)

        print(f"Total Trades      : {total_trades}")
        print(f"Winning Trades    : {winning_trades}")
        print(f"Losing Trades     : {losing_trades}")

        print(f"Win Rate          : {win_rate:.2f}%")

        print(f"Average Trade     : ${avg_trade:,.2f}")
        print(f"Average Winner    : ${avg_win:,.2f}")
        print(f"Average Loser     : ${avg_loss:,.2f}")

        print(f"Best Trade        : ${largest_win:,.2f}")
        print(f"Worst Trade       : ${largest_loss:,.2f}")

        print(f"Profit Factor     : {profit_factor:.2f}")

        print(f"Average Hold      : {avg_hold:.1f} min")

        print(f"Average Leverage  : {avg_leverage:.2f}x")
        print(f"Maximum Leverage  : {max_leverage:.2f}x")

        print(f"Final Equity      : ${final:,.2f}")
        print(f"Total Return      : {ret_pct:.2f}%")
        print(f"Max Drawdown      : {max_dd:.2f}%")

        print("="*50)

        print("[6/6] Plotting Dashboards...")
        self.plot_dashboard(drawdowns)

    def plot_dashboard(self, drawdowns: pd.Series):
        fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=False)
        fig.suptitle('Quantitative Mean-Reversion Risk Dashboard', fontsize=16, fontweight='bold', y=0.98)

        # Equity Curve
        axes[0].plot(self.eq.values, label='Equity ($)', color='#1f77b4', linewidth=1.8)
        axes[0].axhline(y=self.c.initial_capital, color='gray', linestyle='--', alpha=0.7, label='Initial Capital')
        axes[0].set_title('1. Strategy Equity Growth', fontsize=12, fontweight='semibold')
        axes[0].set_ylabel('Capital ($ USD)')
        axes[0].legend(loc='upper left')

        # Monte Carlo Simulation
        sim_horizon = self.c.mc_horizon_trades
        n_sims = self.c.mc_simulations
        
        if len(self.trades) > 0:
            sim_paths = np.zeros((n_sims, sim_horizon))
            sim_paths[:, 0] = self.eq.iloc[-1]
            
            for s in range(n_sims):
                sampled_pnl = np.random.choice(self.trades, size=sim_horizon - 1, replace=True)
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

        # Drawdown Profile
        axes[2].plot(drawdowns.values * -100, color='#e377c2', linewidth=1.2, label='Drawdown %')
        axes[2].fill_between(range(len(drawdowns)), drawdowns.values * -100, 0, color='#e377c2', alpha=0.3)
        axes[2].axhline(y=-self.c.max_drawdown * 100, color='red', linestyle='--', label=f'Max Allowed DD (-{int(self.c.max_drawdown*100)}%)')
        axes[2].set_title('3. Underwater Drawdown Profile', fontsize=12, fontweight='semibold')
        axes[2].set_xlabel('Time Steps (5m Bars)')
        axes[2].set_ylabel('Drawdown (%)')
        axes[2].legend(loc='lower left')

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()


# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================
def main():
    c = Config()
    
    df = DataLoader(c).load()
    df = StatisticalModels(c).compute(df)
    signals = SignalEngine(c).generate(df)
    
    risk_manager = RiskManager(c)
    equity_curve, trades = Backtester(c, df, signals, risk_manager).run()
    
    report = PerformanceReport(c, equity_curve, trades)
    report.generate()

if __name__ == "__main__":
    main()