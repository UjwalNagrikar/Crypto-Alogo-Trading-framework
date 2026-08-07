#!/usr/bin/env python3
"""
Quantitative Mean-Reversion Backtesting Pipeline
Target Asset: BTC/USD 5-minute High-Frequency Data
Mathematical Models: Ornstein-Uhlenbeck Half-Life, Kalman Filter, EWMA Volatility, ATR(14)
"""

import math
import sys
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
    max_leverage: float = 2.0     # Max_Leverage_Cap for the ATR sizing engine (lowered from 3.0 -- see note below)
    max_drawdown: float = 0.20
    dd_cooldown_bars: int = 576   # bars to pause NEW entries after a max_drawdown breach (576 = ~2 days @ 5m).
                                   # Without this, the original circuit breaker is PERMANENT: once drawdown
                                   # crosses max_drawdown, is_trading_allowed() stays False forever because no
                                   # further trades can happen to move capital back up -- that's why trade
                                   # count can collapse to near-zero for the rest of the backtest once tripped.
                                   # After the cooldown elapses, the high-water mark resets to current capital
                                   # so trading resumes with a fresh drawdown budget instead of staying locked.
                                   #
                                   # BUT that reset-and-resume was itself the cause of the -99.96% result: the
                                   # 36 breaches printed in your run each let the strategy re-arm at FULL risk
                                   # against a fresh 20%-drawdown budget, over and over. 0.8^36 = 0.000325 --
                                   # almost exactly the $5.49/$15,000 = 0.000366 ending ratio. The strategy
                                   # didn't lose money in one big move; it lost ~20% in a straight line, 36
                                   # separate times, because nothing ever asked "is this strategy actually
                                   # working right now?" The two knobs below answer that question.
    cooldown_growth_factor: float = 2.0   # each successive breach multiplies the cooldown length by this factor
    max_cooldown_bars: int = 5760         # cap on any single escalated cooldown (5760 bars = ~20 days @ 5m)
    max_total_breaches: int = 6           # hard kill-switch: after this many breaches in one run, stop trading
                                           # entirely for the rest of the backtest and report final capital.
                                           # Escalation turns breach #1..#6 into cooldowns of roughly 2, 4, 8,
                                           # 16, 20(capped), 20(capped) days -- so a strategy that keeps failing
                                           # gets throttled hard well before it reaches breach #6, instead of
                                           # bleeding 20% on a ~2-day cycle for the whole backtest.

    # --- ATR-Based Position Sizing Engine ---
    # Dollar Risk        = Equity * risk_percent
    # Stop Loss Distance = atr_k * ATR(atr_period), floored at price * min_stop_pct
    # Raw Size           = Dollar Risk / Stop Loss Distance
    # Effective Leverage = (Units * Price) / Equity  -> capped at max_leverage
    # Units              = floored to step_size, then checked against min_notional
    #
    # Your run showed average leverage 2.66x against a 3.0x cap -- the leverage
    # cap was binding on almost every trade (ATR-implied stop distance is a
    # small % of BTC's price, so raw_units/risk_percent nearly always wanted
    # more than max_leverage allows). That decouples actual $ risk per trade
    # from risk_percent: realized loss on a stop-out is really
    #   ~ max_leverage * capital * (atr_k * ATR / price)
    # and at atr_k=2.5 that produced an avg loser ($19.63) 55% BIGGER than the
    # avg winner ($12.43) -- even at a 50.7% win rate that's a losing formula.
    # Cutting atr_k below shrinks that loss close to linearly, since the
    # leverage cap is what's actually binding, not risk_percent.
    atr_period: float = 14        # ATR lookback, in 5m bars
    atr_k: float = 1.3            # stop distance = this many ATRs from entry (lowered from 2.5 -- see note above)
    risk_percent: float = 0.015   # fraction of equity risked per trade if the stop is hit
    min_stop_pct: float = 0.0015  # floor on stop distance (as a % of price) so it never collapses in dead-vol regimes
    step_size: float = 0.0001     # exchange lot/step size the final BTC quantity is floored to
    min_notional: float = 10.0    # exchange minimum order value in USD; sizes below this are skipped
    tp_r_multiple: float = 2.0    # take-profit distance = tp_r_multiple * stop-loss distance (R-multiple)
    max_hold_bars: int = 90       # time-stop: force-close a trade after this many bars (~7.5hr @ 5m, ~3x
                                   # max_half_life) if it hasn't hit SL/TP/mean-reversion yet. If price hasn't
                                   # reverted within a few half-lives, the OU thesis for that trade has likely
                                   # broken down -- this caps how long a stalled loser can sit open and drift
                                   # instead of only relying on the ATR stop to eventually catch it.
    verbose_orders: bool = False  # if True, print each bracket order (Entry/SL/TP) as it's placed
    print_trade_log: bool = True  # if True, main() prints every closed trade to the terminal after the report
    min_bars_between_trades: int = 12   # cooldown after any exit before a new entry is allowed (~1hr @ 5m).
                                         # Cuts overtrading/whipsaw re-entries and is the main lever for
                                         # trade frequency alongside z_entry below.

    # Mathematical Model Parameters
    ou_window: int = 40
    vol_span: int = 20
    kalman_delta: float = 0.0001  # Tighter process noise for fast mean reversion

    # Edge Thresholds
    # --- These four (plus min_bars_between_trades above) are the trade-FREQUENCY dials ---
    #   Lower z_entry           -> more bars qualify as "extreme" -> MORE trades
    #   Wider half-life bounds  -> more regimes pass the mean-reversion filter -> MORE trades
    #   Narrower/higher z_entry -> fewer, more selective entries -> FEWER trades
    #
    # Your run: 2730 trades/yr at z_entry=2.0. Target is 1200-1600/yr, roughly a
    # 45-55% cut. Raised to 2.3 as a starting point + the entry cooldown above;
    # this combination is a starting point, NOT a guaranteed hit -- your
    # dataset's actual z-distribution determines the real trade count, so
    # re-run and nudge z_entry by +-0.1 (frequency is fairly sensitive to it)
    # until you land in range.
    z_entry: float = 2.3          # Normalized entry cutoff (raised from 2.0)
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
    """Calculates Kalman Filter Residuals, OU Decay Half-Life, EWMA Volatility, and ATR(14)."""
    def __init__(self, c: Config):
        self.c = c

    def compute(self, df: pd.DataFrame) -> pd.DataFrame:
        print("      Computing continuous mathematical signals...")
        df = df.copy()
        p = df['close'].values.astype(float)

        self._log_returns(df, p)
        self._volatility(df)
        self._average_true_range(df)
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

    def _average_true_range(self, df: pd.DataFrame):
        """
        Wilder's Average True Range over self.c.atr_period bars -- this is the
        volatility measure that now drives position sizing, the trailing stop,
        and the bracket-order SL/TP legs (replacing the old EWMA-vol distance).

        True Range needs high/low/close. If the feed doesn't have 'high'/'low'
        columns, falls back to a close-to-close proxy (|close_t - close_{t-1}|),
        which understates true intrabar range -- a real OHLC feed is strongly
        preferred for live sizing.
        """
        close = df['close'].values.astype(float)
        n = len(close)
        has_hl = 'high' in df.columns and 'low' in df.columns

        if has_hl:
            high = df['high'].values.astype(float)
            low = df['low'].values.astype(float)
            prev_close = np.roll(close, 1)
            prev_close[0] = close[0]
            tr = np.maximum.reduce([
                high - low,
                np.abs(high - prev_close),
                np.abs(low - prev_close),
            ])
        else:
            print("      WARNING: 'high'/'low' columns not found -- using close-to-close "
                  "proxy for True Range. Supply an OHLC feed for accurate ATR sizing.")
            tr = np.zeros(n)
            tr[1:] = np.abs(close[1:] - close[:-1])

        period = int(self.c.atr_period)
        # Wilder's smoothing is equivalent to an EWM with alpha = 1/period.
        atr = pd.Series(tr).ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean().values

        df['true_range'] = tr
        df['atr'] = atr

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
# 4. ATR-BASED RISK MANAGER
# ==========================================
class RiskManager:
    """
    ATR-based position-sizing engine, evaluated fresh on every 5-minute bar close:

        Dollar Risk         = Equity * Risk_Percent
        Stop Loss Distance  = atr_k * ATR(atr_period), floored at price * min_stop_pct
        Raw Position Size   = Dollar Risk / Stop Loss Distance
        Effective Leverage  = (Units * Price) / Equity
            -> if Effective Leverage > Max_Leverage_Cap:
                   Units = (Equity * Max_Leverage_Cap) / Price
        Units               = floored to the exchange step_size
        Notional            = Units * Price
            -> if Notional < min_notional: the trade is skipped (size = 0)

    The same stop_distance also drives the trailing stop in the Backtester and
    feeds build_bracket_order() to compute the SL/TP legs of the entry order.

    NOTE on the drawdown circuit breaker (is_trading_allowed): a naive
    `drawdown < max_drawdown` check is a ONE-WAY door -- once drawdown crosses
    the limit, no new trades can open, so capital can never move again, so
    drawdown never falls back below the limit either. That silently locks
    trading for the rest of the backtest.

    A plain cooldown-and-reset fixes THAT bug but creates a different one: it
    lets a strategy that has gone into a losing regime re-arm at full risk
    against a fresh 20%-drawdown budget indefinitely, bleeding ~20% per cycle
    forever (this is exactly what produced the -99.96% result -- 36 breaches,
    each one a fresh ~20% haircut: 0.8^36 = 0.000325 vs the observed
    0.000366 final/initial ratio). So this version adds two more things on
    top of the cooldown-and-reset:

      1. ESCALATION: each successive breach multiplies the cooldown length by
         cooldown_growth_factor (capped at max_cooldown_bars). A strategy
         that keeps failing gets throttled harder and harder instead of
         retrying on the same ~2-day cycle every time.
      2. KILL SWITCH: after max_total_breaches breaches in one run, trading
         halts entirely for the rest of the backtest. Repeated breaches are
         treated as evidence the strategy isn't working right now, not as
         routine noise to reset past.
    """
    def __init__(self, c: Config):
        self.c = c
        self.capital = c.initial_capital
        self.peak_capital = c.initial_capital
        self.cooldown_end_bar = None   # None = not currently halted
        self.breach_count = 0
        self.halted_bars = 0
        self.halted_permanently = False

    def update_pnl(self, pnl: float):
        self.capital += pnl
        self.peak_capital = max(self.peak_capital, self.capital)

    def is_trading_allowed(self, bar_idx: int) -> bool:
        """Drawdown circuit breaker with escalating cooldowns + a hard kill switch (see class docstring)."""
        if self.halted_permanently:
            return False

        if self.cooldown_end_bar is not None:
            if bar_idx < self.cooldown_end_bar:
                self.halted_bars += 1
                return False
            self.peak_capital = self.capital
            self.cooldown_end_bar = None
            return True

        drawdown = (self.peak_capital - self.capital) / self.peak_capital
        if drawdown >= self.c.max_drawdown:
            self.breach_count += 1
            self.halted_bars += 1

            if self.breach_count >= self.c.max_total_breaches:
                self.halted_permanently = True
                print(f"      [KILL SWITCH] {self.breach_count} drawdown breaches reached -- "
                      f"halting all further trading. Capital: ${self.capital:,.2f}")
                return False

            cooldown = min(
                self.c.dd_cooldown_bars * (self.c.cooldown_growth_factor ** (self.breach_count - 1)),
                self.c.max_cooldown_bars,
            )
            self.cooldown_end_bar = bar_idx + int(cooldown)
            return False

        return True

    @staticmethod
    def floor_to_step(quantity: float, step_size: float) -> float:
        """Rounds a quantity DOWN to the nearest exchange lot/step size."""
        if step_size <= 0 or quantity <= 0:
            return max(quantity, 0.0)
        return math.floor(quantity / step_size) * step_size

    def get_atr_stop_distance(self, price: float, atr: float) -> float:
        """Stop Loss Distance = atr_k * ATR(14), floored so it never collapses to ~0 in dead-vol regimes."""
        if atr is None or np.isnan(atr) or atr <= 0:
            return 0.0
        atr_distance = self.c.atr_k * atr
        floor_distance = price * self.c.min_stop_pct
        return max(atr_distance, floor_distance)

    def get_position_size(self, price: float, atr: float, z_score: float, bar_idx: int) -> tuple[float, float, float]:
        """
        Returns (units, effective_leverage, stop_distance). units == 0.0 whenever
        no trade should be taken: trading halted, ATR unavailable, or the final
        floored quantity fails the exchange min-notional check.
        """
        if not self.is_trading_allowed(bar_idx):
            return 0.0, 1.0, 0.0

        stop_distance = self.get_atr_stop_distance(price, atr)
        if stop_distance <= 0:
            return 0.0, 1.0, 0.0

        # --- Step 1: Dollar Risk & raw size ---
        dollar_risk = self.capital * self.c.risk_percent
        raw_units = dollar_risk / stop_distance

        # --- Step 2: Effective Leverage check & override ---
        pos_value = raw_units * price
        effective_leverage = pos_value / self.capital

        if effective_leverage > self.c.max_leverage:
            units = (self.capital * self.c.max_leverage) / price
        else:
            units = raw_units

        # --- Step 3: floor to exchange step size, verify min notional ---
        units = self.floor_to_step(units, self.c.step_size)
        notional = units * price

        if units <= 0 or notional < self.c.min_notional:
            return 0.0, 1.0, 0.0

        effective_leverage = (units * price) / self.capital  # recompute off the floored qty for accurate reporting
        return units, effective_leverage, stop_distance

    def build_bracket_order(self, entry_price: float, direction: int, stop_distance: float) -> dict:
        """
        Builds the bracket order legs for a new position.
        direction: +1 for long, -1 for short.
        Take-profit distance = tp_r_multiple * stop_distance (a fixed R-multiple target).
        """
        tp_distance = stop_distance * self.c.tp_r_multiple
        if direction == 1:
            sl = entry_price - stop_distance
            tp = entry_price + tp_distance
        else:
            sl = entry_price + stop_distance
            tp = entry_price - tp_distance

        return {
            "direction": "LONG" if direction == 1 else "SHORT",
            "entry": entry_price,
            "stop_loss": sl,
            "take_profit": tp,
            "stop_distance": stop_distance,
            "tp_distance": tp_distance,
        }


# ==========================================
# 5. BACKTEST ENGINE
# ==========================================
class Backtester:
    """Executes trades with friction, slippage, mean-reversion exits, and an ATR-based bracket (SL/TP)."""
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
        stop_distance = 0.0
        trail_stop = None
        take_profit = None
        peak_price = None
        last_exit_bar = -self.c.min_bars_between_trades  # allow an entry on bar 1 if signaled

        p = self.df['close'].values
        z = self.df['ou_zscore'].values
        a = self.df['atr'].values

        for i in range(1, len(self.df)):
            sig = self.signals[i]
            price = p[i]
            z_val = z[i]

            # --- Update trailing stop & check take-profit / time-stop for an open position ---
            stopped_out = False
            hit_tp = False
            timed_out = False
            if pos_size != 0:
                timed_out = (i - entry_idx) >= self.c.max_hold_bars
                if pos_dir == 1:
                    peak_price = max(peak_price, price)
                    trail_stop = peak_price - stop_distance
                    stopped_out = price <= trail_stop
                    hit_tp = price >= take_profit
                else:
                    peak_price = min(peak_price, price)
                    trail_stop = peak_price + stop_distance
                    stopped_out = price >= trail_stop
                    hit_tp = price <= take_profit

            # Priority when multiple conditions trip on the same bar: TP (best case) > price
            # stop (risk control) > time-stop (thesis stalled) > signal flip > mean-reversion.
            should_exit = False
            exit_reason = None
            if pos_dir == 1 and (z_val >= -self.c.z_exit or sig == -1 or stopped_out or hit_tp or timed_out):
                should_exit = True
                exit_reason = ('take_profit' if hit_tp else
                               'trailing_stop' if stopped_out else
                               'time_stop' if timed_out else
                               'signal_flip' if sig == -1 else 'mean_reversion')
            elif pos_dir == -1 and (z_val <= self.c.z_exit or sig == 1 or stopped_out or hit_tp or timed_out):
                should_exit = True
                exit_reason = ('take_profit' if hit_tp else
                               'trailing_stop' if stopped_out else
                               'time_stop' if timed_out else
                               'signal_flip' if sig == 1 else 'mean_reversion')

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
                    'leverage': lev,
                    'exit_reason': exit_reason,
                    'entry_price': entry_price,
                    'stop_loss': trail_stop if pos_dir == 1 else trail_stop,
                    'take_profit': take_profit,
                    'exit_price': exit_price,
                })
                pos_size, entry_price, pos_dir = 0.0, 0.0, 0
                trail_stop, take_profit, peak_price, stop_distance = None, None, None, 0.0
                last_exit_bar = i

            # Entry Order
            cooldown_elapsed = (i - last_exit_bar) >= self.c.min_bars_between_trades
            if sig != 0 and pos_size == 0 and cooldown_elapsed:
                sz, l, sd = self.rm.get_position_size(price, a[i], z_val, i)
                if sz > 0:
                    entry_price = price + (price * self.c.slippage_bps * sig)
                    fee = (sz * entry_price) * self.c.taker_fee
                    self.rm.update_pnl(-fee)

                    pos_size, pos_dir, lev, entry_idx = sz, sig, l, i
                    stop_distance = sd
                    peak_price = entry_price

                    bracket = self.rm.build_bracket_order(entry_price, sig, stop_distance)
                    trail_stop = bracket['stop_loss']
                    take_profit = bracket['take_profit']

                    if self.c.verbose_orders:
                        print(f"      [BRACKET] {bracket['direction']:<5} qty={sz:.6f}  "
                              f"entry={bracket['entry']:.2f}  sl={bracket['stop_loss']:.2f}  "
                              f"tp={bracket['take_profit']:.2f}  leverage={l:.2f}x")

            equity_curve.append(self.rm.capital)

        trades_df = pd.DataFrame(trades) if len(trades) > 0 else pd.DataFrame(
            columns=['pnl', 'hold_bars', 'leverage', 'exit_reason', 'entry_price', 'stop_loss', 'take_profit', 'exit_price']
        )

        if self.rm.breach_count > 0:
            halted_pct = self.rm.halted_bars / len(self.df) * 100
            status = "KILL SWITCH TRIPPED" if self.rm.halted_permanently else "cooldowns only"
            print(f"      Drawdown breaches: {self.rm.breach_count}  |  "
                  f"bars halted: {self.rm.halted_bars} ({halted_pct:.1f}% of period, escalating cooldowns, {status})")

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

        exit_breakdown = None
        if "exit_reason" in self.trades_df.columns:
            exit_breakdown = (
                self.trades_df.groupby("exit_reason")["pnl"]
                .agg(["count", "mean", "sum"])
                .to_dict("index")
            )

        # Annualized trade frequency -- bar-count based, so it works even
        # without timestamps, and is comparable across different sample lengths.
        total_bars = len(self.eq) if self.eq is not None else 0
        bars_per_year = (365.25 * 24 * 60) / self.c.bar_minutes
        trades_per_year = (total_trades / total_bars * bars_per_year) if total_bars > 0 else 0.0

        return {
            "label": self.label,
            "period_start": period_start,
            "period_end": period_end,
            "total_trades": total_trades,
            "trades_per_year": trades_per_year,
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
            "exit_breakdown": exit_breakdown,
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
        print(f"Trades / Year     : {metrics['trades_per_year']:.0f}  (annualized, bar-count based)")
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
        if metrics.get("exit_breakdown"):
            print("-" * 50)
            print("Exit Reason Breakdown:")
            for reason, stats in metrics["exit_breakdown"].items():
                print(f"  {reason:<16} count={stats['count']:>5.0f}  "
                      f"avg=${stats['mean']:>9,.2f}  total=${stats['sum']:>12,.2f}")
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


def run_z_entry_sweep(
    base_config: Config,
    df: pd.DataFrame,
    candidates: list[float],
    target_range: tuple[float, float] = (1200, 1600),
) -> pd.DataFrame:
    """
    z_entry=2.3 is a starting point, not a guaranteed hit -- the real trade
    count depends on your dataset's actual z-score distribution, which I
    don't have. This runs the full OOS backtest once per candidate z_entry
    and reports trades/year + key stats for each, so you can read off
    whichever value actually lands inside target_range on YOUR data instead
    of guessing blind.

    Usage: python mean_reversion_backtest.py --sweep
    """
    rows = []
    for z in candidates:
        c = replace(base_config, z_entry=z)
        _, _, metrics = run_period_backtest(c, df.copy(), label=f"z_entry={z}")
        if metrics is None:
            rows.append({"z_entry": z, "trades_per_year": 0, "win_rate": None,
                         "profit_factor": None, "total_return": None,
                         "max_drawdown": None, "in_target": False})
            continue
        tpy = round(metrics["trades_per_year"])
        rows.append({
            "z_entry": z,
            "trades_per_year": tpy,
            "win_rate": round(metrics["win_rate"], 2),
            "profit_factor": round(metrics["profit_factor"], 2),
            "total_return": round(metrics["total_return"], 2),
            "max_drawdown": round(metrics["max_drawdown"], 2),
            "in_target": target_range[0] <= tpy <= target_range[1],
        })

    sweep_df = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print(f"  Z_ENTRY SWEEP  (target trades/year: {target_range[0]}-{target_range[1]})")
    print("=" * 78)
    print(sweep_df.to_string(index=False))
    print("=" * 78)
    hits = sweep_df[sweep_df["in_target"]]
    if len(hits) > 0:
        print(f"In target range: z_entry in {hits['z_entry'].tolist()}")
    else:
        print("Nothing in range yet -- widen the candidates list based on the trend above "
              "(higher z_entry = fewer trades) and re-run.")
    return sweep_df


def print_trade_log(trades_df: pd.DataFrame, label: str = "Trade Log") -> None:
    """
    Prints every trade in trades_df to the terminal (no row truncation),
    numbered in execution order. Rounds price/PnL columns for readability;
    doesn't touch the underlying trades_df.
    """
    if len(trades_df) == 0:
        print(f"\n[{label}] No trades to display.")
        return

    display_df = trades_df.copy()
    display_df.insert(0, 'trade_#', range(1, len(display_df) + 1))

    for col in ['pnl', 'entry_price', 'stop_loss', 'take_profit', 'exit_price']:
        if col in display_df.columns:
            display_df[col] = display_df[col].round(2)
    if 'leverage' in display_df.columns:
        display_df['leverage'] = display_df['leverage'].round(3)

    print("\n" + "=" * 110)
    print(f"  {label.upper()}  ({len(display_df)} trades)")
    print("=" * 110)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 200):
        print(display_df.to_string(index=False))
    print("=" * 110)


# ==========================================
# MAIN EXECUTION PIPELINE
# ==========================================
def main():
    c = Config()

    loader = DataLoader(c)
    full_df = loader.load()

    # Use chronological split; only backtest the most recent OOS window
    _, oos_df = loader.split_is_oos(full_df)

    if "--sweep" in sys.argv:
        candidates = [1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.8, 3.0]
        run_z_entry_sweep(c, oos_df, candidates)
        return

    print("[2/3] Running Out-of-Sample backtest (1 year)...")
    oos_df = StatisticalModels(c).compute(oos_df)
    signals = SignalEngine(c).generate(oos_df)
    oos_equity, oos_trades = Backtester(c, oos_df, signals, RiskManager(c)).run()

    print("[3/3] Out-of-Sample results...")
    PerformanceReport(
        c, oos_equity, oos_trades, label="Out-of-Sample (1Y)", df=oos_df
    ).generate(plot=True)

    if c.print_trade_log:
        print_trade_log(oos_trades, label="Out-of-Sample Trade Log")

if __name__ == "__main__":
    main()
