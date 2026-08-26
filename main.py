#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
BTCUSD / BTCUSDT 5-MINUTE  --  NON-ML STATISTICAL TRADING & BACKTESTING SYSTEM
================================================================================

A fully self-contained (single-file) quantitative research system that tests
whether 5-minute BTC price data contains a repeatable, economically meaningful
statistical edge AFTER realistic trading costs.

DESIGN PRINCIPLES
-----------------
* ZERO machine learning.  No sklearn / torch / tensorflow / xgboost / lightgbm,
  no trees, no neural nets, no learned weights, no ML clustering / regime models.
  Only mathematics, probability, statistics and classical time-series methods.
* ZERO conventional technical indicators (no RSI / MACD / EMA / SMA / ATR /
  Bollinger / Stochastic / ...).  Raw OHLCV and statistical transforms only.
  (Rolling means / std here are used strictly as *distribution statistics* to
   form z-scores and variances -- never as moving-average crossover signals.)
* Strict causality: every feature at bar t uses information available up to and
  including the close of bar t; a signal formed at t is executed at open[t+1].
* The final 1 year is a locked OUT-OF-SAMPLE (OOS) set.  All calibration
  (empirical probability tables, bucket edges, volatility thresholds) is fit on
  the IN-SAMPLE (IS) years ONLY, then frozen and applied unchanged to OOS.

CORE ENGINE
-----------
A non-ML *empirical probability engine*: for each strategy we discretise the
current statistical state into buckets, and from the IS history we measure the
empirical triple-barrier outcome distribution (P(TP), P(SL), P(timeout),
expected net return) of taking that strategy's proposed direction from that
state.  A trade is only taken when the frozen empirical expected value (net of
all modelled costs) is positive AND a directional probability edge exists AND
the deterministic regime filter agrees AND no structural break is active.

Everything (config, loading, validation, features, strategies, probability
engine, regime & change-point detection, ensemble, event-driven backtest,
costs, risk, walk-forward, robustness, Monte-Carlo, plots, CSV, report) lives
in THIS FILE and runs with:  python main.py
================================================================================
"""

from __future__ import annotations

import os
import sys
import json
import math
import time
import warnings
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless: we only save PNGs
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from scipy import stats as spstats

warnings.filterwarnings("ignore", category=RuntimeWarning)   # we guard math explicitly
pd.options.mode.chained_assignment = None


# =============================================================================
# 1. CONFIGURATION  (frozen before OOS; nothing here is tuned on OOS data)
# =============================================================================
@dataclass
class Config:
    # ---- data ----
    data_path: str = "btcusdt_5m_full.csv"
    bar_minutes: int = 5
    oos_days: int = 365                 # final 1 year is OOS
    min_history_bars: int = 5000        # fail if fewer bars than this

    # ---- capital / costs (realistic BTC perp taker assumptions) ----
    initial_capital: float = 10_000.0
    taker_fee: float = 0.0004           # 4 bps per side
    maker_fee: float = 0.0002           # 2 bps per side (reference only)
    slippage: float = 0.0001            # 1 bp per side
    half_spread: float = 0.0001         # 1 bp per side (dataset has no quotes -> assumed)
    funding_per_8h: float = 0.0001      # 1 bp / 8h, charged pro-rata as a cost (assumed)

    # ---- risk / sizing ----
    risk_per_trade: float = 0.005       # 0.5% of equity risked to the stop
    leverage_max: float = 3.0
    use_fixed_leverage: bool = True     # user-requested: flat leverage on every trade
    fixed_leverage: float = 3.0         # notional = fixed_leverage * equity (ignores risk sizing)
    target_trades_per_year: float = 900.0   # operational frequency target (800-1000); calibrated on IS only
    daily_loss_limit: float = 0.03      # halt new entries after -3% on the day
    max_consec_losses: int = 6
    consec_cooldown_bars: int = 24      # pause entries after a loss streak

    # ---- statistical windows (in bars; 288 bars = 1 day) ----
    win_dist: int = 288                 # return/price distribution window (z-scores)
    win_vol: int = 96                   # realized-vol window (8h)
    win_hurst: int = 288
    win_ac: int = 288                   # autocorrelation window
    win_ou: int = 288                   # OU / AR(1) window
    mom_horizons: tuple = (6, 12, 24)   # multi-horizon momentum (30m,1h,2h)

    # ---- barriers (volatility-scaled, NOT fixed arbitrary thresholds) ----
    k_sl: float = 6.0                   # stop = k_sl * per-bar realized vol
    k_tp: float = 8.0                   # target = k_tp * per-bar realized vol
    max_hold: int = 24                  # max holding period (bars) = 2h

    # ---- regime thresholds (deterministic) ----
    hurst_trend: float = 0.55
    hurst_mr: float = 0.45
    vol_high_q: float = 0.80
    vol_low_q: float = 0.20
    vol_extreme_q: float = 0.985        # extreme vol -> UNSTABLE (no trade)

    # ---- empirical probability engine gating ----
    n_bucket: int = 10                  # decile buckets per strategy signal
    min_samples: int = 150              # min observations for a probability cell
    ev_min: float = 0.0                 # required expected net return per trade
    prob_edge: float = 0.02             # required win-rate edge over 0.5

    # ---- change-point detection ----
    cusum_k: float = 0.5                # CUSUM slack (in std units)
    cusum_h: float = 10.0               # CUSUM threshold
    ph_delta: float = 0.5               # Page-Hinkley magnitude tolerance
    ph_lambda: float = 50.0             # Page-Hinkley threshold
    cp_cooldown_bars: int = 12          # no-trade window after a break

    # ---- validation / simulation ----
    wf_folds: int = 4                   # walk-forward folds on IS
    mc_sims: int = 10_000               # Monte-Carlo simulations (>= 10k required)
    mc_batch: int = 500                 # batch size to bound memory
    seed: int = 42

    # ---- evaluation thresholds (verdict only; NOT optimisation targets) ----
    pass_pf: float = 1.30
    pass_sharpe: float = 1.0
    pass_mc_ploss: float = 0.35

    @property
    def bars_per_day(self) -> int:
        return int(round(24 * 60 / self.bar_minutes))

    @property
    def bars_per_year(self) -> float:
        return 365.25 * self.bars_per_day


CFG = Config()

# strategy identifiers
S_MR, S_MOM, S_VOL, S_OU = 0, 1, 2, 3
STRAT_NAMES = {S_MR: "MEAN_REVERSION", S_MOM: "MOMENTUM",
               S_VOL: "VOL_EXPANSION", S_OU: "ORNSTEIN_UHLENBECK"}

# regime codes
R_NOTRADE, R_MR, R_TREND, R_HIGHVOL, R_LOWVOL, R_UNSTABLE = 0, 1, 2, 3, 4, 5
N_REGIME = 6
REGIME_NAMES = {R_NOTRADE: "NO_TRADE", R_MR: "MEAN_REVERTING", R_TREND: "TRENDING",
                R_HIGHVOL: "HIGH_VOLATILITY", R_LOWVOL: "LOW_VOLATILITY",
                R_UNSTABLE: "UNSTABLE"}

EPS = 1e-12
_T0 = time.time()


def log(msg: str = "") -> None:
    """Timestamped progress line."""
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


def hr(title: str = "") -> None:
    print("=" * 78)
    if title:
        print(title)
        print("=" * 78)


# =============================================================================
# 2. VECTORISED CAUSAL ROLLING STATISTICS
#    (all use a trailing window ending at bar t -> strictly causal)
# =============================================================================
def _rsum(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=w).sum()


def rolling_corr(x: pd.Series, y: pd.Series, w: int) -> pd.Series:
    """Causal Pearson correlation of x,y over trailing window w."""
    sx, sy = _rsum(x, w), _rsum(y, w)
    sxx, syy, sxy = _rsum(x * x, w), _rsum(y * y, w), _rsum(x * y, w)
    num = w * sxy - sx * sy
    den = np.sqrt((w * sxx - sx * sx).clip(lower=0) * (w * syy - sy * sy).clip(lower=0))
    return num / den.replace(0.0, np.nan)


def rolling_beta(x: pd.Series, y: pd.Series, w: int) -> pd.Series:
    """Causal OLS slope of x on y (x = a + b*y): b = cov(x,y)/var(y)."""
    sx, sy = _rsum(x, w), _rsum(y, w)
    syy, sxy = _rsum(y * y, w), _rsum(x * y, w)
    cov = w * sxy - sx * sy
    var = (w * syy - sy * sy)
    return cov / var.replace(0.0, np.nan)


def shift_neg(a: np.ndarray, k: int) -> np.ndarray:
    """Forward shift: out[t] = a[t+k], tail padded with NaN (used for barrier scan)."""
    out = np.full(a.shape, np.nan, dtype=float)
    if 0 < k < len(a):
        out[:-k] = a[k:]
    elif k == 0:
        out[:] = a
    return out


# =============================================================================
# 3. DATA LOADING + SCHEMA INSPECTION  (never blindly assume column names)
# =============================================================================
COLUMN_ALIASES = {
    "timestamp": ["timestamp", "datetime", "date", "time", "open_time", "ts", "date_time"],
    "open":      ["open", "o", "open_price"],
    "high":      ["high", "h", "high_price"],
    "low":       ["low", "l", "low_price"],
    "close":     ["close", "c", "close_price", "price"],
    "volume":    ["volume", "vol", "v", "base_volume", "base_asset_volume"],
    # optional microstructure / derivatives fields
    "quote_volume":     ["quote_volume", "quote_asset_volume", "quote_vol"],
    "trade_count":      ["trade_count", "trades", "number_of_trades", "count", "n_trades"],
    "taker_buy_volume": ["taker_buy_volume", "taker_buy_base_volume", "taker_buy_base_asset_volume"],
    "taker_sell_volume": ["taker_sell_volume", "taker_sell_base_volume"],
    "open_interest":    ["open_interest", "oi"],
    "funding_rate":     ["funding_rate", "funding"],
    "mark_price":       ["mark_price", "mark"],
    "index_price":      ["index_price", "index"],
    "bid":              ["bid", "best_bid", "bid_price"],
    "ask":              ["ask", "best_ask", "ask_price"],
    "spread":           ["spread"],
}
REQUIRED = ["timestamp", "open", "high", "low", "close", "volume"]


def detect_schema(columns) -> dict:
    """Map canonical field -> actual column name using alias table (case-insensitive)."""
    lower = {c.lower().strip(): c for c in columns}
    mapping = {}
    for canon, aliases in COLUMN_ALIASES.items():
        for a in aliases:
            if a in lower:
                mapping[canon] = lower[a]
                break
    return mapping


def load_data(cfg: Config) -> tuple[pd.DataFrame, dict]:
    hr("STEP 1/9  DATASET INSPECTION & LOADING")
    if not os.path.exists(cfg.data_path):
        raise FileNotFoundError(
            f"FATAL: data file not found: '{cfg.data_path}'. Place the 5-minute "
            f"BTC CSV next to main.py or edit Config.data_path.")

    header = pd.read_csv(cfg.data_path, nrows=0)
    log(f"Raw columns detected: {list(header.columns)}")
    schema = detect_schema(header.columns)
    log("Schema mapping (canonical -> actual):")
    for k, v in schema.items():
        print(f"    {k:18s} <- {v}")

    missing = [c for c in REQUIRED if c not in schema]
    if missing:  # FAIL LOUDLY
        raise ValueError(f"FATAL: could not identify required OHLCV columns: {missing}. "
                         f"Detected mapping: {schema}")

    usecols = [schema[c] for c in schema]
    df = pd.read_csv(cfg.data_path, usecols=usecols)
    inv = {v: k for k, v in schema.items()}
    df.rename(columns=inv, inplace=True)
    log(f"Loaded {len(df):,} rows.")

    # parse timestamp -> tz-aware UTC -> naive UTC index (deterministic, no DST)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    if df["timestamp"].isna().any():
        raise ValueError("FATAL: unparseable timestamps encountered.")
    df["timestamp"] = df["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)

    for c in ["open", "high", "low", "close", "volume",
              "quote_volume", "trade_count", "taker_buy_volume", "taker_sell_volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    optional_present = [c for c in schema if c not in REQUIRED]
    log(f"Optional fields present: {optional_present if optional_present else 'none'}")
    for f in ["funding_rate", "open_interest", "mark_price", "index_price", "spread", "bid", "ask"]:
        if f not in df.columns:
            pass
    log("NOTE: funding_rate / bid-ask / spread NOT in dataset -> modelled as "
        "configurable cost assumptions (see Config).")
    return df, schema


# =============================================================================
# 4. DATA VALIDATION  (fail loudly on structural corruption; warn on the rest)
# =============================================================================
def validate_data(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    hr("STEP 2/9  DATA VALIDATION")
    n0 = len(df)

    if df["timestamp"].duplicated().any():
        dups = int(df["timestamp"].duplicated().sum())
        log(f"WARNING: {dups} duplicate timestamps -> keeping first occurrence.")
        df = df[~df["timestamp"].duplicated(keep="first")]

    if not df["timestamp"].is_monotonic_increasing:
        log("WARNING: timestamps not sorted -> sorting chronologically.")
        df = df.sort_values("timestamp")
    df = df.reset_index(drop=True)

    ohlc = ["open", "high", "low", "close"]
    nan_mask = df[ohlc + ["volume"]].isna().any(axis=1)
    inf_mask = ~np.isfinite(df[ohlc + ["volume"]].to_numpy()).all(axis=1)
    bad_val = nan_mask | inf_mask
    if bad_val.any():
        log(f"WARNING: {int(bad_val.sum())} rows with NaN/Inf in OHLCV -> dropped.")
        df = df[~bad_val].reset_index(drop=True)

    hi = df["high"].to_numpy(); lo = df["low"].to_numpy()
    op = df["open"].to_numpy(); cl = df["close"].to_numpy(); vo = df["volume"].to_numpy()
    ohlc_bad = ((hi < np.maximum(op, cl) - 1e-6) | (lo > np.minimum(op, cl) + 1e-6) |
                (hi < lo) | (op <= 0) | (cl <= 0) | (hi <= 0) | (lo <= 0))
    if ohlc_bad.any():
        log(f"WARNING: {int(ohlc_bad.sum())} rows violate OHLC relationships -> dropped "
            f"(not silently repaired).")
        df = df[~ohlc_bad].reset_index(drop=True)

    if (df["volume"] < 0).any():
        log(f"WARNING: {int((df['volume'] < 0).sum())} negative-volume rows -> dropped.")
        df = df[df["volume"] >= 0].reset_index(drop=True)

    # timestamp-gap diagnostics
    step = pd.Timedelta(minutes=cfg.bar_minutes)
    d = df["timestamp"].diff()
    gaps = d[d > step]
    if len(gaps) > 0:
        big = gaps[gaps > 3 * step]
        log(f"INFO: {len(gaps)} gaps > 1 bar ({len(big)} exceed 3 bars). "
            f"Largest gap = {gaps.max()}. Gaps are tolerated (no synthetic bars inserted).")

    if len(df) < cfg.min_history_bars:
        raise ValueError(f"FATAL: only {len(df)} valid bars (< {cfg.min_history_bars}).")

    log(f"Validation complete: {len(df):,} clean bars ({n0 - len(df):,} removed).")
    log(f"Range: {df['timestamp'].iloc[0]}  ->  {df['timestamp'].iloc[-1]}  (UTC)")
    return df


def split_is_oos(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, int]:
    """Chronological split: last `oos_days` = OOS. Returns df + integer OOS start index."""
    hr("STEP 3/9  CHRONOLOGICAL IS / OOS SPLIT")
    end = df["timestamp"].iloc[-1]
    oos_start_ts = end - pd.Timedelta(days=cfg.oos_days)
    oos_start = int(np.searchsorted(df["timestamp"].to_numpy(),
                                    np.datetime64(oos_start_ts), side="left"))
    is_n, oos_n = oos_start, len(df) - oos_start
    log(f"IS : {df['timestamp'].iloc[0]}  ->  {df['timestamp'].iloc[oos_start-1]}  "
        f"({is_n:,} bars)")
    log(f"OOS: {df['timestamp'].iloc[oos_start]}  ->  {df['timestamp'].iloc[-1]}  "
        f"({oos_n:,} bars)  [LOCKED until strategy freeze]")
    return df, oos_start


# =============================================================================
# 5. STATISTICAL FEATURES  (all causal; no technical indicators)
# =============================================================================
def compute_features(df: pd.DataFrame, cfg: Config) -> dict:
    hr("STEP 4/9  STATISTICAL FEATURE ENGINE (causal, non-ML, no indicators)")
    close = df["close"]
    p = np.log(close)                                   # log price
    r = p.diff()                                        # log returns
    F: dict[str, np.ndarray] = {}

    # ---- return distribution & z-score / equilibrium deviation ----
    log("  returns, rolling distribution, z-scores ...")
    m = p.rolling(cfg.win_dist, min_periods=cfg.win_dist).mean()
    sd = p.rolling(cfg.win_dist, min_periods=cfg.win_dist).std()
    zmr = (p - m) / sd.replace(0.0, np.nan)             # deviation from stat. equilibrium
    rmean = r.rolling(cfg.win_dist, min_periods=cfg.win_dist).mean()
    rstd = r.rolling(cfg.win_dist, min_periods=cfg.win_dist).std()
    rz = (r - rmean) / rstd.replace(0.0, np.nan)        # standardised last return

    # ---- realized volatility (per-bar) & higher moments ----
    log("  realized volatility, skewness, kurtosis ...")
    rv = r.rolling(cfg.win_vol, min_periods=cfg.win_vol).std()
    skew = r.rolling(cfg.win_dist, min_periods=cfg.win_dist).skew()
    kurt = r.rolling(cfg.win_dist, min_periods=cfg.win_dist).kurt()

    # ---- multi-horizon momentum (standardised cumulative returns) ----
    log("  multi-horizon momentum & acceleration ...")
    mom_terms = []
    for h in cfg.mom_horizons:
        cr = p - p.shift(h)                             # h-bar log return
        z_h = cr / (rv * math.sqrt(h)).replace(0.0, np.nan)
        mom_terms.append(z_h)
    mom = sum(mom_terms) / len(mom_terms)
    # return acceleration: change in short momentum
    accel = (p - 2 * p.shift(cfg.mom_horizons[0]) + p.shift(2 * cfg.mom_horizons[0]))
    accel = accel / (rv * math.sqrt(cfg.mom_horizons[0])).replace(0.0, np.nan)

    # ---- autocorrelation (lag-1) & directional persistence ----
    log("  autocorrelation & directional persistence ...")
    ac1 = rolling_corr(r, r.shift(1), cfg.win_ac)
    up = (r > 0).astype(float)
    p_up = up.rolling(cfg.win_dist, min_periods=cfg.win_dist).mean()   # P(up)
    # directional persistence: correlation of sign(r_t), sign(r_{t-1})
    sgn = np.sign(r)
    dir_persist = rolling_corr(sgn, sgn.shift(1), cfg.win_ac)

    # ---- Shannon entropy of direction (uncertainty of the sign process) ----
    with np.errstate(divide="ignore", invalid="ignore"):
        pu = p_up.clip(1e-6, 1 - 1e-6)
        entropy = -(pu * np.log2(pu) + (1 - pu) * np.log2(1 - pu))

    # ---- Hurst exponent (generalised structure-function, vectorised) ----
    log("  Hurst exponent (structure-function) ...")
    lags = [1, 2, 4, 8, 16]
    lx = np.log(np.array(lags, dtype=float))
    lxbar = lx.mean()
    denomx = float(((lx - lxbar) ** 2).sum())
    ly_cols = []
    for k in lags:
        dk = p - p.shift(k)
        sdk = dk.rolling(cfg.win_hurst, min_periods=cfg.win_hurst).std()
        ly_cols.append(np.log(sdk.replace(0.0, np.nan)))
    LY = pd.concat(ly_cols, axis=1)
    lybar = LY.mean(axis=1)
    hnum = None
    for j in range(len(lags)):
        term = (lx[j] - lxbar) * (ly_cols[j] - lybar)
        hnum = term if hnum is None else (hnum + term)
    hurst = hnum / denomx

    # ---- Ornstein-Uhlenbeck: AR(1) on the equilibrium deviation ----
    log("  Ornstein-Uhlenbeck mean-reversion speed & half-life ...")
    beta = rolling_beta(zmr, zmr.shift(1), cfg.win_ou)      # AR(1) coefficient
    beta_c = beta.clip(-0.999999, 0.999999)
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = -np.log(beta_c.where(beta_c > 0))           # mean-reversion speed (per bar)
        half_life = np.log(2.0) / kappa                     # bars to revert halfway
    # significance proxy: t-stat of AR(1) autocorrelation
    t_ac = ac1 * np.sqrt((cfg.win_ac - 2) / (1 - (ac1 ** 2)).clip(lower=1e-6))

    # ---- order-flow imbalance (from taker buy/sell volume, if present) ----
    if "taker_buy_volume" in df.columns and "taker_sell_volume" in df.columns:
        tb = df["taker_buy_volume"]; tsell = df["taker_sell_volume"]
        ofi = (tb - tsell) / (tb + tsell).replace(0.0, np.nan)
        ofi_mean = ofi.rolling(cfg.win_vol, min_periods=cfg.win_vol).mean()
    else:
        ofi = pd.Series(np.zeros(len(df)))
        ofi_mean = pd.Series(np.zeros(len(df)))

    # store as numpy
    def arr(s):
        return np.asarray(s, dtype=float)

    F.update(dict(
        logp=arr(p), ret=arr(r), zmr=arr(zmr), rz=arr(rz), rv=arr(rv),
        skew=arr(skew), kurt=arr(kurt), mom=arr(mom), accel=arr(accel),
        ac1=arr(ac1), p_up=arr(p_up), dir_persist=arr(dir_persist),
        entropy=arr(entropy), hurst=arr(hurst), beta=arr(beta),
        kappa=arr(kappa), half_life=arr(half_life), t_ac=arr(t_ac),
        ofi=arr(ofi), ofi_mean=arr(ofi_mean),
    ))
    # warmup mask: first bar at which all core features are finite
    core = np.vstack([F["zmr"], F["rv"], F["mom"], F["ac1"], F["hurst"]])
    warm = np.where(np.isfinite(core).all(axis=0))[0]
    F["warmup"] = int(warm[0]) if len(warm) else len(df)
    log(f"  features ready. warm-up = {F['warmup']} bars.")
    return F


# =============================================================================
# 6. CHANGE-POINT DETECTION  (CUSUM + Page-Hinkley on standardised returns)
# =============================================================================
def detect_change_points(F: dict, cfg: Config) -> np.ndarray:
    """Return boolean cooldown array: True = structural break active -> no trade."""
    z = F["rz"].copy()
    z = np.where(np.isfinite(z), z, 0.0)
    n = len(z)
    cooldown = np.zeros(n, dtype=bool)

    gp = gn = 0.0                 # two-sided CUSUM accumulators
    ph_m = 0.0; ph_min = 0.0      # Page-Hinkley (mean increase)
    ph_m2 = 0.0; ph_max = 0.0     # Page-Hinkley (mean decrease)
    cd = 0
    k, h = cfg.cusum_k, cfg.cusum_h
    for t in range(n):
        x = z[t]
        gp = max(0.0, gp + x - k)
        gn = max(0.0, gn - x - k)
        ph_m += x - cfg.ph_delta; ph_min = min(ph_min, ph_m)
        ph_m2 += -x - cfg.ph_delta; ph_max = min(ph_max, ph_m2)
        broke = (gp > h) or (gn > h) or \
                ((ph_m - ph_min) > cfg.ph_lambda) or ((ph_m2 - ph_max) > cfg.ph_lambda)
        if broke:
            cd = cfg.cp_cooldown_bars
            gp = gn = 0.0
            ph_m = ph_min = ph_m2 = ph_max = 0.0
        if cd > 0:
            cooldown[t] = True
            cd -= 1
    log(f"  change-point cooldown active on {cooldown.mean()*100:5.2f}% of bars.")
    return cooldown


# =============================================================================
# 7. TRIPLE-BARRIER OUTCOMES  (vectorised; used for empirical calibration)
#    For a decision at bar t: entry fills at open[t+1]; barriers scaled by rv[t].
#    Conservative same-bar rule: if both barriers touch in one bar -> STOP first.
# =============================================================================
def compute_outcomes(df: pd.DataFrame, F: dict, cfg: Config) -> dict:
    hr("STEP 5/9  TRIPLE-BARRIER OUTCOME MATRIX (vectorised)")
    op = df["open"].to_numpy(float); hi = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float); cl = df["close"].to_numpy(float)
    n = len(op)
    entry = shift_neg(op, 1)                    # entry price for a decision at t
    sig = F["rv"]                               # decision-time per-bar volatility
    sl_ret = cfg.k_sl * sig
    tp_ret = cfg.k_tp * sig
    valid_sig = np.isfinite(entry) & np.isfinite(sig) & (sig > 0)

    # round-trip cost + funding modelled in return space (consistent w/ backtest)
    rt_cost = 2.0 * (cfg.taker_fee + cfg.slippage + cfg.half_spread)

    out = {}
    for side, name in [(+1, "long"), (-1, "short")]:
        if side > 0:
            tp_lvl = entry * (1 + tp_ret); sl_lvl = entry * (1 - sl_ret)
        else:
            tp_lvl = entry * (1 - tp_ret); sl_lvl = entry * (1 + sl_ret)
        done = np.zeros(n, dtype=bool)
        etype = np.zeros(n, dtype=np.int8)      # 1 tp, 2 sl, 3 timeout
        exitpx = np.full(n, np.nan)
        hold = np.zeros(n, dtype=np.int32)

        for mstep in range(1, cfg.max_hold + 1):
            him = shift_neg(hi, mstep); lom = shift_neg(lo, mstep); clm = shift_neg(cl, mstep)
            live = valid_sig & (~done) & np.isfinite(him)
            if side > 0:
                hit_sl = live & (lom <= sl_lvl)
                hit_tp = live & (him >= tp_lvl) & (~hit_sl)   # stop-first
            else:
                hit_sl = live & (him >= sl_lvl)
                hit_tp = live & (lom <= tp_lvl) & (~hit_sl)
            for mask, lvl_arr, code in [(hit_sl, sl_lvl, 2), (hit_tp, tp_lvl, 1)]:
                if mask.any():
                    exitpx[mask] = lvl_arr[mask]; etype[mask] = code
                    hold[mask] = mstep; done[mask] = True
            if mstep == cfg.max_hold:
                to = valid_sig & (~done) & np.isfinite(clm)
                exitpx[to] = clm[to]; etype[to] = 3; hold[to] = cfg.max_hold; done[to] = True

        with np.errstate(invalid="ignore", divide="ignore"):
            gross = side * (exitpx / entry - 1.0)
        funding = cfg.funding_per_8h * (hold * cfg.bar_minutes) / (8 * 60.0)
        net = gross - rt_cost - funding
        net = np.where(done & valid_sig, net, np.nan)
        out[name] = dict(gross=gross, net=net, etype=etype, hold=hold,
                         exitpx=exitpx, entry=entry)
    out["valid"] = valid_sig & (out["long"]["etype"] > 0)
    log(f"  outcomes computed for {int(out['valid'].sum()):,} decidable bars.")
    return out


# =============================================================================
# 8. PER-STRATEGY STATE (statistical conditioning variable; NO hard-coded side)
#    Direction is decided empirically from the frozen outcome tables (sec. 9-10).
#    Bucketing uses ONLY causal features.
# =============================================================================
def strategy_states(F: dict, cfg: Config) -> dict:
    """Per-strategy *conditioning variable* only.

    We deliberately do NOT hard-code a trade direction here.  Whether a given
    statistical state favours a long or a short is decided empirically from the
    frozen triple-barrier outcome tables (see fit_calibration / generate_signals):
    both directions are measured, and a side is taken only if its net expected
    value survives costs.  This keeps the engine faithful to the brief
    ("empirical conditional probabilities") and removes analyst priors about
    whether extremes revert or continue.

    Each strategy simply supplies the signed statistical state used to bucket
    the conditional distribution:
      * MEAN_REVERSION / OU  -> z-score of price vs its rolling equilibrium
      * MOMENTUM             -> standardised multi-horizon return
      * VOL_EXPANSION        -> standardised most-recent return (shock size/sign)
    """
    zmr, mom, rz = F["zmr"], F["mom"], F["rz"]
    return {
        S_MR:  dict(var=zmr),
        S_MOM: dict(var=mom),
        S_VOL: dict(var=rz),
        S_OU:  dict(var=zmr),
    }


def ou_valid_mask(F: dict, cfg: Config) -> np.ndarray:
    """OU logic only where mean-reversion is statistically supported."""
    beta, hl, t_ac = F["beta"], F["half_life"], F["t_ac"]
    return (np.isfinite(beta) & (beta > 0) & (beta < 1) &
            np.isfinite(hl) & (hl >= 2) & (hl <= cfg.max_hold * 3) &
            (t_ac < -1.5))            # significant negative lag-1 autocorr


def vol_flags(F: dict, cfg: Config, edges: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rv = F["rv"]
    hi = rv > edges["vol_high"]; lo = rv < edges["vol_low"]; ext = rv > edges["vol_extreme"]
    return hi, lo, ext


def regime_labels(F: dict, cfg: Config, edges: dict, cp_cooldown: np.ndarray) -> np.ndarray:
    """Deterministic regime classification from Hurst / autocorrelation / volatility."""
    hurst, ac1 = F["hurst"], F["ac1"]
    vhi, vlo, vext = vol_flags(F, cfg, edges)
    n = len(hurst)
    reg = np.full(n, R_NOTRADE, dtype=np.int8)
    trending = (hurst > cfg.hurst_trend) & (ac1 > 0)
    meanrev = (hurst < cfg.hurst_mr) & (ac1 < 0)
    finite = np.isfinite(hurst) & np.isfinite(ac1)
    # priority order
    reg = np.where(finite & meanrev & (~vhi), R_MR, reg)
    reg = np.where(finite & trending, R_TREND, reg)
    reg = np.where(finite & (reg == R_NOTRADE) & vhi, R_HIGHVOL, reg)
    reg = np.where(finite & (reg == R_NOTRADE) & vlo, R_LOWVOL, reg)
    reg = np.where(cp_cooldown | vext, R_UNSTABLE, reg)   # overrides everything
    return reg


# =============================================================================
# 9. EMPIRICAL PROBABILITY ENGINE  (fit on training rows ONLY, then frozen)
# =============================================================================
def fit_calibration(df: pd.DataFrame, F: dict, OUT: dict, states: dict,
                    train_mask: np.ndarray, cfg: Config) -> dict:
    """Build frozen bucket edges, volatility thresholds, and empirical outcome tables."""
    idx = np.where(train_mask)[0]

    # (a) volatility thresholds from training realized-vol distribution
    rv_tr = F["rv"][idx]
    rv_tr = rv_tr[np.isfinite(rv_tr)]
    edges = dict(
        vol_high=float(np.quantile(rv_tr, cfg.vol_high_q)),
        vol_low=float(np.quantile(rv_tr, cfg.vol_low_q)),
        vol_extreme=float(np.quantile(rv_tr, cfg.vol_extreme_q)),
    )

    # (b) decile bucket edges per strategy signal variable (training only)
    bucket_edges = {}
    qs = np.linspace(0, 1, cfg.n_bucket + 1)[1:-1]
    for s, sd in states.items():
        v = sd["var"][idx]
        v = v[np.isfinite(v)]
        e = np.quantile(v, qs) if len(v) else np.zeros(cfg.n_bucket - 1)
        bucket_edges[s] = e

    calib = dict(edges=edges, bucket_edges=bucket_edges)

    # regime labels on training rows require cooldown; recompute change-points once globally
    cp = calib.get("_cp")
    # NOTE: change-points are data-driven & causal -> computed once outside; passed via F
    reg = regime_labels(F, cfg, edges, F["_cp_cooldown"])

    # (c) empirical outcome tables, fit on training decisions only.
    # dims: [strategy, dir_idx(0=short,1=long), regime, bucket].  BOTH sides are
    # measured from the same states; the traded direction is chosen later from
    # whichever side's frozen expected value survives costs.
    shape = (4, 2, N_REGIME, cfg.n_bucket)
    sum_ret = np.zeros(shape); cnt = np.zeros(shape)
    sum_pos = np.zeros(shape); sum_tp = np.zeros(shape); sum_sl = np.zeros(shape)
    # coarse back-off tables
    c_shape = (4, 2, cfg.n_bucket)
    c_ret = np.zeros(c_shape); c_cnt = np.zeros(c_shape)
    c_pos = np.zeros(c_shape); c_tp = np.zeros(c_shape); c_sl = np.zeros(c_shape)
    cc_shape = (4, 2)
    cc_ret = np.zeros(cc_shape); cc_cnt = np.zeros(cc_shape); cc_pos = np.zeros(cc_shape)

    ou_ok = ou_valid_mask(F, cfg)
    tradeable = reg != R_UNSTABLE          # never calibrate on unstable / break bars
    base_valid = OUT["valid"] & train_mask & tradeable

    for s, sd in states.items():
        var = sd["var"]
        bucket = np.clip(np.digitize(var, bucket_edges[s]).astype(int),
                         0, cfg.n_bucket - 1)                    # 0..n_bucket-1
        dvalid = base_valid & np.isfinite(var)
        if s == S_OU:
            dvalid = dvalid & ou_ok
        rows = np.where(dvalid)[0]
        if len(rows) == 0:
            continue
        reg_r = reg[rows]; buc_r = bucket[rows]
        for di, side in ((1, +1), (0, -1)):     # measure long AND short outcomes
            L = OUT["long"] if side > 0 else OUT["short"]
            net = L["net"][rows]; etp = L["etype"][rows]
            good = np.isfinite(net)
            net_i = net[good]; etp_i = etp[good]
            buc_i = buc_r[good]; reg_i = reg_r[good]
            pos_i = (net_i > 0).astype(float)
            tp_i = (etp_i == 1).astype(float); sl_i = (etp_i == 2).astype(float)

            np.add.at(sum_ret, (s, di, reg_i, buc_i), net_i)
            np.add.at(cnt,     (s, di, reg_i, buc_i), 1.0)
            np.add.at(sum_pos, (s, di, reg_i, buc_i), pos_i)
            np.add.at(sum_tp,  (s, di, reg_i, buc_i), tp_i)
            np.add.at(sum_sl,  (s, di, reg_i, buc_i), sl_i)
            np.add.at(c_ret, (s, di, buc_i), net_i)
            np.add.at(c_cnt, (s, di, buc_i), 1.0)
            np.add.at(c_pos, (s, di, buc_i), pos_i)
            np.add.at(c_tp,  (s, di, buc_i), tp_i)
            np.add.at(c_sl,  (s, di, buc_i), sl_i)
            # coarsest (side-only) cell: scalar target -> accumulate directly
            cc_ret[s, di] += float(net_i.sum())
            cc_cnt[s, di] += float(net_i.size)
            cc_pos[s, di] += float(pos_i.sum())

    with np.errstate(invalid="ignore", divide="ignore"):
        ev = sum_ret / cnt; pup = sum_pos / cnt; ptp = sum_tp / cnt; psl = sum_sl / cnt
        c_ev = c_ret / c_cnt; c_pup = c_pos / c_cnt; c_ptp = c_tp / c_cnt; c_psl = c_sl / c_cnt
        cc_ev = cc_ret / cc_cnt; cc_pup = cc_pos / cc_cnt

    calib.update(dict(
        ev=ev, pup=pup, ptp=ptp, psl=psl, cnt=cnt,
        c_ev=c_ev, c_pup=c_pup, c_ptp=c_ptp, c_psl=c_psl, c_cnt=c_cnt,
        cc_ev=cc_ev, cc_pup=cc_pup, cc_cnt=cc_cnt,
    ))
    return calib


def _lookup_vec(calib, s, dir_idx, reg, buc, cfg):
    """Vectorised deterministic back-off lookup over ALL bars at once.

    For every bar, prefer the fine (regime x bucket) empirical cell; if it lacks
    the minimum sample count, back off to the regime-agnostic (bucket) cell, then
    to the coarsest (side-only) cell.  This is a pure table lookup on the FROZEN
    training statistics -- no fitting occurs here.

    Returns (ev, pup, ptp, psl, cnt), each a length-n array.
    """
    reg = reg.astype(int); buc = buc.astype(int)
    m = cfg.min_samples
    n = len(reg)

    f_cnt = calib["cnt"][s, dir_idx][reg, buc]
    c_cnt = calib["c_cnt"][s, dir_idx][buc]
    cc_cnt = calib["cc_cnt"][s, dir_idx]

    use_fine = f_cnt >= m
    use_coarse = (~use_fine) & (c_cnt >= m)
    use_cc = (~use_fine) & (~use_coarse) & (cc_cnt >= m)

    ev = np.full(n, np.nan); pup = np.full(n, np.nan)
    ptp = np.full(n, np.nan); psl = np.full(n, np.nan); cnt = np.zeros(n)

    if use_fine.any():
        ev[use_fine]  = calib["ev"][s, dir_idx][reg, buc][use_fine]
        pup[use_fine] = calib["pup"][s, dir_idx][reg, buc][use_fine]
        ptp[use_fine] = calib["ptp"][s, dir_idx][reg, buc][use_fine]
        psl[use_fine] = calib["psl"][s, dir_idx][reg, buc][use_fine]
        cnt[use_fine] = f_cnt[use_fine]
    if use_coarse.any():
        ev[use_coarse]  = calib["c_ev"][s, dir_idx][buc][use_coarse]
        pup[use_coarse] = calib["c_pup"][s, dir_idx][buc][use_coarse]
        ptp[use_coarse] = calib["c_ptp"][s, dir_idx][buc][use_coarse]
        psl[use_coarse] = calib["c_psl"][s, dir_idx][buc][use_coarse]
        cnt[use_coarse] = c_cnt[use_coarse]
    if use_cc.any():
        ev[use_cc]  = calib["cc_ev"][s, dir_idx]     # side-scalar cells (no ptp/psl)
        pup[use_cc] = calib["cc_pup"][s, dir_idx]
        cnt[use_cc] = cc_cnt
    return ev, pup, ptp, psl, cnt


# =============================================================================
# 10. SIGNAL GENERATION + DETERMINISTIC ENSEMBLE  (fully vectorised over bars)
# =============================================================================
def generate_signals(df: pd.DataFrame, F: dict, states: dict, calib: dict,
                     cfg: Config, eval_mask: np.ndarray, mode: str = "edge",
                     select_thresh: float = -np.inf) -> dict:
    """Produce the ensemble signal + per-bar logging arrays using FROZEN calibration.

    Direction is chosen EMPIRICALLY: for each strategy we look up the frozen
    expected value of BOTH a long and a short from the current state.  Regime is
    used purely as a conditioning key; the only regime we refuse outright is
    UNSTABLE (extreme vol / active structural break).

    mode="edge"  (the real strategy):
        A (strategy, side) candidate is eligible only if its net EV survives costs
        (ev>ev_min), it shows a genuine directional probability edge
        (pup>=0.5+prob_edge) and the cell is backed by >= min_samples observations.
        Deterministic ensemble (comfortable staying out): gather all eligible
        candidates; if some favour long AND some favour short -> conflict -> NO
        TRADE; otherwise take the single highest-EV candidate on the agreed side.

    mode="baseline"  (transparent no-edge diagnostic, NOT a recommended strategy):
        Drops the ev>ev_min and prob_edge requirements (keeps the sample-count
        validity check, the UNSTABLE exclusion, costs, risk and execution).  Each
        bar simply takes the single highest-EV statistically-valid state (long or
        short).  By construction those states have EV ~ -cost, so this is EXPECTED
        TO LOSE; it exists only to characterise OOS behaviour / costs and to
        populate the required OOS diagnostics rather than hide a null result
        behind zero trades.

    mode="selective"  (baseline + a frequency cap; still no proven edge):
        Same mechanics as baseline, but only bars whose best available EV is
        >= select_thresh are traded.  select_thresh is calibrated on IS ONLY to
        hit an operational trades/year target (a frequency target, NOT a profit
        target) and then frozen.  This keeps only the highest-EV ("least-bad")
        states; it does NOT manufacture an edge and is still expected to lose.
    """
    n = len(df)
    reg = regime_labels(F, cfg, calib["edges"], F["_cp_cooldown"])
    ou_ok = ou_valid_mask(F, cfg)
    tradeable = (reg != R_UNSTABLE) & eval_mask       # regime is a key, not a gate

    S_LIST = [S_MR, S_MOM, S_VOL, S_OU]
    nS = len(S_LIST)
    thr = 0.5 + cfg.prob_edge
    K = nS * 2                                        # candidate columns (strat x side)

    cand_ev = np.full((n, K), -np.inf)
    cand_pup = np.full((n, K), np.nan)
    cand_ptp = np.full((n, K), np.nan)
    cand_psl = np.full((n, K), np.nan)
    col_side = np.zeros(K, dtype=np.int8)
    col_strat = np.zeros(K, dtype=np.int8)

    col = 0
    for s in S_LIST:
        var = states[s]["var"]
        buc = np.clip(np.digitize(var, calib["bucket_edges"][s]).astype(int),
                      0, cfg.n_bucket - 1)
        base = tradeable & np.isfinite(var)
        if s == S_OU:
            base = base & ou_ok            # OU only where AR(1) mean-reversion is valid
        for di, side in ((1, +1), (0, -1)):    # 1 = long, 0 = short
            ev, pup, ptp, psl, cnt = _lookup_vec(calib, s, di, reg, buc, cfg)
            elig = base & np.isfinite(ev) & (cnt >= cfg.min_samples)
            if mode == "edge":
                elig = elig & (ev > cfg.ev_min) & np.isfinite(pup) & (pup >= thr)
            cand_ev[elig, col] = ev[elig]
            cand_pup[:, col] = pup; cand_ptp[:, col] = ptp; cand_psl[:, col] = psl
            col_side[col] = side; col_strat[col] = s
            col += 1

    gbest = cand_ev.max(axis=1); garg = cand_ev.argmax(axis=1)
    has_any = gbest > -np.inf

    if mode == "edge":
        long_cols = np.where(col_side > 0)[0]
        short_cols = np.where(col_side < 0)[0]
        has_long = cand_ev[:, long_cols].max(axis=1) > -np.inf
        has_short = cand_ev[:, short_cols].max(axis=1) > -np.inf
        valid = has_any & ~(has_long & has_short)     # conflicting evidence -> stand aside
    else:
        valid = has_any & (gbest >= select_thresh)    # baseline: -inf; selective: frozen cutoff

    signal = np.zeros(n, dtype=np.int8)
    chosen = np.full(n, -1, dtype=np.int8)
    log_ev = np.full(n, np.nan); log_pup = np.full(n, np.nan)
    log_ptp = np.full(n, np.nan); log_psl = np.full(n, np.nan)

    bars = np.where(valid)[0]
    if len(bars):
        c = garg[bars]
        signal[bars] = col_side[c]
        chosen[bars] = col_strat[c]
        log_ev[bars] = gbest[bars]
        log_pup[bars] = cand_pup[bars, c]
        log_ptp[bars] = cand_ptp[bars, c]
        log_psl[bars] = cand_psl[bars, c]

    return dict(signal=signal, regime=reg, chosen=chosen,
                ev=log_ev, pup=log_pup, ptp=log_ptp, psl=log_psl,
                gbest=gbest)


# =============================================================================
# 11. EVENT-DRIVEN BACKTEST  (one position at a time; entry at open[t+1])
#     Uses precomputed barrier outcomes for the fill path (SL-first) and applies
#     fixed-fractional risk sizing + explicit fees/slippage/funding + risk halts.
# =============================================================================
def backtest(df: pd.DataFrame, F: dict, OUT: dict, SIG: dict,
             lo_idx: int, hi_idx: int, cfg: Config,
             capital: float | None = None) -> dict:
    if capital is None:
        capital = cfg.initial_capital
    op = df["open"].to_numpy(float); cl = df["close"].to_numpy(float)
    ts = df["timestamp"].to_numpy()
    signal = SIG["signal"]; regime = SIG["regime"]; chosen = SIG["chosen"]
    rv = F["rv"]
    Llong, Lshort = OUT["long"], OUT["short"]
    rt_cost = 2.0 * (cfg.taker_fee + cfg.slippage + cfg.half_spread)

    equity = capital
    equity_bar = np.full(len(df), np.nan)
    trades = []
    t = max(lo_idx, F["warmup"] + 1)
    day = None; day_start_eq = equity; halted_today = False
    consec_losses = 0; cooldown = 0

    while t < hi_idx - 1:
        cur_day = pd.Timestamp(ts[t]).date()
        if cur_day != day:
            day = cur_day; day_start_eq = equity; halted_today = False
        equity_bar[t] = equity

        d = signal[t]
        blocked = halted_today or cooldown > 0
        if d == 0 or blocked:
            if cooldown > 0:
                cooldown -= 1
            t += 1
            continue

        side = int(d)
        L = Llong if side > 0 else Lshort
        if not (OUT["valid"][t] and np.isfinite(L["net"][t])):
            t += 1
            continue

        entry_px = L["entry"][t]
        sl_ret = cfg.k_sl * rv[t]
        if not (np.isfinite(entry_px) and np.isfinite(sl_ret) and sl_ret > 0):
            t += 1
            continue

        # position sizing
        if cfg.use_fixed_leverage:
            notional = cfg.fixed_leverage * equity          # flat leverage on every trade
        else:
            risk_amt = equity * cfg.risk_per_trade          # fixed-fractional risk, leverage-capped
            notional = min(risk_amt / sl_ret, cfg.leverage_max * equity)
        qty = notional / entry_px

        gross_ret = L["gross"][t]
        hold = int(L["hold"][t]); etype = int(L["etype"][t])
        exit_idx = min(t + hold, hi_idx - 1)

        gross_pnl = notional * gross_ret
        fees = cfg.taker_fee * notional * 2.0
        slip = (cfg.slippage + cfg.half_spread) * notional * 2.0   # spread folded into slippage
        funding = cfg.funding_per_8h * (hold * cfg.bar_minutes) / (8 * 60.0) * notional
        net_pnl = gross_pnl - fees - slip - funding
        equity += net_pnl

        exit_reason = {1: "take_profit", 2: "stop_loss", 3: "max_hold"}.get(etype, "max_hold")
        entry_fill = entry_px * (1 + side * (cfg.slippage + cfg.half_spread))
        exit_fill = L["exitpx"][t] * (1 - side * (cfg.slippage + cfg.half_spread))
        s_pick = int(chosen[t])
        trades.append(dict(
            timestamp=pd.Timestamp(ts[t]), entry_time=pd.Timestamp(ts[t + 1]),
            exit_time=pd.Timestamp(ts[exit_idx]),
            side=("LONG" if side > 0 else "SHORT"),
            entry_price=entry_fill, exit_price=exit_fill, quantity=qty,
            position_size=notional, gross_pnl=gross_pnl, fees=fees, slippage=slip,
            funding=funding, net_pnl=net_pnl, return_pct=net_pnl / equity_prev_guard(equity, net_pnl),
            holding_time_bars=hold, holding_minutes=hold * cfg.bar_minutes,
            exit_reason=exit_reason,
            strategy=STRAT_NAMES.get(s_pick, "ENSEMBLE"),
            regime=REGIME_NAMES.get(int(regime[t]), "NA"),
            signal_probability=float(SIG["pup"][t]) if np.isfinite(SIG["pup"][t]) else np.nan,
            expected_return=float(SIG["ev"][t]) if np.isfinite(SIG["ev"][t]) else np.nan,
            expected_value=float(SIG["ev"][t]) if np.isfinite(SIG["ev"][t]) else np.nan,
            equity_after=equity,
        ))

        # risk state updates
        if net_pnl < 0:
            consec_losses += 1
        else:
            consec_losses = 0
        if consec_losses >= cfg.max_consec_losses:
            cooldown = cfg.consec_cooldown_bars
            consec_losses = 0
        if equity <= day_start_eq * (1 - cfg.daily_loss_limit):
            halted_today = True

        for j in range(t, min(exit_idx + 1, len(df))):
            equity_bar[j] = equity
        t = exit_idx + 1        # one position at a time -> no overlap

    # fill equity curve forward -- SLICE to the tested window only so that
    # span/CAGR/drawdown-duration reflect the segment, not the whole dataset.
    seg_ts = pd.DatetimeIndex(df["timestamp"].iloc[lo_idx:hi_idx])
    seg_eq = equity_bar[lo_idx:hi_idx].copy()
    if len(seg_eq) and not np.isfinite(seg_eq[0]):
        seg_eq[0] = capital
    eqs = pd.Series(seg_eq, index=seg_ts).ffill().fillna(capital)
    tdf = pd.DataFrame(trades)
    return dict(trades=tdf, equity=eqs, final_equity=equity,
                lo=lo_idx, hi=hi_idx, capital=capital)


def equity_prev_guard(equity, net_pnl):
    prev = equity - net_pnl
    return prev if abs(prev) > EPS else EPS


def calibrate_select_thresh(df, F, OUT, states, calib, cfg, lo, hi) -> dict:
    """Find, ON IS ONLY, the best-EV cutoff that yields ~target_trades_per_year.

    This calibrates a FREQUENCY target (an operational knob), never a profit
    target: we scan candidate EV quantiles, run the IS backtest at each, and keep
    the cutoff whose IS trades/year is closest to cfg.target_trades_per_year.  The
    chosen threshold is returned so it can be FROZEN and applied once to OOS.
    """
    eval_mask = np.zeros(len(df), dtype=bool); eval_mask[lo:hi] = True
    scan = generate_signals(df, F, states, calib, cfg, eval_mask, mode="selective",
                            select_thresh=-np.inf)
    g = scan["gbest"][lo:hi]
    g = g[np.isfinite(g)]
    years = (pd.Timestamp(df["timestamp"].iloc[hi - 1]) -
             pd.Timestamp(df["timestamp"].iloc[lo])).total_seconds() / (86400.0 * 365.25)
    years = max(years, 1e-9)
    if g.size == 0:
        return dict(thresh=np.inf, is_tpy=0.0, is_trades=0, years=years)

    def _eval_q(q):
        thr = -np.inf if q <= 0.0 else float(np.quantile(g, q))
        sig = generate_signals(df, F, states, calib, cfg, eval_mask,
                               mode="selective", select_thresh=thr)
        bt = backtest(df, F, OUT, sig, lo, hi, cfg)
        n = int(len(bt["trades"])); tpy = n / years
        return dict(thresh=thr, is_tpy=tpy, is_trades=n, d=abs(tpy - cfg.target_trades_per_year), q=q)

    # coarse scan to bracket the target, then a fine refine around the best q so the
    # IS frequency lands close to target (a FREQUENCY calibration, IS-only, frozen).
    best = None
    for q in np.concatenate(([0.0], np.linspace(0.55, 0.99, 12))):
        r = _eval_q(q)
        if best is None or r["d"] < best["d"]:
            best = r
    lo_q, hi_q = max(0.0, best["q"] - 0.06), min(0.999, best["q"] + 0.06)
    for q in np.linspace(lo_q, hi_q, 13):
        r = _eval_q(q)
        if r["d"] < best["d"]:
            best = r
    best.pop("d", None); best["years"] = years
    return best


# =============================================================================
# 12. PERFORMANCE METRICS
# =============================================================================
def performance(bt: dict, cfg: Config, label: str = "") -> dict:
    tdf = bt["trades"]; eq = bt["equity"]; cap = bt["capital"]
    m = {"label": label, "initial_capital": cap, "final_capital": float(eq.iloc[-1]),
         "n_trades": int(len(tdf))}
    m["net_pnl"] = m["final_capital"] - cap
    m["total_return"] = m["final_capital"] / cap - 1.0

    span_days = max((eq.index[-1] - eq.index[0]).total_seconds() / 86400.0, 1e-9)
    years = span_days / 365.25
    m["years"] = years
    m["cagr"] = (m["final_capital"] / cap) ** (1 / years) - 1 if m["final_capital"] > 0 else -1.0

    # daily equity -> daily returns (for risk-adjusted ratios)
    daily = eq.resample("1D").last().ffill()
    dret = daily.pct_change().dropna()
    ann = math.sqrt(365.0)
    sd = dret.std()
    m["sharpe"] = float(dret.mean() / sd * ann) if sd > EPS else 0.0
    downside = dret[dret < 0].std()
    m["sortino"] = float(dret.mean() / downside * ann) if (downside and downside > EPS) else 0.0

    # drawdown on the bar-level equity curve
    peak = eq.cummax()
    dd = (eq - peak) / peak
    m["max_drawdown"] = float(dd.min())
    under = dd < -1e-9
    m["max_dd_duration_days"] = _max_underwater_days(eq.index, under.to_numpy())
    m["calmar"] = float(m["cagr"] / abs(m["max_drawdown"])) if m["max_drawdown"] < -EPS else 0.0
    m["recovery_factor"] = float(m["net_pnl"] / abs(m["max_drawdown"] * peak.max())) \
        if m["max_drawdown"] < -EPS else 0.0

    if len(tdf) == 0:
        m.update(dict(win_rate=0, profit_factor=0, expectancy=0, avg_trade=0, median_trade=0,
                      avg_winner=0, avg_loser=0, largest_winner=0, largest_loser=0,
                      trades_per_year=0, trades_per_month=0, trades_per_day=0,
                      long_trades=0, short_trades=0, winning=0, losing=0,
                      avg_hold_min=0, ret_std=0, skew=0, kurt=0, var5=0, es5=0,
                      max_consec_wins=0, max_consec_losses=0, time_in_market=0,
                      pos_neg_ratio=0, long_exposure=0, short_exposure=0,
                      avg_exposure=0, max_exposure=0))
        return m

    pnl = tdf["net_pnl"].to_numpy()
    wins = pnl[pnl > 0]; losses = pnl[pnl < 0]
    m["winning"] = int((pnl > 0).sum()); m["losing"] = int((pnl < 0).sum())
    m["win_rate"] = m["winning"] / len(pnl)
    m["profit_factor"] = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else float("inf")
    m["expectancy"] = float(pnl.mean())
    m["avg_trade"] = float(pnl.mean()); m["median_trade"] = float(np.median(pnl))
    m["avg_winner"] = float(wins.mean()) if len(wins) else 0.0
    m["avg_loser"] = float(losses.mean()) if len(losses) else 0.0
    m["largest_winner"] = float(pnl.max()); m["largest_loser"] = float(pnl.min())
    m["trades_per_year"] = len(pnl) / years
    m["trades_per_month"] = len(pnl) / (years * 12)
    m["trades_per_day"] = len(pnl) / span_days
    m["long_trades"] = int((tdf["side"] == "LONG").sum())
    m["short_trades"] = int((tdf["side"] == "SHORT").sum())
    m["avg_hold_min"] = float(tdf["holding_minutes"].mean())

    rp = tdf["return_pct"].to_numpy()
    m["ret_std"] = float(rp.std())
    m["skew"] = float(spstats.skew(rp)) if len(rp) > 2 else 0.0
    m["kurt"] = float(spstats.kurtosis(rp)) if len(rp) > 3 else 0.0
    m["var5"] = float(np.percentile(rp, 5))
    m["es5"] = float(rp[rp <= np.percentile(rp, 5)].mean()) if (rp <= np.percentile(rp, 5)).any() else 0.0
    m["pos_neg_ratio"] = float(m["winning"] / m["losing"]) if m["losing"] else float("inf")

    # consecutive wins/losses
    signs = np.sign(pnl); mcw = mcl = cw = cls = 0
    for s in signs:
        if s > 0:
            cw += 1; cls = 0; mcw = max(mcw, cw)
        elif s < 0:
            cls += 1; cw = 0; mcl = max(mcl, cls)
        else:
            cw = cls = 0
    m["max_consec_wins"] = int(mcw); m["max_consec_losses"] = int(mcl)

    # exposure
    total_bars = bt["hi"] - bt["lo"]
    bars_in = float(tdf["holding_time_bars"].sum())
    m["time_in_market"] = bars_in / total_bars if total_bars else 0.0
    m["time_out_market"] = 1 - m["time_in_market"]
    long_bars = float(tdf.loc[tdf["side"] == "LONG", "holding_time_bars"].sum())
    short_bars = float(tdf.loc[tdf["side"] == "SHORT", "holding_time_bars"].sum())
    m["long_exposure"] = long_bars / total_bars if total_bars else 0.0
    m["short_exposure"] = short_bars / total_bars if total_bars else 0.0
    m["avg_exposure"] = float((tdf["position_size"] / tdf["equity_after"]).mean())
    m["max_exposure"] = float((tdf["position_size"] / tdf["equity_after"]).max())
    return m


def _max_underwater_days(index, under: np.ndarray) -> float:
    best = cur_start = 0.0
    start_i = None
    for i, u in enumerate(under):
        if u and start_i is None:
            start_i = i
        elif not u and start_i is not None:
            dur = (index[i] - index[start_i]).total_seconds() / 86400.0
            best = max(best, dur); start_i = None
    if start_i is not None:
        dur = (index[-1] - index[start_i]).total_seconds() / 86400.0
        best = max(best, dur)
    return float(best)


# =============================================================================
# 13. WALK-FORWARD VALIDATION  (IS ONLY; re-fit calibration per fold)
# =============================================================================
def walk_forward(df, F, OUT, states, is_end, cfg) -> pd.DataFrame:
    hr("STEP 6/9  WALK-FORWARD VALIDATION (in-sample only)")
    warm = F["warmup"] + 1
    bounds = np.linspace(warm, is_end, cfg.wf_folds + 1).astype(int)
    rows = []
    for k in range(cfg.wf_folds):
        tr_lo, tr_hi = warm, bounds[k]           # anchored (expanding) training
        va_lo, va_hi = bounds[k], bounds[k + 1]
        if k == 0:                                # first fold: split its own range
            mid = (warm + bounds[1]) // 2
            tr_lo, tr_hi, va_lo, va_hi = warm, mid, mid, bounds[1]
        train_mask = np.zeros(len(df), dtype=bool); train_mask[tr_lo:tr_hi] = True
        eval_mask = np.zeros(len(df), dtype=bool); eval_mask[va_lo:va_hi] = True
        calib = fit_calibration(df, F, OUT, states, train_mask, cfg)
        sig = generate_signals(df, F, states, calib, cfg, eval_mask)
        bt = backtest(df, F, OUT, sig, va_lo, va_hi, cfg)
        pm = performance(bt, cfg, label=f"WF fold {k+1}")
        rows.append(dict(fold=k + 1,
                         train=f"{df['timestamp'].iloc[tr_lo].date()}..{df['timestamp'].iloc[tr_hi-1].date()}",
                         valid=f"{df['timestamp'].iloc[va_lo].date()}..{df['timestamp'].iloc[va_hi-1].date()}",
                         trades=pm["n_trades"], ret=pm["total_return"], pf=pm["profit_factor"],
                         sharpe=pm["sharpe"], win=pm["win_rate"], maxdd=pm["max_drawdown"]))
        log(f"  fold {k+1}: trades={pm['n_trades']:5d}  ret={pm['total_return']*100:7.2f}%  "
            f"PF={pm['profit_factor']:.2f}  Sharpe={pm['sharpe']:.2f}  win={pm['win_rate']*100:.1f}%")
    wf = pd.DataFrame(rows)
    if len(wf):
        prof = int((wf["ret"] > 0).sum())
        log(f"  WF summary: {prof}/{len(wf)} folds profitable; "
            f"median PF={wf['pf'].replace(np.inf,np.nan).median():.2f}, "
            f"median Sharpe={wf['sharpe'].median():.2f}")
    return wf


# =============================================================================
# 14. ROBUSTNESS  (IS ONLY; parameter perturbation & cost sensitivity)
# =============================================================================
def _run_is_pipeline(df, base_F, states, is_end, cfg) -> dict:
    """Fit on first 3/4 of IS, evaluate on last 1/4 of IS. Returns metrics."""
    warm = base_F["warmup"] + 1
    cut = warm + int((is_end - warm) * 0.75)
    OUT = compute_outcomes_quiet(df, base_F, cfg)
    train_mask = np.zeros(len(df), dtype=bool); train_mask[warm:cut] = True
    eval_mask = np.zeros(len(df), dtype=bool); eval_mask[cut:is_end] = True
    calib = fit_calibration(df, base_F, OUT, states, train_mask, cfg)
    sig = generate_signals(df, base_F, states, calib, cfg, eval_mask)
    bt = backtest(df, base_F, OUT, sig, cut, is_end, cfg)
    return performance(bt, cfg)


def compute_outcomes_quiet(df, F, cfg):
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return compute_outcomes(df, F, cfg)


def robustness(df, F, states, is_end, cfg) -> pd.DataFrame:
    hr("STEP 7/9  ROBUSTNESS & SENSITIVITY (in-sample only)")
    rows = []

    def record(name, cfg2, F2=None):
        Fu = F if F2 is None else F2
        pm = _run_is_pipeline(df, Fu, states, is_end, cfg2)
        rows.append(dict(variant=name, trades=pm["n_trades"], ret=pm["total_return"],
                         pf=pm["profit_factor"], sharpe=pm["sharpe"], maxdd=pm["max_drawdown"]))
        log(f"  {name:24s} trades={pm['n_trades']:5d}  ret={pm['total_return']*100:7.2f}%  "
            f"PF={pm['profit_factor']:.2f}  Sharpe={pm['sharpe']:.2f}")

    record("baseline", cfg)
    # barrier perturbations (only affect outcomes -> cheap; recomputed inside pipeline)
    for mult, tag in [(0.85, "k_sl-15%"), (1.15, "k_sl+15%")]:
        c = Config(**{**asdict(cfg), "k_sl": cfg.k_sl * mult}); record(tag, c)
    for mult, tag in [(0.85, "k_tp-15%"), (1.15, "k_tp+15%")]:
        c = Config(**{**asdict(cfg), "k_tp": cfg.k_tp * mult}); record(tag, c)
    # gating perturbations
    for val, tag in [(0.0, "prob_edge=0"), (0.04, "prob_edge=0.04")]:
        c = Config(**{**asdict(cfg), "prob_edge": val}); record(tag, c)
    # transaction-cost sensitivity (2x costs)
    c = Config(**{**asdict(cfg), "taker_fee": cfg.taker_fee * 2, "slippage": cfg.slippage * 2,
                  "half_spread": cfg.half_spread * 2}); record("costs x2", c)
    # window perturbation (needs feature recompute)
    c = Config(**{**asdict(cfg), "win_dist": int(cfg.win_dist * 1.5),
                  "win_vol": int(cfg.win_vol * 1.5)})
    F2 = compute_features_quiet(df, c); F2["_cp_cooldown"] = detect_cp_quiet(F2, c)
    record("windows x1.5", c, F2)

    rob = pd.DataFrame(rows)
    prof = int((rob["ret"] > 0).sum())
    log(f"  robustness: {prof}/{len(rob)} variants profitable on IS validation slice.")
    return rob


def compute_features_quiet(df, cfg):
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return compute_features(df, cfg)


def detect_cp_quiet(F, cfg):
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return detect_change_points(F, cfg)


# =============================================================================
# 15. MONTE-CARLO  (bootstrap resampling of OOS trade returns)
# =============================================================================
def monte_carlo(trade_returns: np.ndarray, cap: float, cfg: Config) -> dict:
    hr("STEP 8/9  MONTE-CARLO SIMULATION (bootstrap of OOS trades)")
    rng = np.random.default_rng(cfg.seed)
    n = len(trade_returns)
    if n < 5:
        log("  too few OOS trades for meaningful Monte-Carlo.")
        return dict(insufficient=True, n_trades=n)
    finals = np.empty(cfg.mc_sims); maxdds = np.empty(cfg.mc_sims)
    sample_paths = None
    done = 0
    while done < cfg.mc_sims:
        b = min(cfg.mc_batch, cfg.mc_sims - done)
        idx = rng.integers(0, n, size=(b, n))
        R = trade_returns[idx]                       # (b, n) resampled per-trade returns
        eq = cap * np.cumprod(1.0 + R, axis=1)
        eq = np.hstack([np.full((b, 1), cap), eq])   # prepend starting capital
        finals[done:done + b] = eq[:, -1]
        peak = np.maximum.accumulate(eq, axis=1)
        dd = (eq - peak) / peak
        maxdds[done:done + b] = dd.min(axis=1)
        if sample_paths is None:
            sample_paths = eq[:min(200, b)].copy()
        done += b
        log(f"  simulated {done:,}/{cfg.mc_sims:,} paths ...") if done % (cfg.mc_batch * 4) == 0 else None

    res = dict(insufficient=False, n_trades=n, finals=finals, maxdds=maxdds,
               sample_paths=sample_paths, cap=cap)
    res["median_final"] = float(np.median(finals))
    res["p5"] = float(np.percentile(finals, 5)); res["p25"] = float(np.percentile(finals, 25))
    res["p75"] = float(np.percentile(finals, 75)); res["p95"] = float(np.percentile(finals, 95))
    res["prob_loss"] = float((finals < cap).mean())
    res["prob_above_cap"] = float((finals >= cap).mean())
    res["prob_above_10pct"] = float((finals >= cap * 1.10).mean())
    res["prob_above_25pct"] = float((finals >= cap * 1.25).mean())
    res["median_maxdd"] = float(np.median(maxdds)); res["p95_maxdd"] = float(np.percentile(maxdds, 5))
    log(f"  P(loss)={res['prob_loss']*100:.1f}%  median final=${res['median_final']:,.0f}  "
        f"[5th ${res['p5']:,.0f}, 95th ${res['p95']:,.0f}]")
    return res


# =============================================================================
# 16. VISUALISATION  (8 PNG files; all generated from actual account equity)
# =============================================================================
def make_plots(df, is_bt, oos_bt, is_pm, oos_pm, mc, oos_start, cfg):
    hr("STEP 9/9  VISUALISATION & EXPORT")
    boundary = df["timestamp"].iloc[oos_start]

    # 1. equity_curve.png : IS, OOS, and combined with boundary
    #    NOTE: the principled EV>0 engine takes 0 trades (flat), so these equity
    #    curves depict the SELECTIVE ~900/yr + fixed-3x diagnostic on IS and OOS.
    lev = f"{cfg.fixed_leverage:.0f}x" if cfg.use_fixed_leverage else "risk-sized"
    tag = f"SELECTIVE ~{cfg.target_trades_per_year:.0f}/yr + {lev}"
    fig, ax = plt.subplots(3, 1, figsize=(13, 12))
    ax[0].plot(is_bt["equity"].index, is_bt["equity"].values, color="#1f77b4", lw=0.8)
    ax[0].set_title(f"In-Sample Equity Curve  [{tag} -- diagnostic, EV>0 engine took 0 trades]")
    ax[0].set_ylabel("Equity ($)")
    ax[1].plot(oos_bt["equity"].index, oos_bt["equity"].values, color="#2ca02c", lw=0.9)
    ax[1].set_title(f"Out-of-Sample Equity Curve  [{tag} -- diagnostic]")
    ax[1].set_ylabel("Equity ($)")
    combined = pd.concat([is_bt["equity"], oos_bt["equity"]])
    ax[2].plot(combined.index, combined.values, color="#333333", lw=0.7)
    ax[2].axvline(boundary, color="red", ls="--", lw=1.2, label="OOS boundary")
    ax[2].text(boundary, ax[2].get_ylim()[1], "  OUT-OF-SAMPLE", color="red", va="top", fontsize=9)
    ax[2].set_title(f"Combined IS + OOS Equity  [{tag}; the EV>0 engine stays flat at $10,000]")
    ax[2].set_ylabel("Equity ($)"); ax[2].legend(loc="upper left")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig("equity_curve.png", dpi=110); plt.close(fig)

    # 2. drawdown.png (OOS)
    eq = oos_bt["equity"]; dd = (eq - eq.cummax()) / eq.cummax() * 100
    fig, a = plt.subplots(figsize=(13, 4))
    a.fill_between(dd.index, dd.values, 0, color="#d62728", alpha=0.5)
    a.set_title("Out-of-Sample Drawdown Curve"); a.set_ylabel("Drawdown (%)")
    a.grid(alpha=0.3); fig.tight_layout(); fig.savefig("drawdown.png", dpi=110); plt.close(fig)

    # 3. oos_monthly_returns.png
    monthly = oos_bt["equity"].resample("ME").last().ffill().pct_change().dropna() * 100
    fig, a = plt.subplots(figsize=(13, 4))
    if len(monthly):
        colors = ["#2ca02c" if v >= 0 else "#d62728" for v in monthly.values]
        a.bar(range(len(monthly)), monthly.values, color=colors)
        a.set_xticks(range(len(monthly)))
        a.set_xticklabels([d.strftime("%Y-%m") for d in monthly.index], rotation=45, ha="right", fontsize=8)
    a.axhline(0, color="k", lw=0.8); a.set_title("OOS Monthly Returns (%)"); a.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig("oos_monthly_returns.png", dpi=110); plt.close(fig)

    # 4. oos_trade_distribution.png
    fig, a = plt.subplots(figsize=(11, 5))
    if len(oos_bt["trades"]):
        a.hist(oos_bt["trades"]["net_pnl"].values, bins=60, color="#1f77b4", alpha=0.8, edgecolor="k", lw=0.3)
        a.axvline(0, color="red", ls="--")
    a.set_title("OOS Trade Net P&L Distribution"); a.set_xlabel("Net P&L ($)"); a.set_ylabel("Count")
    a.grid(alpha=0.3); fig.tight_layout(); fig.savefig("oos_trade_distribution.png", dpi=110); plt.close(fig)

    # 5. oos_cumulative_pnl.png
    fig, a = plt.subplots(figsize=(13, 4))
    if len(oos_bt["trades"]):
        cum = oos_bt["trades"]["net_pnl"].cumsum().values
        a.plot(range(len(cum)), cum, color="#2ca02c", lw=1.0)
        a.axhline(0, color="k", lw=0.8)
    a.set_title("OOS Cumulative P&L (by trade #)"); a.set_xlabel("Trade #"); a.set_ylabel("Cumulative Net P&L ($)")
    a.grid(alpha=0.3); fig.tight_layout(); fig.savefig("oos_cumulative_pnl.png", dpi=110); plt.close(fig)

    # 6-8. Monte-Carlo
    if not mc.get("insufficient"):
        fig, a = plt.subplots(figsize=(12, 5))
        sp = mc["sample_paths"]
        for i in range(min(150, len(sp))):
            a.plot(sp[i], color="#1f77b4", alpha=0.06, lw=0.6)
        a.axhline(mc["cap"], color="k", ls="--", lw=1)
        a.set_title(f"Monte-Carlo Equity Paths (sample of {min(150,len(sp))} of {cfg.mc_sims:,})  [SIMULATED]")
        a.set_xlabel("Trade #"); a.set_ylabel("Equity ($)"); a.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig("monte_carlo_equity_paths.png", dpi=110); plt.close(fig)

        fig, a = plt.subplots(figsize=(11, 5))
        a.hist(mc["finals"], bins=80, color="#9467bd", alpha=0.8, edgecolor="k", lw=0.3)
        a.axvline(mc["cap"], color="red", ls="--", label="initial capital")
        a.axvline(mc["median_final"], color="k", ls="-", label="median")
        a.set_title(f"Monte-Carlo Final Equity Distribution ({cfg.mc_sims:,} sims)  [SIMULATED]")
        a.set_xlabel("Final Equity ($)"); a.set_ylabel("Frequency"); a.legend(); a.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig("monte_carlo_final_equity.png", dpi=110); plt.close(fig)

        fig, a = plt.subplots(figsize=(11, 5))
        a.hist(mc["maxdds"] * 100, bins=80, color="#d62728", alpha=0.8, edgecolor="k", lw=0.3)
        a.axvline(mc["median_maxdd"] * 100, color="k", ls="-", label="median")
        a.set_title(f"Monte-Carlo Max-Drawdown Distribution ({cfg.mc_sims:,} sims)  [SIMULATED]")
        a.set_xlabel("Max Drawdown (%)"); a.set_ylabel("Frequency"); a.legend(); a.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig("monte_carlo_drawdown.png", dpi=110); plt.close(fig)

    log("  saved: equity_curve.png, drawdown.png, oos_monthly_returns.png, "
        "oos_trade_distribution.png, oos_cumulative_pnl.png,")
    log("         monte_carlo_equity_paths.png, monte_carlo_final_equity.png, monte_carlo_drawdown.png")


# =============================================================================
# 17. LOOK-AHEAD SELF-CHECK  (numeric causality probe on a couple of features)
# =============================================================================
def lookahead_check(df, cfg) -> bool:
    """Recompute features on a truncated prefix; causal features must match exactly."""
    M = min(len(df), cfg.win_dist * 6)
    full = compute_features_quiet(df, cfg)
    trunc = compute_features_quiet(df.iloc[:M].copy(), cfg)
    ok = True
    for key in ["zmr", "rv", "mom", "hurst", "ac1"]:
        a = full[key][:M - 1]; b = trunc[key][:M - 1]
        both = np.isfinite(a) & np.isfinite(b)
        if both.sum() > 0 and not np.allclose(a[both], b[both], atol=1e-8, rtol=1e-6):
            ok = False
    return ok


# =============================================================================
# 18. TRADE LOG PRINTING + CSV EXPORT
# =============================================================================
def print_trade_samples(tdf: pd.DataFrame, n_detail: int = 3):
    if len(tdf) == 0:
        print("  (no OOS trades)")
        return
    print("\n  OOS trade table (first 12 rows):")
    cols = ["entry_time", "exit_time", "side", "entry_price", "exit_price",
            "net_pnl", "return_pct", "exit_reason", "strategy", "regime"]
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        show = tdf[cols].head(12).copy()
        show["entry_time"] = show["entry_time"].dt.strftime("%Y-%m-%d %H:%M")
        show["exit_time"] = show["exit_time"].dt.strftime("%Y-%m-%d %H:%M")
        show["return_pct"] = (show["return_pct"] * 100).round(3)
        for c in ["entry_price", "exit_price", "net_pnl"]:
            show[c] = show[c].round(2)
        print(show.to_string(index=False))

    for i in range(min(n_detail, len(tdf))):
        r = tdf.iloc[i]
        print("\n" + "=" * 60)
        print(f"OUT-OF-SAMPLE TRADE #{i+1:03d}")
        print("=" * 60)
        print(f"Entry Time:       {r['entry_time']}")
        print(f"Exit Time:        {r['exit_time']}")
        print(f"Side:             {r['side']}")
        print(f"Entry Price:      {r['entry_price']:.2f}")
        print(f"Exit Price:       {r['exit_price']:.2f}")
        print(f"Position Size:    ${r['position_size']:.2f}")
        print(f"Gross P&L:        ${r['gross_pnl']:.2f}")
        print(f"Fees:             ${r['fees']:.4f}")
        print(f"Slippage+Spread:  ${r['slippage']:.4f}")
        print(f"Funding:          ${r['funding']:.4f}")
        print(f"Net P&L:          ${r['net_pnl']:.2f}")
        print(f"Return:           {r['return_pct']*100:.3f}%")
        print(f"Holding Time:     {r['holding_minutes']:.0f} min")
        print(f"Exit Reason:      {r['exit_reason']}")
        print(f"Strategy:         {r['strategy']}")
        print(f"Regime:           {r['regime']}")
        print(f"Probability:      {r['signal_probability']:.3f}")
        print(f"Expected Return:  {r['expected_return']:.5f}")
        print(f"Expected Value:   {r['expected_value']:.5f}")
        print("=" * 60)


# =============================================================================
# 19. FINAL REPORT
# =============================================================================
def calib_gate_summary(calib: dict, cfg: Config) -> dict:
    """Summarise how many FROZEN fine cells clear the EV/probability gate.

    Pure inspection of the frozen training tables -- tunes nothing.  Used so the
    report's headline statistics are computed, never hard-coded.
    """
    thr = 0.5 + cfg.prob_edge
    cnt = calib["cnt"]; ev = calib["ev"]; pup = calib["pup"]   # dims [strat, side, regime, bucket]
    well = (cnt >= cfg.min_samples) & np.isfinite(ev)
    clear = well & (ev > cfg.ev_min) & np.isfinite(pup) & (pup >= thr)
    well_ev = ev[well]
    return dict(
        n_well=int(well.sum()),
        n_clear=int(clear.sum()),
        median_ev=float(np.median(well_ev)) if well_ev.size else float("nan"),
        best_ev=float(np.max(well_ev)) if well_ev.size else float("nan"),
        frac_pos=float((well_ev > 0).mean()) if well_ev.size else float("nan"),
        rt_cost=2.0 * (cfg.taker_fee + cfg.slippage + cfg.half_spread),
    )


def final_report(cfg, dinfo, oos_pm, mc, wf, rob, la_ok, verdict, suitable, engine_pm, cstat):
    hr()
    print("BTCUSD 5-MINUTE NON-ML STATISTICAL TRADING SYSTEM")
    print("FINAL OUT-OF-SAMPLE REPORT")
    hr()
    print(f"""
Data:
  5-year dataset ({dinfo['start']} -> {dinfo['end']})
  {dinfo['is_bars']:,}-bar development (in-sample)
  {dinfo['oos_bars']:,}-bar out-of-sample (final 1 year)

Strategy:            Non-ML Statistical Ensemble (MeanRev / Momentum / VolExp / OU)
Machine Learning:    STRICTLY DISABLED
Technical Indicators: NONE
""")
    print("-" * 60)
    print("PRINCIPLED ENGINE (EV>0 gate) -- THE ACTUAL STRATEGY")
    print("-" * 60)
    print(f"""
OOS Trades Taken:    {engine_pm['n_trades']:,}
OOS Net P&L:         ${engine_pm['net_pnl']:,.2f}
OOS Total Return:    {engine_pm['total_return']*100:.2f}%

The engine requires a frozen empirical expected value that survives all modelled
costs (round-trip ~{cstat['rt_cost']*100:.3f}%) plus a directional probability edge.
On the full 4-year in-sample calibration, {cstat['n_clear']} of {cstat['n_well']} well-sampled
statistical states clear that bar (median well-sampled cell EV = {cstat['median_ev']*100:+.4f}%,
best = {cstat['best_ev']*100:+.4f}%, only {cstat['frac_pos']*100:.2f}% even have EV>0).  The engine
therefore correctly takes {'no positions at all' if engine_pm['n_trades'] == 0 else 'the few positions shown above'}.
PRIMARY FINDING: at 5-minute frequency the conditional edge does not exceed
transaction costs.
""")
    print("-" * 60)
    print("OUT-OF-SAMPLE PERFORMANCE -- SELECTIVE ~900/YR + FIXED 3x (DIAGNOSTIC ONLY)")
    print("-" * 60)
    print(f"""
NOTE: the block below is NOT a proven strategy.  It is the user-requested trading
configuration: keep only the highest-EV ("least-bad") states via a cutoff
calibrated ON IS to ~{cfg.target_trades_per_year:.0f} trades/yr (a FREQUENCY target, frozen before
OOS), sized at a fixed {cfg.fixed_leverage:.0f}x leverage.  Selecting fewer, higher-ranked states
does NOT create a positive edge (the best IS state EV is only {cstat['best_ev']*100:+.4f}%, within
noise of zero), and {cfg.fixed_leverage:.0f}x leverage multiplies the per-trade outcome -- including
losses.  It is shown to report the result honestly and to populate the OOS
diagnostics, never as a tradeable edge.""")
    pf = oos_pm["profit_factor"]
    pf_s = "inf" if math.isinf(pf) else f"{pf:.2f}"
    print(f"""
Initial Capital:     ${oos_pm['initial_capital']:,.2f}
Final Capital:       ${oos_pm['final_capital']:,.2f}
Net P&L:             ${oos_pm['net_pnl']:,.2f}
Total Return:        {oos_pm['total_return']*100:.2f}%
CAGR:                {oos_pm['cagr']*100:.2f}%

Total Trades:        {oos_pm['n_trades']:,}
Trades/Year:         {oos_pm['trades_per_year']:,.0f}
Trades/Month:        {oos_pm['trades_per_month']:,.1f}
Trades/Day:          {oos_pm['trades_per_day']:,.2f}

Win Rate:            {oos_pm['win_rate']*100:.2f}%
Profit Factor:       {pf_s}
Expectancy:          ${oos_pm['expectancy']:.2f}/trade

Sharpe:              {oos_pm['sharpe']:.2f}
Sortino:             {oos_pm['sortino']:.2f}
Calmar:              {oos_pm['calmar']:.2f}

Maximum Drawdown:    {oos_pm['max_drawdown']*100:.2f}%
Max DD Duration:     {oos_pm['max_dd_duration_days']:.1f} days

Average Winner:      ${oos_pm['avg_winner']:.2f}
Average Loser:       ${oos_pm['avg_loser']:.2f}
Largest Winner:      ${oos_pm['largest_winner']:.2f}
Largest Loser:       ${oos_pm['largest_loser']:.2f}

Long Trades:         {oos_pm['long_trades']:,}
Short Trades:        {oos_pm['short_trades']:,}

Average Holding:     {oos_pm['avg_hold_min']:.0f} min
Time in Market:      {oos_pm['time_in_market']*100:.1f}%
""")
    print("-" * 60)
    print("ROBUSTNESS")
    print("-" * 60)
    if not mc.get("insufficient"):
        print(f"""
Monte Carlo Simulations:      {cfg.mc_sims:,}
Probability of Loss:          {mc['prob_loss']*100:.1f}%
Median Simulated Final Equity:${mc['median_final']:,.0f}
5th Percentile:               ${mc['p5']:,.0f}
95th Percentile:              ${mc['p95']:,.0f}""")
    else:
        print("\nMonte Carlo: insufficient OOS trades for simulation.")
    if len(wf):
        wf_prof = int((wf['ret'] > 0).sum())
        print(f"\nWalk-Forward (IS): {wf_prof}/{len(wf)} folds profitable, "
              f"median Sharpe {wf['sharpe'].median():.2f}")
    if len(rob):
        rob_prof = int((rob['ret'] > 0).sum())
        print(f"Robustness (IS):   {rob_prof}/{len(rob)} parameter variants profitable")
    print("\n" + "-" * 60)
    print("VALIDATION")
    print("-" * 60)
    print(f"""
Look-Ahead Bias:                 {'NONE DETECTED' if la_ok else 'WARNING - CHECK FAILED'}
Machine Learning:                NOT USED
Technical Indicators:            NOT USED
OOS Data Used During Optimization: NO
""")
    print("-" * 60)
    print("STATUS")
    print("-" * 60)
    print(f"""
OOS TEST RESULT:     {verdict}
Live Paper Testing:  {suitable}
""")
    hr()


# =============================================================================
# 20. MAIN
# =============================================================================
def main():
    np.random.seed(CFG.seed)
    hr("BTCUSD 5-MINUTE NON-ML STATISTICAL TRADING SYSTEM")
    print("Machine Learning: STRICTLY DISABLED   |   Technical Indicators: NONE")
    print(f"Dependencies: numpy {np.__version__}, pandas {pd.__version__}")

    # ---- load / validate / split ----
    df, schema = load_data(CFG)
    df = validate_data(df, CFG)
    df, oos_start = split_is_oos(df, CFG)

    dinfo = dict(start=df["timestamp"].iloc[0], end=df["timestamp"].iloc[-1],
                 is_bars=oos_start, oos_bars=len(df) - oos_start, total=len(df))

    # ---- reproducibility / config banner ----
    hr("CONFIGURATION (frozen before OOS)")
    print(f"""  Data Start:      {dinfo['start']}
  Data End:        {dinfo['end']}
  IS Start:        {df['timestamp'].iloc[0]}
  IS End:          {df['timestamp'].iloc[oos_start-1]}
  OOS Start:       {df['timestamp'].iloc[oos_start]}
  OOS End:         {dinfo['end']}
  Number of Candles: {len(df):,}
  Initial Capital: ${CFG.initial_capital:,.2f}
  Taker Fee:       {CFG.taker_fee*100:.3f}%   Maker Fee: {CFG.maker_fee*100:.3f}%
  Slippage:        {CFG.slippage*100:.3f}%   Half-Spread: {CFG.half_spread*100:.3f}%
  Funding/8h:      {CFG.funding_per_8h*100:.3f}%
  Leverage:        {('FIXED ' + format(CFG.fixed_leverage, '.1f') + 'x') if CFG.use_fixed_leverage else ('risk-sized, cap ' + format(CFG.leverage_max, '.1f') + 'x')}   Risk/Trade: {CFG.risk_per_trade*100:.2f}%
  Freq Target:     {CFG.target_trades_per_year:.0f} trades/yr (calibrated on IS only)
  Random Seed:     {CFG.seed}""")

    # ---- features + change points (computed once, causal) ----
    F = compute_features(df, CFG)
    F["_cp_cooldown"] = detect_change_points(F, CFG)
    OUT = compute_outcomes(df, F, CFG)
    states = strategy_states(F, CFG)

    is_end = oos_start

    # ---- development: walk-forward + robustness (IS ONLY) ----
    wf = walk_forward(df, F, OUT, states, is_end, CFG)
    rob = robustness(df, F, states, is_end, CFG)

    # ---- look-ahead self check ----
    la_ok = lookahead_check(df, CFG)
    log(f"Look-ahead causality self-check: {'PASS' if la_ok else 'FAIL'}")

    # =====================================================================
    # FREEZE: calibrate on the FULL 4-year IS, then run OOS exactly once.
    # =====================================================================
    hr("FREEZE STRATEGY / PARAMETERS / THRESHOLDS -> RUN OOS ONCE")
    train_mask = np.zeros(len(df), dtype=bool); train_mask[F["warmup"] + 1:is_end] = True
    calib = fit_calibration(df, F, OUT, states, train_mask, CFG)
    cstat = calib_gate_summary(calib, CFG)
    log(f"Frozen calibration: {cstat['n_clear']}/{cstat['n_well']} well-sampled states clear the "
        f"EV>0 + prob-edge gate (median cell EV {cstat['median_ev']*100:+.4f}%, "
        f"cost hurdle {cstat['rt_cost']*100:.3f}%).")

    # FREQUENCY calibration (IS ONLY): pick the best-EV cutoff that yields the
    # requested ~800-1000 trades/year, then FREEZE it for the OOS run.  This tunes
    # only how often we trade -- not whether the trades are profitable.
    sel = calibrate_select_thresh(df, F, OUT, states, calib, CFG, F["warmup"] + 1, is_end)
    sel_thresh = sel["thresh"]
    log(f"Frozen frequency cutoff: best-EV >= {sel_thresh*100:+.4f}%  "
        f"(IS achieves {sel['is_tpy']:.0f} trades/yr; target {CFG.target_trades_per_year:.0f}; "
        f"fixed {CFG.fixed_leverage:.0f}x leverage).")

    # in-sample run (full-IS calibration) for the IS equity curve / comparison
    is_eval = np.zeros(len(df), dtype=bool); is_eval[F["warmup"] + 1:is_end] = True
    is_sig = generate_signals(df, F, states, calib, CFG, is_eval)
    is_bt = backtest(df, F, OUT, is_sig, F["warmup"] + 1, is_end, CFG)
    is_pm = performance(is_bt, CFG, "IN-SAMPLE")
    log(f"IS (reference): trades={is_pm['n_trades']}, return={is_pm['total_return']*100:.2f}%, "
        f"PF={is_pm['profit_factor']:.2f}, Sharpe={is_pm['sharpe']:.2f}")

    # =====================================================================
    # OUT-OF-SAMPLE, evaluated exactly once on the FROZEN calibration.
    # Two views on the SAME frozen tables (no OOS optimisation of anything):
    #   (1) principled EV>0 engine -> the actual edge test / headline verdict;
    #   (2) user-requested trading config: SELECTIVE (frozen frequency cutoff ->
    #       ~800-1000 trades/yr) with FIXED 3x leverage.  The selectivity keeps
    #       only the highest-EV ("least-bad") states; it does NOT create an edge,
    #       so it is still expected to lose -- reported honestly, not hidden.
    # =====================================================================
    oos_eval = np.zeros(len(df), dtype=bool); oos_eval[oos_start:] = True

    edge_sig = generate_signals(df, F, states, calib, CFG, oos_eval, mode="edge")
    edge_bt = backtest(df, F, OUT, edge_sig, oos_start, len(df), CFG)
    edge_pm = performance(edge_bt, CFG, "OOS-ENGINE")
    log(f"OOS principled engine (EV>0 gate): {edge_pm['n_trades']} trades "
        f"-> no cost-surviving edge; the engine stays flat.")

    sel_sig = generate_signals(df, F, states, calib, CFG, oos_eval,
                               mode="selective", select_thresh=sel_thresh)
    sel_bt = backtest(df, F, OUT, sel_sig, oos_start, len(df), CFG)
    sel_pm = performance(sel_bt, CFG, "OOS-SELECTIVE-3x")
    log(f"OOS selective (~{CFG.target_trades_per_year:.0f}/yr target, fixed "
        f"{CFG.fixed_leverage:.0f}x): trades={sel_pm['n_trades']:,} "
        f"({sel_pm['trades_per_year']:.0f}/yr), return={sel_pm['total_return']*100:.2f}%, "
        f"PF={sel_pm['profit_factor']:.2f}, Sharpe={sel_pm['sharpe']:.2f}  "
        f"[selecting least-bad states cannot manufacture an edge -- reported for honesty]")

    # Populated OOS diagnostics (trade log / metrics / Monte-Carlo / plots) use the
    # user-requested selective+leveraged config; the headline verdict uses the engine.
    oos_bt, oos_pm = sel_bt, sel_pm

    # ---- export OOS trades (selective config; the EV>0 engine produced none) ----
    tdf = oos_bt["trades"]
    if len(tdf):
        tdf.to_csv("out_of_sample_trades.csv", index=False)
        log(f"Saved {len(tdf):,} OOS trades -> out_of_sample_trades.csv")
    else:
        pd.DataFrame().to_csv("out_of_sample_trades.csv", index=False)
        log("No OOS trades generated -> wrote empty out_of_sample_trades.csv")

    # ---- Monte-Carlo on OOS trade returns ----
    tret = tdf["return_pct"].to_numpy() if len(tdf) else np.array([])
    mc = monte_carlo(tret, CFG.initial_capital, CFG)

    # ---- plots ----
    # Plot the SELECTIVE (frozen cutoff, 3x) config on BOTH IS and OOS so the equity
    # curves depict one consistent strategy across the boundary (EV>0 engine is flat).
    is_sel_sig = generate_signals(df, F, states, calib, CFG, is_eval,
                                  mode="selective", select_thresh=sel_thresh)
    is_sel_bt = backtest(df, F, OUT, is_sel_sig, F["warmup"] + 1, is_end, CFG)
    make_plots(df, is_sel_bt, oos_bt, is_pm, oos_pm, mc, oos_start, CFG)

    # ---- OOS trade log ----
    hr("OUT-OF-SAMPLE TRADE LOG  (selective ~900/yr + 3x -- principled engine took 0 trades)")
    print_trade_samples(tdf)

    # ---- save frozen config ----
    with open("frozen_config.json", "w") as fh:
        json.dump({**asdict(CFG),
                   "data_start": str(dinfo["start"]), "data_end": str(dinfo["end"]),
                   "oos_start": str(df["timestamp"].iloc[oos_start]),
                   "n_candles": int(len(df))}, fh, indent=2)
    log("Saved frozen_config.json")

    # ---- verdict (deterministic; evaluation thresholds, NOT tuned) ----
    # The strategy is the principled EV>0 engine.  If it declines to trade, that
    # itself is a decisive negative finding (no cost-surviving edge exists), which
    # the losing baseline then confirms empirically.  We NEVER relax the gate to
    # manufacture a passing result.
    engine_n = edge_pm["n_trades"]
    ploss = mc.get("prob_loss", 1.0) if not mc.get("insufficient") else 1.0
    if engine_n == 0:
        verdict = "FAIL - no repeatable, cost-surviving edge (engine took 0 trades)"
        suitable = "NOT SUITABLE"
    elif engine_n < 30:
        verdict = "INCONCLUSIVE (engine took too few OOS trades)"
        suitable = "NOT SUITABLE"
    elif edge_pm["net_pnl"] > 0 and edge_pm["profit_factor"] >= CFG.pass_pf \
            and edge_pm["sharpe"] >= CFG.pass_sharpe and ploss <= CFG.pass_mc_ploss:
        verdict = "PASS"
        suitable = "SUITABLE (proceed to paper trading with caution)"
    elif edge_pm["net_pnl"] <= 0 or edge_pm["profit_factor"] < 1.0:
        verdict = "FAIL"
        suitable = "NOT SUITABLE"
    else:
        verdict = "INCONCLUSIVE"
        suitable = "INCONCLUSIVE (marginal edge; more testing required)"

    final_report(CFG, dinfo, oos_pm, mc, wf, rob, la_ok, verdict, suitable, edge_pm, cstat)
    log("DONE.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n" + "!" * 78)
        print(f"FATAL ERROR: {type(e).__name__}: {e}")
        print("!" * 78)
        import traceback
        traceback.print_exc()
        sys.exit(1)