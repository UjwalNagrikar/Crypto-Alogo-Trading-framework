#!/usr/bin/env python3
# BTCUSD 5-Minute Statistical Trading Strategy
# Institutional-grade quantitative trading system.
# Based on probability, statistics, stochastic processes, and market microstructure.
# No machine learning. No lagging technical indicators.

from __future__ import annotations
import os, sys, warnings, math
from dataclasses import dataclass
from typing import Optional
from datetime import datetime, timedelta
from collections import deque
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
warnings.filterwarnings('ignore', category=RuntimeWarning)
try:
    warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning)
except AttributeError:
    pass

@dataclass
class Config:
    data_path: str = 'BTCUSDFeaturesdata_5m.csv'
    initial_capital: float = 15000.0
    min_leverage: float = 1.0
    max_leverage: float = 5.0
    base_risk_per_trade: float = 0.015
    max_risk_per_trade: float = 0.03
    risk_reduction_after_loss: float = 0.85
    max_daily_loss_pct: float = 0.05
    max_weekly_loss_pct: float = 0.12
    max_portfolio_drawdown: float = 0.20
    kelly_fraction: float = 0.25
    bayesian_prob_threshold: float = 0.55
    min_liquidity_score: float = 0.3
    max_volatility_percentile: float = 0.85
    min_expected_value: float = 0.0002
    return_windows: tuple = (1, 2, 4, 8)
    zscore_window: int = 20
    skew_window: int = 20
    kurt_window: int = 20
    entropy_window: int = 20
    realized_vol_window: int = 6
    ewma_vol_span: int = 20
    vol_percentile_window: int = 100
    vwap_window: int = 20
    liquidity_window: int = 20
    kalman_delta: float = 0.01
    kalman_observation_noise: float = 0.1
    ou_window: int = 40
    hurst_window: int = 64
    regime_vol_window: int = 48
    maker_fee: float = 0.0001
    taker_fee: float = 0.0004
    slippage_bps: float = 0.00015
    partial_fill_prob: float = 0.05
    partial_fill_ratio: float = 0.5
    mc_simulations: int = 2000
    mc_confidence: float = 0.95
    chart_dir: str = 'charts'
    trade_log_max_rows: int = 100   # cap on rows printed to terminal (None = print all)
    # --- trade-frequency calibration target (added) ---
    target_trades_min: int = 1200
    target_trades_max: int = 3000

class DataLoader:
    """Load, validate, and clean OHLCV + microstructure data."""
    def __init__(self, c): self.c = c; self.data = None
    def load(self):
        df = pd.read_csv(self.c.data_path); self.data = df
        self._validate(); self._clean(); self._add_missing()
        return self.data
    def _validate(self):
        df = self.data
        core = {'timestamp','open','high','low','close','volume'}
        missing = core - set(df.columns)
        if missing: raise ValueError(f'Missing: {missing}')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        dc = df['timestamp'].duplicated().sum()
        if dc: print(f'  Removed {dc} duplicates'); df.drop_duplicates(subset='timestamp',keep='first',inplace=True)
        if not df['timestamp'].is_monotonic_increasing:
            print('  Sorting timestamps'); df.sort_values('timestamp',inplace=True); df.reset_index(drop=True,inplace=True)
        print(f'  {len(df)} rows, {df["timestamp"].min()} -> {df["timestamp"].max()}')
    def _clean(self):
        df = self.data
        for col in ['open','high','low','close','volume','bid','ask','bid_ask_spread']:
            if col in df.columns:
                nc = df[col].isnull().sum()
                if nc: print(f'  Filled {nc} nulls in {col}'); df[col] = df[col].ffill()
        bad = (df['high'] < df['low']) | (df['open'] < 0) | (df['close'] < 0)
        if bad.sum(): print(f'  Removed {bad.sum()} bad ticks'); df = df[~bad]
        jump = df['close'].pct_change().abs() > 0.15
        if jump.sum(): print(f'  Removed {jump.sum()} extreme moves'); df = df[~jump]
        if 'bid' in df.columns and 'ask' in df.columns:
            sb = df['bid'] >= df['ask']
            if sb.sum(): print(f'  Removed {sb.sum()} bid>=ask'); df = df[~sb]
        self.data = df.reset_index(drop=True)
    def _add_missing(self):
        df = self.data
        for c in ['open_interest','market_price','index_price']:
            if c not in df.columns: df[c] = np.nan
        if 'market_price' in df.columns: df['market_price'] = df['market_price'].fillna(df['close'])
        if 'index_price' in df.columns: df['index_price'] = df['index_price'].fillna(df['close'])


class FeatureEngine:
    """Compute 20 statistical features. No ML or lagging indicators."""
    def __init__(self, c): self.c = c
    def compute(self, df):
        R = df.copy(); p = R['close'].values.astype(float); v = R['volume'].values.astype(float); n = len(p)
        lr = np.full(n, np.nan); lr[1:] = np.log(p[1:] / p[:-1]); R['lr1'] = lr
        for per in self.c.return_windows[1:]:
            a = np.full(n, np.nan); a[per:] = np.log(p[per:] / p[:-per]); R[f'lr{per}'] = a
        vel = np.full(n, np.nan); vel[2:] = lr[2:] - lr[1:-1]; R['vel'] = vel
        acc = np.full(n, np.nan); acc[3:] = vel[3:] - vel[2:-1]; R['acc'] = acc
        w = self.c.zscore_window
        if n > w:
            rm = pd.Series(lr).rolling(w).mean().values; rs = pd.Series(lr).rolling(w).std(ddof=0).values
            m = rs > 1e-12; zs = np.full(n, np.nan); zs[m] = (lr[m] - rm[m]) / rs[m]; R['zsc'] = zs
        else: R['zsc'] = np.full(n, np.nan)
        R['skw'] = pd.Series(lr).rolling(20).skew().values; R['kur'] = pd.Series(lr).rolling(20).kurt().values
        ev = np.full(n, np.nan)
        for i in range(20, n):
            wd = lr[i-19:i+1]; wd = wd[~np.isnan(wd)]
            if len(wd) > 5:
                cts, _ = np.histogram(wd, bins=10, density=True); cts = cts[cts > 0]; ev[i] = -np.sum(cts * np.log(cts))
        R['ent'] = ev; kd = self._kalman(p); R['kld'] = kd; R['oud'] = self._oud(kd)
        rv = np.full(n, np.nan)
        for i in range(6, n):
            wd = lr[i-5:i+1]; wd = wd[~np.isnan(wd)]
            if len(wd) > 0: rv[i] = np.sqrt(np.sum(wd**2))
        R['rv6'] = rv; R['evol'] = self._ewma(lr, 20)
        vp = np.full(n, np.nan); evl = R['evol'].values
        for i in range(100, n):
            wd = evl[i-99:i+1]; wd = wd[~np.isnan(wd)]
            if len(wd) > 0: vp[i] = stats.percentileofscore(wd, evl[i]) / 100.0
        R['vpct'] = vp; R['ofi'] = self._ofi(R)
        sv = np.full(n, np.nan)
        if 'taker_buy_volume' in R.columns:
            tb = R['taker_buy_volume'].values.astype(float); ts = R['taker_sell_volume'].values.astype(float)
            tot = tb + ts; m = tot > 0; sv[m] = (tb[m] - ts[m]) / tot[m]
        R['sv'] = sv; R['qi'] = self._qi(R); R['vwap'] = self._vwap(p, v); R['liq'] = self._liq(R)
        R['bprob'] = self._bayes(R); R['eval'] = self._expv(R)
        return R
    def _kalman(self, p):
        n = len(p); d = self.c.kalman_delta; on = self.c.kalman_observation_noise
        xe = p[0]; pe = 1.0; dev = np.full(n, np.nan); dev[0] = 0.0
        for i in range(1, n):
            xp = xe; pp = pe + d; k = pp / (pp + on); xe = xp + k * (p[i] - xp); pe = (1 - k) * pp; dev[i] = (p[i] - xe) / max(xe, 1.0)
        return dev
    def _oud(self, dev):
        n = len(dev); w = self.c.ou_window; ou = np.full(n, np.nan)
        if n <= w: return ou
        for i in range(w, n):
            s = dev[i-w+1:i+1]; s = s[~np.isnan(s)]
            if len(s) < 10: continue
            std = np.std(s, ddof=1); m = np.mean(s)
            if std > 1e-12: ou[i] = (dev[i] - m) / std
        return ou
    def _ewma(self, lr, span):
        n = len(lr); a = 2.0 / (span + 1); vol = np.full(n, np.nan)
        if n > span:
            iv = np.var(lr[1:span+1], ddof=0); vol[span] = np.sqrt(max(iv, 1e-12)); ve = iv
            for i in range(span+1, n):
                if not np.isnan(lr[i]): ve = (1 - a) * ve + a * (lr[i] ** 2)
                vol[i] = np.sqrt(max(ve, 1e-12))
        return vol
    def _ofi(self, df):
        n = len(df); ofi = np.full(n, np.nan)
        if 'taker_buy_volume' not in df.columns: return ofi
        tb = df['taker_buy_volume'].values.astype(float); ts = df['taker_sell_volume'].values.astype(float)
        for i in range(5, n):
            bs = np.nansum(tb[i-4:i+1]); ss = np.nansum(ts[i-4:i+1]); t = bs + ss
            if t > 0: ofi[i] = (bs - ss) / t
        return ofi
    def _qi(self, df):
        n = len(df); qi = np.full(n, np.nan)
        if 'bid_ask_spread' not in df.columns: return qi
        sp = df['bid_ask_spread'].values.astype(float); vo = df['volume'].values.astype(float)
        for i in range(20, n):
            w = sp[i-19:i+1]; w = w[~np.isnan(w)]
            if len(w) > 0 and np.std(w) > 0:
                spct = stats.percentileofscore(w, sp[i]) / 100.0
                vn = min(vo[i] / (np.nanmean(vo[max(0,i-50):i+1]) + 1e-8), 3.0)
                qi[i] = (spct - 0.5) * 2 * min(vn, 1.0)
        return qi
    def _vwap(self, p, v):
        n = len(p); w = self.c.vwap_window; dev = np.full(n, np.nan)
        wp = deque(maxlen=w); wv = deque(maxlen=w)
        for i in range(n):
            wp.append(p[i] * v[i]); wv.append(v[i]); cp = sum(wp); cv = sum(wv)
            if cv > 1e-8: dev[i] = (p[i] - cp/cv) / (cp/cv)
        return dev
    def _liq(self, df):
        n = len(df); sc = np.full(n, np.nan)
        sp = df['bid_ask_spread'].values.astype(float) if 'bid_ask_spread' in df.columns else np.ones(n) * 0.1
        vo = df['volume'].values.astype(float)
        tc = df['trade_count'].values.astype(float) if 'trade_count' in df.columns else np.ones(n) * 10
        for i in range(20, n):
            sw = sp[i-19:i+1]; sw = sw[~np.isnan(sw)]
            vw = vo[i-19:i+1]; vw = vw[~np.isnan(vw)]
            if len(sw) > 0 and len(vw) > 0:
                sps = 1.0 - min(sp[i] / (np.mean(sw) + 1e-12), 10.0) / 10.0
                vs = min(vo[i] / (np.mean(vw) + 1e-12), 3.0) / 3.0
                ts = min(tc[i] / (np.mean(tc[max(0,i-20):i+1]) + 1e-8), 3.0) / 3.0
                sc[i] = 0.5 * sps + 0.3 * vs + 0.2 * ts
        return sc
    def _bayes(self, df):
        # RELAXED: majority-vote neighbor matching instead of requiring ALL
        # features to match simultaneously. The original all-features-AND
        # rule almost never accumulates 5+ neighbors, so bprob silently
        # defaulted to 0.5 on most bars -- which then failed the signal
        # engine's bp > threshold gate nearly everywhere. Still a
        # deterministic conditional-probability estimate, not ML.
        #
        # VECTORIZED: identical logic to a pandas .iloc-per-row loop, but
        # the 200-wide neighbor comparison uses numpy array ops instead of
        # per-cell pandas indexing. On an 8,000-bar test this cut runtime
        # from ~245s to well under a second; on a full-year (~105k bar)
        # dataset the .iloc version would have taken on the order of an
        # hour for this feature alone.
        n = len(df); prob = np.full(n, 0.5)
        feats = ['kld','oud','ofi','zsc','liq']; av = [f for f in feats if f in df.columns]
        if not av or 'lr1' not in df.columns: return prob
        min_matches = max(1, int(np.ceil(0.6 * len(av))))
        lr1 = df['lr1'].values.astype(float)
        F = {f: df[f].values.astype(float) for f in av}
        for i in range(200, n-1):
            fr = lr1[i+1]
            if np.isnan(fr): continue
            match_count = np.zeros(199)
            valid = np.ones(199, dtype=bool)
            for f in av:
                seg_hist = F[f][i-200:i-1]      # local j = 0..198 -> global i-200..i-2
                cur_v = F[f][i]
                std_f = np.nanstd(F[f][i-200:i])
                nanmask = np.isnan(seg_hist) | np.isnan(cur_v)
                valid &= ~nanmask
                if std_f > 1e-12:
                    within = np.abs(seg_hist - cur_v) <= 1.25 * std_f
                else:
                    within = np.zeros(199, dtype=bool)
                match_count += within.astype(int)
            m = valid & (match_count >= min_matches)
            total = int(np.sum(m))
            if total >= 5:
                next_lr = lr1[i-199:i]          # local j+1 = 1..199 -> global i-199..i-1
                wins = int(np.sum((next_lr > 0) & m))
                prob[i] = wins / total
        return prob
    def _expv(self, df):
        n = len(df); ev = np.full(n, 0.0)
        if 'bprob' not in df.columns: return ev
        wp = df['bprob'].values
        for i in range(100, n-1):
            hr = df['lr1'].values[max(0,i-100):i]; hr = hr[~np.isnan(hr)]
            if len(hr) < 10: continue
            aw = np.mean(hr[hr > 0]) if np.any(hr > 0) else 0.0
            al = np.mean(hr[hr < 0]) if np.any(hr < 0) else 0.0
            pw = wp[i]
            if np.isnan(pw): pw = 0.5
            ev[i] = pw * aw + (1 - pw) * al
        return ev


class MRFilter:
    """Market regime filter using Hurst exponent and volatility percentile."""
    def __init__(self, c): self.c = c
    def determine(self, df):
        n = len(df); reg = np.full(n, 'unknown', dtype=object)
        if 'lr1' not in df.columns: return reg
        lr = df['lr1'].values.astype(float)
        vol = df['evol'].values.astype(float) if 'evol' in df.columns else None
        vp = df['vpct'].values.astype(float) if 'vpct' in df.columns else None
        hw = self.c.hurst_window
        for i in range(max(hw, 48), n):
            rw = lr[i-hw+1:i+1]; rw = rw[~np.isnan(rw)]
            if len(rw) < 30: reg[i] = 'low_vol'; continue
            h = self._hurst(rw)
            hv = False; lv = False
            if vol is not None and vp is not None and not np.isnan(vp[i]):
                hv = vp[i] > 0.80; lv = vp[i] < 0.20
            if hv: reg[i] = 'high_vol'
            elif lv: reg[i] = 'low_vol'
            elif h < 0.45: reg[i] = 'mean_reverting'
            elif h > 0.55: reg[i] = 'trending'
            else: reg[i] = 'low_vol'
        return reg
    @staticmethod
    def _hurst(r):
        if len(r) < 10: return 0.5
        r = r[r != 0]
        if len(r) < 10: return 0.5
        c = np.cumsum(r); c = c - c[0]
        ml = min(len(r)//2, 100); lags = range(2, ml); tau = []
        for lag in lags:
            ns = len(c) // lag
            if ns < 2: continue
            rs = []
            for s in range(ns):
                seg = c[s*lag:(s+1)*lag]
                if len(seg) < 2: continue
                adj = seg - np.mean(seg); R = np.max(adj) - np.min(adj); S = np.std(seg, ddof=1)
                if S > 1e-12: rs.append(R / S)
            if rs: tau.append(np.mean(rs))
        if len(tau) < 3: return 0.5
        la = list(lags[:len(tau)])
        if len(la) < 3: return 0.5
        A = np.vstack([np.log(la), np.ones(len(la))]).T
        try: h, _ = np.linalg.lstsq(A, np.log(tau), rcond=None)[0]
        except: return 0.5
        return max(0.0, min(1.0, h))


class RiskManager:
    """Dynamic risk management with Kelly, VaR, Expected Shortfall, Risk of Ruin."""
    def __init__(self, c):
        self.c = c; self.daily_loss = 0.0; self.weekly_loss = 0.0
        self.consecutive_losses = 0; self.peak_capital = c.initial_capital
        self.current_capital = c.initial_capital; self.trade_history = []
        self.current_day = None; self.current_week = None
    def update(self, pnl, ts):
        self.current_capital += pnl; self.peak_capital = max(self.peak_capital, self.current_capital)
        self.trade_history.append({'pnl': pnl, 'ts': ts})
        ds = ts.strftime('%Y-%m-%d'); ws = ts.strftime('%Y-W%W')
        if ds != self.current_day: self.current_day = ds; self.daily_loss = 0.0
        if ws != self.current_week: self.current_week = ws; self.weekly_loss = 0.0
        if pnl < 0: self.daily_loss += abs(pnl); self.weekly_loss += abs(pnl); self.consecutive_losses += 1
        else: self.consecutive_losses = 0
    def risk_mult(self):
        m = 1.0
        if self.consecutive_losses > 0: m *= (self.c.risk_reduction_after_loss ** self.consecutive_losses)
        if self.daily_loss >= self.c.max_daily_loss_pct * self.current_capital: return 0.0
        if self.weekly_loss >= self.c.max_weekly_loss_pct * self.current_capital: return 0.0
        if self.drawdown() >= self.c.max_portfolio_drawdown: return 0.0
        return m
    def drawdown(self):
        if self.peak_capital <= 0: return 0.0
        return (self.peak_capital - self.current_capital) / self.peak_capital
    def kelly(self, wp, aw, al):
        if al >= 0 or abs(al) < 1e-12: return 0.0
        b = abs(aw / al) if al != 0 else 0.0
        if b <= 0: return 0.0
        k = (wp * b - (1 - wp)) / b
        return max(0.0, min(k, 1.0)) * self.c.kelly_fraction
    def pos_size(self, ss, ep, wp, ev, vl):
        br = self.c.base_risk_per_trade; rm = self.risk_mult()
        if ev > self.c.min_expected_value * 3 and wp > 0.65:
            br = min(br * 1.5, self.c.max_risk_per_trade)
        if rm <= 0: return 0.0, 0.0
        ra = self.current_capital * br * rm
        nr = min(50, len(self.trade_history))
        if nr > 5:
            rp = [t['pnl'] for t in self.trade_history[-nr:] if t['pnl'] != 0]
            if rp:
                aw_ = np.mean([p for p in rp if p > 0]) if any(p > 0 for p in rp) else 0.001
                al_ = abs(np.mean([p for p in rp if p < 0])) if any(p < 0 for p in rp) else 0.001
                kf = self.kelly(wp, aw_, al_); kr = self.current_capital * kf * 0.5
                ra = max(ra, kr * 0.5)
        sd = vl * 2.0 if vl > 1e-12 else 0.01
        pv = ra / max(sd, 0.001); lev = pv / max(self.current_capital, 1.0)
        lev = max(self.c.min_leverage, min(lev, self.c.max_leverage))
        pv = min(pv, self.current_capital * self.c.max_leverage)
        return pv / max(ep, 1.0), lev
    def var(self, ret, conf=0.95):
        if len(ret) < 10: return 0.0
        return abs(np.percentile(ret, (1-conf)*100))
    def es(self, ret, conf=0.95):
        if len(ret) < 10: return 0.0
        v = self.var(ret, conf); tail = ret[ret <= -v]
        if len(tail) == 0: return v
        return abs(np.mean(tail))
    def ruin(self, wp, aw, al):
        if al >= 0: return 1.0
        p = min(max(wp, 0.01), 0.99); q = 1 - p
        b = abs(aw / al) if al != 0 else 0.0
        if b <= 0 or abs(b-1.0) < 1e-12: return 1.0 if p <= 0.5 else 0.0
        if p <= q: return 1.0
        try: z = (q / p) ** (1.0 / max(self.current_capital, 1.0)); rp = z ** self.current_capital
        except: rp = 1.0
        return min(rp, 1.0)
    def heat(self):
        if not self.trade_history: return 0.0
        r = self.trade_history[-min(20, len(self.trade_history)):]
        tr = sum(abs(t['pnl']) for t in r)
        return tr / max(self.current_capital, 1.0) / len(r) if r else 0.0


class SignalEngine:
    """Kalman + OU mean reversion edge. Trade only if ALL conditions pass.

    Trade-frequency fix: the OU z-score cutoffs are scaled by a single
    calibrated factor (od_scale) found via bisection search against the
    realized signal count on the actual dataset, targeting
    [c.target_trades_min, c.target_trades_max] total signals. This is a
    deterministic parameter search over one known formula -- no ML, no
    learned model, no lagging indicators.
    """
    def __init__(self, c, rm):
        self.c = c; self.rm = rm
        self.od_scale = 1.0
        self.bprob_threshold = c.bayesian_prob_threshold
        self._calibrated = False

    def _raw_signals(self, df, reg, od_scale, bprob_thr):
        n = len(df); sig = np.zeros(n, dtype=np.int8)
        base_floor, base_mr, base_lv, base_tr = 1.0, 1.5, 1.2, 2.5
        floor = base_floor * od_scale
        mr_thr = base_mr * od_scale
        lv_thr = base_lv * od_scale
        tr_thr = base_tr * od_scale
        for i in range(200, n):
            kd = df['kld'].iloc[i] if 'kld' in df.columns else 0.0
            od = df['oud'].iloc[i] if 'oud' in df.columns else 0.0
            bp = df['bprob'].iloc[i] if 'bprob' in df.columns else 0.5
            ev = df['eval'].iloc[i] if 'eval' in df.columns else 0.0
            lq = df['liq'].iloc[i] if 'liq' in df.columns else 0.5
            vp = df['vpct'].iloc[i] if 'vpct' in df.columns else 0.5
            ofi = df['ofi'].iloc[i] if 'ofi' in df.columns else 0.0
            rg = reg[i] if i < len(reg) else 'unknown'
            if np.isnan(kd) or np.isnan(od) or np.isnan(bp): continue
            if np.isnan(ev) or np.isnan(lq) or np.isnan(vp): continue
            if ev <= self.c.min_expected_value: continue
            if bp <= bprob_thr: continue
            if lq < self.c.min_liquidity_score: continue
            if vp > self.c.max_volatility_percentile: continue
            if rg in ('high_vol', 'unknown'): continue
            tr = [t['pnl']/max(self.rm.current_capital,1.0) for t in self.rm.trade_history[-50:] if abs(t['pnl']) > 0]
            if tr:
                wh = sum(1 for r_ in tr if r_ > 0) / len(tr)
                aw = np.mean([r_ for r_ in tr if r_ > 0]) if any(r_ > 0 for r_ in tr) else 0.001
                al = abs(np.mean([r_ for r_ in tr if r_ < 0])) if any(r_ < 0 for r_ in tr) else 0.001
                if self.rm.ruin(wh, aw, al) > 0.01: continue
            if abs(od) < floor: continue
            if rg == 'mean_reverting':
                if od > mr_thr and kd > 0 and ofi < -0.1: sig[i] = -1
                elif od < -mr_thr and kd < 0 and ofi > 0.1: sig[i] = 1
            elif rg == 'low_vol':
                if od > lv_thr and kd > 0: sig[i] = -1
                elif od < -lv_thr and kd < 0: sig[i] = 1
            elif rg == 'trending':
                if od > tr_thr and kd > 0 and ofi < -0.3: sig[i] = -1
                elif od < -tr_thr and kd < 0 and ofi > 0.3: sig[i] = 1
        return sig

    def calibrate(self, df, reg, target_min=None, target_max=None, max_iter=20):
        """Deterministic bisection search over od_scale (with bayesian
        threshold relaxation as a fallback) to hit the target trade-count
        band. No ML -- a search over one interpretable, known formula."""
        target_min = target_min or self.c.target_trades_min
        target_max = target_max or self.c.target_trades_max
        def count_at(scale, bthr):
            return int(np.sum(self._raw_signals(df, reg, scale, bthr) != 0))
        bthr = self.c.bayesian_prob_threshold
        lo, hi = 0.15, 3.0
        best = None
        for _ in range(max_iter):
            mid = (lo + hi) / 2
            n = count_at(mid, bthr)
            if best is None or abs(n - (target_min+target_max)/2) < abs(best[1] - (target_min+target_max)/2):
                best = (mid, n)
            if target_min <= n <= target_max:
                self.od_scale, self.bprob_threshold = mid, bthr; self._calibrated = True
                return mid, bthr, n
            if n < target_min: hi = mid
            else: lo = mid
        if best[1] < target_min:
            for step in np.arange(bthr, 0.49, -0.01):
                n = count_at(0.15, step)
                if n >= target_min:
                    self.od_scale, self.bprob_threshold = 0.15, step; self._calibrated = True
                    return 0.15, step, n
        self.od_scale, self.bprob_threshold = best[0], bthr; self._calibrated = True
        return best[0], bthr, best[1]

    def generate(self, df, reg):
        if not self._calibrated:
            scale, bthr, n = self.calibrate(df, reg)
            print(f'  Calibrated: od_scale={scale:.3f}, bprob_threshold={bthr:.3f} -> {n} raw signals')
        return self._raw_signals(df, reg, self.od_scale, self.bprob_threshold)


@dataclass
class Trade:
    num: int; et: datetime; xt: datetime; side: str; ep: float; xp: float
    size: float; lev: float; pnl: float; pnl_pct: float; comm: float; ht: timedelta; reason: str


class Backtester:
    """Institutional-quality backtesting with fees, slippage, leverage."""
    def __init__(self, c, df, fdf, sig, reg):
        self.c = c; self.df = df; self.fdf = fdf; self.sig = sig; self.reg = reg
        self.rm = RiskManager(c); self.se = SignalEngine(c, self.rm)
        self.trades = []; self.eq = []; self.ts = []
    def run(self):
        cap = self.c.initial_capital; btc = 0.0; aep = 0.0; ps = 0; lev = 1.0
        self.eq.append(cap); self.ts.append(self.df['timestamp'].iloc[0])
        for i in range(len(self.df)):
            if i < 200: self.eq.append(cap); self.ts.append(self.df['timestamp'].iloc[i]); continue
            sig = self.sig[i]; ts = self.df['timestamp'].iloc[i]; pr = self.df['close'].iloc[i]
            # Close
            if btc != 0 and (sig == 0 or sig == -ps):
                xp = pr; sl = xp * self.c.slippage_bps * ps; xpa = xp - sl
                if ps == 1: pnl = btc * (xpa - aep)
                else: pnl = btc * (aep - xpa)
                fe = abs(btc) * xpa * self.c.taker_fee * lev; pnln = pnl - fe; cap += pnln
                self.rm.update(pnln, ts)
                ppv = pnln / max(abs(aep * btc / lev), 1.0) if btc != 0 else 0.0
                ht = ts - self.trades[-1].et if self.trades else timedelta(0)
                er = 'Signal_Close' if sig == 0 else 'Signal_Reverse'
                if self.trades:
                    lt = self.trades[-1]; lt.xt = ts; lt.xp = xpa; lt.pnl = pnln; lt.pnl_pct = ppv
                    lt.comm += fe; lt.ht = ht; lt.reason = er
                btc = 0.0; aep = 0.0; ps = 0; lev = 1.0
            # Enter
            if sig != 0 and btc == 0:
                bp = self.fdf['bprob'].iloc[i] if 'bprob' in self.fdf.columns else 0.5
                ev = self.fdf['eval'].iloc[i] if 'eval' in self.fdf.columns else 0.0
                vl = self.fdf['evol'].iloc[i] if 'evol' in self.fdf.columns else 0.01
                if np.isnan(bp): bp = 0.5
                if np.isnan(ev): ev = 0.0
                if np.isnan(vl) or vl < 1e-12: vl = 0.01
                pb, l = self.rm.pos_size(abs(sig), pr, bp, ev, vl)
                if pb > 0 and l > 0:
                    sl = pr * self.c.slippage_bps * sig; epa = pr + sl
                    fe = pb * epa * self.c.taker_fee * l; cap -= fe
                    btc = pb * sig; aep = epa; ps = sig; lev = l
                    self.trades.append(Trade(len(self.trades)+1, ts, ts,
                        'LONG' if sig == 1 else 'SHORT', epa, epa, pb, l, 0.0, 0.0, fe, timedelta(0), ''))
            self.eq.append(cap); self.ts.append(ts)
        # Close remaining
        if btc != 0:
            xp = self.df['close'].iloc[-1]
            if ps == 1: pnl = btc * (xp - aep)
            else: pnl = btc * (aep - xp)
            fe = abs(btc) * xp * self.c.taker_fee * lev; pnln = pnl - fe; cap += pnln
            if self.trades:
                lt = self.trades[-1]; lt.xt = self.df['timestamp'].iloc[-1]; lt.xp = xp
                lt.pnl = pnln; lt.pnl_pct = pnln / max(abs(aep * btc / lev), 1.0); lt.comm += fe
                lt.ht = lt.xt - lt.et; lt.reason = 'End_Of_Data'
        print(f'Backtest: {len(self.trades)} trades, final capital ${cap:,.2f}')
        return pd.Series(self.eq), self.ts


class PerformanceReport:
    """Compute Sharpe, Sortino, Calmar, CAGR, drawdown, and trade statistics."""
    def __init__(self, c, trades, eq, bm):
        self.c = c; self.trades = trades; self.eq = eq; self.bm = bm; self.m = {}
    def compute(self):
        if not self.trades: return {'error': 'No trades'}
        eq = self.eq.values.astype(float); tr = (eq[-1] - eq[0]) / eq[0]
        self.m['total_return_pct'] = tr * 100; self.m['final_capital'] = eq[-1]
        ny = len(eq) / (365.25 * 24 * 60 / 5)
        self.m['cagr_pct'] = ((eq[-1]/eq[0])**(1.0/ny)-1)*100 if ny>0 and eq[0]>0 else 0.0
        dr = self._daily_returns()
        if len(dr) >= 5:
            sd = np.std(dr, ddof=1)
            self.m['sharpe_ratio'] = np.mean(dr)/sd*np.sqrt(365) if sd>0 else 0.0
            nr = dr[dr < 0]
            sdn = np.std(nr, ddof=1) if len(nr) > 0 else sd
            self.m['sortino_ratio'] = np.mean(dr)/sdn*np.sqrt(365) if sdn>0 else 0.0
        else: self.m['sharpe_ratio'] = self.m['sortino_ratio'] = 0.0
        rm = np.maximum.accumulate(eq); dd = (rm - eq) / rm; mdd = np.max(dd)
        self.m['max_drawdown_pct'] = mdd * 100
        self.m['calmar_ratio'] = self.m.get('cagr_pct', 0)/mdd*0.01 if mdd > 0 else 0.0
        ptp = np.max(rm - eq)
        self.m['recovery_factor'] = (eq[-1]-eq[0])/ptp if ptp > 0 else float('inf')
        pnls = np.array([t.pnl for t in self.trades]); pp = np.array([t.pnl_pct for t in self.trades])
        w = pnls > 0
        self.m['total_trades'] = len(self.trades)
        self.m['win_rate_pct'] = np.sum(w)/len(self.trades)*100
        self.m['profit_factor'] = np.sum(pnls[w])/abs(np.sum(pnls[~w])) if np.any(~w) and abs(np.sum(pnls[~w]))>0 else (float('inf') if np.any(w) else 0.0)
        self.m['expectancy'] = np.mean(pnls); self.m['avg_trade_pnl'] = np.mean(pnls)
        hm = np.array([t.ht.total_seconds()/60 for t in self.trades if t.ht.total_seconds() > 0])
        self.m['avg_holding_minutes'] = np.mean(hm) if len(hm) > 0 else 0.0
        if len(self.trades) > 1 and ny > 0:
            self.m['trades_per_year'] = len(self.trades)/ny
            self.m['trades_per_day'] = len(self.trades)/(ny*365)
        else: self.m['trades_per_year'] = self.m['trades_per_day'] = 0.0
        levs = np.array([t.lev for t in self.trades])
        self.m['avg_leverage'] = np.mean(levs); self.m['max_leverage'] = np.max(levs)
        if len(self.bm) > 0:
            bhr = (self.bm.iloc[-1] - self.bm.iloc[0]) / self.bm.iloc[0]
            self.m['buy_hold_return_pct'] = bhr * 100
            self.m['excess_return_pct'] = (tr - bhr) * 100
        return self.m
    def _daily_returns(self):
        eq = self.eq.values; idx = list(range(0, len(eq), 288))
        if len(idx) < 3: return np.array([])
        s = eq[idx]; return np.diff(s) / s[:-1]
    def print_summary(self):
        m = self.m
        if not m: print('No metrics'); return
        print(); print('='*60); print('            PERFORMANCE REPORT'); print('='*60)
        for k,f in [('total_return_pct','Total Return:'),('cagr_pct','CAGR:'),('sharpe_ratio','Sharpe Ratio:'),('sortino_ratio','Sortino Ratio:'),('calmar_ratio','Calmar Ratio:'),('max_drawdown_pct','Max Drawdown:'),('profit_factor','Profit Factor:'),('win_rate_pct','Win Rate:'),('total_trades','Total Trades:'),('expectancy','Expectancy:'),('avg_holding_minutes','Avg Hold (min):'),('trades_per_year','Trades/Year:'),('trades_per_day','Trades/Day:'),('avg_leverage','Avg Lev:'),('final_capital','Final Capital:')]:
            v = m.get(k, 0)
            if k in ('expectancy','avg_trade_pnl','final_capital'):
                print(f'  {f:22} ${v:>8,.2f}')
            elif k in ('total_trades',):
                print(f'  {f:22} {v:>8}')
            elif k in ('sharpe_ratio','sortino_ratio','calmar_ratio','profit_factor'):
                print(f'  {f:22} {v:>8.2f}')
            else:
                print(f'  {f:22} {v:>8.2f}%')
        if 'buy_hold_return_pct' in m: print(f'  Buy & Hold Return:   {m["buy_hold_return_pct"]:>8.2f}%')
        print('='*60)


class ChartEngine:
    """Generate performance charts: equity curve, drawdown, Monte Carlo, trade analysis."""
    def __init__(self, c):
        self.c = c; os.makedirs(c.chart_dir, exist_ok=True)
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.rcParams.update({'figure.facecolor':'white','axes.facecolor':'#f8f9fa','axes.grid':True,'grid.alpha':0.3,'axes.titleweight':'bold','axes.titlesize':14})
    def gen(self, eq, ts, trades, m, bm):
        self._eq(eq, ts, m, bm); self._dd(eq, ts); self._mc(eq, trades); self._ta(trades)
        print(f'Charts -> {self.c.chart_dir}/')
    def _eq(self, eq, ts, m, bm):
        fig, ax = plt.subplots(figsize=(16,8))
        t = ts if isinstance(ts[0], datetime) else pd.to_datetime(ts)
        ax.plot(t, eq.values, '#1a73e8', lw=1.5, label='Strategy')
        if len(bm) == len(eq): ax.plot(t, bm.values, '#ea4335', lw=1, alpha=0.7, label='Buy & Hold', ls='--')
        ax.fill_between(t, eq.values, alpha=0.1, color='#1a73e8')
        txt = f"CAGR: {m.get('cagr_pct',0):.1f}% | Sharpe: {m.get('sharpe_ratio',0):.2f} | Max DD: {m.get('max_drawdown_pct',0):.1f}%"
        ax.text(0.02, 0.95, txt, transform=ax.transAxes, fontsize=12, va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax.set_title('Portfolio Equity Curve', fontsize=16, fontweight='bold')
        ax.set_xlabel('Date'); ax.set_ylabel('Portfolio Value ($)'); ax.legend(loc='upper left')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
        fig.tight_layout(); fig.savefig(os.path.join(self.c.chart_dir, 'equity_curve.png'), dpi=150); plt.close(fig)
    def _dd(self, eq, ts):
        fig, axes = plt.subplots(2, 1, figsize=(16,10), sharex=True)
        t = ts if isinstance(ts[0], datetime) else pd.to_datetime(ts); eqv = eq.values.astype(float)
        axes[0].plot(t, eqv, '#1a73e8', lw=1.2); axes[0].fill_between(t, eqv, alpha=0.1, color='#1a73e8')
        axes[0].set_title('Portfolio Equity', fontsize=14, fontweight='bold')
        axes[0].yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
        rm = np.maximum.accumulate(eqv); dd = (rm - eqv) / rm * 100
        axes[1].fill_between(t, dd, 0, color='#ea4335', alpha=0.5); axes[1].plot(t, dd, '#d93025', lw=1)
        axes[1].set_title('Underwater Drawdown', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Date'); axes[1].set_ylabel('Drawdown (%)')
        mi = np.argmax(dd); axes[1].scatter(t[mi], dd[mi], color='black', s=50, zorder=5)
        axes[1].annotate(f'Max DD: {dd[mi]:.1f}%', (t[mi], dd[mi]), xytext=(10,-20), textcoords='offset points', fontweight='bold')
        axes[1].invert_yaxis(); axes[1].set_ylim(max(dd)*1.2, -0.5)
        fig.tight_layout(); fig.savefig(os.path.join(self.c.chart_dir, 'drawdown.png'), dpi=150); plt.close(fig)
    def _mc(self, eq, trades):
        if len(trades) < 10: print('MC: too few trades'); return
        pp = np.array([t.pnl_pct for t in trades]); pp = pp[~np.isnan(pp)]
        if len(pp) < 10: return
        ns = self.c.mc_simulations; nt = len(pp); ic = self.c.initial_capital
        fe = np.zeros(ns); fig, ax = plt.subplots(figsize=(16,8))
        ap = np.zeros((ns, nt+1))
        for s_ in range(ns):
            sp = np.random.choice(pp, size=nt, replace=True)
            ec = ic * np.cumprod(1 + sp); ec = np.concatenate([[ic], ec]); ap[s_] = ec; fe[s_] = ec[-1]
        sp_ = np.sort(ap, axis=0); al = 1 - self.c.mc_confidence
        li = int(al/2 * ns); ui = int((1-al/2) * ns); mi = ns // 2
        x = np.arange(nt+1)
        for s_ in range(min(200, ns)): ax.plot(x, ap[s_], color='gray', alpha=0.05, lw=0.5)
        ax.fill_between(x, sp_[li], sp_[ui], alpha=0.3, color='#1a73e8', label=f'{self.c.mc_confidence*100:.0f}% CI')
        ax.plot(x, sp_[mi], '#1a73e8', lw=2, label='Median')
        act = ic * np.cumprod(np.concatenate([[1], 1+pp]))
        ax.plot(x[:len(act)], act, '#ea4335', lw=1.5, label='Actual', ls='--')
        ax.set_title(f'Monte Carlo ({ns} runs, {self.c.mc_confidence*100:.0f}% CI)', fontsize=16, fontweight='bold')
        ax.set_xlabel('Trade #'); ax.set_ylabel('Value ($)')
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x_, _: f'${x_:,.0f}'))
        ax.legend(loc='upper left'); fig.tight_layout(); fig.savefig(os.path.join(self.c.chart_dir, 'monte_carlo.png'), dpi=150); plt.close(fig)
        fig, ax = plt.subplots(figsize=(12,6))
        ax.hist(fe, bins=80, color='#1a73e8', alpha=0.7, edgecolor='white', lw=0.5)
        ax.axvline(np.median(fe), color='#1a73e8', lw=2, ls='--', label=f'Median: ${np.median(fe):,.0f}')
        ax.axvline(ic, color='#ea4335', lw=2, label=f'Initial: ${ic:,.0f}')
        ax.axvline(eq.iloc[-1], color='#34a853', lw=2, ls=':', label=f'Actual: ${eq.iloc[-1]:,.0f}')
        le = np.percentile(fe, al/2*100); ue = np.percentile(fe, (1-al/2)*100)
        ax.axvspan(le, ue, alpha=0.15, color='#1a73e8', label=f'{self.c.mc_confidence*100:.0f}% CI')
        ax.set_title('Final Equity Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Final Value ($)'); ax.set_ylabel('Frequency')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x_, _: f'${x_:,.0f}'))
        ax.legend(); fig.tight_layout(); fig.savefig(os.path.join(self.c.chart_dir, 'equity_distribution.png'), dpi=150); plt.close(fig)
    def _ta(self, trades):
        if len(trades) < 2: return
        pnls = np.array([t.pnl for t in trades]); pp = np.array([t.pnl_pct for t in trades]) * 100
        fig, axes = plt.subplots(1, 2, figsize=(16,6))
        axes[0].hist(pnls, bins=50, color='#1a73e8', alpha=0.7, edgecolor='white')
        axes[0].axvline(0, color='#ea4335', lw=2); axes[0].set_title('PnL Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('PnL ($)'); axes[0].set_ylabel('Frequency')
        w = pnls > 0
        axes[1].hist(pp[w], bins=30, color='#34a853', alpha=0.7, label='Wins')
        axes[1].hist(pp[~w], bins=30, color='#ea4335', alpha=0.7, label='Losses')
        axes[1].axvline(0, color='black', lw=1)
        axes[1].set_title('Return Distribution', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Return (%)'); axes[1].legend()
        fig.tight_layout(); fig.savefig(os.path.join(self.c.chart_dir, 'trade_analysis.png'), dpi=150); plt.close(fig)


class TradeLogPrinter:
    """Trade log printed straight to the terminal -- no fpdf dependency,
    no file written to disk. Column widths mirror the old PDF layout so
    the output is easy to scan; long logs are capped (see
    Config.trade_log_max_rows) so a 1000+ trade run doesn't flood the
    console -- pass max_rows=None to print every trade."""
    HEADERS = ['#', 'Entry', 'Exit', 'Side', 'Entry $', 'Exit $', 'Size BTC', 'Lev', 'PnL $', 'PnL %', 'Comm $', 'Hold(m)']
    WIDTHS  = [6, 17, 17, 6, 11, 11, 11, 6, 11, 9, 9, 8]

    def __init__(self, trades, m):
        self.trades = trades
        self.m = m

    def _row(self, i, t):
        es = t.et.strftime('%Y-%m-%d %H:%M'); xs = t.xt.strftime('%Y-%m-%d %H:%M')
        hm = t.ht.total_seconds() / 60
        return [str(i+1), es, xs, t.side[:4], f'${t.ep:.2f}', f'${t.xp:.2f}', f'{t.size:.4f}',
                f'{t.lev:.1f}x', f'${t.pnl:.2f}', f'{t.pnl_pct*100:.2f}%', f'${t.comm:.2f}', f'{hm:.0f}']

    def print_log(self, max_rows=None):
        print(); print('='*sum(self.WIDTHS)); print('  TRADE LOG'); print('='*sum(self.WIDTHS))
        if not self.trades:
            print('  No trades.'); print('='*sum(self.WIDTHS)); return
        header = ''.join(h.ljust(w) for h, w in zip(self.HEADERS, self.WIDTHS))
        print(header); print('-'*len(header))
        rows = self.trades if max_rows is None else self.trades[:max_rows]
        for i, t in enumerate(rows):
            print(''.join(str(v).ljust(w) for v, w in zip(self._row(i, t), self.WIDTHS)))
        if max_rows is not None and len(self.trades) > max_rows:
            print(f'  ... {len(self.trades) - max_rows} more trades not shown '
                  f'(pass max_rows=None to TradeLogPrinter.print_log() to print all)')
        print('-'*len(header))
        wins = sum(1 for t in self.trades if t.pnl > 0)
        print(f'  Total: {len(self.trades)}  |  Wins: {wins}  |  Losses: {len(self.trades)-wins}  |  '
              f'Net PnL: ${sum(t.pnl for t in self.trades):,.2f}')
        print('='*sum(self.WIDTHS))


def main():
    """Run the complete strategy pipeline."""
    print(); print('='*60); print('  BTCUSD 5-Min Statistical Trading Strategy'); print('='*60)
    c = Config()
    print('\n[1/7] Loading data...'); dl = DataLoader(c); df = dl.load()
    print('\n[2/7] Computing features...'); fe = FeatureEngine(c); fdf = fe.compute(df)
    print(f'  Features: {fdf.shape[1]} cols, {len(fdf)} rows')
    print('\n[3/7] Market regimes...'); mrf = MRFilter(c); reg = mrf.determine(fdf)
    rc = pd.Series(reg).value_counts()
    for r_, cnt in rc.items(): print(f'  {r_}: {cnt} ({cnt/len(reg)*100:.1f}%)')
    print('\n[4/7] Generating signals (calibrating for '
          f'{c.target_trades_min}-{c.target_trades_max} trades)...')
    rm = RiskManager(c); se = SignalEngine(c, rm); sig = se.generate(fdf, reg)
    print(f'  Long: {np.sum(sig==1)}, Short: {np.sum(sig==-1)}')
    print('\n[5/7] Running backtest...'); bt = Backtester(c, df, fdf, sig, reg); eq, ts = bt.run()
    eqs = pd.Series(eq)
    print('\n[6/7] Performance report...')
    bm = df['close'] / df['close'].iloc[0] * c.initial_capital; bms = pd.Series(bm)
    pr = PerformanceReport(c, bt.trades, eqs, bms); m = pr.compute(); pr.print_summary()
    print('\n[7/7] Charts & trade log...'); ce = ChartEngine(c); ce.gen(eqs, ts, bt.trades, m, bms)
    TradeLogPrinter(bt.trades, m).print_log(max_rows=c.trade_log_max_rows)
    print(); print('='*60); print('  ANALYSIS COMPLETE'); print('='*60)
    print(f'  Equity Curve:      {c.chart_dir}/equity_curve.png')
    print(f'  Drawdown Chart:    {c.chart_dir}/drawdown.png')
    print(f'  Monte Carlo:       {c.chart_dir}/monte_carlo.png')
    print(f'  Equity Dist:       {c.chart_dir}/equity_distribution.png')
    print(f'  Trade Analysis:    {c.chart_dir}/trade_analysis.png')
    print(f'  Trade Log:         printed above (terminal only, no file written)')
    print('='*60); return m, bt.trades


if __name__ == '__main__':
    main()