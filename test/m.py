import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# CONFIG
CSV_FILE = "/content/btcusdt_5m_full.csv"
INITIAL_CAPITAL = 10_000.0
OUTPUT_DIR = Path.cwd()
IN_SAMPLE_YEARS = 4

# Regime / structure
REGIME_LOOKBACK = 40
BREAKOUT_LOOKBACK = 20
RANGE_LOOKBACK = 20

# Risk
RISK_PER_TRADE = 0.005  # 0.5%
MAX_POSITION_NOTIONAL = 1.0  # max 1x equity

# Reward / stop
MOMENTUM_RR = 2.0
# Execution costs
FEE = 0.0005  # 0.05%
SLIPPAGE = 0.0002  # 0.02%

# Trade management
MAX_HOLD_BARS = 100
COOLDOWN_BARS = 3

# Minimum breakout strength
BREAKOUT_BUFFER = 0.001  # 0.10%

# Minimum range quality
MAX_RANGE_TREND_RATIO = 0.35

# LOAD DATA
df = pd.read_csv(CSV_FILE)
df.columns = [x.lower().strip() for x in df.columns]

time_candidates = ["timestamp", "datetime", "date", "time"]
time_col = None

for c in time_candidates:
    if c in df.columns:
        time_col = c
        break

if time_col is None:
    raise ValueError(
        "No timestamp column found. Use timestamp, datetime, date or time."
    )

required = ["open", "high", "low", "close", "volume"]
for c in required:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")

df[time_col] = pd.to_datetime(df[time_col])
df = (
    df.sort_values(time_col)
    .drop_duplicates(subset=[time_col])
    .reset_index(drop=True)
)
df = df.dropna(subset=required).reset_index(drop=True)

# CHRONOLOGICAL TRAIN / TEST SPLIT
split_date = df[time_col].iloc[0] + pd.DateOffset(years=IN_SAMPLE_YEARS)
in_sample_df = df[df[time_col] < split_date].copy()
out_of_sample_df = df[df[time_col] >= split_date].copy()

if in_sample_df.empty or out_of_sample_df.empty:
    raise ValueError(
        f"Dataset must contain more than {IN_SAMPLE_YEARS} years of data "
        "for an out-of-sample backtest."
    )

in_sample_df.to_csv(OUTPUT_DIR / "in_sample_first_4_years.csv", index=False)
out_of_sample_df.to_csv(OUTPUT_DIR / "out_of_sample_unseen_data.csv", index=False)
out_of_sample_start = len(in_sample_df)

# PRICE STRUCTURE
# Previous structural high / low
df["prev_high"] = df["high"].rolling(BREAKOUT_LOOKBACK).max().shift(1)
df["prev_low"] = df["low"].rolling(BREAKOUT_LOOKBACK).min().shift(1)

# Larger regime range
df["regime_high"] = df["high"].rolling(REGIME_LOOKBACK).max().shift(1)
df["regime_low"] = df["low"].rolling(REGIME_LOOKBACK).min().shift(1)

# Range
df["range_size"] = df["regime_high"] - df["regime_low"]

# Price location inside range
df["range_position"] = (df["close"] - df["regime_low"]) / df["range_size"]

# Recent directional movement
df["direction_move"] = df["close"] - df["close"].shift(REGIME_LOOKBACK)
df["direction_ratio"] = abs(df["direction_move"]) / df["range_size"]

# Candle characteristics
df["candle_range"] = df["high"] - df["low"]
df["body"] = abs(df["close"] - df["open"])
df["body_ratio"] = df["body"] / df["candle_range"].replace(0, np.nan)

# Where candle closes within its own range
df["close_location"] = (df["close"] - df["low"]) / df[
    "candle_range"
].replace(0, np.nan)

# BACKTEST STATE
cash = INITIAL_CAPITAL
position = None
trades = []
equity_records = []
cooldown = 0


# HELPERS
def calculate_quantity(equity, entry, stop):
    risk_dollars = equity * RISK_PER_TRADE
    risk_per_coin = abs(entry - stop)
    if risk_per_coin <= 0:
        return 0, 0

    qty = risk_dollars / risk_per_coin

    # Cap notional exposure
    max_qty = equity * MAX_POSITION_NOTIONAL / entry
    qty = min(qty, max_qty)

    actual_risk = qty * risk_per_coin
    return qty, actual_risk


def apply_entry_slippage(price, side):
    if side == "LONG":
        return price * (1 + SLIPPAGE)
    return price * (1 - SLIPPAGE)


def apply_exit_slippage(price, side):
    if side == "LONG":
        return price * (1 - SLIPPAGE)
    return price * (1 + SLIPPAGE)


# MAIN LOOP
for i in range(out_of_sample_start, len(df)):
    row = df.iloc[i]
    current_time = row[time_col]
    open_price = float(row["open"])
    high = float(row["high"])
    low = float(row["low"])
    close = float(row["close"])

    equity_before = cash

    
    # MANAGE POSITION
    if position is not None:
        position["bars"] += 1
        exit_price = None
        exit_reason = None
        side = position["side"]

        # LONG
        if side == "LONG":
            stop = position["stop"]
            target = position["target"]

            if low <= stop:
                exit_price = apply_exit_slippage(stop, side)
                exit_reason = "STOP"
            elif high >= target:
                exit_price = apply_exit_slippage(target, side)
                exit_reason = "TARGET"
        
        # SHORT
        else:
            stop = position["stop"]
            target = position["target"]

            if high >= stop:
                exit_price = apply_exit_slippage(stop, side)
                exit_reason = "STOP"
            elif low <= target:
                exit_price = apply_exit_slippage(target, side)
                exit_reason = "TARGET"

        # TIME EXIT
        if exit_price is None and position["bars"] >= MAX_HOLD_BARS:
            exit_price = apply_exit_slippage(close, side)
            exit_reason = "TIME_EXIT"

        # EXIT
        if exit_price is not None:
            entry = position["entry"]
            qty = position["qty"]

            if side == "LONG":
                gross_pnl = (exit_price - entry) * qty
            else:
                gross_pnl = (entry - exit_price) * qty

            entry_fee = entry * qty * FEE
            exit_fee = exit_price * qty * FEE
            net_pnl = gross_pnl - entry_fee - exit_fee

            cash += net_pnl

            trades.append(
                {
                    "entry_time": position["entry_time"],
                    "exit_time": current_time,
                    "side": side,
                    "regime": position["regime"],
                    "entry": entry,
                    "exit": exit_price,
                    "stop": position["stop"],
                    "target": position["target"],
                    "qty": qty,
                    "risk": position["risk"],
                    "pnl": net_pnl,
                    "R": net_pnl / position["risk"]
                    if position["risk"] > 0
                    else 0,
                    "bars": position["bars"],
                    "reason": exit_reason,
                }
            )

            position = None
            cooldown = COOLDOWN_BARS

    # COOLDOWN
    if cooldown > 0:
        cooldown -= 1

    # IF FLAT, FIND REGIME
    if position is None and cooldown == 0:
        regime_high = row["regime_high"]
        regime_low = row["regime_low"]
        range_size = row["range_size"]
        direction_ratio = row["direction_ratio"]
        range_position = row["range_position"]

        if (
            pd.isna(regime_high)
            or pd.isna(regime_low)
            or pd.isna(range_size)
            or range_size <= 0
        ):
            regime = "UNKNOWN"
        else:
            # TREND / MOMENTUM REGIME
            strong_direction = direction_ratio >= MAX_RANGE_TREND_RATIO

            # Price is moving meaningfully through its recent range.
            if strong_direction:
                if row["direction_move"] > 0:
                    regime = "BULL_TREND"
                else:
                    regime = "BEAR_TREND"
            # RANGE REGIME
            else:
                regime = "RANGE"

        # MOMENTUM LONG
        if regime == "BULL_TREND":
            breakout_level = row["prev_high"]
            breakout = close > breakout_level * (1 + BREAKOUT_BUFFER)
            strong_close = row["close_location"] >= 0.70
            strong_body = row["body_ratio"] >= 0.50

            if breakout and strong_close and strong_body:
                entry = open_price

                # Structural stop
                recent_low = df["low"].iloc[i - 10 : i].min()
                stop = float(recent_low)

                if stop < entry:
                    risk_distance = entry - stop
                    target = entry + risk_distance * MOMENTUM_RR
                    equity = cash

                    qty, risk = calculate_quantity(equity, entry, stop)

                    if qty > 0:
                        entry = apply_entry_slippage(entry, "LONG")
                        position = {
                            "side": "LONG",
                            "regime": "MOMENTUM",
                            "entry_time": current_time,
                            "entry": entry,
                            "stop": stop,
                            "target": target,
                            "qty": qty,
                            "risk": risk,
                            "bars": 0,
                        }

        # MOMENTUM SHORT
        elif regime == "BEAR_TREND":
            breakdown_level = row["prev_low"]
            breakdown = close < breakdown_level * (1 - BREAKOUT_BUFFER)
            strong_close = row["close_location"] <= 0.30
            strong_body = row["body_ratio"] >= 0.50

            if breakdown and strong_close and strong_body:
                entry = open_price

                recent_high = df["high"].iloc[i - 10 : i].max()
                stop = float(recent_high)

                if stop > entry:
                    risk_distance = stop - entry
                    target = entry - risk_distance * MOMENTUM_RR
                    equity = cash

                    qty, risk = calculate_quantity(equity, entry, stop)

                    if qty > 0:
                        entry = apply_entry_slippage(entry, "SHORT")
                        position = {
                            "side": "SHORT",
                            "regime": "MOMENTUM",
                            "entry_time": current_time,
                            "entry": entry,
                            "stop": stop,
                            "target": target,
                            "qty": qty,
                            "risk": risk,
                            "bars": 0,
                        }

    # MARK TO MARKET
    current_equity = cash
    if position is not None:
        if position["side"] == "LONG":
            unrealized = (close - position["entry"]) * position["qty"]
        else:
            unrealized = (position["entry"] - close) * position["qty"]
        current_equity += unrealized

    equity_records.append({"time": current_time, "equity": current_equity})

# DATAFRAME RESULTS
equity_df = pd.DataFrame(equity_records)
trades_df = pd.DataFrame(trades)

if len(trades_df) == 0:
    print("NO TRADES")
    raise SystemExit

# EQUITY
final_equity = equity_df["equity"].iloc[-1]
net_profit = final_equity - INITIAL_CAPITAL
total_return = (final_equity / INITIAL_CAPITAL - 1) * 100

# WIN RATE
wins = trades_df[trades_df["pnl"] > 0]
losses = trades_df[trades_df["pnl"] < 0]

win_rate = (len(wins) / len(trades_df)) * 100
gross_profit = wins["pnl"].sum()
gross_loss = abs(losses["pnl"].sum())
profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
expectancy = trades_df["pnl"].mean()

# DRAW DOWN
equity_df["peak"] = equity_df["equity"].cummax()
equity_df["drawdown"] = equity_df["equity"] - equity_df["peak"]
equity_df["drawdown_pct"] = (
    equity_df["drawdown"] / equity_df["peak"]
) * 100

max_dd = equity_df["drawdown"].min()
max_dd_pct = equity_df["drawdown_pct"].min()

# SHARPE / SORTINO
returns = equity_df["equity"].pct_change().dropna()

if returns.std() > 0:
    sharpe = (returns.mean() / returns.std()) * np.sqrt(365)
else:
    sharpe = 0

downside = returns[returns < 0]
if len(downside) > 1 and downside.std() > 0:
    sortino = (returns.mean() / downside.std()) * np.sqrt(365)
else:
    sortino = 0

# CAGR
start = equity_df["time"].iloc[0]
end = equity_df["time"].iloc[-1]
years = (end - start).total_seconds() / (365.25 * 24 * 60 * 60)

if years > 0:
    cagr = ((final_equity / INITIAL_CAPITAL) ** (1 / years) - 1) * 100
else:
    cagr = 0

# REGIME PERFORMANCE
momentum = trades_df[trades_df["regime"] == "MOMENTUM"]


def regime_stats(data):
    if len(data) == 0:
        return {"trades": 0, "pnl": 0, "winrate": 0}
    return {
        "trades": len(data),
        "pnl": data["pnl"].sum(),
        "winrate": (len(data[data["pnl"] > 0]) / len(data)) * 100,
    }


mom_stats = regime_stats(momentum)


def plot_monte_carlo(trades_data, initial_capital, simulations=5000, seed=42):
    trade_pnl = trades_data["pnl"].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    sampled_pnl = rng.choice(
        trade_pnl,
        size=(simulations, len(trade_pnl)),
        replace=True,
    )
    simulated_equity = initial_capital + np.cumsum(sampled_pnl, axis=1)
    simulated_equity = np.column_stack(
        [np.full(simulations, initial_capital), simulated_equity]
    )

    percentile_5, percentile_50, percentile_95 = np.percentile(
        simulated_equity,
        [5, 50, 95],
        axis=0,
    )
    trade_number = np.arange(simulated_equity.shape[1])

    plt.figure(figsize=(15, 7))
    for path in simulated_equity[:100]:
        plt.plot(trade_number, path, color="steelblue", alpha=0.08, linewidth=0.8)
    plt.fill_between(
        trade_number,
        percentile_5,
        percentile_95,
        color="cornflowerblue",
        alpha=0.25,
        label="5th-95th percentile",
    )
    plt.plot(
        trade_number,
        percentile_50,
        color="navy",
        linewidth=2,
        label="Median simulated equity",
    )
    plt.axhline(
        initial_capital,
        color="black",
        linestyle="--",
        alpha=0.6,
        label="Initial capital",
    )
    plt.title("Monte Carlo Trade-Sequence Simulation")
    plt.xlabel("Completed trades")
    plt.ylabel("Simulated equity ($)")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "monte_carlo_simulation.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()

# PRINT REPORT
print()
print("=" * 65)
print("BTCUSD REGIME + MOMENTUM")
print("=" * 65)
print(f"In-Sample Period  : {in_sample_df[time_col].iloc[0]} to {in_sample_df[time_col].iloc[-1]}")
print(f"Out-of-Sample     : {out_of_sample_df[time_col].iloc[0]} to {out_of_sample_df[time_col].iloc[-1]}")
print("Results below use out-of-sample data only.")
print(f"Initial Capital   : ${INITIAL_CAPITAL:,.2f}")
print(f"Final Equity      : ${final_equity:,.2f}")
print(f"Net Profit        : ${net_profit:,.2f}")
print(f"Total Return      : {total_return:.2f}%")
print(f"CAGR              : {cagr:.2f}%")
print("-" * 65)
print(f"Total Trades      : {len(trades_df)}")
print(f"Winning Trades    : {len(wins)}")
print(f"Losing Trades     : {len(losses)}")
print(f"Win Rate          : {win_rate:.2f}%")
print(f"Profit Factor     : {profit_factor:.2f}")
print(f"Expectancy / Trade: ${expectancy:.2f}")
print("-" * 65)
print(f"Max Drawdown      : ${max_dd:,.2f}")
print(f"Max Drawdown %    : {max_dd_pct:.2f}%")
print(f"Sharpe            : {sharpe:.2f}")
print(f"Sortino           : {sortino:.2f}")
print("-" * 65)
print("MOMENTUM")
print(f"  Trades          : {mom_stats['trades']}")
print(f"  P&L             : ${mom_stats['pnl']:,.2f}")
print(f"  Win Rate        : {mom_stats['winrate']:.2f}%")
print("=" * 65)

# SAVE RESULTS
trades_df.to_csv("btc_regime_trades.csv", index=False)
equity_df.to_csv("btc_regime_equity.csv", index=False)

# EQUITY CURVE
plt.figure(figsize=(15, 7))
plt.plot(
    equity_df["time"],
    equity_df["equity"],
    color="blue",
    linewidth=1.4,
)
plt.axhline(INITIAL_CAPITAL, color="black", linestyle="--", alpha=0.5)
plt.title("BTCUSD Regime Strategy - Equity Curve")
plt.xlabel("Date")
plt.ylabel("Equity ($)")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "equity_curve.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

# MONTE CARLO SIMULATION
plot_monte_carlo(trades_df, INITIAL_CAPITAL)

# DRAWDOWN
plt.figure(figsize=(15, 5))
plt.fill_between(
    equity_df["time"],
    equity_df["drawdown_pct"],
    0,
    color="red",
    alpha=0.35,
)
plt.plot(equity_df["time"], equity_df["drawdown_pct"], color="red")
plt.title("BTCUSD Strategy Drawdown")
plt.xlabel("Date")
plt.ylabel("Drawdown (%)")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "max_drawdown.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()

# OUT-OF-SAMPLE TRADES
plt.figure(figsize=(15, 7))
plt.plot(
    out_of_sample_df[time_col],
    out_of_sample_df["close"],
    color="steelblue",
    linewidth=1.0,
    label="Out-of-sample close",
)

long_trades = trades_df[trades_df["side"] == "LONG"]
short_trades = trades_df[trades_df["side"] == "SHORT"]

plt.scatter(
    long_trades["entry_time"],
    long_trades["entry"],
    marker="^",
    color="green",
    s=45,
    label="Long entry",
    zorder=3,
)
plt.scatter(
    short_trades["entry_time"],
    short_trades["entry"],
    marker="v",
    color="darkorange",
    s=45,
    label="Short entry",
    zorder=3,
)
plt.scatter(
    trades_df["exit_time"],
    trades_df["exit"],
    marker="x",
    color="red",
    s=45,
    label="Exit",
    zorder=3,
)
plt.title("Out-of-Sample Trades")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "out_of_sample_trades.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close()
