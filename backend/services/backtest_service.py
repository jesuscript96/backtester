"""
Runs backtests per ticker-date pair using translated strategy signals.
Pure numpy simulation — no vectorbt dependency.

Memory-optimised streaming architecture:
  Single-pass loop over groupby — signal generation + portfolio simulation
  happen for one day at a time so peak RSS stays well under 512 MB.
"""

import ctypes
import gc
import logging
import sys
import time

import numpy as np
import pandas as pd

from backend.services.strategy_engine import translate_strategy
from backend.services.portfolio_sim import simulate

logger = logging.getLogger("backtester.engine")



def _release_memory():
    if sys.platform == "linux":
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass


def run_backtest(
    intraday_df: pd.DataFrame,
    qualifying_df: pd.DataFrame,
    strategy_def: dict,
    init_cash: float = 10000.0,
    fees: float = 0.0,
    slippage: float = 0.0,
) -> dict:
    t_total = time.time()
    grouped = intraday_df.groupby(["ticker", "date"])
    n_groups = grouped.ngroups
    logger.info(f"[INIT] groupby done, {n_groups} groups")

    qual_lookup = _build_qualifying_lookup(qualifying_df)

    day_items: list[tuple | None] = []
    for (ticker, date), day_df in grouped:
        day_df = day_df.sort_values("timestamp").reset_index(drop=True)
        if len(day_df) < 5:
            continue
        day_items.append((
            ticker,
            str(date),
            {
                "open": day_df["open"].values,
                "high": day_df["high"].values,
                "low": day_df["low"].values,
                "close": day_df["close"].values,
                "volume": day_df["volume"].values,
                "timestamp": day_df["timestamp"].values,
            },
            qual_lookup.get((ticker, date), {}),
        ))

    del grouped, intraday_df, qualifying_df, qual_lookup
    gc.collect()
    _release_memory()
    logger.info(f"[INIT] extracted {len(day_items)} days to numpy, big DFs freed")

    all_trades: list[dict] = []
    all_candles: list[dict] = []
    all_equity: list[dict] = []
    day_results: list[dict] = []
    days_with_entries = 0
    t1 = time.time()

    for i in range(len(day_items)):
        ticker, date, arrays, daily_stats = day_items[i]
        day_items[i] = None

        mini_df = pd.DataFrame(arrays)

        try:
            signals = translate_strategy(mini_df, strategy_def, daily_stats)
        except Exception:
            del mini_df
            continue
        if not signals["entries"].any():
            del mini_df, signals
            continue

        entries_arr = signals["entries"].values if hasattr(signals["entries"], "values") else np.asarray(signals["entries"])
        exits_arr = signals["exits"].values if hasattr(signals["exits"], "values") else np.asarray(signals["exits"])

        try:
            sim_result = simulate(
                close=arrays["close"],
                open_=arrays["open"],
                high=arrays["high"],
                low=arrays["low"],
                entries=entries_arr,
                exits=exits_arr,
                direction=signals["direction"],
                init_cash=init_cash,
                fees=fees,
                slippage=slippage,
                sl_stop=signals["sl_stop"],
                sl_trail=signals["sl_trail"],
                tp_stop=signals["tp_stop"],
                accumulate=signals.get("accept_reentries", False),
            )
        except Exception as exc:
            logger.warning(f"[STREAM] day {ticker} {date} failed: {exc}")
            del mini_df, signals
            continue

        del mini_df, signals

        eq_vals = sim_result["equity"]
        raw_trades = sim_result["trades"]

        if not raw_trades:
            del sim_result
            continue

        timestamps = pd.Series(pd.to_datetime(arrays["timestamp"]))
        ts_epoch = (timestamps.astype("int64") // 10**9).values

        candles_dict = {
            "ticker": ticker,
            "date": date,
            "candles": _build_candles(ts_epoch, arrays),
        }

        trades_records = _enrich_trades(raw_trades, timestamps, ticker, date, strategy_def)

        equity = _extract_equity_from_values(eq_vals, timestamps)
        equity_dict = {"ticker": ticker, "date": date, "equity": equity}

        stats = _extract_day_stats_from_values(eq_vals, ticker, date, trades_records)

        all_candles.append(candles_dict)
        all_trades.extend(trades_records)
        all_equity.append(equity_dict)
        day_results.append(stats)
        days_with_entries += 1

        del sim_result, eq_vals, raw_trades, arrays, daily_stats

        if days_with_entries % 20 == 0:
            logger.info(
                f"[STREAM] {days_with_entries} days processed, "
                f"{i+1}/{len(day_items)} scanned ({round(time.time()-t1, 2)}s)"
            )

    del day_items
    gc.collect()
    _release_memory()
    logger.info(
        f"[STREAM] done: {days_with_entries} days with entries "
        f"({round(time.time()-t1, 2)}s)"
    )

    t4 = time.time()
    aggregate = _aggregate_metrics(day_results, all_trades)
    global_eq, global_dd = _compute_global_equity_and_drawdown(all_equity, init_cash)
    logger.info(f"[AGG] aggregate+equity done ({round(time.time()-t4, 2)}s)")

    logger.info(
        f"[DONE] {len(day_results)} days, {len(all_trades)} trades, "
        f"total={round(time.time()-t_total, 2)}s"
    )

    return {
        "aggregate_metrics": aggregate,
        "day_results": day_results,
        "candles": all_candles,
        "trades": all_trades,
        "equity_curves": all_equity,
        "global_equity": global_eq,
        "global_drawdown": global_dd,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_qualifying_lookup(qualifying_df: pd.DataFrame) -> dict:
    if qualifying_df.empty:
        return {}
    lookup: dict = {}
    for _, row in qualifying_df.iterrows():
        lookup[(row["ticker"], row["date"])] = row.to_dict()
    return lookup


def _build_candles(ts_epoch: np.ndarray, arrays: dict) -> list[dict]:
    n = len(ts_epoch)
    candles = [None] * n
    for j in range(n):
        candles[j] = {
            "time": int(ts_epoch[j]),
            "open": float(arrays["open"][j]),
            "high": float(arrays["high"][j]),
            "low": float(arrays["low"][j]),
            "close": float(arrays["close"][j]),
            "volume": int(arrays["volume"][j]),
        }
    return candles


def _enrich_trades(
    raw_trades: list[dict],
    timestamps: pd.Series,
    ticker: str,
    date: str,
    strategy_def: dict,
) -> list[dict]:
    if not raw_trades:
        return []

    max_idx = len(timestamps) - 1
    result = []
    for t in raw_trades:
        ei = min(t["entry_idx"], max_idx)
        xi = min(t["exit_idx"], max_idx)
        entry_ts = timestamps.iloc[ei]
        exit_ts = timestamps.iloc[xi]

        r_multiple = _compute_r_multiple(
            t["entry_price"], t["exit_price"], t["direction"], strategy_def
        )

        result.append({
            "ticker": ticker,
            "date": date,
            "entry_time": str(entry_ts),
            "exit_time": str(exit_ts),
            "entry_idx": t["entry_idx"],
            "exit_idx": t["exit_idx"],
            "entry_price": t["entry_price"],
            "exit_price": t["exit_price"],
            "pnl": t["pnl"],
            "return_pct": t["return_pct"],
            "direction": t["direction"],
            "status": t["status"],
            "size": t["size"],
            "exit_reason": t["exit_reason"],
            "r_multiple": r_multiple,
            "entry_hour": entry_ts.hour,
            "entry_weekday": entry_ts.weekday(),
        })
    return result


def _compute_r_multiple(
    entry_price: float, exit_price: float, direction: str, strategy_def: dict
) -> float | None:
    rm = strategy_def.get("risk_management", {})
    sl_pct = None
    if rm.get("use_hard_stop"):
        sl_cfg = rm.get("hard_stop") or {}
        sl_pct = sl_cfg.get("value")
    if not sl_pct or sl_pct <= 0:
        return None
    r_risk = entry_price * (sl_pct / 100)
    if r_risk <= 0:
        return None
    is_long = "long" in direction.lower()
    pnl_per_share = (exit_price - entry_price) if is_long else (entry_price - exit_price)
    return round(pnl_per_share / r_risk, 2)


# ---------------------------------------------------------------------------
# Equity extraction
# ---------------------------------------------------------------------------

_MAX_EQUITY_POINTS = 100


def _extract_equity_from_values(eq_vals: np.ndarray, timestamps: pd.Series) -> list[dict]:
    try:
        n = min(len(eq_vals), len(timestamps))
        if n == 0:
            return []
        ts_epoch = (timestamps.iloc[:n].astype("int64") // 10**9).values.astype(int)
        vals = eq_vals[:n].astype(np.float64)
        if n > _MAX_EQUITY_POINTS:
            idx = np.linspace(0, n - 1, _MAX_EQUITY_POINTS, dtype=int)
            ts_epoch = ts_epoch[idx]
            vals = vals[idx]
        return [{"time": int(t), "value": float(v)} for t, v in zip(ts_epoch, vals)]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Global equity & drawdown
# ---------------------------------------------------------------------------

def _compute_global_equity_and_drawdown(
    all_equity: list[dict],
    init_cash: float,
) -> tuple[list[dict], list[dict]]:
    if not all_equity:
        return [], []

    day_arrays = []
    carry = 0.0
    for day_eq in all_equity:
        points = day_eq.get("equity", [])
        if not points:
            continue
        day_vals = np.array([pt["value"] for pt in points])
        day_start = day_vals[0]
        offset = carry - day_start + init_cash if not day_arrays else carry - day_start
        day_vals = day_vals + offset
        day_arrays.append(day_vals)
        carry = day_vals[-1]

    if not day_arrays:
        return [], []

    values = np.concatenate(day_arrays)
    running_max = np.maximum.accumulate(values)
    dd_pct = np.where(running_max > 0, (values / running_max - 1) * 100, 0.0)

    base_ts = 1_000_000_000
    times = base_ts + np.arange(len(values)) * 60
    values_rounded = np.round(values, 2)
    dd_rounded = np.round(dd_pct, 4)

    global_equity = [
        {"time": int(t), "value": float(v)} for t, v in zip(times, values_rounded)
    ]
    global_drawdown = [
        {"time": int(t), "value": float(d)} for t, d in zip(times, dd_rounded)
    ]

    return global_equity, global_drawdown


# ---------------------------------------------------------------------------
# Per-day statistics
# ---------------------------------------------------------------------------

def _extract_day_stats_from_values(
    eq_vals: np.ndarray,
    ticker: str,
    date: str,
    trades_records: list[dict],
) -> dict:
    empty = {
        "ticker": ticker, "date": date,
        "total_return_pct": 0, "max_drawdown_pct": 0, "win_rate_pct": 0,
        "total_trades": 0, "profit_factor": 0, "sharpe_ratio": 0,
        "sortino_ratio": 0, "expectancy": 0, "best_trade_pct": 0,
        "worst_trade_pct": 0, "init_value": 0, "end_value": 0,
    }
    try:
        eq_arr = np.asarray(eq_vals, dtype=np.float64)
        if len(eq_arr) == 0:
            return empty

        start_val = float(eq_arr[0])
        end_val = float(eq_arr[-1])
        total_ret = (end_val / start_val - 1) * 100 if start_val > 0 else 0.0

        running_max = np.maximum.accumulate(eq_arr)
        dd_pct = np.where(running_max > 0, (eq_arr / running_max - 1) * 100, 0.0)
        max_dd = float(np.min(dd_pct))

        n_trades = len(trades_records)
        pnls = np.array([t["pnl"] for t in trades_records]) if trades_records else np.array([])
        wins = pnls[pnls > 0] if len(pnls) else np.array([])
        losses = pnls[pnls <= 0] if len(pnls) else np.array([])

        win_rate = (len(wins) / n_trades * 100) if n_trades > 0 else 0.0
        sum_wins = float(wins.sum()) if len(wins) else 0.0
        sum_losses = float(np.abs(losses.sum())) if len(losses) else 0.0
        profit_factor = (sum_wins / sum_losses) if sum_losses > 0 else 0.0
        expectancy = float(pnls.mean()) if len(pnls) else 0.0

        rets_pct = np.array([t["return_pct"] for t in trades_records]) if trades_records else np.array([])
        best_trade = float(rets_pct.max()) if len(rets_pct) else 0.0
        worst_trade = float(rets_pct.min()) if len(rets_pct) else 0.0

        bar_returns = np.diff(eq_arr) / np.where(eq_arr[:-1] != 0, eq_arr[:-1], 1.0)
        std = float(np.std(bar_returns)) if len(bar_returns) > 1 else 0.0
        mean_r = float(np.mean(bar_returns)) if len(bar_returns) > 0 else 0.0
        ann_factor = np.sqrt(252 * 390)
        sharpe = (mean_r / std * ann_factor) if std > 0 else 0.0
        down_returns = bar_returns[bar_returns < 0]
        down_std = float(np.std(down_returns)) if len(down_returns) > 1 else 0.0
        sortino = (mean_r / down_std * ann_factor) if down_std > 0 else 0.0
    except Exception:
        return empty

    return {
        "ticker": ticker,
        "date": date,
        "total_return_pct": _safe_float(total_ret),
        "max_drawdown_pct": _safe_float(max_dd),
        "win_rate_pct": _safe_float(win_rate),
        "total_trades": n_trades,
        "profit_factor": _safe_float(profit_factor),
        "sharpe_ratio": _safe_float(sharpe),
        "sortino_ratio": _safe_float(sortino),
        "expectancy": _safe_float(expectancy),
        "best_trade_pct": _safe_float(best_trade),
        "worst_trade_pct": _safe_float(worst_trade),
        "init_value": _safe_float(start_val),
        "end_value": _safe_float(end_val),
    }


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def _aggregate_metrics(day_results: list[dict], trades: list[dict]) -> dict:
    if not day_results:
        return {
            "total_days": 0,
            "total_trades": 0,
            "win_rate_pct": 0,
            "avg_return_per_day_pct": 0,
            "total_return_pct": 0,
            "avg_sharpe": 0,
            "avg_max_dd_pct": 0,
            "avg_profit_factor": 0,
            "avg_pnl": 0,
            "total_pnl": 0,
        }

    total_days = len(day_results)
    total_trades = sum(d.get("total_trades", 0) for d in day_results)

    pnls = np.array([t.get("pnl", 0) for t in trades]) if trades else np.array([])
    winning_trades = int((pnls > 0).sum()) if len(pnls) else 0
    total_closed = len(pnls)
    win_rate = (winning_trades / total_closed * 100) if total_closed > 0 else 0

    returns = np.array([d.get("total_return_pct", 0) or 0 for d in day_results])
    avg_return = float(returns.mean()) if len(returns) else 0
    total_return = float(np.prod(1 + returns / 100) * 100 - 100) if len(returns) else 0

    sharpes = np.array([d.get("sharpe_ratio", 0) or 0 for d in day_results])
    avg_sharpe = float(sharpes.mean())

    dds = np.array([d.get("max_drawdown_pct", 0) or 0 for d in day_results])
    avg_dd = float(dds.mean())

    pfs = [d.get("profit_factor") for d in day_results if d.get("profit_factor") is not None and d.get("profit_factor") > 0]
    avg_pf = float(np.mean(pfs)) if pfs else 0

    avg_pnl = float(pnls.mean()) if len(pnls) else 0

    return {
        "total_days": total_days,
        "total_trades": total_trades,
        "win_rate_pct": round(win_rate, 2),
        "avg_return_per_day_pct": round(avg_return, 4),
        "total_return_pct": round(total_return, 4),
        "avg_sharpe": round(avg_sharpe, 4),
        "avg_max_dd_pct": round(avg_dd, 4),
        "avg_profit_factor": round(avg_pf, 4),
        "avg_pnl": round(avg_pnl, 2),
        "total_pnl": round(float(pnls.sum()), 2) if len(pnls) else 0,
    }


def _safe_float(val) -> float | None:
    try:
        v = float(val)
        return v if not np.isnan(v) and not np.isinf(v) else None
    except (TypeError, ValueError):
        return None
