"""
Runs vectorbt backtests per ticker-date pair using translated strategy signals.
Aggregates results across all qualifying days.

Performance strategy:
  - Phase 1: Generate signals for all days in parallel (ThreadPoolExecutor)
  - Phase 2: Group compatible days by (bar_count, risk_params)
  - Phase 3: Execute batched Portfolio.from_signals() per group (one vbt call per group)
  - Phase 4: Extract per-column results from batched portfolios
"""

import logging
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import vectorbt as vbt

from backend.services.strategy_engine import translate_strategy

logger = logging.getLogger("backtester.engine")

_MAX_WORKERS = int(os.getenv("BACKTEST_WORKERS", min(os.cpu_count() or 2, 4)))
_MIN_BATCH_SIZE = 2


def run_backtest(
    intraday_df: pd.DataFrame,
    qualifying_df: pd.DataFrame,
    strategy_def: dict,
    init_cash: float = 10000.0,
    fees: float = 0.0,
    slippage: float = 0.0,
) -> dict:
    """
    Run backtest for each (ticker, date) pair independently.
    Returns aggregated metrics + per-day details.
    """
    t_total = time.time()
    grouped = intraday_df.groupby(["ticker", "date"])
    logger.info(f"[PHASE 0] groupby done, {grouped.ngroups} groups, workers={_MAX_WORKERS}")

    t0 = time.time()
    qual_lookup = _build_qualifying_lookup(qualifying_df)
    logger.info(f"[PHASE 0] qualifying lookup built ({round(time.time()-t0, 2)}s)")

    # Phase 1 — generate signals for all days in parallel
    signal_inputs = []
    for (ticker, date), day_df in grouped:
        day_df = day_df.sort_values("timestamp").reset_index(drop=True)
        if len(day_df) < 5:
            continue
        daily_stats = qual_lookup.get((ticker, date), {})
        signal_inputs.append((ticker, str(date), day_df, daily_stats))

    logger.info(f"[PHASE 1] generating signals for {len(signal_inputs)} days...")
    t1 = time.time()
    prepared: list[tuple] = []
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as executor:
        future_map = {
            executor.submit(translate_strategy, inp[2], strategy_def, inp[3]): inp
            for inp in signal_inputs
        }
        for future in as_completed(future_map):
            ticker, date, day_df, daily_stats = future_map[future]
            try:
                signals = future.result()
            except Exception:
                continue
            if not signals["entries"].any():
                continue
            prepared.append((ticker, date, day_df, signals))
    logger.info(f"[PHASE 1] signals done: {len(prepared)} days with entries ({round(time.time()-t1, 2)}s)")

    # Phase 2 — group by (bar_count, risk_params) for batching
    t2 = time.time()
    batches: dict[tuple, list] = defaultdict(list)
    for item in prepared:
        _ticker, _date, _day_df, _signals = item
        key = _batch_key(_day_df, _signals)
        batches[key].append(item)
    batch_sizes = [len(v) for v in batches.values()]
    logger.info(f"[PHASE 2] {len(batches)} batch groups, sizes={batch_sizes} ({round(time.time()-t2, 2)}s)")

    all_trades: list[dict] = []
    all_candles: list[dict] = []
    all_equity: list[dict] = []
    day_results: list[dict] = []

    # Phase 3 + 4 — execute and extract
    t3 = time.time()
    batch_i = 0
    for batch_key, batch_items in batches.items():
        tb = time.time()
        if len(batch_items) >= _MIN_BATCH_SIZE:
            results = _process_batch(batch_items, init_cash, fees, slippage, strategy_def)
        else:
            results = [
                _process_single_prepared(item, init_cash, fees, slippage, strategy_def)
                for item in batch_items
            ]
        batch_i += 1
        logger.info(f"[PHASE 3] batch {batch_i}/{len(batches)} ({len(batch_items)} days) ({round(time.time()-tb, 2)}s)")

        for r in results:
            if r is None:
                continue
            candles, trades, equity, stats = r
            all_candles.append(candles)
            all_trades.extend(trades)
            all_equity.append(equity)
            day_results.append(stats)
    logger.info(f"[PHASE 3] all batches done ({round(time.time()-t3, 2)}s)")

    t4 = time.time()
    aggregate = _aggregate_metrics(day_results, all_trades)
    global_eq, global_dd = _compute_global_equity_and_drawdown(all_equity, init_cash)
    logger.info(f"[PHASE 4] aggregate+equity done ({round(time.time()-t4, 2)}s)")

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
    """Pre-index qualifying stats by (ticker, date) for O(1) access."""
    if qualifying_df.empty:
        return {}
    lookup: dict = {}
    for _, row in qualifying_df.iterrows():
        lookup[(row["ticker"], row["date"])] = row.to_dict()
    return lookup


def _batch_key(day_df: pd.DataFrame, signals: dict) -> tuple:
    """Grouping key for compatible days that can share one Portfolio call."""
    sl = signals["sl_stop"]
    tp = signals["tp_stop"]
    return (
        len(day_df),
        signals["direction"],
        round(sl, 8) if sl is not None else None,
        signals["sl_trail"],
        round(tp, 8) if tp is not None else None,
        signals.get("accept_reentries", False),
    )


# ---------------------------------------------------------------------------
# Batch processing — single Portfolio.from_signals() for N days
# ---------------------------------------------------------------------------

def _process_batch(
    batch_items: list[tuple],
    init_cash: float,
    fees: float,
    slippage: float,
    strategy_def: dict,
) -> list[tuple | None]:
    """Run a batch of same-length, same-risk-params days in one vbt call."""
    col_names = [f"{t}_{d}" for t, d, _, _ in batch_items]
    first_signals = batch_items[0][3]

    close_dict = {}
    entries_dict = {}
    exits_dict = {}
    open_dict = {}
    high_dict = {}
    low_dict = {}

    for col, (_t, _d, day_df, signals) in zip(col_names, batch_items):
        close_dict[col] = day_df["close"].values
        open_dict[col] = day_df["open"].values
        high_dict[col] = day_df["high"].values
        low_dict[col] = day_df["low"].values
        entries_dict[col] = signals["entries"].values
        exits_dict[col] = signals["exits"].values

    pf_kwargs: dict = {
        "close": pd.DataFrame(close_dict),
        "entries": pd.DataFrame(entries_dict),
        "exits": pd.DataFrame(exits_dict),
        "direction": first_signals["direction"],
        "init_cash": init_cash,
        "fees": fees,
        "slippage": slippage,
        "open": pd.DataFrame(open_dict),
        "high": pd.DataFrame(high_dict),
        "low": pd.DataFrame(low_dict),
        "freq": "1min",
    }

    sl_stop = first_signals["sl_stop"]
    if sl_stop is not None:
        pf_kwargs["sl_stop"] = sl_stop
        if first_signals["sl_trail"]:
            pf_kwargs["sl_trail"] = True
    if first_signals["tp_stop"] is not None:
        pf_kwargs["tp_stop"] = first_signals["tp_stop"]
    if not first_signals.get("accept_reentries", False):
        pf_kwargs["accumulate"] = False

    try:
        pf = vbt.Portfolio.from_signals(**pf_kwargs)
    except Exception:
        return [
            _process_single_prepared(item, init_cash, fees, slippage, strategy_def)
            for item in batch_items
        ]

    # Extract per-column results
    try:
        all_records = pf.trades.records_readable
    except Exception:
        all_records = pd.DataFrame()

    equity_df = pf.value()

    results: list[tuple | None] = []
    for col, (ticker, date, day_df, signals) in zip(col_names, batch_items):
        timestamps = pd.to_datetime(day_df["timestamp"])

        # Candles
        ts_epoch = (timestamps.astype("int64") // 10**9).values
        candle_df = pd.DataFrame({
            "time": ts_epoch.astype(int),
            "open": day_df["open"].values.astype(float),
            "high": day_df["high"].values.astype(float),
            "low": day_df["low"].values.astype(float),
            "close": day_df["close"].values.astype(float),
            "volume": day_df["volume"].values.astype(int),
        })
        candles_dict = {"ticker": ticker, "date": date, "candles": candle_df.to_dict(orient="records")}

        # Trades from batch records
        if not all_records.empty and "Column" in all_records.columns:
            col_records = all_records[all_records["Column"] == col]
        else:
            col_records = pd.DataFrame()
        trades_records = _extract_trades_from_records(
            col_records, timestamps, ticker, date, strategy_def, len(day_df)
        )

        # Equity from batch equity DataFrame
        col_equity_vals = equity_df[col].values if col in equity_df.columns else np.array([])
        equity = _extract_equity_from_values(col_equity_vals, timestamps)
        equity_dict = {"ticker": ticker, "date": date, "equity": equity}

        # Stats
        stats = _extract_day_stats_from_values(col_equity_vals, ticker, date, trades_records)

        results.append((candles_dict, trades_records, equity_dict, stats))

    return results


# ---------------------------------------------------------------------------
# Single-day processing (fallback for batches of size 1)
# ---------------------------------------------------------------------------

def _process_single_prepared(
    item: tuple,
    init_cash: float,
    fees: float,
    slippage: float,
    strategy_def: dict,
) -> tuple | None:
    """Process a single prepared (ticker, date, day_df, signals)."""
    ticker, date, day_df, signals = item

    entries = signals["entries"]
    exits = signals["exits"]
    sl_stop = signals["sl_stop"]
    sl_trail = signals["sl_trail"]
    tp_stop = signals["tp_stop"]

    close = day_df["close"].values
    open_ = day_df["open"].values
    high = day_df["high"].values
    low = day_df["low"].values
    timestamps = pd.to_datetime(day_df["timestamp"])

    pf_kwargs: dict = {
        "close": pd.Series(close, name="close"),
        "entries": entries.values,
        "exits": exits.values,
        "direction": signals["direction"],
        "init_cash": init_cash,
        "fees": fees,
        "slippage": slippage,
        "open": pd.Series(open_),
        "high": pd.Series(high),
        "low": pd.Series(low),
        "freq": "1min",
    }

    if sl_stop is not None:
        pf_kwargs["sl_stop"] = sl_stop
        if sl_trail:
            pf_kwargs["sl_trail"] = True
    if tp_stop is not None:
        pf_kwargs["tp_stop"] = tp_stop
    if not signals.get("accept_reentries", False):
        pf_kwargs["accumulate"] = False

    try:
        pf = vbt.Portfolio.from_signals(**pf_kwargs)
    except Exception:
        return None

    ts_epoch = (timestamps.astype("int64") // 10**9).values
    candle_df = pd.DataFrame({
        "time": ts_epoch.astype(int),
        "open": open_.astype(float),
        "high": high.astype(float),
        "low": low.astype(float),
        "close": close.astype(float),
        "volume": day_df["volume"].values.astype(int),
    })
    candles_dict = {"ticker": ticker, "date": date, "candles": candle_df.to_dict(orient="records")}

    try:
        records = pf.trades.records_readable
    except Exception:
        records = pd.DataFrame()

    trades_records = _extract_trades_from_records(
        records, timestamps, ticker, date, strategy_def, len(day_df)
    )

    eq_vals = np.asarray(pf.value(), dtype=np.float64)
    equity = _extract_equity_from_values(eq_vals, timestamps)
    equity_dict = {"ticker": ticker, "date": date, "equity": equity}

    stats = _extract_day_stats_from_values(eq_vals, ticker, date, trades_records)

    return candles_dict, trades_records, equity_dict, stats


# ---------------------------------------------------------------------------
# Trade extraction (vectorized, works for both single & batch)
# ---------------------------------------------------------------------------

def _extract_trades_from_records(
    records: pd.DataFrame,
    timestamps: pd.Series,
    ticker: str,
    date: str,
    strategy_def: dict | None = None,
    total_bars: int = 0,
) -> list[dict]:
    """Build trade dicts from a (possibly filtered) records_readable DataFrame."""
    try:
        if records.empty:
            return []

        cols = records.columns
        entry_idx_col = "Entry Timestamp" if "Entry Timestamp" in cols else "Entry Index"
        exit_idx_col = "Exit Timestamp" if "Exit Timestamp" in cols else "Exit Index"
        entry_price_col = "Avg Entry Price" if "Avg Entry Price" in cols else "Entry Price"
        exit_price_col = "Avg Exit Price" if "Avg Exit Price" in cols else "Exit Price"

        entry_indices = records[entry_idx_col].astype(int).values
        exit_indices = records[exit_idx_col].astype(int).values

        max_idx = len(timestamps) - 1
        entry_clipped = np.minimum(entry_indices, max_idx)
        exit_clipped = np.minimum(exit_indices, max_idx)

        entry_ts = timestamps.iloc[entry_clipped].values
        exit_ts = timestamps.iloc[exit_clipped].values

        entry_prices = records[entry_price_col].astype(float).values
        exit_prices = records[exit_price_col].astype(float).values
        directions = records["Direction"].astype(str).values if "Direction" in cols else np.full(len(records), "Long")
        pnls = records["PnL"].astype(float).values if "PnL" in cols else np.zeros(len(records))
        returns_pct = (records["Return"].astype(float).values * 100) if "Return" in cols else np.zeros(len(records))
        statuses = records["Status"].astype(str).values if "Status" in cols else np.full(len(records), "Closed")
        sizes = records["Size"].astype(float).values if "Size" in cols else np.zeros(len(records))

        entry_dts = pd.to_datetime(entry_ts)

        exit_reasons = _determine_exit_reasons_vec(
            entry_prices, exit_prices, directions, exit_indices, total_bars, strategy_def or {}
        )
        r_multiples = _compute_r_multiples_vec(
            entry_prices, exit_prices, directions, strategy_def or {}
        )

        result_df = pd.DataFrame({
            "ticker": ticker,
            "date": date,
            "entry_time": pd.Series(entry_ts).astype(str).values,
            "exit_time": pd.Series(exit_ts).astype(str).values,
            "entry_idx": entry_indices,
            "exit_idx": exit_indices,
            "entry_price": entry_prices,
            "exit_price": exit_prices,
            "pnl": pnls,
            "return_pct": returns_pct,
            "direction": directions,
            "status": statuses,
            "size": sizes,
            "exit_reason": exit_reasons,
            "r_multiple": r_multiples,
            "entry_hour": entry_dts.hour,
            "entry_weekday": entry_dts.weekday,
        })
        return result_df.to_dict(orient="records")
    except Exception:
        return []


def _determine_exit_reasons_vec(
    entry_prices: np.ndarray,
    exit_prices: np.ndarray,
    directions: np.ndarray,
    exit_indices: np.ndarray,
    total_bars: int,
    strategy_def: dict,
) -> np.ndarray:
    """Vectorized exit reason assignment with priority: EOD > SL > TP > Trailing > Signal."""
    n = len(entry_prices)
    rm = strategy_def.get("risk_management", {})
    is_long = np.array(["long" in d.lower() for d in directions])
    reasons = np.full(n, "Signal", dtype=object)

    trailing = rm.get("trailing_stop") or {}
    if trailing.get("active"):
        reasons[:] = "Trailing"

    tp_pct = None
    if rm.get("use_take_profit"):
        tp_cfg = rm.get("take_profit") or {}
        tp_pct = tp_cfg.get("value")
    if tp_pct and tp_pct > 0:
        tp_long = is_long & (exit_prices >= entry_prices * (1 + tp_pct / 100))
        tp_short = (~is_long) & (exit_prices <= entry_prices * (1 - tp_pct / 100))
        reasons[tp_long | tp_short] = "TP"

    sl_pct = None
    if rm.get("use_hard_stop"):
        sl_cfg = rm.get("hard_stop") or {}
        sl_pct = sl_cfg.get("value")
    if sl_pct and sl_pct > 0:
        sl_long = is_long & (exit_prices <= entry_prices * (1 - sl_pct / 100))
        sl_short = (~is_long) & (exit_prices >= entry_prices * (1 + sl_pct / 100))
        reasons[sl_long | sl_short] = "SL"

    if total_bars > 0:
        reasons[exit_indices >= total_bars - 1] = "EOD"

    return reasons


def _compute_r_multiples_vec(
    entry_prices: np.ndarray,
    exit_prices: np.ndarray,
    directions: np.ndarray,
    strategy_def: dict,
) -> list:
    """Vectorized R-multiple computation."""
    rm = strategy_def.get("risk_management", {})
    sl_cfg = rm.get("hard_stop") or {}
    sl_pct = sl_cfg.get("value") if rm.get("use_hard_stop") else None

    if not sl_pct or sl_pct <= 0:
        return [None] * len(entry_prices)

    r_risk = entry_prices * (sl_pct / 100)
    is_long = np.array(["long" in d.lower() for d in directions])
    pnl_per_share = np.where(is_long, exit_prices - entry_prices, entry_prices - exit_prices)

    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(r_risk > 0, np.round(pnl_per_share / r_risk, 2), np.nan)
    return [None if np.isnan(v) else float(v) for v in result]


# ---------------------------------------------------------------------------
# Equity extraction (vectorized)
# ---------------------------------------------------------------------------

def _extract_equity_from_values(eq_vals: np.ndarray, timestamps: pd.Series) -> list[dict]:
    """Build equity list[dict] from a numpy array of equity values."""
    try:
        n = min(len(eq_vals), len(timestamps))
        if n == 0:
            return []
        ts_epoch = (timestamps.iloc[:n].astype("int64") // 10**9).values.astype(int)
        vals = eq_vals[:n].astype(np.float64)
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
    """Chain per-day equity curves into a global curve and compute drawdown."""
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
# Per-day statistics (from pre-extracted equity values)
# ---------------------------------------------------------------------------

def _extract_day_stats_from_values(
    eq_vals: np.ndarray,
    ticker: str,
    date: str,
    trades_records: list[dict],
) -> dict:
    """Compute day statistics from an equity array and trade records."""
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
