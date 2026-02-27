"""
Runs vectorbt backtests per ticker-date pair using translated strategy signals.
Aggregates results across all qualifying days.
"""

import numpy as np
import pandas as pd
import vectorbt as vbt
from backend.services.strategy_engine import translate_strategy
# #region agent log
import time as _time, json as _json
_LOG = "/Users/jvch/Desktop/AutomatoWebs/BacktesterMVP/.cursor/debug-448660.log"
def _dlog(msg, data=None, hid=""):
    line = {"sessionId":"448660","hypothesisId":hid,"location":"backtest_service.py","message":msg,"data":data or {},"timestamp":int(_time.time()*1000)}
    try:
        with open(_LOG, "a") as f: f.write(_json.dumps(line)+"\n")
    except: pass
    print(f"[DBG-{hid}] {msg} {data or ''}")
# #endregion


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
    all_trades = []
    all_candles = []
    all_equity = []
    day_results = []

    grouped = intraday_df.groupby(["ticker", "date"])

    # #region agent log
    _t_total_loop = _time.time()
    _t_signals_sum = 0.0
    _t_portfolio_sum = 0.0
    _t_candles_sum = 0.0
    _t_trades_sum = 0.0
    _t_equity_sum = 0.0
    _t_stats_sum = 0.0
    _n_groups = 0
    _n_processed = 0
    _dlog("LOOP_START", {"total_groups": grouped.ngroups}, "HA")
    # #endregion

    for (ticker, date), day_df in grouped:
        day_df = day_df.sort_values("timestamp").reset_index(drop=True)
        if len(day_df) < 5:
            continue

        daily_row = qualifying_df[
            (qualifying_df["ticker"] == ticker)
            & (qualifying_df["date"] == date)
        ]
        daily_stats = daily_row.iloc[0].to_dict() if not daily_row.empty else {}

        # #region agent log
        _ts0 = _time.time()
        # #endregion
        try:
            signals = translate_strategy(day_df, strategy_def, daily_stats)
        except Exception:
            continue
        # #region agent log
        _t_signals_sum += _time.time() - _ts0
        # #endregion

        entries = signals["entries"]
        exits = signals["exits"]
        direction = signals["direction"]
        sl_stop = signals["sl_stop"]
        sl_trail = signals["sl_trail"]
        tp_stop = signals["tp_stop"]

        if not entries.any():
            continue

        # #region agent log
        _n_processed += 1
        # #endregion

        close = day_df["close"].values
        open_ = day_df["open"].values
        high = day_df["high"].values
        low = day_df["low"].values

        close_s = pd.Series(close, name="close")
        timestamps = pd.to_datetime(day_df["timestamp"])

        pf_kwargs = {
            "close": close_s,
            "entries": entries.values,
            "exits": exits.values,
            "direction": direction,
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

        # #region agent log
        _tp0 = _time.time()
        # #endregion
        try:
            pf = vbt.Portfolio.from_signals(**pf_kwargs)
        except Exception as e:
            print(f"Error running backtest for {ticker} {date}: {e}")
            continue
        # #region agent log
        _t_portfolio_sum += _time.time() - _tp0
        _tc0 = _time.time()
        # #endregion

        ts_epoch = (pd.to_datetime(day_df["timestamp"]).astype("int64") // 10**9).values
        candles = [
            {"time": int(t), "open": float(o), "high": float(h), "low": float(l), "close": float(c), "volume": int(v)}
            for t, o, h, l, c, v in zip(
                ts_epoch, day_df["open"].values, day_df["high"].values,
                day_df["low"].values, day_df["close"].values, day_df["volume"].values,
            )
        ]

        # #region agent log
        _t_candles_sum += _time.time() - _tc0
        _ttr0 = _time.time()
        # #endregion
        trades_records = _extract_trades(
            pf, timestamps, ticker, str(date), strategy_def, len(day_df)
        )
        # #region agent log
        _t_trades_sum += _time.time() - _ttr0
        _te0 = _time.time()
        # #endregion
        equity = _extract_equity(pf, timestamps)
        # #region agent log
        _t_equity_sum += _time.time() - _te0
        _tst0 = _time.time()
        # #endregion

        stats = _extract_day_stats(pf, ticker, str(date), trades_records)
        # #region agent log
        _t_stats_sum += _time.time() - _tst0
        # #endregion

        all_candles.append({"ticker": ticker, "date": str(date), "candles": candles})
        all_trades.extend(trades_records)
        all_equity.append({"ticker": ticker, "date": str(date), "equity": equity})
        day_results.append(stats)

    # #region agent log
    _dlog("LOOP_DONE", {
        "total_loop_s": round(_time.time()-_t_total_loop, 2),
        "groups_total": grouped.ngroups,
        "groups_processed": _n_processed,
        "signals_s": round(_t_signals_sum, 2),
        "portfolio_s": round(_t_portfolio_sum, 2),
        "candles_s": round(_t_candles_sum, 2),
        "trades_extract_s": round(_t_trades_sum, 2),
        "equity_extract_s": round(_t_equity_sum, 2),
        "stats_s": round(_t_stats_sum, 2),
    }, "HA")
    # #endregion

    aggregate = _aggregate_metrics(day_results, all_trades)
    # #region agent log
    _tg0 = _time.time()
    # #endregion
    global_eq, global_dd = _compute_global_equity_and_drawdown(
        all_equity, init_cash
    )
    # #region agent log
    _dlog("POST_PROCESS", {"global_eq_dd_s": round(_time.time()-_tg0, 2), "n_trades": len(all_trades), "n_day_results": len(day_results), "n_candles_groups": len(all_candles)}, "HD")
    # #endregion

    return {
        "aggregate_metrics": aggregate,
        "day_results": day_results,
        "candles": all_candles,
        "trades": all_trades,
        "equity_curves": all_equity,
        "global_equity": global_eq,
        "global_drawdown": global_dd,
    }


def _extract_trades(
    pf: vbt.Portfolio,
    timestamps: pd.Series,
    ticker: str,
    date: str,
    strategy_def: dict | None = None,
    total_bars: int = 0,
) -> list[dict]:
    trades = []
    try:
        records = pf.trades.records_readable
        for _, t in records.iterrows():
            entry_idx = int(t.get("Entry Timestamp", t.get("Entry Index", 0)))
            exit_idx = int(t.get("Exit Timestamp", t.get("Exit Index", 0)))

            entry_ts = timestamps.iloc[min(entry_idx, len(timestamps) - 1)]
            exit_ts = timestamps.iloc[min(exit_idx, len(timestamps) - 1)]

            entry_price = float(t.get("Avg Entry Price", t.get("Entry Price", 0)))
            exit_price = float(t.get("Avg Exit Price", t.get("Exit Price", 0)))
            direction = str(t.get("Direction", "Long"))

            entry_dt = pd.Timestamp(entry_ts)

            trades.append({
                "ticker": ticker,
                "date": date,
                "entry_time": str(entry_ts),
                "exit_time": str(exit_ts),
                "entry_idx": entry_idx,
                "exit_idx": exit_idx,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl": float(t.get("PnL", 0)),
                "return_pct": float(t.get("Return", 0)) * 100,
                "direction": direction,
                "status": str(t.get("Status", "Closed")),
                "size": float(t.get("Size", 0)),
                "exit_reason": _determine_exit_reason(
                    entry_price, exit_price, direction,
                    exit_idx, total_bars, strategy_def or {},
                ),
                "r_multiple": _compute_r_multiple(
                    entry_price, exit_price, direction, strategy_def or {},
                ),
                "entry_hour": entry_dt.hour,
                "entry_weekday": entry_dt.weekday(),
            })
    except Exception:
        pass
    return trades


def _determine_exit_reason(
    entry_price: float,
    exit_price: float,
    direction: str,
    exit_idx: int,
    total_bars: int,
    strategy_def: dict,
) -> str:
    rm = strategy_def.get("risk_management", {})
    is_long = "long" in direction.lower()

    if total_bars > 0 and exit_idx >= total_bars - 1:
        return "EOD"

    sl_pct = None
    if rm.get("use_hard_stop"):
        sl_cfg = rm.get("hard_stop") or {}
        sl_pct = sl_cfg.get("value")
    if sl_pct and sl_pct > 0:
        if is_long and exit_price <= entry_price * (1 - sl_pct / 100):
            return "SL"
        elif not is_long and exit_price >= entry_price * (1 + sl_pct / 100):
            return "SL"

    tp_pct = None
    if rm.get("use_take_profit"):
        tp_cfg = rm.get("take_profit") or {}
        tp_pct = tp_cfg.get("value")
    if tp_pct and tp_pct > 0:
        if is_long and exit_price >= entry_price * (1 + tp_pct / 100):
            return "TP"
        elif not is_long and exit_price <= entry_price * (1 - tp_pct / 100):
            return "TP"

    trailing = rm.get("trailing_stop") or {}
    if trailing.get("active"):
        return "Trailing"

    return "Signal"


def _compute_r_multiple(
    entry_price: float,
    exit_price: float,
    direction: str,
    strategy_def: dict,
) -> float | None:
    rm = strategy_def.get("risk_management", {})
    sl_cfg = rm.get("hard_stop") or {}
    sl_pct = sl_cfg.get("value") if rm.get("use_hard_stop") else None
    if not sl_pct or sl_pct <= 0:
        return None
    r_risk = entry_price * (sl_pct / 100)
    if r_risk <= 0:
        return None
    is_long = "long" in direction.lower()
    pnl_per_share = (exit_price - entry_price) if is_long else (entry_price - exit_price)
    return round(pnl_per_share / r_risk, 2)


def _extract_equity(pf: vbt.Portfolio, timestamps: pd.Series) -> list[dict]:
    equity = []
    try:
        values = pf.value()
        for i, v in enumerate(values):
            if i < len(timestamps):
                equity.append({
                    "time": int(pd.Timestamp(timestamps.iloc[i]).timestamp()),
                    "value": float(v),
                })
    except Exception:
        pass
    return equity


def _compute_global_equity_and_drawdown(
    all_equity: list[dict],
    init_cash: float,
) -> tuple[list[dict], list[dict]]:
    """Chain per-day equity curves into a global curve and compute drawdown."""
    if not all_equity:
        return [], []

    global_values = []
    carry = 0.0

    for day_eq in all_equity:
        points = day_eq.get("equity", [])
        if not points:
            continue
        day_start = points[0]["value"]
        offset = carry - day_start + init_cash if not global_values else carry - day_start
        for pt in points:
            global_values.append(pt["value"] + offset)
        carry = global_values[-1]

    if not global_values:
        return [], []

    values = np.array(global_values)
    running_max = np.maximum.accumulate(values)
    dd_pct = np.where(running_max > 0, (values / running_max - 1) * 100, 0.0)

    base_ts = 1_000_000_000
    global_equity = [
        {"time": base_ts + i * 60, "value": round(float(v), 2)}
        for i, v in enumerate(values)
    ]
    global_drawdown = [
        {"time": base_ts + i * 60, "value": round(float(d), 4)}
        for i, d in enumerate(dd_pct)
    ]

    return global_equity, global_drawdown


def _extract_day_stats(
    pf: vbt.Portfolio, ticker: str, date: str, trades_records: list[dict]
) -> dict:
    try:
        eq = pf.value()
        start_val = float(eq.iloc[0]) if hasattr(eq, "iloc") else float(eq[0])
        end_val = float(eq.iloc[-1]) if hasattr(eq, "iloc") else float(eq[-1])
        total_ret = (end_val / start_val - 1) * 100 if start_val > 0 else 0.0

        eq_arr = np.asarray(eq, dtype=np.float64)
        running_max = np.maximum.accumulate(eq_arr)
        dd_pct = np.where(running_max > 0, (eq_arr / running_max - 1) * 100, 0.0)
        max_dd = float(np.min(dd_pct))

        n_trades = len(trades_records)
        pnls = [t["pnl"] for t in trades_records]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]

        win_rate = (len(wins) / n_trades * 100) if n_trades > 0 else 0.0
        sum_wins = sum(wins) if wins else 0.0
        sum_losses = abs(sum(losses)) if losses else 0.0
        profit_factor = (sum_wins / sum_losses) if sum_losses > 0 else 0.0
        expectancy = float(np.mean(pnls)) if pnls else 0.0

        rets_pct = [t["return_pct"] for t in trades_records]
        best_trade = max(rets_pct) if rets_pct else 0.0
        worst_trade = min(rets_pct) if rets_pct else 0.0

        bar_returns = np.diff(eq_arr) / np.where(eq_arr[:-1] != 0, eq_arr[:-1], 1.0)
        std = float(np.std(bar_returns)) if len(bar_returns) > 1 else 0.0
        mean_r = float(np.mean(bar_returns)) if len(bar_returns) > 0 else 0.0
        ann_factor = np.sqrt(252 * 390)
        sharpe = (mean_r / std * ann_factor) if std > 0 else 0.0
        down_returns = bar_returns[bar_returns < 0]
        down_std = float(np.std(down_returns)) if len(down_returns) > 1 else 0.0
        sortino = (mean_r / down_std * ann_factor) if down_std > 0 else 0.0
    except Exception:
        return {
            "ticker": ticker, "date": date,
            "total_return_pct": 0, "max_drawdown_pct": 0, "win_rate_pct": 0,
            "total_trades": 0, "profit_factor": 0, "sharpe_ratio": 0,
            "sortino_ratio": 0, "expectancy": 0, "best_trade_pct": 0,
            "worst_trade_pct": 0, "init_value": 0, "end_value": 0,
        }

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

    winning_trades = sum(1 for t in trades if t.get("pnl", 0) > 0)
    total_closed = len(trades) if trades else 0
    win_rate = (winning_trades / total_closed * 100) if total_closed > 0 else 0

    returns = [d.get("total_return_pct", 0) or 0 for d in day_results]
    avg_return = np.mean(returns) if returns else 0
    total_return = np.prod([1 + r / 100 for r in returns]) * 100 - 100 if returns else 0

    sharpes = [d.get("sharpe_ratio", 0) or 0 for d in day_results]
    avg_sharpe = np.mean(sharpes) if sharpes else 0

    dds = [d.get("max_drawdown_pct", 0) or 0 for d in day_results]
    avg_dd = np.mean(dds) if dds else 0

    pfs = [d.get("profit_factor") for d in day_results if d.get("profit_factor") is not None and d.get("profit_factor") > 0]
    avg_pf = np.mean(pfs) if pfs else 0

    pnls = [t.get("pnl", 0) for t in trades]
    avg_pnl = np.mean(pnls) if pnls else 0

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
        "total_pnl": round(sum(pnls), 2),
    }


def _safe_float(val) -> float | None:
    try:
        v = float(val)
        return v if not np.isnan(v) and not np.isinf(v) else None
    except (TypeError, ValueError):
        return None


def _safe_int(val) -> int:
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0
