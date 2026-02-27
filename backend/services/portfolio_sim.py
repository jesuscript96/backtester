"""
Lightweight numpy portfolio simulator.
Replaces vbt.Portfolio.from_signals() with ~0 memory overhead per day.

Supports: long/short, stop-loss (fixed & trailing), take-profit, fees, slippage.
Equity model: init_cash + sum(realized_pnl) + unrealized_pnl
"""

import numpy as np


def simulate(
    close: np.ndarray,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    entries: np.ndarray,
    exits: np.ndarray,
    direction: str = "longonly",
    init_cash: float = 10000.0,
    fees: float = 0.0,
    slippage: float = 0.0,
    sl_stop: float | None = None,
    sl_trail: bool = False,
    tp_stop: float | None = None,
    accumulate: bool = False,
) -> dict:
    n = len(close)
    is_long = direction == "longonly"

    equity = np.empty(n, dtype=np.float64)
    trades: list[dict] = []

    realized_pnl = 0.0
    in_position = False
    entry_price = 0.0
    entry_idx = 0
    entry_fee_amount = 0.0
    size = 0.0
    trail_extreme = 0.0

    for i in range(n):
        # --- check exits before entries ---
        if in_position:
            exit_triggered = False
            exit_price = close[i]
            exit_reason = "Signal"

            if is_long:
                price_for_sl = low[i]
                price_for_tp = high[i]
            else:
                price_for_sl = high[i]
                price_for_tp = low[i]

            # stop-loss / trailing stop
            if sl_stop is not None:
                if sl_trail:
                    if is_long:
                        trail_extreme = max(trail_extreme, high[i])
                        sl_level = trail_extreme * (1 - sl_stop)
                        if price_for_sl <= sl_level:
                            exit_triggered = True
                            exit_price = max(sl_level, low[i])
                            exit_reason = "Trailing"
                    else:
                        trail_extreme = min(trail_extreme, low[i])
                        sl_level = trail_extreme * (1 + sl_stop)
                        if price_for_sl >= sl_level:
                            exit_triggered = True
                            exit_price = min(sl_level, high[i])
                            exit_reason = "Trailing"
                else:
                    if is_long:
                        sl_level = entry_price * (1 - sl_stop)
                        if price_for_sl <= sl_level:
                            exit_triggered = True
                            exit_price = max(sl_level, low[i])
                            exit_reason = "SL"
                    else:
                        sl_level = entry_price * (1 + sl_stop)
                        if price_for_sl >= sl_level:
                            exit_triggered = True
                            exit_price = min(sl_level, high[i])
                            exit_reason = "SL"

            # take-profit
            if not exit_triggered and tp_stop is not None:
                if is_long:
                    tp_level = entry_price * (1 + tp_stop)
                    if price_for_tp >= tp_level:
                        exit_triggered = True
                        exit_price = min(tp_level, high[i])
                        exit_reason = "TP"
                else:
                    tp_level = entry_price * (1 - tp_stop)
                    if price_for_tp <= tp_level:
                        exit_triggered = True
                        exit_price = max(tp_level, low[i])
                        exit_reason = "TP"

            # signal exit
            if not exit_triggered and exits[i]:
                exit_triggered = True
                exit_price = close[i]
                exit_reason = "Signal"

            # end-of-day forced close
            if not exit_triggered and i == n - 1:
                exit_triggered = True
                exit_price = close[i]
                exit_reason = "EOD"

            if exit_triggered:
                slip = exit_price * slippage
                net_exit = (exit_price - slip) if is_long else (exit_price + slip)
                exit_fee = abs(net_exit * size) * fees

                if is_long:
                    pnl = (net_exit - entry_price) * size - exit_fee - entry_fee_amount
                else:
                    pnl = (entry_price - net_exit) * size - exit_fee - entry_fee_amount

                realized_pnl += pnl
                capital_at_risk = entry_price * size + entry_fee_amount
                ret_pct = (pnl / capital_at_risk) * 100 if capital_at_risk > 0 else 0.0

                trades.append({
                    "entry_idx": entry_idx,
                    "exit_idx": i,
                    "entry_price": round(entry_price, 6),
                    "exit_price": round(net_exit, 6),
                    "pnl": round(pnl, 4),
                    "return_pct": round(ret_pct, 4),
                    "direction": "Long" if is_long else "Short",
                    "status": "Closed",
                    "size": round(size, 6),
                    "exit_reason": exit_reason,
                })
                in_position = False
                size = 0.0
                entry_fee_amount = 0.0

        # --- check entries ---
        if not in_position and entries[i] and i < n - 1:
            available_cash = init_cash + realized_pnl
            if available_cash <= 0:
                equity[i] = init_cash + realized_pnl
                continue

            ep = open_[i + 1]
            slip = ep * slippage
            entry_price = (ep + slip) if is_long else (ep - slip)
            if entry_price <= 0:
                equity[i] = init_cash + realized_pnl
                continue

            entry_fee_rate = 1 + fees
            size = available_cash / (entry_price * entry_fee_rate)
            if size <= 0:
                equity[i] = init_cash + realized_pnl
                continue

            entry_fee_amount = abs(entry_price * size) * fees
            realized_pnl -= entry_fee_amount
            in_position = True
            entry_idx = i + 1
            trail_extreme = entry_price

        # --- equity ---
        if in_position:
            if is_long:
                unrealized = (close[i] - entry_price) * size
            else:
                unrealized = (entry_price - close[i]) * size
            equity[i] = init_cash + realized_pnl + unrealized
        else:
            equity[i] = init_cash + realized_pnl

    return {"equity": equity, "trades": trades}
