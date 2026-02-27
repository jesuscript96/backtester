"""
Monte Carlo simulation via trade PnL shuffling.
Produces percentile equity curves and risk metrics.
"""

import numpy as np


def run_montecarlo(
    pnls: list[float],
    init_cash: float = 10000.0,
    simulations: int = 1000,
) -> dict:
    pnl_arr = np.array(pnls)
    n_trades = len(pnl_arr)

    rng = np.random.default_rng()
    all_curves = np.empty((simulations, n_trades + 1))
    all_curves[:, 0] = init_cash

    for i in range(simulations):
        shuffled = rng.permutation(pnl_arr)
        all_curves[i, 1:] = init_cash + np.cumsum(shuffled)

    pct_keys = [5, 25, 50, 75, 95]
    percentiles: dict[str, list[float]] = {}
    for p in pct_keys:
        curve = np.percentile(all_curves, p, axis=0)
        base_ts = 1_000_000_000
        percentiles[f"p{p}"] = [
            {"time": base_ts + j * 86400, "value": round(float(v), 2)}
            for j, v in enumerate(curve)
        ]

    final_balances = all_curves[:, -1]
    max_dds = np.empty(simulations)
    for i in range(simulations):
        curve = all_curves[i]
        running_max = np.maximum.accumulate(curve)
        dd = (curve - running_max) / np.where(running_max > 0, running_max, 1)
        max_dds[i] = float(dd.min()) * 100

    ruin_threshold = init_cash * 0.1
    ruin_count = np.sum(np.any(all_curves < ruin_threshold, axis=1))

    return {
        "percentiles": percentiles,
        "ruin_probability": round(float(ruin_count / simulations) * 100, 2),
        "worst_drawdown": round(float(max_dds.min()), 2),
        "median_drawdown": round(float(np.median(max_dds)), 2),
        "final_balance_percentiles": {
            f"p{p}": round(float(np.percentile(final_balances, p)), 2)
            for p in pct_keys
        },
    }
