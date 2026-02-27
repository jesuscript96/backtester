"""
Translates a strategy JSON definition into boolean signal arrays.
Recursively evaluates ConditionGroups (AND/OR) to produce entry/exit signals.
"""

import numpy as np
import pandas as pd
from backend.services.indicators import compute_indicator, detect_candle_pattern


def translate_strategy(
    df: pd.DataFrame,
    strategy_def: dict,
    daily_stats: dict | None = None,
) -> dict:
    """
    Translate strategy definition JSON into simulation parameters.

    Returns dict with:
        entries: pd.Series[bool]
        exits: pd.Series[bool]
        direction: str ('longonly' or 'shortonly')
        sl_stop: float | None
        sl_trail: bool
        tp_stop: float | None
        init_cash: passed through
    """
    bias = strategy_def.get("bias", "long")
    direction = "longonly" if bias == "long" else "shortonly"

    entry_logic = strategy_def.get("entry_logic", {})
    exit_logic = strategy_def.get("exit_logic", {})
    risk = strategy_def.get("risk_management", {})

    entry_tf = entry_logic.get("timeframe", "1m")
    exit_tf = exit_logic.get("timeframe", "1m")

    entry_df = _resample_if_needed(df, entry_tf)
    exit_df = _resample_if_needed(df, exit_tf)
    # #region agent log
    import json as _json6d, time as _time6d
    try:
        with open('/Users/jvch/Desktop/AutomatoWebs/BacktesterMVP/.cursor/debug-6a6030.log','a') as _f6d:
            _f6d.write(_json6d.dumps({"sessionId":"6a6030","location":"strategy_engine.py:translate_strategy","message":"df_index_types","data":{"entry_tf":entry_tf,"exit_tf":exit_tf,"entry_df_index_type":str(type(entry_df.index)),"exit_df_index_type":str(type(exit_df.index)),"df_index_type":str(type(df.index))},"timestamp":int(_time6d.time()*1000),"hypothesisId":"C"})+'\n')
    except Exception:
        pass
    # #endregion

    entry_cache: dict = {}
    exit_cache: dict = entry_cache if entry_tf == exit_tf else {}

    entries = _evaluate_condition_group(
        entry_logic.get("root_condition", {}), entry_df, daily_stats, entry_cache
    )
    exits = _evaluate_condition_group(
        exit_logic.get("root_condition", {}), exit_df, daily_stats, exit_cache
    )

    if entry_tf != "1m":
        entries = entries.reindex(df.index, method="ffill").fillna(False)
    if exit_tf != "1m":
        exits = exits.reindex(df.index, method="ffill").fillna(False)

    risk_cache: dict = entry_cache if entry_tf == "1m" else {}
    sl_stop, sl_trail, tp_stop = _parse_risk_management(risk, df, daily_stats, risk_cache)

    return {
        "entries": entries.astype(bool),
        "exits": exits.astype(bool),
        "direction": direction,
        "sl_stop": sl_stop,
        "sl_trail": sl_trail,
        "tp_stop": tp_stop,
        "accept_reentries": risk.get("accept_reentries", False),
    }


def _resample_if_needed(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    if timeframe == "1m":
        return df

    tf_map = {"5m": "5min", "15m": "15min", "30m": "30min", "1h": "1h", "1d": "1D"}
    freq = tf_map.get(timeframe, "1min")

    ts = pd.to_datetime(df["timestamp"])
    resampled = df.set_index(ts).resample(freq).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "timestamp": "first",
    }).dropna(subset=["open"])

    return resampled.reset_index(drop=True)


def _evaluate_condition_group(
    group: dict,
    df: pd.DataFrame,
    daily_stats: dict | None,
    cache: dict | None = None,
) -> pd.Series:
    """Recursively evaluate a ConditionGroup with AND/OR logic."""
    if not group:
        return pd.Series(True, index=df.index)

    operator = group.get("operator", "AND")
    conditions = group.get("conditions", [])

    if not conditions:
        return pd.Series(True, index=df.index)

    results = []
    for cond in conditions:
        cond_type = cond.get("type", "")
        if cond_type == "group" or ("conditions" in cond and "operator" in cond):
            result = _evaluate_condition_group(cond, df, daily_stats, cache)
        else:
            result = _evaluate_single_condition(cond, df, daily_stats, cache)
        results.append(result)

    if not results:
        return pd.Series(True, index=df.index)

    combined = results[0]
    for r in results[1:]:
        if operator == "AND":
            combined = combined & r
        else:
            combined = combined | r

    return combined


def _evaluate_single_condition(
    cond: dict,
    df: pd.DataFrame,
    daily_stats: dict | None,
    cache: dict | None = None,
) -> pd.Series:
    cond_type = cond.get("type", "")

    if cond_type == "indicator_comparison":
        return _eval_indicator_comparison(cond, df, daily_stats, cache)
    elif cond_type == "price_level_distance":
        return _eval_price_level_distance(cond, df, daily_stats, cache)
    elif cond_type == "candle_pattern":
        return _eval_candle_pattern(cond, df)
    else:
        return pd.Series(True, index=df.index)


def _eval_indicator_comparison(
    cond: dict,
    df: pd.DataFrame,
    daily_stats: dict | None,
    cache: dict | None = None,
) -> pd.Series:
    source_cfg = cond.get("source", {})
    target_cfg = cond.get("target", {})
    comparator = cond.get("comparator", "GREATER_THAN")

    source_series = compute_indicator(
        name=source_cfg.get("name", "Close"),
        df=df,
        period=source_cfg.get("period"),
        offset=source_cfg.get("offset", 0),
        daily_stats=daily_stats,
        cache=cache,
    )

    if isinstance(target_cfg, (int, float)):
        target_series = pd.Series(float(target_cfg), index=df.index)
    elif isinstance(target_cfg, dict):
        target_series = compute_indicator(
            name=target_cfg.get("name", "Close"),
            df=df,
            period=target_cfg.get("period"),
            offset=target_cfg.get("offset", 0),
            daily_stats=daily_stats,
            cache=cache,
        )
    else:
        target_series = pd.Series(float(target_cfg), index=df.index)

    return _apply_comparator(source_series, target_series, comparator)


def _eval_price_level_distance(
    cond: dict,
    df: pd.DataFrame,
    daily_stats: dict | None,
    cache: dict | None = None,
) -> pd.Series:
    source_name = cond.get("source", "Close")
    level_name = cond.get("level", "Pre-Market High")
    comparator = cond.get("comparator", "DISTANCE_LESS_THAN")
    value_pct = cond.get("value_pct", 1.0)

    source_series = compute_indicator(name=source_name, df=df, daily_stats=daily_stats, cache=cache)
    level_series = compute_indicator(name=level_name, df=df, daily_stats=daily_stats, cache=cache)

    distance_pct = abs(source_series - level_series) / level_series.replace(0, np.nan) * 100

    if comparator == "DISTANCE_LESS_THAN":
        return distance_pct <= value_pct
    elif comparator == "DISTANCE_GREATER_THAN":
        return distance_pct >= value_pct
    else:
        return _apply_comparator(distance_pct, pd.Series(value_pct, index=df.index), comparator)


def _eval_candle_pattern(cond: dict, df: pd.DataFrame) -> pd.Series:
    return detect_candle_pattern(
        df=df,
        pattern=cond.get("pattern", "GREEN_VOLUME"),
        lookback=cond.get("lookback", 0),
        consecutive_count=cond.get("consecutive_count", 1),
    )


def _apply_comparator(
    source: pd.Series,
    target: pd.Series,
    comparator: str,
) -> pd.Series:
    if comparator == "GREATER_THAN":
        return source > target
    elif comparator == "LESS_THAN":
        return source < target
    elif comparator == "GREATER_THAN_OR_EQUAL":
        return source >= target
    elif comparator == "LESS_THAN_OR_EQUAL":
        return source <= target
    elif comparator == "EQUAL":
        return source == target
    elif comparator == "CROSSES_ABOVE":
        return (source.shift(1) <= target.shift(1)) & (source > target)
    elif comparator == "CROSSES_BELOW":
        return (source.shift(1) >= target.shift(1)) & (source < target)
    elif comparator == "DISTANCE_GREATER_THAN":
        return abs(source - target) / target.replace(0, np.nan) * 100 > target
    elif comparator == "DISTANCE_LESS_THAN":
        return abs(source - target) / target.replace(0, np.nan) * 100 < target
    else:
        return source > target


def _parse_risk_management(
    risk: dict,
    df: pd.DataFrame,
    daily_stats: dict | None,
    cache: dict | None = None,
) -> tuple[float | None, bool, float | None]:
    sl_stop = None
    sl_trail = False
    tp_stop = None

    if risk.get("use_hard_stop") and risk.get("hard_stop"):
        hs = risk["hard_stop"]
        hs_type = hs.get("type", "Percentage")
        hs_value = hs.get("value", 0)

        if hs_type == "Percentage":
            sl_stop = hs_value / 100.0
        elif hs_type == "Fixed Amount":
            first_close = df["close"].iloc[0] if not df.empty else 1
            sl_stop = hs_value / first_close if first_close > 0 else None
        elif hs_type == "ATR Multiplier":
            atr = compute_indicator("ATR", df, period=14, daily_stats=daily_stats, cache=cache)
            avg_atr = atr.dropna().mean()
            first_close = df["close"].iloc[0] if not df.empty else 1
            sl_stop = (avg_atr * hs_value) / first_close if first_close > 0 else None
        elif hs_type == "Market Structure (HOD/LOD)":
            sl_stop = None

    trailing = risk.get("trailing_stop", {})
    if trailing.get("active"):
        sl_trail = True
        if trailing.get("type") == "Percentage" and trailing.get("buffer_pct"):
            sl_stop = trailing["buffer_pct"] / 100.0

    if risk.get("use_take_profit") and risk.get("take_profit"):
        tp = risk["take_profit"]
        if tp.get("type") == "Percentage":
            tp_stop = tp.get("value", 0) / 100.0

    return sl_stop, sl_trail, tp_stop
