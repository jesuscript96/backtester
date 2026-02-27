"""
Computes technical indicators from OHLCV data.
Uses vectorbt built-in indicators where possible, pandas_ta for the rest.
"""

import numpy as np
import pandas as pd
import vectorbt as vbt

try:
    import pandas_ta as ta
except ImportError:
    ta = None


def compute_indicator(
    name: str,
    df: pd.DataFrame,
    period: int | None = None,
    offset: int = 0,
    daily_stats: dict | None = None,
) -> pd.Series:
    """
    Compute a single indicator series from OHLCV DataFrame.

    df must have columns: open, high, low, close, volume, timestamp
    daily_stats provides pre-computed values like pm_high, pm_low, previous_close etc.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    volume = df["volume"]

    result = _compute_raw(name, close, high, low, open_, volume, period, daily_stats, df)

    if offset and offset != 0:
        result = result.shift(offset)

    return result


def _to_series(result, index) -> pd.Series:
    """Convert vectorbt output (may be DataFrame or Series) to a plain Series."""
    if isinstance(result, pd.DataFrame):
        vals = result.iloc[:, 0].values
        return pd.Series(vals, index=index)
    if isinstance(result, pd.Series):
        return pd.Series(result.values, index=index)
    return pd.Series(result, index=index)


def _compute_raw(
    name: str,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    open_: pd.Series,
    volume: pd.Series,
    period: int | None,
    daily_stats: dict | None,
    df: pd.DataFrame,
) -> pd.Series:
    ds = daily_stats or {}

    # --- Price series (direct) ---
    if name == "Close":
        return close
    if name == "Open":
        return open_
    if name == "High":
        return high
    if name == "Low":
        return low
    if name == "Volume":
        return volume.astype(float)

    # --- Built-in vectorbt indicators ---
    if name == "SMA":
        return _to_series(vbt.MA.run(close, window=period or 20, ewm=False).ma, close.index)
    if name == "EMA":
        return _to_series(vbt.MA.run(close, window=period or 20, ewm=True).ma, close.index)
    if name == "RSI":
        return _to_series(vbt.RSI.run(close, window=period or 14).rsi, close.index)
    if name == "MACD":
        return _to_series(vbt.MACD.run(close).macd, close.index)
    if name == "ATR":
        return _to_series(vbt.ATR.run(high, low, close, window=period or 14).atr, close.index)

    # --- pandas_ta indicators ---
    if name == "WMA" and ta:
        return ta.wma(close, length=period or 20)
    if name == "ADX" and ta:
        adx_df = ta.adx(high, low, close, length=period or 14)
        return adx_df.iloc[:, 0] if adx_df is not None else pd.Series(np.nan, index=close.index)
    if name == "Williams %R" and ta:
        return ta.willr(high, low, close, length=period or 14)
    if name == "VWAP":
        typical = (high + low + close) / 3
        cum_tp_vol = (typical * volume).cumsum()
        cum_vol = volume.cumsum()
        return cum_tp_vol / cum_vol.replace(0, np.nan)
    if name == "AVWAP":
        typical = (high + low + close) / 3
        cum_tp_vol = (typical * volume).cumsum()
        cum_vol = volume.cumsum()
        return cum_tp_vol / cum_vol.replace(0, np.nan)

    # --- Level indicators (scalar values broadcast to series) ---
    if name == "Pre-Market High":
        return pd.Series(ds.get("pm_high", np.nan), index=close.index)
    if name == "Pre-Market Low":
        return pd.Series(ds.get("pm_low", np.nan), index=close.index)
    if name == "High of Day":
        return high.cummax()
    if name == "Low of Day":
        return low.cummin()
    if name == "Yesterday High":
        return pd.Series(ds.get("yesterday_high", np.nan), index=close.index)
    if name == "Yesterday Low":
        return pd.Series(ds.get("yesterday_low", np.nan), index=close.index)
    if name == "Yesterday Close":
        return pd.Series(ds.get("previous_close", np.nan), index=close.index)

    # --- Custom computed ---
    if name == "Accumulated Volume":
        return volume.cumsum().astype(float)
    if name == "Consecutive Red Candles":
        is_red = (close < open_).astype(int)
        groups = is_red.ne(is_red.shift()).cumsum()
        return is_red.groupby(groups).cumsum()
    if name == "Consecutive Higher Highs":
        hh = (high > high.shift(1)).astype(int)
        groups = hh.ne(hh.shift()).cumsum()
        return hh.groupby(groups).cumsum()
    if name == "Consecutive Lower Lows":
        ll = (low < low.shift(1)).astype(int)
        groups = ll.ne(ll.shift()).cumsum()
        return ll.groupby(groups).cumsum()

    if name == "Ret % PM":
        pm_h = ds.get("pm_high", np.nan)
        prev_c = ds.get("previous_close", np.nan)
        val = (pm_h - prev_c) / prev_c * 100 if prev_c and prev_c > 0 else np.nan
        return pd.Series(val, index=close.index)
    if name == "Ret % RTH":
        return (close - open_.iloc[0]) / open_.iloc[0] * 100 if open_.iloc[0] > 0 else pd.Series(np.nan, index=close.index)
    if name == "Ret % AM":
        return (close - open_.iloc[0]) / open_.iloc[0] * 100 if open_.iloc[0] > 0 else pd.Series(np.nan, index=close.index)

    if name == "Time of Day":
        return pd.to_datetime(df["timestamp"]).dt.hour * 60 + pd.to_datetime(df["timestamp"]).dt.minute

    if name == "Max N Bars":
        return pd.Series(range(len(close)), index=close.index, dtype=float)

    return pd.Series(np.nan, index=close.index)


def detect_candle_pattern(
    df: pd.DataFrame,
    pattern: str,
    lookback: int = 0,
    consecutive_count: int = 1,
) -> pd.Series:
    """Detect candle pattern and return boolean Series."""
    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]
    body = abs(close - open_)
    full_range = high - low + 1e-10

    if pattern == "GREEN_VOLUME":
        signal = close > open_
    elif pattern == "GREEN_VOLUME_PLUS":
        signal = (close > open_) & (volume > volume.shift(1))
    elif pattern == "RED_VOLUME":
        signal = close < open_
    elif pattern == "RED_VOLUME_PLUS":
        signal = (close < open_) & (volume > volume.shift(1))
    elif pattern == "DOJI":
        signal = (body / full_range) < 0.1
    elif pattern == "HAMMER":
        lower_wick = pd.DataFrame({"o": open_, "c": close}).min(axis=1) - low
        signal = (lower_wick >= 2 * body) & (body / full_range < 0.4)
    elif pattern == "SHOOTING_STAR":
        upper_wick = high - pd.DataFrame({"o": open_, "c": close}).max(axis=1)
        signal = (upper_wick >= 2 * body) & (body / full_range < 0.4)
    else:
        signal = pd.Series(False, index=close.index)

    if lookback > 0:
        signal = signal.shift(lookback).fillna(False).astype(bool)

    if consecutive_count > 1:
        rolling_sum = signal.astype(int).rolling(window=consecutive_count, min_periods=consecutive_count).sum()
        signal = rolling_sum >= consecutive_count

    return signal.astype(bool)
