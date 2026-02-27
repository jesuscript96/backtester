"""
Computes technical indicators from OHLCV data.
Uses vectorbt built-in indicators where possible, numba for custom computations,
TA-Lib with pandas_ta fallback for the rest.
"""

import numpy as np
import pandas as pd
import vectorbt as vbt
from numba import njit

try:
    import talib as _talib
except ImportError:
    _talib = None

try:
    import pandas_ta as ta
except ImportError:
    ta = None


# ---------------------------------------------------------------------------
# Numba-accelerated kernels
# ---------------------------------------------------------------------------

@njit(cache=True)
def _vwap_nb(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    n = len(close)
    result = np.empty(n, dtype=np.float64)
    cum_tp_vol = 0.0
    cum_vol = 0.0
    for i in range(n):
        tp = (high[i] + low[i] + close[i]) / 3.0
        cum_tp_vol += tp * volume[i]
        cum_vol += volume[i]
        result[i] = cum_tp_vol / cum_vol if cum_vol != 0 else np.nan
    return result


@njit(cache=True)
def _consecutive_count_nb(signal: np.ndarray) -> np.ndarray:
    """Count consecutive True values in a boolean signal array."""
    n = len(signal)
    result = np.zeros(n, dtype=np.float64)
    count = 0.0
    for i in range(n):
        if signal[i]:
            count += 1.0
        else:
            count = 0.0
        result[i] = count
    return result


@njit(cache=True)
def _hammer_nb(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    n = len(close)
    result = np.empty(n, dtype=np.bool_)
    for i in range(n):
        body = abs(close[i] - open_[i])
        full_range = high[i] - low[i] + 1e-10
        lower_wick = min(open_[i], close[i]) - low[i]
        result[i] = (lower_wick >= 2 * body) and (body / full_range < 0.4)
    return result


@njit(cache=True)
def _shooting_star_nb(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> np.ndarray:
    n = len(close)
    result = np.empty(n, dtype=np.bool_)
    for i in range(n):
        body = abs(close[i] - open_[i])
        full_range = high[i] - low[i] + 1e-10
        upper_wick = high[i] - max(open_[i], close[i])
        result[i] = (upper_wick >= 2 * body) and (body / full_range < 0.4)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_indicator(
    name: str,
    df: pd.DataFrame,
    period: int | None = None,
    offset: int = 0,
    daily_stats: dict | None = None,
    cache: dict | None = None,
) -> pd.Series:
    """
    Compute a single indicator series from OHLCV DataFrame.

    df must have columns: open, high, low, close, volume, timestamp
    daily_stats provides pre-computed values like pm_high, pm_low, previous_close etc.
    cache: optional dict keyed by (name, period, offset) to avoid redundant computation.
    """
    cache_key = (name, period, offset)
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    volume = df["volume"]

    result = _compute_raw(name, close, high, low, open_, volume, period, daily_stats, df)

    if offset and offset != 0:
        result = result.shift(offset)

    if cache is not None:
        cache[cache_key] = result

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

    # --- TA-Lib (fast C) with pandas_ta fallback ---
    if name == "WMA":
        if _talib is not None:
            return pd.Series(_talib.WMA(close.values, timeperiod=period or 20), index=close.index)
        if ta is not None:
            return ta.wma(close, length=period or 20)
    if name == "ADX":
        if _talib is not None:
            return pd.Series(
                _talib.ADX(high.values, low.values, close.values, timeperiod=period or 14),
                index=close.index,
            )
        if ta is not None:
            adx_df = ta.adx(high, low, close, length=period or 14)
            return adx_df.iloc[:, 0] if adx_df is not None else pd.Series(np.nan, index=close.index)
    if name == "Williams %R":
        if _talib is not None:
            return pd.Series(
                _talib.WILLR(high.values, low.values, close.values, timeperiod=period or 14),
                index=close.index,
            )
        if ta is not None:
            return ta.willr(high, low, close, length=period or 14)

    # --- VWAP / AVWAP (numba-accelerated) ---
    if name in ("VWAP", "AVWAP"):
        vals = _vwap_nb(high.values, low.values, close.values, volume.values.astype(np.float64))
        return pd.Series(vals, index=close.index)

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

    # --- Custom computed (numba-accelerated consecutives) ---
    if name == "Accumulated Volume":
        return volume.cumsum().astype(float)
    if name == "Consecutive Red Candles":
        signal = (close.values < open_.values)
        return pd.Series(_consecutive_count_nb(signal), index=close.index)
    if name == "Consecutive Higher Highs":
        hh = np.empty(len(high), dtype=np.bool_)
        hh[0] = False
        hh[1:] = high.values[1:] > high.values[:-1]
        return pd.Series(_consecutive_count_nb(hh), index=close.index)
    if name == "Consecutive Lower Lows":
        ll = np.empty(len(low), dtype=np.bool_)
        ll[0] = False
        ll[1:] = low.values[1:] < low.values[:-1]
        return pd.Series(_consecutive_count_nb(ll), index=close.index)

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
        ts = pd.to_datetime(df["timestamp"])
        return ts.dt.hour * 60 + ts.dt.minute

    if name == "Max N Bars":
        return pd.Series(np.arange(len(close), dtype=float), index=close.index)

    return pd.Series(np.nan, index=close.index)


def detect_candle_pattern(
    df: pd.DataFrame,
    pattern: str,
    lookback: int = 0,
    consecutive_count: int = 1,
) -> pd.Series:
    """Detect candle pattern and return boolean Series."""
    close = df["close"].values
    open_ = df["open"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values
    idx = df.index

    if pattern == "GREEN_VOLUME":
        sig = close > open_
    elif pattern == "GREEN_VOLUME_PLUS":
        vol_up = np.empty(len(volume), dtype=np.bool_)
        vol_up[0] = False
        vol_up[1:] = volume[1:] > volume[:-1]
        sig = (close > open_) & vol_up
    elif pattern == "RED_VOLUME":
        sig = close < open_
    elif pattern == "RED_VOLUME_PLUS":
        vol_up = np.empty(len(volume), dtype=np.bool_)
        vol_up[0] = False
        vol_up[1:] = volume[1:] > volume[:-1]
        sig = (close < open_) & vol_up
    elif pattern == "DOJI":
        body = np.abs(close - open_)
        full_range = high - low + 1e-10
        sig = (body / full_range) < 0.1
    elif pattern == "HAMMER":
        sig = _hammer_nb(open_, high, low, close)
    elif pattern == "SHOOTING_STAR":
        sig = _shooting_star_nb(open_, high, low, close)
    else:
        return pd.Series(False, index=idx)

    signal = pd.Series(sig, index=idx)

    if lookback > 0:
        signal = signal.shift(lookback).fillna(False).astype(bool)

    if consecutive_count > 1:
        rolling_sum = signal.astype(int).rolling(window=consecutive_count, min_periods=consecutive_count).sum()
        signal = rolling_sum >= consecutive_count

    return signal.astype(bool)
