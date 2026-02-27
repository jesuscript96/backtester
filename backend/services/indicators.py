"""
Computes technical indicators from OHLCV data.
Pure numpy/pandas implementations — no vectorbt dependency.
"""

import numpy as np
import pandas as pd

import os as _os

_HAS_NUMBA = False

try:
    import talib as _talib
except ImportError:
    _talib = None

try:
    import pandas_ta as ta
except ImportError:
    ta = None


# ---------------------------------------------------------------------------
# Pure-numpy indicator implementations
# ---------------------------------------------------------------------------

def _sma(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(values), np.nan)
    if len(values) < window:
        return out
    cs = np.cumsum(values)
    out[window - 1] = cs[window - 1] / window
    out[window:] = (cs[window:] - cs[:-window]) / window
    return out


def _ema(values: np.ndarray, window: int) -> np.ndarray:
    alpha = 2.0 / (window + 1)
    out = np.empty(len(values))
    out[:] = np.nan
    first_valid = window - 1
    if first_valid >= len(values):
        return out
    out[first_valid] = np.mean(values[:window])
    for i in range(first_valid + 1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def _rsi(close: np.ndarray, window: int) -> np.ndarray:
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = _ema(gain, window)
    avg_loss = _ema(loss, window)
    with np.errstate(divide="ignore", invalid="ignore"):
        rs = avg_gain / np.where(avg_loss != 0, avg_loss, np.nan)
        rsi = 100.0 - 100.0 / (1.0 + rs)
    out = np.full(len(close), np.nan)
    out[1:] = rsi
    return out


def _macd(close: np.ndarray, fast: int = 12, slow: int = 26) -> np.ndarray:
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    return ema_fast - ema_slow


def _atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> np.ndarray:
    n = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    return _ema(tr, window)


def _vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    typical = (high + low + close) / 3.0
    cum_tp_vol = np.cumsum(typical * volume)
    cum_vol = np.cumsum(volume)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(cum_vol != 0, cum_tp_vol / cum_vol, np.nan)


def _consecutive_count(signal: np.ndarray) -> np.ndarray:
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


def _hammer(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    body = np.abs(close - open_)
    full_range = high - low + 1e-10
    lower_wick = np.minimum(open_, close) - low
    return (lower_wick >= 2 * body) & (body / full_range < 0.4)


def _shooting_star(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    body = np.abs(close - open_)
    full_range = high - low + 1e-10
    upper_wick = high - np.maximum(open_, close)
    return (upper_wick >= 2 * body) & (body / full_range < 0.4)


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

    if name == "SMA":
        return pd.Series(_sma(close.values, period or 20), index=close.index)
    if name == "EMA":
        return pd.Series(_ema(close.values, period or 20), index=close.index)
    if name == "RSI":
        return pd.Series(_rsi(close.values, period or 14), index=close.index)
    if name == "MACD":
        return pd.Series(_macd(close.values), index=close.index)
    if name == "ATR":
        return pd.Series(_atr(high.values, low.values, close.values, period or 14), index=close.index)

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

    if name in ("VWAP", "AVWAP"):
        vals = _vwap(high.values, low.values, close.values, volume.values.astype(np.float64))
        return pd.Series(vals, index=close.index)

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

    if name == "Accumulated Volume":
        return volume.cumsum().astype(float)
    if name == "Consecutive Red Candles":
        signal = (close.values < open_.values)
        return pd.Series(_consecutive_count(signal), index=close.index)
    if name == "Consecutive Higher Highs":
        hh = np.empty(len(high), dtype=np.bool_)
        hh[0] = False
        hh[1:] = high.values[1:] > high.values[:-1]
        return pd.Series(_consecutive_count(hh), index=close.index)
    if name == "Consecutive Lower Lows":
        ll = np.empty(len(low), dtype=np.bool_)
        ll[0] = False
        ll[1:] = low.values[1:] < low.values[:-1]
        return pd.Series(_consecutive_count(ll), index=close.index)

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
        sig = _hammer(open_, high, low, close)
    elif pattern == "SHOOTING_STAR":
        sig = _shooting_star(open_, high, low, close)
    else:
        return pd.Series(False, index=idx)

    signal = pd.Series(sig, index=idx)

    if lookback > 0:
        signal = signal.shift(lookback).fillna(False).astype(bool)

    if consecutive_count > 1:
        rolling_sum = signal.astype(int).rolling(window=consecutive_count, min_periods=consecutive_count).sum()
        signal = rolling_sum >= consecutive_count

    return signal.astype(bool)
