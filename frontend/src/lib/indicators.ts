import type { CandleData } from "./api";
import { type Time } from "lightweight-charts";

export interface IndicatorDataPoint {
    time: Time;
    value: number;
}

export interface MACDDataPoint {
    time: Time;
    macd: number;
    signal: number;
    histogram: number;
}

// 1. Simple Moving Average (SMA)
export function calculateSMA(data: CandleData[], period: number): IndicatorDataPoint[] {
    const result: IndicatorDataPoint[] = [];
    if (data.length < period || period <= 0) return result;

    let sum = 0;
    for (let i = 0; i < data.length; i++) {
        sum += data[i].close;
        if (i >= period) {
            sum -= data[i - period].close;
        }
        if (i >= period - 1) {
            result.push({ time: data[i].time as Time, value: sum / period });
        }
    }
    return result;
}

// 2. Exponential Moving Average (EMA)
export function calculateEMA(data: CandleData[], period: number): IndicatorDataPoint[] {
    const result: IndicatorDataPoint[] = [];
    if (data.length < period || period <= 0) return result;

    const k = 2 / (period + 1);
    let ema = data[0].close; // Initial EMA could be SMA of first 'period' elements, but starting with first close is common

    // Compute SMA for the first 'period' to use as the initial EMA seed
    let sum = 0;
    for (let i = 0; i < period; i++) {
        sum += data[i].close;
    }
    ema = sum / period;
    result.push({ time: data[period - 1].time as Time, value: ema });

    for (let i = period; i < data.length; i++) {
        ema = (data[i].close - ema) * k + ema;
        result.push({ time: data[i].time as Time, value: ema });
    }
    return result;
}

// 3. Relative Strength Index (RSI)
export function calculateRSI(data: CandleData[], period: number): IndicatorDataPoint[] {
    const result: IndicatorDataPoint[] = [];
    if (data.length <= period || period <= 0) return result;

    let sumGain = 0;
    let sumLoss = 0;

    // First averages
    for (let i = 1; i <= period; i++) {
        const diff = data[i].close - data[i - 1].close;
        if (diff >= 0) sumGain += diff;
        else sumLoss += Math.abs(diff);
    }

    let avgGain = sumGain / period;
    let avgLoss = sumLoss / period;

    if (avgLoss === 0) {
        result.push({ time: data[period].time as Time, value: 100 });
    } else {
        let rs = avgGain / avgLoss;
        result.push({ time: data[period].time as Time, value: 100 - (100 / (1 + rs)) });
    }

    // Smoothed averages
    for (let i = period + 1; i < data.length; i++) {
        const diff = data[i].close - data[i - 1].close;
        const gain = diff >= 0 ? diff : 0;
        const loss = diff < 0 ? Math.abs(diff) : 0;

        avgGain = (avgGain * (period - 1) + gain) / period;
        avgLoss = (avgLoss * (period - 1) + loss) / period;

        if (avgLoss === 0) {
            result.push({ time: data[i].time as Time, value: 100 });
        } else {
            const rs = avgGain / avgLoss;
            result.push({ time: data[i].time as Time, value: 100 - (100 / (1 + rs)) });
        }
    }

    return result;
}

// 4. Moving Average Convergence Divergence (MACD)
// Using standard defaults: fast=12, slow=26, signal=9
export function calculateMACD(data: CandleData[]): MACDDataPoint[] {
    const result: MACDDataPoint[] = [];
    const fastPeriod = 12;
    const slowPeriod = 26;
    const signalPeriod = 9;

    if (data.length < slowPeriod) return result;

    const fastEMA = calculateEMA(data, fastPeriod);
    const slowEMA = calculateEMA(data, slowPeriod);

    // Align EMAs (slow EMA starts later)
    const macdLinePoints: { time: Time, value: number }[] = [];

    let slowIdx = 0;
    for (let i = 0; i < fastEMA.length; i++) {
        if (slowIdx < slowEMA.length && fastEMA[i].time === slowEMA[slowIdx].time) {
            macdLinePoints.push({
                time: fastEMA[i].time,
                value: fastEMA[i].value - slowEMA[slowIdx].value
            });
            slowIdx++;
        }
    }

    // Calculate Signal Line (EMA of MACD line)
    // We need to write a generic EMA for a stream of numbers
    if (macdLinePoints.length < signalPeriod) return result;

    const signalLine: { time: Time, value: number }[] = [];
    const k = 2 / (signalPeriod + 1);

    let sum = 0;
    for (let i = 0; i < signalPeriod; i++) {
        sum += macdLinePoints[i].value;
    }
    let ema = sum / signalPeriod;
    signalLine.push({ time: macdLinePoints[signalPeriod - 1].time, value: ema });

    for (let i = signalPeriod; i < macdLinePoints.length; i++) {
        ema = (macdLinePoints[i].value - ema) * k + ema;
        signalLine.push({ time: macdLinePoints[i].time, value: ema });
    }

    // Combine MACD, Signal, and Histogram
    let sigIdx = 0;
    for (let i = 0; i < macdLinePoints.length; i++) {
        if (sigIdx < signalLine.length && macdLinePoints[i].time === signalLine[sigIdx].time) {
            const macd = macdLinePoints[i].value;
            const signal = signalLine[sigIdx].value;
            result.push({
                time: macdLinePoints[i].time,
                macd: macd,
                signal: signal,
                histogram: macd - signal
            });
            sigIdx++;
        }
    }

    return result;
}

// 5. Volume Weighted Average Price (VWAP)
// Assumes intraday data that resets daily. 
// Uses a simplistic daily reset approach based on Date string prefix.
export function calculateVWAP(data: CandleData[]): IndicatorDataPoint[] {
    const result: IndicatorDataPoint[] = [];
    if (data.length === 0) return result;

    let cumulativeTPV = 0; // Cumulative Typical Price * Volume
    let cumulativeVolume = 0;
    let currentDay = "";

    for (let i = 0; i < data.length; i++) {
        const item = data[i];

        // We expect `item.time` to be a unix timestamp (number) or date string.
        // In our backend, it's typically a unix timestamp or ISO string.
        // If it's a number, convert to date to extract YYYY-MM-DD.
        let dateStr = "";
        if (typeof item.time === 'number') {
            dateStr = new Date(item.time * 1000).toISOString().split('T')[0];
        } else {
            dateStr = String(item.time).split('T')[0];
        }

        if (dateStr !== currentDay) {
            currentDay = dateStr;
            cumulativeTPV = 0;
            cumulativeVolume = 0;
        }

        const typicalPrice = (item.high + item.low + item.close) / 3;
        cumulativeTPV += typicalPrice * item.volume;
        cumulativeVolume += item.volume;

        if (cumulativeVolume > 0) {
            result.push({ time: item.time as Time, value: cumulativeTPV / cumulativeVolume });
        }
    }

    return result;
}
