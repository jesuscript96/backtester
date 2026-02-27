"use client";

import { useEffect, useRef, useState } from "react";
import {
  createChart,
  CandlestickSeries,
  LineSeries,
  HistogramSeries,
  type IChartApi,
  type ISeriesApi,
  type CandlestickData,
  type SeriesMarker,
  type Time,
} from "lightweight-charts";
import { createSeriesMarkers } from "lightweight-charts";
import type { CandleData, TradeRecord, EquityPoint } from "@/lib/api";
import {
  calculateSMA,
  calculateEMA,
  calculateRSI,
  calculateMACD,
  calculateVWAP,
} from "@/lib/indicators";

interface ChartProps {
  candles: CandleData[];
  trades: TradeRecord[];
  equity: EquityPoint[];
  ticker: string;
  date: string;
}

export default function Chart({ candles, trades, equity, ticker, date }: ChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  // Indicators State
  const [activeIndicators, setActiveIndicators] = useState<Set<string>>(new Set());
  const [smaPeriod, setSmaPeriod] = useState(20);
  const [emaPeriod, setEmaPeriod] = useState(20);
  const [rsiPeriod, setRsiPeriod] = useState(14);

  const toggleIndicator = (id: string) => {
    setActiveIndicators((prev: Set<string>) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  useEffect(() => {
    if (!chartContainerRef.current || candles.length === 0) return;

    if (chartRef.current) {
      chartRef.current.remove();
    }

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 480,
      layout: {
        background: { color: "#ffffff" },
        textColor: "#333",
      },
      grid: {
        vertLines: { color: "#f0f0f0" },
        horzLines: { color: "#f0f0f0" },
      },
      crosshair: {
        mode: 0,
      },
      rightPriceScale: {
        borderColor: "#e2e8f0",
      },
      timeScale: {
        borderColor: "#e2e8f0",
        timeVisible: true,
        secondsVisible: false,
      },
    });

    chartRef.current = chart;

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#10b981",
      downColor: "#ef4444",
      borderDownColor: "#ef4444",
      borderUpColor: "#10b981",
      wickDownColor: "#ef4444",
      wickUpColor: "#10b981",
    });

    const sorted = [...candles].sort((a, b) => a.time - b.time);
    const deduped = sorted.filter((c, i) => i === 0 || c.time !== sorted[i - 1].time);

    const candleData: CandlestickData<Time>[] = deduped.map((c) => ({
      time: c.time as Time,
      open: c.open,
      high: c.high,
      low: c.low,
      close: c.close,
    }));

    candleSeries.setData(candleData);

    const volumeSeries = chart.addSeries(HistogramSeries, {
      priceFormat: { type: "volume" },
      priceScaleId: "volume",
    });

    chart.priceScale("volume").applyOptions({
      scaleMargins: { top: 0.85, bottom: 0 },
    });

    volumeSeries.setData(
      deduped.map((c) => ({
        time: c.time as Time,
        value: c.volume,
        color: c.close >= c.open ? "rgba(16,185,129,0.3)" : "rgba(239,68,68,0.3)",
      }))
    );

    if (trades.length > 0) {
      const markers: SeriesMarker<Time>[] = [];

      for (const t of trades) {
        if (t.entry_idx >= 0 && t.entry_idx < deduped.length) {
          const isLong = t.direction.toLowerCase().includes("long");
          markers.push({
            time: deduped[t.entry_idx].time as Time,
            position: isLong ? "belowBar" : "aboveBar",
            color: isLong ? "#10b981" : "#ef4444",
            shape: isLong ? "arrowUp" : "arrowDown",
            text: `${isLong ? "L" : "S"} $${t.entry_price.toFixed(2)}`,
          });
        }
        if (t.exit_idx >= 0 && t.exit_idx < deduped.length && t.status === "Closed") {
          markers.push({
            time: deduped[t.exit_idx].time as Time,
            position: "aboveBar",
            color: t.pnl >= 0 ? "#10b981" : "#ef4444",
            shape: "circle",
            text: `${t.pnl >= 0 ? "+" : ""}$${t.pnl.toFixed(2)}`,
          });
        }
      }

      markers.sort((a, b) => (a.time as number) - (b.time as number));
      createSeriesMarkers(candleSeries, markers);
    }

    if (equity.length > 0) {
      const equitySeries = chart.addSeries(LineSeries, {
        color: "#3b82f6",
        lineWidth: 2,
        priceScaleId: "equity",
        lastValueVisible: false,
        priceLineVisible: false,
      });

      chart.priceScale("equity").applyOptions({
        scaleMargins: { top: 0, bottom: 0.7 },
      });

      const eqSorted = [...equity].sort((a, b) => a.time - b.time);
      const eqDeduped = eqSorted.filter((e, i) => i === 0 || e.time !== eqSorted[i - 1].time);
      equitySeries.setData(
        eqDeduped.map((e) => ({ time: e.time as Time, value: e.value }))
      );
    }

    // --- ADD INDICATORS OVERLAYS ---
    if (activeIndicators.has("SMA")) {
      const smaData = calculateSMA(candles, smaPeriod);
      if (smaData.length > 0) {
        const series = chart.addSeries(LineSeries, {
          color: "#f59e0b", // Amber
          lineWidth: 2,
        });
        series.setData(smaData);
      }
    }

    if (activeIndicators.has("EMA")) {
      const emaData = calculateEMA(candles, emaPeriod);
      if (emaData.length > 0) {
        const series = chart.addSeries(LineSeries, {
          color: "#a855f7", // Purple
          lineWidth: 2,
        });
        series.setData(emaData);
      }
    }

    if (activeIndicators.has("VWAP")) {
      const vwapData = calculateVWAP(candles);
      if (vwapData.length > 0) {
        const series = chart.addSeries(LineSeries, {
          color: "#ec4899", // Pink
          lineWidth: 2,
        });
        series.setData(vwapData);
      }
    }

    // --- ADD OSCILLATORS (Independent Scales) ---
    // If we have RSI or MACD, we need to create new price scales at the bottom
    let oscCount = 0;
    if (activeIndicators.has("RSI")) oscCount++;
    if (activeIndicators.has("MACD")) oscCount++;

    if (activeIndicators.has("RSI")) {
      chart.priceScale("rsi").applyOptions({
        scaleMargins: {
          top: 0.8, // Push to bottom
          bottom: 0,
        },
      });

      const rsiData = calculateRSI(candles, rsiPeriod);
      if (rsiData.length > 0) {
        const series = chart.addSeries(LineSeries, {
          color: "#3b82f6", // Blue
          lineWidth: 2,
          priceScaleId: "rsi",
        });
        series.setData(rsiData);
        // Create 30/70 reference lines
        series.createPriceLine({ price: 70, color: "#ef4444", lineWidth: 1, lineStyle: 2 });
        series.createPriceLine({ price: 30, color: "#10b981", lineWidth: 1, lineStyle: 2 });
      }
    }

    if (activeIndicators.has("MACD")) {
      chart.priceScale("macd").applyOptions({
        scaleMargins: {
          top: 0.8, // Push to bottom, sharing space or overlapping slightly depending on implementation
          bottom: 0,
        },
      });

      const macdData = calculateMACD(candles);
      if (macdData.length > 0) {
        const macdSeries = chart.addSeries(LineSeries, {
          color: "#2563eb",
          lineWidth: 2,
          priceScaleId: "macd",
        });
        macdSeries.setData(macdData.map(d => ({ time: d.time, value: d.macd })));

        const signalSeries = chart.addSeries(LineSeries, {
          color: "#f59e0b",
          lineWidth: 1,
          priceScaleId: "macd",
        });
        signalSeries.setData(macdData.map(d => ({ time: d.time, value: d.signal })));

        const histSeries = chart.addSeries(HistogramSeries, {
          priceScaleId: "macd",
        });
        histSeries.setData(macdData.map(d => ({
          time: d.time,
          value: d.histogram,
          color: d.histogram >= 0 ? "rgba(16,185,129,0.5)" : "rgba(239,68,68,0.5)"
        })));
      }
    }


    chart.timeScale().fitContent();

    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
      chartRef.current = null;
    };
  }, [candles, trades, equity, activeIndicators, smaPeriod, emaPeriod, rsiPeriod]);

  return (
    <div className="bg-white rounded-lg border border-[var(--border)] overflow-hidden">

      {/* TOOLBAR */}
      <div className="px-4 py-2 border-b border-[var(--border)] flex flex-wrap items-center justify-between gap-3 bg-gray-50">
        <div className="flex items-center gap-3">
          <span className="font-semibold text-sm">{ticker}</span>
          <span className="text-xs text-[var(--muted)]">{date}</span>
          <span className="text-xs px-1.5 py-0.5 bg-gray-200 rounded text-gray-700">1m</span>
        </div>

        <div className="flex items-center gap-2 text-xs">
          {/* SMA */}
          <div className={`flex items-center border rounded overflow-hidden transition-colors ${activeIndicators.has("SMA") ? "border-amber-500 bg-amber-50" : "border-gray-300 bg-white"}`}>
            <button onClick={() => toggleIndicator("SMA")} className={`px-2 py-1 font-medium ${activeIndicators.has("SMA") ? "text-amber-700" : "text-gray-600 hover:bg-gray-100"}`}>
              SMA
            </button>
            {activeIndicators.has("SMA") && (
              <input type="number" value={smaPeriod} onChange={e => setSmaPeriod(Number(e.target.value))} className="w-12 pl-1 pr-0 py-1 border-l border-amber-200 bg-transparent text-amber-900 outline-none" min={1} />
            )}
          </div>

          {/* EMA */}
          <div className={`flex items-center border rounded overflow-hidden transition-colors ${activeIndicators.has("EMA") ? "border-purple-500 bg-purple-50" : "border-gray-300 bg-white"}`}>
            <button onClick={() => toggleIndicator("EMA")} className={`px-2 py-1 font-medium ${activeIndicators.has("EMA") ? "text-purple-700" : "text-gray-600 hover:bg-gray-100"}`}>
              EMA
            </button>
            {activeIndicators.has("EMA") && (
              <input type="number" value={emaPeriod} onChange={e => setEmaPeriod(Number(e.target.value))} className="w-12 pl-1 pr-0 py-1 border-l border-purple-200 bg-transparent text-purple-900 outline-none" min={1} />
            )}
          </div>

          {/* RSI */}
          <div className={`flex items-center border rounded overflow-hidden transition-colors ${activeIndicators.has("RSI") ? "border-blue-500 bg-blue-50" : "border-gray-300 bg-white"}`}>
            <button onClick={() => toggleIndicator("RSI")} className={`px-2 py-1 font-medium ${activeIndicators.has("RSI") ? "text-blue-700" : "text-gray-600 hover:bg-gray-100"}`}>
              RSI
            </button>
            {activeIndicators.has("RSI") && (
              <input type="number" value={rsiPeriod} onChange={e => setRsiPeriod(Number(e.target.value))} className="w-12 pl-1 pr-0 py-1 border-l border-blue-200 bg-transparent text-blue-900 outline-none" min={1} />
            )}
          </div>

          {/* MACD */}
          <button
            onClick={() => toggleIndicator("MACD")}
            className={`px-2 py-1 border rounded font-medium transition-colors ${activeIndicators.has("MACD") ? "border-blue-500 bg-blue-50 text-blue-700" : "border-gray-300 bg-white text-gray-600 hover:bg-gray-100"}`}
          >
            MACD
          </button>

          {/* VWAP */}
          <button
            onClick={() => toggleIndicator("VWAP")}
            className={`px-2 py-1 border rounded font-medium transition-colors ${activeIndicators.has("VWAP") ? "border-pink-500 bg-pink-50 text-pink-700" : "border-gray-300 bg-white text-gray-600 hover:bg-gray-100"}`}
          >
            VWAP
          </button>
        </div>
      </div>

      <div ref={chartContainerRef} style={{ width: "100%", height: "480px" }} />
    </div>
  );
}
