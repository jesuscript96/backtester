"use client";

import { useEffect, useRef } from "react";
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
  }, [candles, trades, equity]);

  return (
    <div className="bg-white rounded-lg border border-[var(--border)] overflow-hidden">
      <div className="px-4 py-2 border-b border-[var(--border)] flex items-center gap-3">
        <span className="font-semibold text-sm">{ticker}</span>
        <span className="text-xs text-[var(--muted)]">{date}</span>
        <span className="text-xs text-[var(--muted)]">1m</span>
      </div>
      <div ref={chartContainerRef} />
    </div>
  );
}
