"use client";

import { useEffect, useRef } from "react";
import {
  createChart,
  BaselineSeries,
  type IChartApi,
  type Time,
} from "lightweight-charts";
import type { DrawdownPoint } from "@/lib/api";

interface DrawdownTabProps {
  globalDrawdown: DrawdownPoint[];
}

export default function DrawdownTab({ globalDrawdown }: DrawdownTabProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!containerRef.current || !globalDrawdown.length) return;

    if (chartRef.current) {
      chartRef.current.remove();
    }

    const chart = createChart(containerRef.current, {
      width: containerRef.current.clientWidth,
      height: 400,
      layout: { background: { color: "#ffffff" }, textColor: "#333" },
      grid: {
        vertLines: { color: "#f0f0f0" },
        horzLines: { color: "#f0f0f0" },
      },
      rightPriceScale: {
        borderColor: "#e2e8f0",
      },
      timeScale: { borderColor: "#e2e8f0", timeVisible: false },
    });
    chartRef.current = chart;

    const series = chart.addSeries(BaselineSeries, {
      baseValue: { type: "price", price: 0 },
      topLineColor: "rgba(16,185,129,0.5)",
      topFillColor1: "rgba(16,185,129,0.05)",
      topFillColor2: "rgba(16,185,129,0.02)",
      bottomLineColor: "#ef4444",
      bottomFillColor1: "rgba(239,68,68,0.05)",
      bottomFillColor2: "rgba(239,68,68,0.4)",
      lineWidth: 2,
    });

    series.setData(
      globalDrawdown.map((p) => ({ time: p.time as Time, value: p.value }))
    );

    chart.timeScale().fitContent();

    const handleResize = () => {
      if (containerRef.current) {
        chart.applyOptions({ width: containerRef.current.clientWidth });
      }
    };
    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
      chartRef.current = null;
    };
  }, [globalDrawdown]);

  if (!globalDrawdown.length) {
    return <p className="text-sm text-[var(--muted)]">Sin datos de drawdown</p>;
  }

  const maxDD = Math.min(...globalDrawdown.map((d) => d.value));

  return (
    <div>
      <div className="mb-3 flex items-center gap-4">
        <span className="text-xs text-[var(--muted)] uppercase tracking-wide">
          Max Drawdown
        </span>
        <span className="text-sm font-semibold text-[var(--danger)]">
          {maxDD.toFixed(2)}%
        </span>
      </div>
      <div ref={containerRef} />
    </div>
  );
}
