"use client";

import { useEffect, useRef, useMemo } from "react";
import {
  createChart,
  AreaSeries,
  HistogramSeries,
  type IChartApi,
  type Time,
} from "lightweight-charts";
import type { GlobalEquityPoint, TradeRecord } from "@/lib/api";

interface EquityCurveTabProps {
  globalEquity: GlobalEquityPoint[];
  trades: TradeRecord[];
}

export default function EquityCurveTab({ globalEquity, trades }: EquityCurveTabProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  const openPositions = useMemo(() => {
    if (!globalEquity.length || !trades.length) return [];
    const timeSet = new Set(globalEquity.map((p) => p.time));
    const counts = new Map<number, number>();
    for (const t of timeSet) counts.set(t, 0);

    for (const trade of trades) {
      const entryTs = Math.floor(new Date(trade.entry_time).getTime() / 1000);
      const exitTs = Math.floor(new Date(trade.exit_time).getTime() / 1000);
      for (const t of timeSet) {
        if (t >= entryTs && t <= exitTs) {
          counts.set(t, (counts.get(t) || 0) + 1);
        }
      }
    }
    return globalEquity.map((p) => ({
      time: p.time as Time,
      value: counts.get(p.time) || 0,
      color:
        (counts.get(p.time) || 0) > 0
          ? "rgba(59,130,246,0.25)"
          : "rgba(59,130,246,0.05)",
    }));
  }, [globalEquity, trades]);

  useEffect(() => {
    if (!containerRef.current || !globalEquity.length) return;

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
      rightPriceScale: { borderColor: "#e2e8f0" },
      timeScale: { borderColor: "#e2e8f0", timeVisible: false },
    });
    chartRef.current = chart;

    const equitySeries = chart.addSeries(AreaSeries, {
      lineColor: "#3b82f6",
      topColor: "rgba(59,130,246,0.4)",
      bottomColor: "rgba(59,130,246,0.05)",
      lineWidth: 2,
    });
    // #region agent log
    const _eq = globalEquity;
    const _dups: {idx: number, t: number, prev: number}[] = [];
    const _nonMono: {idx: number, t: number, prev: number}[] = [];
    for (let _i = 1; _i < _eq.length; _i++) {
      if (_eq[_i].time === _eq[_i-1].time) _dups.push({idx: _i, t: _eq[_i].time, prev: _eq[_i-1].time});
      if (_eq[_i].time < _eq[_i-1].time) _nonMono.push({idx: _i, t: _eq[_i].time, prev: _eq[_i-1].time});
    }
    fetch('http://127.0.0.1:7313/ingest/621ccf43-8399-497f-9568-7644af923671',{method:'POST',headers:{'Content-Type':'application/json','X-Debug-Session-Id':'d50eeb'},body:JSON.stringify({sessionId:'d50eeb',location:'EquityCurveTab.tsx:73',message:'globalEquity timestamps audit',runId:'post-fix',data:{total:_eq.length,first5:_eq.slice(0,5).map(p=>p.time),last5:_eq.slice(-5).map(p=>p.time),dupCount:_dups.length,nonMonoCount:_nonMono.length,firstDup:_dups[0]||null,firstNonMono:_nonMono[0]||null},timestamp:Date.now(),hypothesisId:'H1-H2-H3-H4'})}).catch(()=>{});
    // #endregion
    equitySeries.setData(
      globalEquity.map((p) => ({ time: p.time as Time, value: p.value }))
    );

    if (openPositions.length) {
      const posSeries = chart.addSeries(HistogramSeries, {
        priceFormat: { type: "volume" },
        priceScaleId: "positions",
      });
      chart.priceScale("positions").applyOptions({
        scaleMargins: { top: 0.85, bottom: 0 },
      });
      posSeries.setData(openPositions);
    }

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
  }, [globalEquity, openPositions]);

  if (!globalEquity.length) {
    return <p className="text-sm text-[var(--muted)]">Sin datos de equity</p>;
  }

  return <div ref={containerRef} />;
}
