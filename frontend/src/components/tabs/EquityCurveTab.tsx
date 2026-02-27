"use client";

import { useEffect, useRef, useMemo, useState } from "react";
import {
  createChart,
  AreaSeries,
  HistogramSeries,
  BaselineSeries,
  type IChartApi,
  type Time,
} from "lightweight-charts";
import type { GlobalEquityPoint, DrawdownPoint, TradeRecord } from "@/lib/api";

interface EquityCurveTabProps {
  globalEquity: GlobalEquityPoint[];
  globalDrawdown: DrawdownPoint[];
  trades: TradeRecord[];
  initCash: number;
  riskR: number;
}

export default function EquityCurveTab({ globalEquity, globalDrawdown, trades, initCash, riskR }: EquityCurveTabProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const ddChartRef = useRef<IChartApi | null>(null);

  type ViewMode = "$" | "%" | "R";
  const [viewMode, setViewMode] = useState<ViewMode>("$");

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

    // Split container into two divs for the two charts
    containerRef.current.innerHTML = `
      <div id="equity-chart-container" style="width: 100%; height: 400px; margin-bottom: 1rem;"></div>
      <div id="dd-chart-container" style="width: 100%; height: 150px;"></div>
    `;

    const equityContainer = containerRef.current.querySelector("#equity-chart-container") as HTMLElement;
    const ddContainer = containerRef.current.querySelector("#dd-chart-container") as HTMLElement;

    // --- Equity Chart ---
    const chart = createChart(equityContainer, {
      width: equityContainer.clientWidth,
      height: 400,
      layout: { background: { color: "#ffffff" }, textColor: "#333" },
      grid: {
        vertLines: { color: "#f0f0f0" },
        horzLines: { color: "#f0f0f0" },
      },
      rightPriceScale: { borderColor: "#e2e8f0" },
      timeScale: { borderColor: "#e2e8f0", timeVisible: true },
    });
    chartRef.current = chart;

    const equitySeries = chart.addSeries(AreaSeries, {
      lineColor: "#3b82f6",
      topColor: "rgba(59,130,246,0.4)",
      bottomColor: "rgba(59,130,246,0.05)",
      lineWidth: 2,
    });
    equitySeries.setData(
      globalEquity.map((p) => {
        let val = p.value;
        if (viewMode === "%") {
          val = ((p.value / initCash) - 1) * 100;
        } else if (viewMode === "R") {
          val = riskR > 0 ? (p.value - initCash) / riskR : 0;
        }
        return { time: p.time as Time, value: val };
      })
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



    // --- Drawdown Chart ---
    let ddChart: IChartApi | null = null;
    let drawdownSeries: any = null;

    if (globalDrawdown && globalDrawdown.length) {
      ddChart = createChart(ddContainer, {
        width: ddContainer.clientWidth,
        height: 150,
        layout: { background: { color: "#ffffff" }, textColor: "#333" },
        grid: {
          vertLines: { color: "#f0f0f0" },
          horzLines: { color: "#f0f0f0" },
        },
        rightPriceScale: { borderColor: "#e2e8f0" },
        timeScale: { borderColor: "#e2e8f0", timeVisible: true },
      });
      ddChartRef.current = ddChart;

      drawdownSeries = ddChart.addSeries(BaselineSeries, {
        baseValue: { type: "price", price: 0 },
        topLineColor: "rgba(16,185,129,0.5)",
        topFillColor1: "rgba(16,185,129,0.05)",
        topFillColor2: "rgba(16,185,129,0.02)",
        bottomLineColor: "#ef4444",
        bottomFillColor1: "rgba(239,68,68,0.05)",
        bottomFillColor2: "rgba(239,68,68,0.4)",
        lineWidth: 2,
      });

      drawdownSeries.setData(
        globalDrawdown.map((p) => {
          let val = p.value; // Drawdown is natively in % from the backend
          if (viewMode === "R") {
            // Convert % drawdown to absolute $ drawdown, then divide by R
            // Since drawdown is negative %, (p.value/100) * initCash gives $ drawdown 
            // We use the account high water mark, but roughly p.value% of current is close enough
            // Since we don't have the HWM per day, we'll approximate absolute drawdown displacement
            // Actually, risk_r view for drawdown is just % DD converted to R units via initCash
            val = riskR > 0 ? ((p.value / 100) * initCash) / riskR : 0;
          } else if (viewMode === "$") {
            val = (p.value / 100) * initCash;
          }
          return { time: p.time as Time, value: val };
        })
      );

      // Synchronize horizontal scrolling
      chart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
        if (range && ddChart) {
          ddChart.timeScale().setVisibleLogicalRange(range);
        }
      });

      ddChart.timeScale().subscribeVisibleLogicalRangeChange((range) => {
        if (range) {
          chart.timeScale().setVisibleLogicalRange(range);
        }
      });

      // Synchronize crosshair
      chart.subscribeCrosshairMove((param) => {
        if (!param.time || !ddChart || !drawdownSeries) return;
        ddChart.setCrosshairPosition(param.point?.x || 0, param.time, drawdownSeries);
      });

      ddChart.subscribeCrosshairMove((param) => {
        if (!param.time || !equitySeries) return;
        chart.setCrosshairPosition(param.point?.x || 0, param.time, equitySeries);
      });
    }

    chart.timeScale().fitContent();
    if (ddChart) ddChart.timeScale().fitContent();

    const handleResize = () => {
      if (equityContainer) {
        chart.applyOptions({ width: equityContainer.clientWidth });
      }
      if (ddContainer && ddChart) {
        ddChart.applyOptions({ width: ddContainer.clientWidth });
      }
    };
    window.addEventListener("resize", handleResize);

    return () => {
      window.removeEventListener("resize", handleResize);
      chart.remove();
      if (ddChart) ddChart.remove();
      chartRef.current = null;
      ddChartRef.current = null;
    };
  }, [globalEquity, globalDrawdown, openPositions, viewMode, initCash, riskR]);

  if (!globalEquity.length) {
    return <p className="text-sm text-[var(--muted)]">Sin datos de equity</p>;
  }

  const maxDD = globalDrawdown && globalDrawdown.length > 0
    ? Math.min(...globalDrawdown.map((d) => d.value))
    : 0;

  return (
    <div>
      {globalDrawdown && globalDrawdown.length > 0 && (
        <div className="mb-3 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <span className="text-xs text-[var(--muted)] uppercase tracking-wide">
              Max Drawdown
            </span>
            <span className="text-sm font-semibold text-[var(--danger)]">
              {maxDD.toFixed(2)}%
            </span>
          </div>

          <div className="flex bg-gray-100 p-1 rounded-md text-sm border border-gray-200">
            {(["$", "%", "R"] as ViewMode[]).map((mode) => (
              <button
                key={mode}
                onClick={() => setViewMode(mode)}
                className={`px-3 py-1 rounded transition-colors ${viewMode === mode
                    ? "bg-white text-gray-900 shadow-sm font-medium"
                    : "text-gray-500 hover:text-gray-700"
                  }`}
              >
                {mode}
              </button>
            ))}
          </div>
        </div>
      )}
      <div ref={containerRef} />
    </div>
  );
}
