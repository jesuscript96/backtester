"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import {
  createChart,
  LineSeries,
  type IChartApi,
  type Time,
} from "lightweight-charts";
import { runMonteCarlo, type TradeRecord, type MonteCarloResult } from "@/lib/api";

interface PortfolioTabProps {
  trades: TradeRecord[];
  initCash: number;
}

const PERCENTILE_COLORS: Record<string, string> = {
  p5: "#ef4444",
  p25: "#f97316",
  p50: "#3b82f6",
  p75: "#10b981",
  p95: "#059669",
};

const PERCENTILE_LABELS: Record<string, string> = {
  p5: "P5 (peor)",
  p25: "P25",
  p50: "P50 (mediana)",
  p75: "P75",
  p95: "P95 (mejor)",
};

export default function PortfolioTab({ trades, initCash }: PortfolioTabProps) {
  const [result, setResult] = useState<MonteCarloResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [simCount, setSimCount] = useState(1000);
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  const handleRun = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const pnls = trades.map((t) => t.pnl);
      const data = await runMonteCarlo({
        pnls,
        init_cash: initCash,
        simulations: simCount,
      });
      setResult(data);
    } catch (err: unknown) {
      const msg =
        err && typeof err === "object" && "message" in err
          ? (err as { message: string }).message
          : "Error desconocido";
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, [trades, initCash, simCount]);

  useEffect(() => {
    if (!result || !containerRef.current) return;

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

    for (const [key, color] of Object.entries(PERCENTILE_COLORS)) {
      const curveData = result.percentiles[key];
      if (!curveData) continue;

      const series = chart.addSeries(LineSeries, {
        color,
        lineWidth: key === "p50" ? 3 : 1,
        lastValueVisible: false,
        priceLineVisible: false,
      });
      series.setData(
        curveData.map((p) => ({ time: p.time as Time, value: p.value }))
      );
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
  }, [result]);

  if (!trades.length) {
    return <p className="text-sm text-[var(--muted)]">Sin trades para simular</p>;
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <label className="text-xs text-[var(--muted)]">Simulaciones:</label>
          <input
            type="number"
            min={100}
            max={10000}
            step={100}
            value={simCount}
            onChange={(e) => setSimCount(Number(e.target.value))}
            className="px-2 py-1 text-sm border border-[var(--border)] rounded w-24 focus:outline-none focus:ring-1 focus:ring-[var(--accent)]"
          />
        </div>
        <button
          onClick={handleRun}
          disabled={loading}
          className="px-4 py-1.5 text-sm font-medium rounded-md bg-[var(--accent)] text-white hover:bg-[var(--accent-hover)] disabled:opacity-50 transition-colors"
        >
          {loading ? "Simulando..." : "Ejecutar Monte Carlo"}
        </button>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-md p-3">
          <p className="text-sm text-red-700">{error}</p>
        </div>
      )}

      {loading && (
        <div className="flex items-center justify-center h-40">
          <svg
            className="animate-spin h-6 w-6 text-[var(--accent)]"
            viewBox="0 0 24 24"
          >
            <circle
              className="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              strokeWidth="4"
              fill="none"
            />
            <path
              className="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
            />
          </svg>
        </div>
      )}

      {result && !loading && (
        <>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="border border-[var(--border)] rounded-md p-3">
              <p className="text-xs text-[var(--muted)] uppercase">
                Prob. de Ruina
              </p>
              <p className="text-lg font-semibold text-[var(--danger)]">
                {result.ruin_probability}%
              </p>
            </div>
            <div className="border border-[var(--border)] rounded-md p-3">
              <p className="text-xs text-[var(--muted)] uppercase">
                Peor DD
              </p>
              <p className="text-lg font-semibold text-[var(--danger)]">
                {result.worst_drawdown.toFixed(1)}%
              </p>
            </div>
            <div className="border border-[var(--border)] rounded-md p-3">
              <p className="text-xs text-[var(--muted)] uppercase">
                DD Mediano
              </p>
              <p className="text-lg font-semibold">
                {result.median_drawdown.toFixed(1)}%
              </p>
            </div>
            <div className="border border-[var(--border)] rounded-md p-3">
              <p className="text-xs text-[var(--muted)] uppercase">
                Balance Final (P50)
              </p>
              <p className="text-lg font-semibold">
                ${result.final_balance_percentiles.p50?.toLocaleString()}
              </p>
            </div>
          </div>

          <div className="flex gap-4 text-xs flex-wrap">
            {Object.entries(PERCENTILE_LABELS).map(([key, label]) => (
              <div key={key} className="flex items-center gap-1.5">
                <span
                  className="w-3 h-0.5 rounded-full inline-block"
                  style={{ backgroundColor: PERCENTILE_COLORS[key] }}
                />
                <span className="text-[var(--muted)]">{label}</span>
              </div>
            ))}
          </div>

          <div ref={containerRef} />

          <div className="overflow-x-auto">
            <table className="text-xs w-full">
              <thead>
                <tr className="border-b border-[var(--border)]">
                  <th className="px-2 py-1 text-left text-[var(--muted)]">
                    Percentil
                  </th>
                  <th className="px-2 py-1 text-right text-[var(--muted)]">
                    Balance Final
                  </th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(result.final_balance_percentiles).map(
                  ([key, val]) => (
                    <tr key={key} className="border-b border-[var(--border)]">
                      <td className="px-2 py-1">
                        {PERCENTILE_LABELS[key] || key}
                      </td>
                      <td className="px-2 py-1 text-right font-mono">
                        ${val.toLocaleString()}
                      </td>
                    </tr>
                  )
                )}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
