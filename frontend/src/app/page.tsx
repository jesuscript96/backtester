"use client";

import { useState, useRef } from "react";
import BacktestPanel from "@/components/BacktestPanel";
import Chart from "@/components/Chart";
import MetricsCard from "@/components/MetricsCard";
import ResultsTabs from "@/components/ResultsTabs";
import DaySelector from "@/components/DaySelector";
import { runBacktest, type BacktestResult } from "@/lib/api";

export default function Home() {
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedDay, setSelectedDay] = useState(0);
  const initCashRef = useRef(10000);

  const handleRun = async (params: {
    dataset_id: string;
    strategy_id: string;
    init_cash: number;
    fees: number;
    slippage: number;
  }) => {
    setLoading(true);
    setError(null);
    setResult(null);
    setSelectedDay(0);
    initCashRef.current = params.init_cash;

    try {
      const data = await runBacktest(params);
      setResult(data);
    } catch (err: unknown) {
      let msg = "Error desconocido";
      if (err && typeof err === "object" && "response" in err) {
        const axiosErr = err as { response?: { data?: { detail?: string } } };
        msg = axiosErr.response?.data?.detail || "Error del servidor";
      } else if (err && typeof err === "object" && "message" in err) {
        const errMsg = (err as { message: string }).message;
        if (errMsg.includes("timeout")) {
          msg = "Timeout: el backtest tardo demasiado. Prueba con un dataset mas pequeno.";
        } else if (errMsg.includes("Network")) {
          msg = "Error de red: verifica que el backend este corriendo en localhost:8000";
        } else {
          msg = errMsg;
        }
      }
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const currentCandles = result?.candles?.[selectedDay];
  const currentEquity = result?.equity_curves?.[selectedDay];
  const currentTrades = result?.trades?.filter(
    (t) =>
      currentCandles &&
      t.ticker === currentCandles.ticker &&
      t.date === currentCandles.date
  );

  return (
    <div className="min-h-screen bg-[var(--background)]">
      <header className="border-b border-[var(--border)] bg-white px-6 py-3">
        <div className="flex items-center gap-3">
          <h1 className="text-lg font-bold text-[var(--foreground)]">BacktesterMVP</h1>
          <span className="text-xs px-2 py-0.5 rounded bg-blue-100 text-blue-700 font-medium">
            VectorBT
          </span>
        </div>
      </header>

      <div className="flex h-[calc(100vh-53px)]">
        <aside className="w-80 min-w-80 border-r border-[var(--border)] p-4 overflow-y-auto space-y-4 bg-[var(--background)]">
          <BacktestPanel onRun={handleRun} loading={loading} />

          {result && (
            <DaySelector
              days={result.candles}
              selectedIdx={selectedDay}
              onSelect={setSelectedDay}
            />
          )}
        </aside>

        <main className="flex-1 overflow-y-auto p-4 space-y-4">
          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <p className="text-sm text-red-700">{error}</p>
            </div>
          )}

          {!result && !loading && !error && (
            <div className="flex items-center justify-center h-full">
              <div className="text-center space-y-2">
                <div className="text-5xl opacity-20">📊</div>
                <p className="text-[var(--muted)] text-sm">
                  Selecciona un dataset y una estrategia para ejecutar el backtest
                </p>
              </div>
            </div>
          )}

          {loading && (
            <div className="flex items-center justify-center h-64">
              <div className="text-center space-y-3">
                <svg className="animate-spin h-8 w-8 text-[var(--accent)] mx-auto" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                </svg>
                <p className="text-sm text-[var(--muted)]">
                  Ejecutando backtest con VectorBT...
                </p>
              </div>
            </div>
          )}

          {result && (
            <>
              <MetricsCard metrics={result.aggregate_metrics} />

              {currentCandles && currentCandles.candles.length > 0 && (
                <Chart
                  candles={currentCandles.candles}
                  trades={currentTrades || []}
                  equity={currentEquity?.equity || []}
                  ticker={currentCandles.ticker}
                  date={currentCandles.date}
                />
              )}

              <ResultsTabs result={result} initCash={initCashRef.current} />
            </>
          )}
        </main>
      </div>
    </div>
  );
}
