"use client";

import { useEffect, useState } from "react";
import type { Dataset, Strategy } from "@/lib/api";
import { fetchDatasets, fetchStrategies } from "@/lib/api";

interface BacktestPanelProps {
  onRun: (params: {
    dataset_id: string;
    strategy_id: string;
    init_cash: number;
    risk_r: number;
    fees: number;
    slippage: number;
  }) => void;
  loading: boolean;
}

export default function BacktestPanel({ onRun, loading }: BacktestPanelProps) {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [strategies, setStrategies] = useState<Strategy[]>([]);
  const [selectedDataset, setSelectedDataset] = useState("");
  const [selectedStrategy, setSelectedStrategy] = useState("");
  const [initCash, setInitCash] = useState(10000);
  const [riskR, setRiskR] = useState(100);
  const [fees, setFees] = useState(0);
  const [slippage, setSlippage] = useState(0);
  const [loadingData, setLoadingData] = useState(true);
  const [loadError, setLoadError] = useState(false);

  const loadData = async () => {
    setLoadingData(true);
    setLoadError(false);
    let failed = false;
    try {
      console.log("[INIT] Fetching datasets...");
      const d = await fetchDatasets();
      console.log("[INIT] Datasets OK:", d.length);
      setDatasets(d);
      if (d.length > 0) setSelectedDataset(d[0].id);
    } catch (e) {
      console.error("[INIT] Datasets FAILED:", e);
      failed = true;
    }
    try {
      console.log("[INIT] Fetching strategies...");
      const s = await fetchStrategies();
      console.log("[INIT] Strategies OK:", s.length);
      setStrategies(s);
      if (s.length > 0) setSelectedStrategy(s[0].id);
    } catch (e) {
      console.error("[INIT] Strategies FAILED:", e);
      failed = true;
    }
    setLoadError(failed);
    setLoadingData(false);
  };

  useEffect(() => {
    loadData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleRun = () => {
    if (!selectedDataset || !selectedStrategy) return;
    onRun({
      dataset_id: selectedDataset,
      strategy_id: selectedStrategy,
      init_cash: initCash,
      risk_r: riskR,
      fees: fees / 100,
      slippage: slippage / 100,
    });
  };

  const selectedStrat = strategies.find((s) => s.id === selectedStrategy);
  const selectedDs = datasets.find((d) => d.id === selectedDataset);

  return (
    <div className="bg-white rounded-lg border border-[var(--border)] p-4 space-y-4">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-[var(--muted)]">
        Configuracion
      </h2>

      {loadError && (
        <div className="flex items-center gap-2 bg-red-50 border border-red-200 rounded-md px-3 py-2">
          <span className="text-xs text-red-600 flex-1">Error al conectar con el servidor</span>
          <button
            onClick={loadData}
            className="text-xs font-medium text-red-700 underline hover:no-underline"
          >
            Reintentar
          </button>
        </div>
      )}

      <div className="space-y-3">
        <div>
          <label className="block text-xs font-medium mb-1 text-[var(--muted)]">
            Dataset
          </label>
          {loadingData ? (
            <div className="h-9 bg-gray-100 rounded animate-pulse" />
          ) : (
            <select
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value)}
              className="w-full border border-[var(--border)] rounded-md px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-[var(--accent)]"
            >
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name} ({d.pair_count} pares)
                </option>
              ))}
            </select>
          )}
        </div>

        <div>
          <label className="block text-xs font-medium mb-1 text-[var(--muted)]">
            Estrategia
          </label>
          {loadingData ? (
            <div className="h-9 bg-gray-100 rounded animate-pulse" />
          ) : (
            <select
              value={selectedStrategy}
              onChange={(e) => setSelectedStrategy(e.target.value)}
              className="w-full border border-[var(--border)] rounded-md px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-[var(--accent)]"
            >
              {strategies.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name}
                </option>
              ))}
            </select>
          )}
          {selectedStrat?.description && (
            <p className="text-xs text-[var(--muted)] mt-1">{selectedStrat.description}</p>
          )}
        </div>

        <div className="grid grid-cols-2 gap-2 mb-2">
          <div>
            <label className="block text-xs font-medium mb-1 text-[var(--muted)]">
              Capital ($)
            </label>
            <input
              type="number"
              value={initCash}
              onChange={(e) => setInitCash(Number(e.target.value))}
              className="w-full border border-[var(--border)] rounded-md px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-[var(--accent)]"
            />
          </div>
          <div>
            <label className="block text-xs font-medium mb-1 text-[var(--muted)]">
              Riesgo 1R ($)
            </label>
            <input
              type="number"
              value={riskR}
              onChange={(e) => setRiskR(Number(e.target.value))}
              className="w-full border border-[var(--border)] rounded-md px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-[var(--accent)]"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <div>
            <label className="block text-xs font-medium mb-1 text-[var(--muted)]">
              Fees (%)
            </label>
            <input
              type="number"
              step="0.01"
              value={fees}
              onChange={(e) => setFees(Number(e.target.value))}
              className="w-full border border-[var(--border)] rounded-md px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-[var(--accent)]"
            />
          </div>
          <div>
            <label className="block text-xs font-medium mb-1 text-[var(--muted)]">
              Slippage (%)
            </label>
            <input
              type="number"
              step="0.01"
              value={slippage}
              onChange={(e) => setSlippage(Number(e.target.value))}
              className="w-full border border-[var(--border)] rounded-md px-3 py-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-[var(--accent)]"
            />
          </div>
        </div>
      </div>

      <button
        onClick={handleRun}
        disabled={loading || !selectedDataset || !selectedStrategy}
        className="w-full bg-[var(--accent)] hover:bg-[var(--accent-hover)] disabled:opacity-50 disabled:cursor-not-allowed text-white font-medium py-2.5 rounded-md text-sm transition-colors"
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2">
            <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
            </svg>
            Ejecutando backtest...
          </span>
        ) : (
          "Ejecutar Backtest"
        )}
      </button>
    </div>
  );
}
