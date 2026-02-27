"use client";

import type { AggregateMetrics } from "@/lib/api";

interface MetricsCardProps {
  metrics: AggregateMetrics;
}

function Stat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="flex flex-col">
      <span className="text-xs text-[var(--muted)] uppercase tracking-wide">{label}</span>
      <span className={`text-lg font-semibold ${color || ""}`}>{value}</span>
    </div>
  );
}

export default function MetricsCard({ metrics }: MetricsCardProps) {
  const pnlColor = (metrics.total_pnl ?? 0) >= 0 ? "text-[var(--success)]" : "text-[var(--danger)]";
  const retColor = (metrics.total_return_pct ?? 0) >= 0 ? "text-[var(--success)]" : "text-[var(--danger)]";

  return (
    <div className="bg-white rounded-lg border border-[var(--border)] p-4">
      <h2 className="text-sm font-semibold uppercase tracking-wide text-[var(--muted)] mb-3">
        Resultados Agregados
      </h2>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-4">
        <Stat label="Dias" value={String(metrics.total_days ?? 0)} />
        <Stat label="Trades" value={String(metrics.total_trades ?? 0)} />
        <Stat
          label="Win Rate"
          value={`${(metrics.win_rate_pct ?? 0).toFixed(1)}%`}
          color={(metrics.win_rate_pct ?? 0) >= 50 ? "text-[var(--success)]" : "text-[var(--danger)]"}
        />
        <Stat label="Total Return" value={`${(metrics.total_return_pct ?? 0).toFixed(2)}%`} color={retColor} />
        <Stat label="PnL Total" value={`$${(metrics.total_pnl ?? 0).toFixed(2)}`} color={pnlColor} />
        <Stat label="Avg Return/Dia" value={`${(metrics.avg_return_per_day_pct ?? 0).toFixed(3)}%`} />
        <Stat label="Avg PnL/Trade" value={`$${(metrics.avg_pnl ?? 0).toFixed(2)}`} />
        <Stat label="Avg Sharpe" value={(metrics.avg_sharpe ?? 0).toFixed(3)} />
        <Stat label="Avg Max DD" value={`${(metrics.avg_max_dd_pct ?? 0).toFixed(2)}%`} color="text-[var(--danger)]" />
        <Stat label="Avg Profit Factor" value={(metrics.avg_profit_factor ?? 0).toFixed(3)} />
      </div>
    </div>
  );
}
