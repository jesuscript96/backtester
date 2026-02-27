"use client";

import type { TradeRecord } from "@/lib/api";

interface TradeTableProps {
  trades: TradeRecord[];
}

export default function TradeTable({ trades }: TradeTableProps) {
  if (trades.length === 0) {
    return (
      <div className="bg-white rounded-lg border border-[var(--border)] p-4">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-[var(--muted)] mb-2">
          Trades
        </h2>
        <p className="text-sm text-[var(--muted)]">Sin trades</p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg border border-[var(--border)] overflow-hidden">
      <div className="px-4 py-3 border-b border-[var(--border)]">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-[var(--muted)]">
          Trades ({trades.length})
        </h2>
      </div>
      <div className="overflow-x-auto max-h-80 overflow-y-auto">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 sticky top-0">
            <tr>
              <th className="px-3 py-2 text-left text-xs font-medium text-[var(--muted)] uppercase">Ticker</th>
              <th className="px-3 py-2 text-left text-xs font-medium text-[var(--muted)] uppercase">Fecha</th>
              <th className="px-3 py-2 text-left text-xs font-medium text-[var(--muted)] uppercase">Dir</th>
              <th className="px-3 py-2 text-right text-xs font-medium text-[var(--muted)] uppercase">Entry</th>
              <th className="px-3 py-2 text-right text-xs font-medium text-[var(--muted)] uppercase">Exit</th>
              <th className="px-3 py-2 text-right text-xs font-medium text-[var(--muted)] uppercase">Size</th>
              <th className="px-3 py-2 text-right text-xs font-medium text-[var(--muted)] uppercase">PnL</th>
              <th className="px-3 py-2 text-right text-xs font-medium text-[var(--muted)] uppercase">Return</th>
              <th className="px-3 py-2 text-left text-xs font-medium text-[var(--muted)] uppercase">Status</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--border)]">
            {trades.map((t, i) => (
              <tr key={i} className="hover:bg-gray-50 transition-colors">
                <td className="px-3 py-2 font-medium">{t.ticker}</td>
                <td className="px-3 py-2 text-[var(--muted)]">{t.date}</td>
                <td className="px-3 py-2">
                  <span
                    className={`inline-block px-1.5 py-0.5 rounded text-xs font-medium ${
                      t.direction.toLowerCase().includes("long")
                        ? "bg-emerald-100 text-emerald-700"
                        : "bg-red-100 text-red-700"
                    }`}
                  >
                    {t.direction}
                  </span>
                </td>
                <td className="px-3 py-2 text-right font-mono">${t.entry_price.toFixed(2)}</td>
                <td className="px-3 py-2 text-right font-mono">${t.exit_price.toFixed(2)}</td>
                <td className="px-3 py-2 text-right font-mono">{t.size.toFixed(2)}</td>
                <td className={`px-3 py-2 text-right font-mono font-medium ${t.pnl >= 0 ? "text-[var(--success)]" : "text-[var(--danger)]"}`}>
                  {t.pnl >= 0 ? "+" : ""}${t.pnl.toFixed(2)}
                </td>
                <td className={`px-3 py-2 text-right font-mono ${t.return_pct >= 0 ? "text-[var(--success)]" : "text-[var(--danger)]"}`}>
                  {t.return_pct >= 0 ? "+" : ""}{t.return_pct.toFixed(2)}%
                </td>
                <td className="px-3 py-2">
                  <span className={`text-xs ${t.status === "Closed" ? "text-[var(--muted)]" : "text-amber-600"}`}>
                    {t.status}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
