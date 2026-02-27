"use client";

import { useState, useMemo } from "react";
import type { TradeRecord } from "@/lib/api";

interface TradesTabProps {
  trades: TradeRecord[];
}

type SortKey = keyof TradeRecord;
type SortDir = "asc" | "desc";

const EXIT_COLORS: Record<string, string> = {
  SL: "bg-red-100 text-red-700",
  TP: "bg-emerald-100 text-emerald-700",
  Trailing: "bg-amber-100 text-amber-700",
  Signal: "bg-blue-100 text-blue-700",
  EOD: "bg-gray-100 text-gray-600",
};

export default function TradesTab({ trades }: TradesTabProps) {
  const [search, setSearch] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("date");
  const [sortDir, setSortDir] = useState<SortDir>("desc");

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortDir(sortDir === "asc" ? "desc" : "asc");
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  };

  const filtered = useMemo(() => {
    let result = trades;
    if (search.trim()) {
      const q = search.toLowerCase();
      result = result.filter(
        (t) =>
          t.ticker.toLowerCase().includes(q) ||
          t.date.includes(q) ||
          t.exit_reason.toLowerCase().includes(q)
      );
    }
    return [...result].sort((a, b) => {
      const aVal = a[sortKey];
      const bVal = b[sortKey];
      if (aVal == null && bVal == null) return 0;
      if (aVal == null) return 1;
      if (bVal == null) return -1;
      if (typeof aVal === "number" && typeof bVal === "number") {
        return sortDir === "asc" ? aVal - bVal : bVal - aVal;
      }
      const aStr = String(aVal);
      const bStr = String(bVal);
      return sortDir === "asc"
        ? aStr.localeCompare(bStr)
        : bStr.localeCompare(aStr);
    });
  }, [trades, search, sortKey, sortDir]);

  const summary = useMemo(() => {
    const rValues = trades
      .map((t) => t.r_multiple)
      .filter((r): r is number => r !== null);
    return {
      total: trades.length,
      avgR: rValues.length ? rValues.reduce((a, b) => a + b, 0) / rValues.length : null,
      totalPnl: trades.reduce((a, t) => a + t.pnl, 0),
    };
  }, [trades]);

  const SortHeader = ({
    label,
    field,
    align = "left",
  }: {
    label: string;
    field: SortKey;
    align?: "left" | "right";
  }) => (
    <th
      className={`px-3 py-2 text-${align} text-xs font-medium text-[var(--muted)] uppercase cursor-pointer hover:text-[var(--foreground)] select-none`}
      onClick={() => handleSort(field)}
    >
      {label}
      {sortKey === field && (
        <span className="ml-1">{sortDir === "asc" ? "▲" : "▼"}</span>
      )}
    </th>
  );

  if (!trades.length) {
    return <p className="text-sm text-[var(--muted)]">Sin trades</p>;
  }

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <input
          type="text"
          placeholder="Buscar ticker, fecha, exit reason..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="px-3 py-1.5 text-sm border border-[var(--border)] rounded-md w-64 focus:outline-none focus:ring-1 focus:ring-[var(--accent)]"
        />
        <div className="flex gap-4 text-xs text-[var(--muted)]">
          <span>
            Total: <strong>{summary.total}</strong>
          </span>
          {summary.avgR !== null && (
            <span>
              Avg R:{" "}
              <strong
                className={
                  summary.avgR >= 0
                    ? "text-[var(--success)]"
                    : "text-[var(--danger)]"
                }
              >
                {summary.avgR.toFixed(2)}R
              </strong>
            </span>
          )}
          <span>
            PnL:{" "}
            <strong
              className={
                summary.totalPnl >= 0
                  ? "text-[var(--success)]"
                  : "text-[var(--danger)]"
              }
            >
              {summary.totalPnl >= 0 ? "+" : ""}${summary.totalPnl.toFixed(2)}
            </strong>
          </span>
        </div>
      </div>

      <div className="overflow-x-auto max-h-[500px] overflow-y-auto border border-[var(--border)] rounded-md">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 sticky top-0">
            <tr>
              <SortHeader label="Ticker" field="ticker" />
              <SortHeader label="Fecha" field="date" />
              <SortHeader label="Entrada" field="entry_time" />
              <SortHeader label="Salida" field="exit_time" />
              <SortHeader label="Entry $" field="entry_price" align="right" />
              <SortHeader label="Exit $" field="exit_price" align="right" />
              <SortHeader label="Size" field="size" align="right" />
              <SortHeader label="PnL" field="pnl" align="right" />
              <SortHeader label="R" field="r_multiple" align="right" />
              <SortHeader label="Exit" field="exit_reason" />
            </tr>
          </thead>
          <tbody className="divide-y divide-[var(--border)]">
            {filtered.map((t, i) => (
              <tr key={i} className="hover:bg-gray-50 transition-colors">
                <td className="px-3 py-1.5 font-medium">{t.ticker}</td>
                <td className="px-3 py-1.5 text-[var(--muted)]">{t.date}</td>
                <td className="px-3 py-1.5 text-[var(--muted)] font-mono text-xs">
                  {t.entry_time.split(" ").pop()?.slice(0, 8)}
                </td>
                <td className="px-3 py-1.5 text-[var(--muted)] font-mono text-xs">
                  {t.exit_time.split(" ").pop()?.slice(0, 8)}
                </td>
                <td className="px-3 py-1.5 text-right font-mono">
                  ${t.entry_price.toFixed(2)}
                </td>
                <td className="px-3 py-1.5 text-right font-mono">
                  ${t.exit_price.toFixed(2)}
                </td>
                <td className="px-3 py-1.5 text-right font-mono">
                  {t.size.toFixed(2)}
                </td>
                <td
                  className={`px-3 py-1.5 text-right font-mono font-medium ${
                    t.pnl >= 0
                      ? "text-[var(--success)]"
                      : "text-[var(--danger)]"
                  }`}
                >
                  {t.pnl >= 0 ? "+" : ""}${t.pnl.toFixed(2)}
                </td>
                <td
                  className={`px-3 py-1.5 text-right font-mono ${
                    (t.r_multiple || 0) >= 0
                      ? "text-[var(--success)]"
                      : "text-[var(--danger)]"
                  }`}
                >
                  {t.r_multiple !== null ? `${t.r_multiple.toFixed(2)}R` : "—"}
                </td>
                <td className="px-3 py-1.5">
                  <span
                    className={`inline-block px-1.5 py-0.5 rounded text-xs font-medium ${
                      EXIT_COLORS[t.exit_reason] || "bg-gray-100 text-gray-600"
                    }`}
                  >
                    {t.exit_reason}
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
