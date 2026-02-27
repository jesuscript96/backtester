"use client";

import { useMemo } from "react";
import type { DayResult, TradeRecord } from "@/lib/api";

interface PerformanceTabProps {
  dayResults: DayResult[];
  trades: TradeRecord[];
}

interface MonthRow {
  key: string;
  label: string;
  returnPct: number;
  pnl: number;
  trades: number;
  winRate: number;
}

function cellColor(value: number): string {
  if (value > 0) {
    const intensity = Math.min(value / 5, 1);
    return `rgba(16,185,129,${0.1 + intensity * 0.3})`;
  }
  if (value < 0) {
    const intensity = Math.min(Math.abs(value) / 5, 1);
    return `rgba(239,68,68,${0.1 + intensity * 0.3})`;
  }
  return "transparent";
}

export default function PerformanceTab({ dayResults, trades }: PerformanceTabProps) {
  const rows = useMemo(() => {
    const monthMap = new Map<string, { returns: number[]; tradeCount: number; wins: number; totalTrades: number; pnls: number[] }>();

    for (const dr of dayResults) {
      const month = dr.date.slice(0, 7);
      if (!monthMap.has(month)) {
        monthMap.set(month, { returns: [], tradeCount: 0, wins: 0, totalTrades: 0, pnls: [] });
      }
      const m = monthMap.get(month)!;
      m.returns.push(dr.total_return_pct || 0);
      m.tradeCount += dr.total_trades;
    }

    for (const t of trades) {
      const month = t.date.slice(0, 7);
      if (!monthMap.has(month)) continue;
      const m = monthMap.get(month)!;
      m.totalTrades++;
      if (t.pnl > 0) m.wins++;
      m.pnls.push(t.pnl);
    }

    const result: MonthRow[] = [];
    const yearMap = new Map<string, MonthRow[]>();

    const sortedMonths = Array.from(monthMap.keys()).sort();
    for (const key of sortedMonths) {
      const m = monthMap.get(key)!;
      const compoundReturn =
        m.returns.reduce((acc, r) => acc * (1 + r / 100), 1) * 100 - 100;
      const row: MonthRow = {
        key,
        label: key,
        returnPct: compoundReturn,
        pnl: m.pnls.reduce((a, b) => a + b, 0),
        trades: m.tradeCount,
        winRate: m.totalTrades > 0 ? (m.wins / m.totalTrades) * 100 : 0,
      };
      result.push(row);

      const year = key.slice(0, 4);
      if (!yearMap.has(year)) yearMap.set(year, []);
      yearMap.get(year)!.push(row);
    }

    const finalRows: (MonthRow & { isYear?: boolean })[] = [];
    let currentYear = "";
    for (const row of result) {
      const year = row.key.slice(0, 4);
      if (year !== currentYear && currentYear !== "") {
        const yearRows = yearMap.get(currentYear)!;
        const yearReturn =
          yearRows.reduce((acc, r) => acc * (1 + r.returnPct / 100), 1) * 100 - 100;
        const yearPnl = yearRows.reduce((a, r) => a + r.pnl, 0);
        const yearTrades = yearRows.reduce((a, r) => a + r.trades, 0);
        const yearWins = yearRows.reduce(
          (a, r) => a + (r.winRate * r.trades) / 100,
          0
        );
        finalRows.push({
          key: `year-${currentYear}`,
          label: `${currentYear} Total`,
          returnPct: yearReturn,
          pnl: yearPnl,
          trades: yearTrades,
          winRate: yearTrades > 0 ? (yearWins / yearTrades) * 100 : 0,
          isYear: true,
        });
      }
      currentYear = year;
      finalRows.push(row);
    }

    if (currentYear && yearMap.has(currentYear)) {
      const yearRows = yearMap.get(currentYear)!;
      const yearReturn =
        yearRows.reduce((acc, r) => acc * (1 + r.returnPct / 100), 1) * 100 - 100;
      const yearPnl = yearRows.reduce((a, r) => a + r.pnl, 0);
      const yearTrades = yearRows.reduce((a, r) => a + r.trades, 0);
      const yearWins = yearRows.reduce(
        (a, r) => a + (r.winRate * r.trades) / 100,
        0
      );
      finalRows.push({
        key: `year-${currentYear}`,
        label: `${currentYear} Total`,
        returnPct: yearReturn,
        pnl: yearPnl,
        trades: yearTrades,
        winRate: yearTrades > 0 ? (yearWins / yearTrades) * 100 : 0,
        isYear: true,
      });
    }

    return finalRows;
  }, [dayResults, trades]);

  if (!dayResults.length) {
    return <p className="text-sm text-[var(--muted)]">Sin resultados</p>;
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-[var(--border)]">
            <th className="px-3 py-2 text-left text-xs font-medium text-[var(--muted)] uppercase">
              Periodo
            </th>
            <th className="px-3 py-2 text-right text-xs font-medium text-[var(--muted)] uppercase">
              Return %
            </th>
            <th className="px-3 py-2 text-right text-xs font-medium text-[var(--muted)] uppercase">
              PnL
            </th>
            <th className="px-3 py-2 text-right text-xs font-medium text-[var(--muted)] uppercase">
              Trades
            </th>
            <th className="px-3 py-2 text-right text-xs font-medium text-[var(--muted)] uppercase">
              Win Rate
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-[var(--border)]">
          {rows.map((row) => (
            <tr
              key={row.key}
              className={
                (row as MonthRow & { isYear?: boolean }).isYear
                  ? "font-semibold bg-gray-50"
                  : "hover:bg-gray-50 transition-colors"
              }
            >
              <td className="px-3 py-2">{row.label}</td>
              <td
                className="px-3 py-2 text-right font-mono"
                style={{ backgroundColor: cellColor(row.returnPct) }}
              >
                <span
                  className={
                    row.returnPct >= 0
                      ? "text-[var(--success)]"
                      : "text-[var(--danger)]"
                  }
                >
                  {row.returnPct >= 0 ? "+" : ""}
                  {row.returnPct.toFixed(2)}%
                </span>
              </td>
              <td className="px-3 py-2 text-right font-mono">
                <span
                  className={
                    row.pnl >= 0
                      ? "text-[var(--success)]"
                      : "text-[var(--danger)]"
                  }
                >
                  {row.pnl >= 0 ? "+" : ""}${row.pnl.toFixed(2)}
                </span>
              </td>
              <td className="px-3 py-2 text-right">{row.trades}</td>
              <td className="px-3 py-2 text-right">
                {row.winRate.toFixed(1)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
