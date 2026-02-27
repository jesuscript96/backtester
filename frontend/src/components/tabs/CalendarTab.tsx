"use client";

import { useMemo, useState } from "react";
import type { DayResult, TradeRecord } from "@/lib/api";

interface CalendarTabProps {
  dayResults: DayResult[];
  trades: TradeRecord[];
}

const WEEKDAY_LABELS = ["Lun", "Mar", "Mie", "Jue", "Vie", "Sab", "Dom"];

function pnlColor(pnl: number, maxAbs: number): string {
  if (maxAbs === 0) return "transparent";
  const intensity = Math.min(Math.abs(pnl) / maxAbs, 1);
  if (pnl > 0) return `rgba(16,185,129,${0.15 + intensity * 0.5})`;
  if (pnl < 0) return `rgba(239,68,68,${0.15 + intensity * 0.5})`;
  return "rgba(148,163,184,0.1)";
}

export default function CalendarTab({ dayResults, trades }: CalendarTabProps) {
  const pnlByDate = useMemo(() => {
    const map = new Map<string, number>();
    for (const t of trades) {
      map.set(t.date, (map.get(t.date) || 0) + t.pnl);
    }
    return map;
  }, [trades]);

  const months = useMemo(() => {
    const set = new Set<string>();
    for (const dr of dayResults) {
      set.add(dr.date.slice(0, 7));
    }
    return Array.from(set).sort();
  }, [dayResults]);

  const [selectedMonth, setSelectedMonth] = useState(months[0] || "");

  const calendarDays = useMemo(() => {
    if (!selectedMonth) return [];

    const [year, month] = selectedMonth.split("-").map(Number);
    const firstDay = new Date(year, month - 1, 1);
    const lastDay = new Date(year, month, 0);
    const startWeekday = (firstDay.getDay() + 6) % 7;

    const days: (null | { date: string; pnl: number | null })[] = [];

    for (let i = 0; i < startWeekday; i++) {
      days.push(null);
    }

    for (let d = 1; d <= lastDay.getDate(); d++) {
      const dateStr = `${year}-${String(month).padStart(2, "0")}-${String(d).padStart(2, "0")}`;
      const pnl = pnlByDate.has(dateStr) ? pnlByDate.get(dateStr)! : null;
      days.push({ date: dateStr, pnl });
    }

    return days;
  }, [selectedMonth, pnlByDate]);

  const maxAbsPnl = useMemo(() => {
    const values = Array.from(pnlByDate.values());
    return values.length ? Math.max(...values.map(Math.abs), 1) : 1;
  }, [pnlByDate]);

  if (!dayResults.length) {
    return <p className="text-sm text-[var(--muted)]">Sin resultados</p>;
  }

  return (
    <div>
      {months.length > 1 && (
        <div className="flex gap-2 mb-4 flex-wrap">
          {months.map((m) => (
            <button
              key={m}
              onClick={() => setSelectedMonth(m)}
              className={`px-3 py-1 text-xs rounded-md border transition-colors ${
                selectedMonth === m
                  ? "border-[var(--accent)] bg-blue-50 text-[var(--accent)]"
                  : "border-[var(--border)] text-[var(--muted)] hover:border-gray-400"
              }`}
            >
              {m}
            </button>
          ))}
        </div>
      )}

      <div className="grid grid-cols-7 gap-1">
        {WEEKDAY_LABELS.map((label) => (
          <div
            key={label}
            className="text-center text-xs font-medium text-[var(--muted)] py-1"
          >
            {label}
          </div>
        ))}

        {calendarDays.map((day, i) => {
          if (!day) {
            return <div key={`empty-${i}`} className="aspect-square" />;
          }
          const dayNum = parseInt(day.date.split("-")[2]);
          return (
            <div
              key={day.date}
              className="aspect-square rounded-md flex flex-col items-center justify-center text-xs border border-[var(--border)]"
              style={{
                backgroundColor:
                  day.pnl !== null ? pnlColor(day.pnl, maxAbsPnl) : undefined,
              }}
            >
              <span className="text-[10px] text-[var(--muted)]">{dayNum}</span>
              {day.pnl !== null && (
                <span
                  className={`font-mono text-[10px] font-medium ${
                    day.pnl >= 0
                      ? "text-[var(--success)]"
                      : "text-[var(--danger)]"
                  }`}
                >
                  {day.pnl >= 0 ? "+" : ""}
                  {day.pnl.toFixed(0)}
                </span>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
