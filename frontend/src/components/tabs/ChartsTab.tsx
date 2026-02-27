"use client";

import { useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
  ReferenceLine,
} from "recharts";
import type { TradeRecord } from "@/lib/api";

interface ChartsTabProps {
  trades: TradeRecord[];
}

const WEEKDAY_NAMES = ["Lun", "Mar", "Mie", "Jue", "Vie"];

export default function ChartsTab({ trades }: ChartsTabProps) {
  const rHistogram = useMemo(() => {
    const rValues = trades
      .map((t) => t.r_multiple)
      .filter((r): r is number => r !== null);
    if (!rValues.length) return [];

    const bucketSize = 0.5;
    const min = Math.floor(Math.min(...rValues) / bucketSize) * bucketSize;
    const max = Math.ceil(Math.max(...rValues) / bucketSize) * bucketSize;
    const buckets = new Map<number, number>();

    for (let b = min; b <= max; b += bucketSize) {
      buckets.set(parseFloat(b.toFixed(1)), 0);
    }

    for (const r of rValues) {
      const bucket = parseFloat(
        (Math.floor(r / bucketSize) * bucketSize).toFixed(1)
      );
      buckets.set(bucket, (buckets.get(bucket) || 0) + 1);
    }

    return Array.from(buckets.entries())
      .sort((a, b) => a[0] - b[0])
      .map(([r, count]) => ({ r: `${r}R`, value: count, rNum: r }));
  }, [trades]);

  const evByHour = useMemo(() => {
    const hourMap = new Map<number, { total: number; count: number }>();
    for (const t of trades) {
      const h = t.entry_hour;
      if (!hourMap.has(h)) hourMap.set(h, { total: 0, count: 0 });
      const m = hourMap.get(h)!;
      m.total += t.pnl;
      m.count++;
    }
    return Array.from(hourMap.entries())
      .sort((a, b) => a[0] - b[0])
      .map(([hour, data]) => ({
        hour: `${hour}:00`,
        ev: data.count > 0 ? data.total / data.count : 0,
        count: data.count,
      }));
  }, [trades]);

  const evByDay = useMemo(() => {
    const dayMap = new Map<number, { total: number; count: number }>();
    for (const t of trades) {
      const d = t.entry_weekday;
      if (d > 4) continue;
      if (!dayMap.has(d)) dayMap.set(d, { total: 0, count: 0 });
      const m = dayMap.get(d)!;
      m.total += t.pnl;
      m.count++;
    }
    return [0, 1, 2, 3, 4].map((d) => {
      const data = dayMap.get(d) || { total: 0, count: 0 };
      return {
        day: WEEKDAY_NAMES[d],
        ev: data.count > 0 ? data.total / data.count : 0,
        count: data.count,
      };
    });
  }, [trades]);

  if (!trades.length) {
    return <p className="text-sm text-[var(--muted)]">Sin trades para analizar</p>;
  }

  return (
    <div className="space-y-8">
      {rHistogram.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-[var(--muted)] uppercase tracking-wide mb-3">
            Distribucion R-Multiple
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={rHistogram} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis dataKey="r" tick={{ fontSize: 11 }} />
              <YAxis tick={{ fontSize: 11 }} allowDecimals={false} />
              <Tooltip
                formatter={(value) => [`${value} trades`, "Frecuencia"]}
                contentStyle={{ fontSize: 12 }}
              />
              <ReferenceLine x="0R" stroke="#94a3b8" strokeDasharray="3 3" />
              <Bar dataKey="value" radius={[2, 2, 0, 0]}>
                {rHistogram.map((entry, idx) => (
                  <Cell
                    key={idx}
                    fill={entry.rNum >= 0 ? "#10b981" : "#ef4444"}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      <div>
        <h3 className="text-sm font-semibold text-[var(--muted)] uppercase tracking-wide mb-3">
          EV por Hora de Entrada
        </h3>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={evByHour} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="hour" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} tickFormatter={(v: number) => `$${v.toFixed(0)}`} />
            <Tooltip
              formatter={(value, _name, props) => {
                const v = Number(value);
                const count = (props?.payload as { count?: number })?.count ?? 0;
                return [`$${v.toFixed(2)} (${count} trades)`, "Avg PnL"];
              }}
              contentStyle={{ fontSize: 12 }}
            />
            <ReferenceLine y={0} stroke="#94a3b8" />
            <Bar dataKey="ev" radius={[2, 2, 0, 0]}>
              {evByHour.map((entry, idx) => (
                <Cell
                  key={idx}
                  fill={entry.ev >= 0 ? "#10b981" : "#ef4444"}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div>
        <h3 className="text-sm font-semibold text-[var(--muted)] uppercase tracking-wide mb-3">
          EV por Dia de la Semana
        </h3>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={evByDay} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
            <XAxis dataKey="day" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} tickFormatter={(v: number) => `$${v.toFixed(0)}`} />
            <Tooltip
              formatter={(value, _name, props) => {
                const v = Number(value);
                const count = (props?.payload as { count?: number })?.count ?? 0;
                return [`$${v.toFixed(2)} (${count} trades)`, "Avg PnL"];
              }}
              contentStyle={{ fontSize: 12 }}
            />
            <ReferenceLine y={0} stroke="#94a3b8" />
            <Bar dataKey="ev" radius={[2, 2, 0, 0]}>
              {evByDay.map((entry, idx) => (
                <Cell
                  key={idx}
                  fill={entry.ev >= 0 ? "#10b981" : "#ef4444"}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
