"use client";

import type { DayCandles } from "@/lib/api";

interface DaySelectorProps {
  days: DayCandles[];
  selectedIdx: number;
  onSelect: (idx: number) => void;
}

export default function DaySelector({ days, selectedIdx, onSelect }: DaySelectorProps) {
  if (days.length <= 1) return null;

  return (
    <div className="bg-white rounded-lg border border-[var(--border)] p-3">
      <h2 className="text-xs font-semibold uppercase tracking-wide text-[var(--muted)] mb-2">
        Dias ({days.length})
      </h2>
      <div className="flex flex-wrap gap-1.5">
        {days.map((d, i) => (
          <button
            key={`${d.ticker}-${d.date}`}
            onClick={() => onSelect(i)}
            className={`px-2.5 py-1 text-xs rounded-md transition-colors ${
              i === selectedIdx
                ? "bg-[var(--accent)] text-white"
                : "bg-gray-100 text-[var(--foreground)] hover:bg-gray-200"
            }`}
          >
            {d.ticker} {d.date}
          </button>
        ))}
      </div>
    </div>
  );
}
