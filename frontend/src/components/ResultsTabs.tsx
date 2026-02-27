"use client";

import { useState } from "react";
import type { BacktestResult } from "@/lib/api";
import EquityCurveTab from "@/components/tabs/EquityCurveTab";
import DrawdownTab from "@/components/tabs/DrawdownTab";
import PerformanceTab from "@/components/tabs/PerformanceTab";
import CalendarTab from "@/components/tabs/CalendarTab";
import TradesTab from "@/components/tabs/TradesTab";
import ChartsTab from "@/components/tabs/ChartsTab";
import PortfolioTab from "@/components/tabs/PortfolioTab";

const TABS = [
  { id: "equity", label: "Equity Curve" },
  { id: "drawdown", label: "Drawdown" },
  { id: "performance", label: "Performance" },
  { id: "calendar", label: "Calendar" },
  { id: "trades", label: "Trades" },
  { id: "charts", label: "Charts" },
  { id: "portfolio", label: "Portfolio" },
] as const;

type TabId = (typeof TABS)[number]["id"];

interface ResultsTabsProps {
  result: BacktestResult;
  initCash: number;
}

export default function ResultsTabs({ result, initCash }: ResultsTabsProps) {
  const [activeTab, setActiveTab] = useState<TabId>("equity");

  return (
    <div className="bg-white rounded-lg border border-[var(--border)] overflow-hidden">
      <div className="border-b border-[var(--border)] overflow-x-auto">
        <nav className="flex min-w-max">
          {TABS.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`px-4 py-2.5 text-sm font-medium whitespace-nowrap transition-colors
                ${
                  activeTab === tab.id
                    ? "text-[var(--accent)] border-b-2 border-[var(--accent)]"
                    : "text-[var(--muted)] hover:text-[var(--foreground)]"
                }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      <div className="p-4">
        {activeTab === "equity" && (
          <EquityCurveTab
            globalEquity={result.global_equity}
            trades={result.trades}
          />
        )}
        {activeTab === "drawdown" && (
          <DrawdownTab globalDrawdown={result.global_drawdown} />
        )}
        {activeTab === "performance" && (
          <PerformanceTab dayResults={result.day_results} trades={result.trades} />
        )}
        {activeTab === "calendar" && (
          <CalendarTab dayResults={result.day_results} trades={result.trades} />
        )}
        {activeTab === "trades" && <TradesTab trades={result.trades} />}
        {activeTab === "charts" && <ChartsTab trades={result.trades} />}
        {activeTab === "portfolio" && (
          <PortfolioTab trades={result.trades} initCash={initCash} />
        )}
      </div>
    </div>
  );
}
