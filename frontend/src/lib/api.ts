import axios from "axios";

const api = axios.create({
  baseURL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api",
  timeout: 300000,
});

export interface Dataset {
  id: string;
  name: string;
  pair_count: number;
  created_at: string;
}

export interface Strategy {
  id: string;
  name: string;
  description: string;
  definition: Record<string, unknown>;
}

export interface TradeRecord {
  ticker: string;
  date: string;
  entry_time: string;
  exit_time: string;
  entry_idx: number;
  exit_idx: number;
  entry_price: number;
  exit_price: number;
  pnl: number;
  return_pct: number;
  direction: string;
  status: string;
  size: number;
  exit_reason: string;
  r_multiple: number | null;
  entry_hour: number;
  entry_weekday: number;
}

export interface CandleData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface EquityPoint {
  time: number;
  value: number;
}

export interface DayCandles {
  ticker: string;
  date: string;
  candles: CandleData[];
}

export interface DayEquity {
  ticker: string;
  date: string;
  equity: EquityPoint[];
}

export interface DayResult {
  ticker: string;
  date: string;
  total_return_pct: number | null;
  max_drawdown_pct: number | null;
  win_rate_pct: number | null;
  total_trades: number;
  profit_factor: number | null;
  sharpe_ratio: number | null;
  sortino_ratio: number | null;
  expectancy: number | null;
  best_trade_pct: number | null;
  worst_trade_pct: number | null;
  init_value: number | null;
  end_value: number | null;
}

export interface AggregateMetrics {
  total_days: number;
  total_trades: number;
  win_rate_pct: number;
  avg_return_per_day_pct: number;
  total_return_pct: number;
  avg_sharpe: number;
  avg_max_dd_pct: number;
  avg_profit_factor: number;
  avg_pnl: number;
  total_pnl: number;
}

export interface GlobalEquityPoint {
  time: number;
  value: number;
}

export interface DrawdownPoint {
  time: number;
  value: number;
}

export interface BacktestResult {
  aggregate_metrics: AggregateMetrics;
  day_results: DayResult[];
  candles: DayCandles[];
  trades: TradeRecord[];
  equity_curves: DayEquity[];
  global_equity: GlobalEquityPoint[];
  global_drawdown: DrawdownPoint[];
}

export interface MonteCarloPercentileCurve {
  time: number;
  value: number;
}

export interface MonteCarloResult {
  percentiles: Record<string, MonteCarloPercentileCurve[]>;
  ruin_probability: number;
  worst_drawdown: number;
  median_drawdown: number;
  final_balance_percentiles: Record<string, number>;
}

export async function runMonteCarlo(params: {
  pnls: number[];
  init_cash: number;
  simulations?: number;
}): Promise<MonteCarloResult> {
  const { data } = await api.post("/montecarlo", params);
  return data;
}

export async function fetchDatasets(): Promise<Dataset[]> {
  const { data } = await api.get("/datasets");
  return data;
}

export async function fetchStrategies(): Promise<Strategy[]> {
  const { data } = await api.get("/strategies");
  return data;
}

export async function runBacktest(params: {
  dataset_id: string;
  strategy_id: string;
  init_cash?: number;
  risk_r?: number;
  fees?: number;
  slippage?: number;
}): Promise<BacktestResult> {
  const { data } = await api.post("/backtest", params);
  return data;
}
