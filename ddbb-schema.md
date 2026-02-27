MOTHERDUCK_TOKEN=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJlbWFpbCI6Imp2YWxlbnp1ZWxhLmNodWxpYUBnbWFpbC5jb20iLCJtZFJlZ2lvbiI6ImF3cy11cy1lYXN0LTEiLCJzZXNzaW9uIjoianZhbGVuenVlbGEuY2h1bGlhLmdtYWlsLmNvbSIsInBhdCI6IlVYQXRaLTI2M2JwWEhHSkxSTVFrUVpENHg5RlozY091dTNneTVKOE00RkkiLCJ1c2VySWQiOiJiYzZmNmEzOC00NmU1LTQzNTgtODIyMS0zY2VhZjJhYzM5NzkiLCJpc3MiOiJtZF9wYXQiLCJyZWFkT25seSI6ZmFsc2UsInRva2VuVHlwZSI6InJlYWRfd3JpdGUiLCJpYXQiOjE3NzAzNDAwNDB9.w1g65spA7RDYyYpRKPhMmnJkz87MLb3uWQSsQvLpQfc

La base de datos se llama Massive 

Básicamente vas a querer datos por minuto (no tienen resample) de ciertos dias de ciertos tickers. 

CREATE TABLE intraday_1m(
  ticker VARCHAR,
  volume BIGINT,
  open DOUBLE,
  "close" DOUBLE,
  high DOUBLE,
  low DOUBLE,
  "timestamp" TIMESTAMP,
  transactions BIGINT,
  date DATE
);

Los datos para el backtester se seleccionan mediante datasets: combinaciones explícitas de (ticker, date).

CREATE TABLE datasets(
  id VARCHAR PRIMARY KEY,
  name VARCHAR NOT NULL,
  created_at TIMESTAMP DEFAULT now()
);

CREATE TABLE dataset_pairs(
  dataset_id VARCHAR NOT NULL,
  ticker VARCHAR NOT NULL,
  date DATE NOT NULL,
  PRIMARY KEY (dataset_id, ticker, date)
);

El backtester recibe un dataset_id, hace JOIN con daily_metrics para obtener los daily_stats
(pm_high, pm_low, prev_close, yesterday_high via LAG(), etc.) y con intraday_1m para las velas de 1 minuto.

Las daily_metrics ya están precomputadas en:

CREATE TABLE daily_metrics(
  ticker VARCHAR,
  volume BIGINT,
  open DOUBLE,
  "close" DOUBLE,
  high DOUBLE,
  low DOUBLE,
  "timestamp" TIMESTAMP,
  transactions BIGINT,
  pm_volume BIGINT,
  pm_high DOUBLE,
  pm_low DOUBLE,
  pm_high_time VARCHAR,
  pm_low_time VARCHAR,
  gap_pct DOUBLE,
  pmh_gap_pct DOUBLE,
  pmh_fade_pct DOUBLE,
  rth_volume BIGINT,
  rth_open DOUBLE,
  rth_high DOUBLE,
  rth_low DOUBLE,
  rth_close DOUBLE,
  hod_time VARCHAR,
  lod_time VARCHAR,
  rth_run_pct DOUBLE,
  rth_fade_pct DOUBLE,
  rth_range_pct DOUBLE,
  m15_return_pct DOUBLE,
  m30_return_pct DOUBLE,
  m60_return_pct DOUBLE,
  m180_return_pct DOUBLE,
  close_1559 DOUBLE,
  last_close DOUBLE,
  day_return_pct DOUBLE,
  prev_close DOUBLE,
  eod_volume BIGINT
);

Las estrategias vienen de una tabla massive.strategies

CREATE TABLE strategies(
  id VARCHAR,
  "name" VARCHAR,
  description VARCHAR,
  definition JSON,
  created_at TIMESTAMP,
  updated_at TIMESTAMP
);

y tiene muchos criterios

1. Catálogo de Estrategia (
Strategy
)
El JSON de una estrategia define su comportamiento operativo, criterios de entrada/salida y gestión de riesgo.

Estructura Principal
json
{
  "name": "Nombre de la Estrategia",
  "description": "Descripción opcional",
  "bias": "long", // "long" o "short"
  "entry_logic": {
    "timeframe": "1m", // "1m", "5m", "15m", "30m", "1h", "1d"
    "root_condition": { /* ConditionGroup */ }
  },
  "exit_logic": {
    "timeframe": "1m",
    "root_condition": { /* ConditionGroup */ }
  },
  "risk_management": {
    "use_hard_stop": true,
    "use_take_profit": true,
    "accept_reentries": true,
    "hard_stop": { "type": "Percentage", "value": 2.0 },
    "take_profit": { "type": "Percentage", "value": 6.0 },
    "trailing_stop": { "active": false, "type": "Percentage", "buffer_pct": 0.5 },
    "max_drawdown_daily": null
  },
  "universe_filters": {
    "min_market_cap": null,
    "max_market_cap": null,
    "min_price": null,
    "max_price": null,
    "min_volume": null,
    "max_shares_float": null,
    "require_shortable": true,
    "exclude_dilution": true,
    "whitelist_sectors": []
  }
}
Componentes Lógicos (
ConditionGroup
 y Conditions)
Las condiciones pueden anidarse usando grupos con operadores AND / OR.

A. Comparación de Indicadores (indicator_comparison)
json
{
  "type": "indicator_comparison",
  "source": {
    "name": "SMA",
    "period": 20,
    "offset": 0
  },
  "comparator": "GREATER_THAN", // Ver tabla de comparadores
  "target": {
    "name": "EMA",
    "period": 50
  } // O puede ser un número estático: 70.0
}
B. Distancia a Nivel de Precios (price_level_distance)
json
{
  "type": "price_level_distance",
  "source": "Close", // "Close", "High", "Low"
  "level": "Pre-Market High", // Ver tabla de indicadores de nivel
  "comparator": "DISTANCE_LESS_THAN",
  "value_pct": 1.5 // Distancia máxima del 1.5%
}
C. Patrones de Velas (
candle_pattern
)
json
{
  "type": "candle_pattern",
  "pattern": "GREEN_VOLUME_PLUS", // Ver tabla de patrones
  "lookback": 1,
  "consecutive_count": 3
}
Tablas de Referencia (Enums)
Categoría	Valores Disponibles
Indicadores	SMA, EMA, WMA, VWAP, AVWAP, RSI, MACD, ATR, ADX, Williams %R, Close, Open, High, Low, Pre-Market High, Pre-Market Low, High of Day, Low of Day, Yesterday High, Yesterday Low, Yesterday Close, Volume, Accumulated Volume, Consecutive Red Candles, Consecutive Higher Highs, Consecutive Lower Lows, Ret % PM, Ret % RTH, Ret % AM, Time of Day, Max N Bars, Custom
Comparadores	GREATER_THAN, LESS_THAN, GREATER_THAN_OR_EQUAL, LESS_THAN_OR_EQUAL, EQUAL, CROSSES_ABOVE, CROSSES_BELOW, DISTANCE_GREATER_THAN, DISTANCE_LESS_THAN
Patrones	RED_VOLUME, RED_VOLUME_PLUS, GREEN_VOLUME, GREEN_VOLUME_PLUS, DOJI, HAMMER, SHOOTING_STAR
Riesgo (Tipos)	Fixed Amount, Percentage, ATR Multiplier, Market Structure (HOD/LOD)


Lo digo porque tienes que traducir todo esto para que lo use vectorBT y trading View



