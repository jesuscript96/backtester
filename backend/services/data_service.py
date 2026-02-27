"""
Fetches data from MotherDuck for backtesting.
Datasets define (ticker, date) pairs via two normalized tables;
daily stats come from daily_metrics; intraday candles come from intraday_1m.
"""

import json
import uuid
import pandas as pd
from backend.db.connection import query_df, execute_sql


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def list_strategies() -> list[dict]:
    df = query_df(
        "SELECT id, name, description, definition, created_at, updated_at "
        "FROM strategies ORDER BY updated_at DESC"
    )
    rows = []
    for _, r in df.iterrows():
        definition = r["definition"] if isinstance(r["definition"], dict) else json.loads(r["definition"])
        rows.append({
            "id": r["id"],
            "name": r["name"],
            "description": r["description"],
            "definition": definition,
        })
    return rows


def get_strategy(strategy_id: str) -> dict | None:
    df = query_df(
        "SELECT id, name, description, definition FROM strategies WHERE id = ?",
        [strategy_id],
    )
    if df.empty:
        return None
    r = df.iloc[0]
    definition = r["definition"] if isinstance(r["definition"], dict) else json.loads(r["definition"])
    return {
        "id": r["id"],
        "name": r["name"],
        "description": r["description"],
        "definition": definition,
    }


# ---------------------------------------------------------------------------
# Datasets CRUD
# ---------------------------------------------------------------------------

def list_datasets() -> list[dict]:
    df = query_df("""
        SELECT d.id, d.name, COUNT(dp.ticker) AS pair_count, d.created_at
        FROM datasets d
        LEFT JOIN dataset_pairs dp ON d.id = dp.dataset_id
        GROUP BY d.id, d.name, d.created_at
        ORDER BY d.created_at DESC
    """)
    return df.to_dict(orient="records")


def get_dataset(dataset_id: str) -> dict | None:
    ds = query_df("SELECT id, name, created_at FROM datasets WHERE id = ?", [dataset_id])
    if ds.empty:
        return None
    pairs = query_df(
        "SELECT ticker, date FROM dataset_pairs WHERE dataset_id = ? ORDER BY ticker, date",
        [dataset_id],
    )
    row = ds.iloc[0]
    return {
        "id": row["id"],
        "name": row["name"],
        "created_at": str(row["created_at"]),
        "pairs": pairs.to_dict(orient="records"),
        "pair_count": len(pairs),
    }


def create_dataset(name: str, pairs: list[dict]) -> dict:
    ds_id = str(uuid.uuid4())
    execute_sql("INSERT INTO datasets (id, name) VALUES (?, ?)", [ds_id, name])
    for p in pairs:
        execute_sql(
            "INSERT INTO dataset_pairs (dataset_id, ticker, date) VALUES (?, ?, ?)",
            [ds_id, p["ticker"], p["date"]],
        )
    return {"id": ds_id, "name": name, "pair_count": len(pairs)}


def delete_dataset(dataset_id: str) -> bool:
    execute_sql("DELETE FROM dataset_pairs WHERE dataset_id = ?", [dataset_id])
    execute_sql("DELETE FROM datasets WHERE id = ?", [dataset_id])
    return True


# ---------------------------------------------------------------------------
# Data fetching for backtest
# ---------------------------------------------------------------------------

def fetch_dataset_data(dataset_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        qualifying: daily stats from daily_metrics enriched with yesterday_high/low
        intraday: raw 1m candles from intraday_1m
    """
    qualifying_sql = """
    WITH enriched AS (
        SELECT dm.*,
               CAST(dm."timestamp" AS DATE) AS date,
               LAG(dm.rth_high) OVER (PARTITION BY dm.ticker ORDER BY dm."timestamp") AS yesterday_high,
               LAG(dm.rth_low)  OVER (PARTITION BY dm.ticker ORDER BY dm."timestamp") AS yesterday_low,
               dm.prev_close AS previous_close
        FROM daily_metrics dm
    )
    SELECT e.*
    FROM dataset_pairs dp
    INNER JOIN enriched e ON dp.ticker = e.ticker AND dp.date = e.date
    WHERE dp.dataset_id = ?
    """
    qualifying = query_df(qualifying_sql, [dataset_id])

    if qualifying.empty:
        return qualifying, pd.DataFrame()

    intraday_sql = """
    SELECT i.ticker, i.date, i."timestamp", i.open, i.high, i.low,
           i."close", i.volume, i.transactions
    FROM intraday_1m i
    INNER JOIN dataset_pairs dp ON i.ticker = dp.ticker AND i.date = dp.date
    WHERE dp.dataset_id = ?
    ORDER BY i.ticker, i."timestamp"
    """
    intraday = query_df(intraday_sql, [dataset_id])

    return qualifying, intraday
