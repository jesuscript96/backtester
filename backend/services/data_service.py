"""
Fetches data from MotherDuck for backtesting.
Datasets define (ticker, date) pairs via two normalized tables;
daily stats come from daily_metrics; intraday candles come from intraday_1m.
"""

import json
import logging
import time
import uuid
import pandas as pd
from backend.db.connection import query_df, execute_sql

logger = logging.getLogger("backtester.data")


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
            "created_at": str(r["created_at"]) if pd.notnull(r["created_at"]) else None,
            "updated_at": str(r["updated_at"]) if pd.notnull(r["updated_at"]) else None,
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
    if not df.empty and "created_at" in df.columns:
        df["created_at"] = df["created_at"].astype(str)
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

def fetch_dataset_data(dataset_id: str) -> tuple[pd.DataFrame, list[tuple[str, str]]]:
    """
    Returns:
        qualifying: daily stats from daily_metrics enriched with yesterday_high/low
        pairs: list of (ticker, date) tuples for intraday batch fetching
    """
    t0 = time.time()

    qualifying_sql = """
    WITH relevant_tickers AS (
        SELECT DISTINCT ticker FROM dataset_pairs WHERE dataset_id = ?
    ),
    filtered_dm AS (
        SELECT dm.*,
               CAST(dm."timestamp" AS DATE) AS date
        FROM daily_metrics dm
        WHERE dm.ticker IN (SELECT ticker FROM relevant_tickers)
    ),
    enriched AS (
        SELECT f.*,
               LAG(f.rth_high) OVER (PARTITION BY f.ticker ORDER BY f."timestamp") AS yesterday_high,
               LAG(f.rth_low)  OVER (PARTITION BY f.ticker ORDER BY f."timestamp") AS yesterday_low,
               f.prev_close AS previous_close
        FROM filtered_dm f
    )
    SELECT e.*
    FROM dataset_pairs dp
    INNER JOIN enriched e ON dp.ticker = e.ticker AND dp.date = e.date
    WHERE dp.dataset_id = ?
    """
    qualifying = query_df(qualifying_sql, [dataset_id, dataset_id])
    t_q = time.time()
    logger.info(f"qualifying query: {len(qualifying)} rows ({round(t_q - t0, 2)}s)")

    if qualifying.empty:
        return qualifying, []

    pairs_df = query_df(
        "SELECT ticker, date FROM dataset_pairs WHERE dataset_id = ? ORDER BY ticker, date",
        [dataset_id],
    )
    pairs = [(r["ticker"], r["date"]) for _, r in pairs_df.iterrows()]
    logger.info(f"pairs fetched: {len(pairs)}")

    return qualifying, pairs


def fetch_intraday_batch(
    dataset_id: str, pairs: list[tuple[str, str]]
) -> pd.DataFrame:
    """Fetch intraday candles for a specific batch of (ticker, date) pairs."""
    if not pairs:
        return pd.DataFrame()

    values_clause = ", ".join(
        f"('{t}', '{d}')" for t, d in pairs
    )
    sql = f"""
    SELECT i.ticker, i.date, i."timestamp", i.open, i.high, i.low,
           i."close", i.volume
    FROM intraday_1m i
    WHERE (i.ticker, i.date) IN ({values_clause})
    """
    df = query_df(sql)

    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df[col] = df[col].astype("float32")
    if "volume" in df.columns:
        df["volume"] = df["volume"].astype("int32")

    return df
