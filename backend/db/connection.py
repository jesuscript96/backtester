import duckdb
from backend.config import MOTHERDUCK_TOKEN, MOTHERDUCK_DB

_conn = None


def get_connection() -> duckdb.DuckDBPyConnection:
    global _conn
    if _conn is None:
        _conn = _create_connection()
    return _conn


def _create_connection() -> duckdb.DuckDBPyConnection:
    return duckdb.connect(f"md:{MOTHERDUCK_DB}?motherduck_token={MOTHERDUCK_TOKEN}")


def _reset_connection():
    global _conn
    try:
        if _conn is not None:
            _conn.close()
    except Exception:
        pass
    _conn = None


def query_df(sql: str, params: list | None = None):
    """Execute SQL and return a pandas DataFrame. Auto-reconnects on failure."""
    global _conn
    for attempt in range(2):
        try:
            conn = get_connection()
            if params:
                return conn.execute(sql, params).fetchdf()
            return conn.execute(sql).fetchdf()
        except Exception as e:
            if attempt == 0:
                _reset_connection()
                continue
            raise e


def execute_sql(sql: str, params: list | None = None):
    """Execute SQL statement (INSERT, UPDATE, DELETE). Auto-reconnects on failure."""
    global _conn
    for attempt in range(2):
        try:
            conn = get_connection()
            if params:
                conn.execute(sql, params)
            else:
                conn.execute(sql)
            return
        except Exception as e:
            if attempt == 0:
                _reset_connection()
                continue
            raise e
