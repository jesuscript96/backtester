import gc
import logging
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.data_service import get_strategy, fetch_dataset_data, fetch_day_candles
from backend.services.backtest_service import run_backtest
from backend.services.montecarlo_service import run_montecarlo

logger = logging.getLogger("backtester.backtest")

router = APIRouter(prefix="/api", tags=["backtest"])

MAX_DAYS = 100000


class BacktestRequest(BaseModel):
    dataset_id: str
    strategy_id: str
    init_cash: float = 10000.0
    risk_r: float = 100.0
    fees: float = 0.0
    slippage: float = 0.0


class MonteCarloRequest(BaseModel):
    pnls: list[float]
    init_cash: float = 10000.0
    simulations: int = 1000


@router.post("/backtest")
def run_backtest_endpoint(req: BacktestRequest):
    t0 = time.time()
    logger.info(f"BACKTEST START dataset={req.dataset_id} strategy={req.strategy_id}")

    strategy = get_strategy(req.strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    logger.info(f"  strategy loaded ({round(time.time()-t0, 2)}s)")

    try:
        t_fetch = time.time()
        qualifying, intraday = fetch_dataset_data(req.dataset_id)
        logger.info(
            f"  data fetched: {len(qualifying)} qualifying rows, "
            f"{len(intraday)} intraday rows ({round(time.time()-t_fetch, 2)}s)"
        )
    except Exception as e:
        logger.error(f"  data fetch FAILED: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching data: {str(e)}")

    if intraday.empty:
        raise HTTPException(
            status_code=400,
            detail="No hay datos intradiarios para este dataset",
        )

    unique_days = intraday.groupby(["ticker", "date"]).ngroups
    logger.info(f"  unique days: {unique_days}")
    if unique_days > MAX_DAYS:
        raise HTTPException(
            status_code=400,
            detail=f"Demasiados dias ({unique_days}). Maximo permitido: {MAX_DAYS}.",
        )

    try:
        t_bt = time.time()
        results = run_backtest(
            intraday_df=intraday,
            qualifying_df=qualifying,
            strategy_def=strategy["definition"],
            init_cash=req.init_cash,
            risk_r=req.risk_r,
            fees=req.fees,
            slippage=req.slippage,
        )
        del intraday, qualifying
        gc.collect()

        bt_elapsed = round(time.time() - t_bt, 2)
        total_elapsed = round(time.time() - t0, 2)
        n_trades = len(results.get("trades", []))
        n_days = len(results.get("day_results", []))
        logger.info(
            f"BACKTEST DONE {n_days} days, {n_trades} trades, "
            f"backtest={bt_elapsed}s, total={total_elapsed}s"
        )
    except Exception as e:
        logger.error(f"  backtest FAILED after {round(time.time()-t0, 2)}s: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error en backtest: {str(e)}")

    return results


@router.get("/candles")
def get_candles(dataset_id: str, ticker: str, date: str):
    candles = fetch_day_candles(dataset_id, ticker, date)
    if not candles:
        raise HTTPException(status_code=404, detail="No candle data found")
    return {"ticker": ticker, "date": date, "candles": candles}


@router.post("/montecarlo")
def run_montecarlo_endpoint(req: MonteCarloRequest):
    if not req.pnls:
        raise HTTPException(status_code=400, detail="No trades provided")
    if req.simulations < 100 or req.simulations > 10000:
        raise HTTPException(
            status_code=400, detail="Simulations must be between 100 and 10000"
        )
    try:
        return run_montecarlo(req.pnls, req.init_cash, req.simulations)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error en Monte Carlo: {str(e)}"
        )
