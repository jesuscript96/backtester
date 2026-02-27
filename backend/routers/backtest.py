from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.services.data_service import get_strategy, fetch_dataset_data
from backend.services.backtest_service import run_backtest
from backend.services.montecarlo_service import run_montecarlo

router = APIRouter(prefix="/api", tags=["backtest"])

MAX_DAYS = 500


class BacktestRequest(BaseModel):
    dataset_id: str
    strategy_id: str
    init_cash: float = 10000.0
    fees: float = 0.0
    slippage: float = 0.0


class MonteCarloRequest(BaseModel):
    pnls: list[float]
    init_cash: float = 10000.0
    simulations: int = 1000


@router.post("/backtest")
def run_backtest_endpoint(req: BacktestRequest):
    strategy = get_strategy(req.strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")

    try:
        qualifying, intraday = fetch_dataset_data(req.dataset_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching data: {str(e)}")

    if intraday.empty:
        raise HTTPException(
            status_code=400,
            detail="No hay datos intradiarios para este dataset",
        )

    unique_days = intraday.groupby(["ticker", "date"]).ngroups
    if unique_days > MAX_DAYS:
        raise HTTPException(
            status_code=400,
            detail=f"Demasiados dias ({unique_days}). Maximo permitido: {MAX_DAYS}.",
        )

    try:
        results = run_backtest(
            intraday_df=intraday,
            qualifying_df=qualifying,
            strategy_def=strategy["definition"],
            init_cash=req.init_cash,
            fees=req.fees,
            slippage=req.slippage,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en backtest: {str(e)}")

    return results


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
