from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.services.data_service import get_strategy, fetch_dataset_data
from backend.services.backtest_service import run_backtest
from backend.services.montecarlo_service import run_montecarlo
# #region agent log
import time as _time, json as _json, os as _os
_LOG = "/Users/jvch/Desktop/AutomatoWebs/BacktesterMVP/.cursor/debug-448660.log"
def _dlog(msg, data=None, hid=""):
    line = {"sessionId":"448660","hypothesisId":hid,"location":"backtest.py","message":msg,"data":data or {},"timestamp":int(_time.time()*1000)}
    try:
        with open(_LOG, "a") as f: f.write(_json.dumps(line)+"\n")
    except: pass
    print(f"[DBG-{hid}] {msg} {data or ''}")
# #endregion

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
    # #region agent log
    t0 = _time.time()
    _dlog("BACKTEST_START", {"dataset": req.dataset_id, "strategy": req.strategy_id}, "HA")
    # #endregion
    strategy = get_strategy(req.strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")

    try:
        # #region agent log
        t_fetch = _time.time()
        # #endregion
        qualifying, intraday = fetch_dataset_data(req.dataset_id)
        # #region agent log
        _dlog("DATA_FETCH_DONE", {"elapsed_s": round(_time.time()-t_fetch,2), "qualifying_rows": len(qualifying), "intraday_rows": len(intraday)}, "HB")
        # #endregion
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
        # #region agent log
        t_bt = _time.time()
        # #endregion
        results = run_backtest(
            intraday_df=intraday,
            qualifying_df=qualifying,
            strategy_def=strategy["definition"],
            init_cash=req.init_cash,
            fees=req.fees,
            slippage=req.slippage,
        )
        # #region agent log
        _dlog("BACKTEST_DONE", {"elapsed_s": round(_time.time()-t_bt,2), "total_s": round(_time.time()-t0,2)}, "HA")
        # #endregion
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
