import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import data, backtest
from backend.config import ALLOWED_ORIGINS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("backtester")

app = FastAPI(title="BacktesterMVP", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(data.router)
app.include_router(backtest.router)


@app.on_event("startup")
def on_startup():
    logger.info("=== BacktesterMVP starting ===")
    logger.info(f"ALLOWED_ORIGINS = {ALLOWED_ORIGINS}")
    logger.info("Engine: pure numpy (no vectorbt)")
    # #region agent log
    import resource, sys, json, os
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        rss = ru.ru_maxrss / (1024*1024) if sys.platform == "darwin" else ru.ru_maxrss / 1024
    except Exception:
        rss = -1
    try:
        with open('/proc/self/status') as f:
            for line in f:
                if line.startswith('VmRSS:'):
                    rss = int(line.split()[1]) / 1024
    except Exception:
        pass
    logger.info(f"[DBG:A] startup_memory | RSS={round(rss,1)}MB")
    try:
        lp = os.path.join(os.path.dirname(__file__), '..', '.cursor', 'debug-568c25.log')
        with open(lp, 'a') as f:
            f.write(json.dumps({"sessionId":"568c25","location":"main.py","message":"startup_memory","data":{"rss_mb":round(rss,1)},"timestamp":int(time.time()*1000),"hypothesisId":"A"})+'\n')
    except Exception:
        pass
    # #endregion


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    elapsed = round(time.time() - start, 2)
    logger.info(f"{request.method} {request.url.path} -> {response.status_code} ({elapsed}s)")
    return response


@app.get("/api/health")
def health():
    return {"status": "ok"}
