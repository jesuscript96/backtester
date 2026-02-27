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
