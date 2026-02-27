from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import data, backtest
from backend.config import ALLOWED_ORIGINS

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


@app.get("/api/health")
def health():
    return {"status": "ok"}
