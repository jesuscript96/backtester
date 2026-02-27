from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.services.data_service import (
    list_strategies,
    get_strategy,
    list_datasets,
    get_dataset,
    create_dataset,
    delete_dataset,
)

router = APIRouter(prefix="/api", tags=["data"])


# --- Strategies ---

@router.get("/strategies")
def get_strategies():
    return list_strategies()


@router.get("/strategies/{strategy_id}")
def get_strategy_endpoint(strategy_id: str):
    s = get_strategy(strategy_id)
    if not s:
        raise HTTPException(status_code=404, detail="Strategy not found")
    return s


# --- Datasets ---

class PairItem(BaseModel):
    ticker: str
    date: str


class CreateDatasetRequest(BaseModel):
    name: str
    pairs: list[PairItem]


@router.get("/datasets")
def get_datasets():
    return list_datasets()


@router.get("/datasets/{dataset_id}")
def get_dataset_endpoint(dataset_id: str):
    ds = get_dataset(dataset_id)
    if not ds:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return ds


@router.post("/datasets")
def create_dataset_endpoint(req: CreateDatasetRequest):
    if not req.pairs:
        raise HTTPException(status_code=400, detail="Pairs list cannot be empty")
    return create_dataset(req.name, [p.model_dump() for p in req.pairs])


@router.delete("/datasets/{dataset_id}")
def delete_dataset_endpoint(dataset_id: str):
    delete_dataset(dataset_id)
    return {"message": "Dataset deleted"}
