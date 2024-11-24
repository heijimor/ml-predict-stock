from fastapi import APIRouter
from app.services.stock_manager import StockManager
from app.models.predict_request import PredictionRequest

router = APIRouter()

@router.post("/predict", tags=["predict"])
async def predict(request: PredictionRequest):
  return {
    "ticker": request.ticker
  }
