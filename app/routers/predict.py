from fastapi import APIRouter
from app.services.stock_manager import StockManager
from app.models.predict_request import PredictionRequest
import os

router = APIRouter()

@router.post("/predict", tags=["predict"])
async def predict(request: PredictionRequest):
  stock_manager = StockManager()
  model_path = os.path.join(stock_manager.BASE_DIR, "../../models/lstm_model.h5")
  scaler_path = os.path.join(stock_manager.BASE_DIR, "../../models/scaler.json")
  model, scaler = stock_manager.load_model(model_path, scaler_path)
  recent_data = stock_manager.collect(request.ticker, request.start, request.end)
  print(f'recent_data: \n ${recent_data}')
  predictions = stock_manager.predict(model, scaler, recent_data, 15)
  return { "predictions": predictions }