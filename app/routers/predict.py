from fastapi import APIRouter
from app.services.stock_manager import StockManager
from app.models.predict_request import PredictionRequest
import os

router = APIRouter()

@router.post("/predict", tags=["predict"])
async def predict(request: PredictionRequest):
  stock_manager = StockManager()
  
  # Paths to the model and scaler
  model_path = os.path.join(stock_manager.BASE_DIR, "../../models/lstm_model.h5")
  scaler_path = os.path.join(stock_manager.BASE_DIR, "../../models/scaler.json")

  # Load the model and scaler
  model, scaler = stock_manager.load_model(model_path, scaler_path)

  # Collect recent data based on the provided request
  # recent_data = stock_manager.collect(request.ticker, request.start_date, request.end_date)
  recent_data = stock_manager.collect("AAPL", "2015-01-01", "2016-01-31")

  # Generate predictions using the loaded model and scaler
  predictions = stock_manager.predict(model, scaler, recent_data, 60)

  # Return the predictions as a JSON response
  return {"predictions": predictions.tolist()}