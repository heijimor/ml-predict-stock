from fastapi import APIRouter
from app.services.stock_manager import StockManager
from pydantic import BaseModel

router = APIRouter()

class StockDataResponse(BaseModel):
    date: str
    close_price: float

@router.get("/stock", tags=["stock"])
async def train_stock():
    stock_manager = StockManager()
    data = stock_manager.collect("AAPL", "2015-01-01", "2023-01-01")
    scaled_data, scaler = stock_manager.normalize(data)
    seq_length = 60
    X, y = stock_manager.sequencialize(scaled_data, seq_length)
    X_train, X_test, y_train, y_test = stock_manager.prepare(X, y)
    model = stock_manager.build((X_train.shape[1], 1))
    stock_manager.train(model, X_train, y_train, X_test, y_test)
    stock_manager.save(model)
    stock_manager.evaluate(model, scaler, X_test, y_test)

    response_data = [StockDataResponse(date=str(index.date()), close_price=row['Close']) 
                     for index, row in data.iterrows()]
    return [
      {"data": response_data}
    ]
