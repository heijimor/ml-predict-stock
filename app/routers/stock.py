from fastapi import APIRouter
from app.services.stock_manager import StockManager
from pydantic import BaseModel

router = APIRouter()

class StockDataResponse(BaseModel):
    date: str
    close_price: float

@router.get("/stock", tags=["stock"])
async def get_stock():
    stock_manager = StockManager()
    data = stock_manager.collect("AAPL", "2015-01-01", "2023-01-01")
    response_data = [StockDataResponse(date=str(index.date()), close_price=row['Close']) 
                     for index, row in data.iterrows()]
    return [
      {"data": response_data}
    ]
