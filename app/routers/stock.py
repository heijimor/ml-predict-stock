from fastapi import APIRouter
from app.services.stock_manager import StockManager

router = APIRouter()

@router.get("/stock", tags=["stock"])
async def get_stock():
    stock_manager = StockManager()
    data = stock_manager.collect("AAPL", "2015-01-01", "2023-01-01")

    return [
      {"data": data}
    ]
