import yfinance as yf
import pandas as pd

class StockManager:
  def __init__(self):
    pass
  
  def collect(
    self,
    ticker: str,
    start_date: str,
    end_date: str
  ) -> pd.DataFrame:
    stock = yf.Ticker(ticker)
    data = stock.history(start=start_date, end=end_date)
    return data[['Close']]
  
  def prepare(self):
    pass
  
  def build(self):
    pass
  
  def train(self):
    pass
  
  def save(self):
    pass