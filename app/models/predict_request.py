from pydantic import BaseModel

class PredictionRequest(BaseModel):
    ticker: str  # Ticker da ação
    days: int    # Últimos 'n' dias para previsão
