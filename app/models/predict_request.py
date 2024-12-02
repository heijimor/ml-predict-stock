from pydantic import BaseModel

class PredictionRequest(BaseModel):
    ticker: str
    start: str
    end: str
