from fastapi import FastAPI

from app.routers import stock, predict, metrics

app = FastAPI()
app.include_router(metrics.router)
app.include_router(stock.router)
app.include_router(predict.router)

@app.get("/")
async def root():
  return {"message": "Hello Bigger Applications!"}