from fastapi import APIRouter
from prometheus_client import start_http_server, Summary, Counter, Gauge, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST

router = APIRouter()

@router.get('/metrics')
def metrics():
    return generate_latest(), 200, {'Content-Type': CONTENT_TYPE_LATEST}