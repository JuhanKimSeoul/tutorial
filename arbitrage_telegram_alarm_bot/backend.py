from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
from pydantic import BaseModel
from typing import List, Optional
import logging
from datetime import datetime
import requests
from cachetools import TTLCache
from apscheduler.schedulers.background import BackgroundScheduler
import asyncio
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from main import TradingDataManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

apscheduler_logger = logging.getLogger('apscheduler.executors.default')
apscheduler_logger.setLevel(logging.WARNING)

app = FastAPI()

# #Enable CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.exception_handler(RateLimitExceeded)
async def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded"}
    )
class Position(BaseModel):
    id: int
    exchange: str
    ticker: str
    size: float
    avg_buy_price: float
    position: str

class Order(BaseModel):
    id: int
    exchange: str
    ticker: str
    quoteVolume: float
    orderId: str
    timestamp: str
    position: str

cache = TTLCache(maxsize=100, ttl=1)

def connect_to_database(db_name="orders.db"):
    try:
        conn = sqlite3.connect(db_name)
        return conn
    except sqlite3.Error as e:
        print(f"Database connection failed: {e}")
        return None

def validate_datetime(dt_str: str):
    try:
        datetime.strptime(dt_str, "%Y-%m-%d %H:%M")
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid datetime format: {dt_str}. Expected format: YYYY-MM-DD HH:MM")

@app.get("/", response_class=HTMLResponse)
async def get_tradingview_html():
    with open("tradingview.html", "r") as file:
        return HTMLResponse(content=file.read(), status_code=200)

@app.get("/position", response_class=List[Position])
async def get_position(
    exchange: Optional[str] = Query(None),
    symbol: Optional[str] = Query(None),
):
    res = TradingDataManager('upbit').get_all_position()
    return [ Position(item.get('id'), 
                    item.get('exchange'), 
                    item.get('symbol'), 
                    item.get('size'), 
                    item.get('avg_buy_price'), 
                    item.get('position')) for item in res if (item.get('exchange') == exchange) and (item.get('symbol') == symbol)]

@app.get("/orders", response_model=List[Order])
async def get_orders(
    exchange: Optional[str] = Query(None),
    ticker: Optional[str] = Query(None),
    inq_st_dt: Optional[str] = Query(None),
    inq_close_dt: Optional[str] = Query(None)
):
    logger.info(f"Querying Params\n"
                f"exchange={exchange},\n"
                f"ticker={ticker},\n"
                f"inq_st_dt={inq_st_dt},\n"
                f"inq_close_dt={inq_close_dt}")
    if inq_st_dt:
        validate_datetime(inq_st_dt)
    if inq_close_dt:
        validate_datetime(inq_close_dt)

    conn = connect_to_database()
    cursor = conn.cursor()
    
    query = "SELECT * FROM 'order' WHERE 1=1"
    params = []
    
    if exchange:
        query += " AND exchange = ?"
        params.append(exchange)
    
    if ticker:
        query += " AND ticker = ?"
        params.append(ticker)
    
    if inq_st_dt:
        query += " AND timestamp >= ?"
        params.append(inq_st_dt)
    
    if inq_close_dt:
        query += " AND timestamp <= ?"
        params.append(inq_close_dt)
    
    cursor.execute(query, params)
    rows = cursor.fetchall()
    conn.close()
    
    orders = []
    for row in rows:
        orders.append(Order(
            id=row[0],
            exchange=row[1],
            ticker=row[2],
            quoteVolume=row[3],
            orderId=row[4],
            timestamp=row[5],
            position=row[6],
        ))
    return orders

@app.get("/kline")
@limiter.limit("10/second")
async def get_kline(request: Request, exchange: str, symbol: str, interval: str, to: Optional[str] = None, limit: Optional[int] = None):
    logger.info(f"Querying Kline\n"
                f"exchange={exchange},\n"
                f"symbol={symbol},\n"
                f"interval={interval},\n"
                f"to={to},\n"
                f"limit={limit}")
    
    return await TradingDataManager(exchange).get_ticker_kline(symbol, interval, to, limit)

@app.get("/kline_update")
def get_kline_update(exchange: str, symbol: str, interval: str):
    cache_key = f"{exchange}_{symbol}_{interval}"
    if cache_key in cache:
        return cache[cache_key]
    else:
        raise HTTPException(status_code=404, detail="Data not found in cache")

def schedule_cache_kline():
    exchange = "upbit"  # Example exchange
    symbol = "BTC"  # Example symbol
    interval = '5m'  # Example interval
    data = asyncio.run(TradingDataManager(exchange).get_ticker_kline(symbol, interval, limit=1))
    cache[f"{exchange}_{symbol}_{interval}"] = data
    # logger.info(f"Cache updated: {data}")

if __name__ == "__main__":
    # scheduler = BackgroundScheduler()
    # scheduler.add_job(schedule_cache_kline, 'interval', seconds=1, max_instances=2)
    # scheduler.start()

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
