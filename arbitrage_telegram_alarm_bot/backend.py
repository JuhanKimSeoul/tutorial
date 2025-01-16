from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
from pydantic import BaseModel
from typing import List, Optional
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

class Order(BaseModel):
    id: int
    exchange: str
    ticker: str
    quoteVolume: float
    orderId: str
    timestamp: str
    position: str

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
def get_tradingview_html():
    with open("tradingview.html", "r") as file:
        return HTMLResponse(content=file.read(), status_code=200)

@app.get("/orders", response_model=List[Order])
def get_orders(
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
