from fastapi import FastAPI
import redis
import orjson
# from celery_with_fastapi.celery import fetch_market_data

app = FastAPI()

r = redis.Redis(host='localhost', port=6379, db=0)

@app.get('/cached_market_data/{exchange_id}')
async def get_cached_market_data(exchange_id: str):
    market_data = r.get(f'market_data_{exchange_id}')
    if market_data:
        return orjson.loads(market_data)
    return {'status' : 403, 'message' : 'Market data not found'}

# @app.get('/market_data/{exchange_id}')
# async def get_market_data(exchange_id: str):
#     task = fetch_market_data.delay(exchange_id)
#     return {'status' : 200, 'task_id' : task.id}

@app.get('/orderbook_data/{exchange_id}/{target_symbol}')
async def get_orderbook_data(exchange_id: str, target_symbol: str):
    orderbook_data = r.get(f'orderbook_data_{exchange_id}_{target_symbol}')
    if orderbook_data:
        return orjson.loads(orderbook_data)
    return {'status' : 403, 'message' : 'Orderbook data not found'}