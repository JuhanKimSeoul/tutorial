from celery import Celery
# from celery.schedules import crontab
import ccxt.async_support as ccxt
import asyncio
import redis
import orjson
import logging
import configparser
import json

config = configparser.ConfigParser()
config.read('config.ini')
ex = dict(config['exchanges'])

logging.basicConfig(level=logging.INFO)

app = Celery(
    'tasks',
    broker='amqp://localhost',
    backend='redis://localhost:6379/0'
)

r = redis.Redis(host='localhost', port=6379, db=0)

@app.task(ignore_result=True)
def fetch_market_data(exchange_id):
    task_id = fetch_market_data.request.id
    logging.debug(f'celery-task-meta-{task_id}')
    try:
        exchange_class = getattr(ccxt, exchange_id)
        ex = exchange_class()
        markets = ex.load_markets()
        r.set(f'market_data_{exchange_id}', orjson.dumps(markets))
    except Exception as e:
        return str(e)
    finally:
        ex.close()

@app.task(ignore_result=True)
def fetch_orderbook_data(exchange_id):

    async def _fetch_orderbook_data():
        try:
            exchange_class = getattr(ccxt, exchange_id)
            ex = exchange_class()
            target_symbol = 'BTC/USDT'
            orderbook = await ex.fetch_order_book('BTC/USDT', limit=500)
            logging.debug(json.dumps(orderbook, indent=4))
            r.set(f'orderbook_data_{exchange_id}_{target_symbol}', orjson.dumps(orderbook))
            return orjson.dumps(orderbook)
        except Exception as e:
            return str(e)
        finally:
            await ex.close()
        
    task_id = fetch_orderbook_data.request.id
    logging.debug(f'celery-task-meta-{task_id}')
    return asyncio.run(_fetch_orderbook_data())
    

app.conf.beat_schedule = {
    'fetch-binance-market-data': {
        'task': 'core.fetch_orderbook_data',
        'schedule': 10,
        'args': ('binance',)
    },
    'fetch-coinbase-market-data': {
        'task': 'core.fetch_orderbook_data',
        'schedule': 10,
        'args': ('coinbase',)
    },
    'fetch-bybit-market-data': {
        'task': 'core.fetch_orderbook_data',
        'schedule': 10,
        'args': ('bybit',)
    },
    'fetch-upbit-market-data': {   
        'task': 'core.fetch_orderbook_data',
        'schedule': 10,
        'args': ('upbit',)
    },
}

app.conf.update(
    task_ignore_result=False,
    task_acks_on_failure_or_timeout=True,
    worker_prefetch_multiplier=1,
    task_track_started=True,
)

