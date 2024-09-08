import configparser
import ccxt.async_support as ccxt
import orjson
import asyncio
import logging
import json

logging.basicConfig(level=logging.INFO)

config = configparser.ConfigParser()
config.read('config.ini')
ex = dict(config['exchanges'])

async def load_exchange():
    tasks = []
    tmp = {}
    for k, v in ex.items():
        if v == 'True':
            exchange_class = getattr(ccxt, k)
            exchange = exchange_class()
            tasks.append(asyncio.create_task(exchange.load_markets()))
            tmp[k] = exchange
            
    await asyncio.gather(*tasks)
    for k, exchange in zip(ex, tmp.values()):
        if ex[k] == 'True':
            ex[k] = exchange

    logging.info('Exchanges loaded')

def fetch_orderbook_data(exchange_id):

    async def _fetch_orderbook_data():
        # task_id = fetch_orderbook_data.request.id
        # logging.(f'celery-task-meta-{task_id}')
        try:
            # target_symbol = 'BTC/USDT'
            ex = ccxt.binance()
            orderbook = await ex.fetch_order_book('BTC/USDT', limit=500)
            logging.debug(json.dumps(orderbook, indent=4))
            return orderbook
            # r.set(f'orderbook_data_{exchange_id}_{target_symbol}', orjson.dumps(orderbook))
        except Exception as e:
            return str(e)
        finally:
            await ex.close()

    result = asyncio.run(_fetch_orderbook_data())
    print(result)
    return result
# asyncio.run(load_exchange())
# asyncio.run(fetch_orderbook_data('binance'))
fetch_orderbook_data('binance')