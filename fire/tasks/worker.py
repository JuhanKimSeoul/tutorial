from celery import Celery
from config import settings
import ccxt.async_support as ccxt
import asyncio
from exchange.exchange import *
import itertools
from typing import List
import aioredis
import logging
import json
import time
import traceback

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Adjust the level as needed
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('celery_worker.log'),  # Log file path
        # logging.StreamHandler()  # Log to console
    ]
)

logger = logging.getLogger(__name__)

class CeleryConfig:
    result_backend=settings['redis_url']
    broker_url=settings['broker_url']
    broker_transport_options={
        'region' : 'ap-northeast-2',
        'predefined_queues' : {
            settings['aws_sqs_name'] : {
                'url' : settings['aws_sqs_url'],
                'access_key_id' : settings['aws_access_key_id'],
                'secret_access_key' : settings['aws_secret_access_key']
            }
        }
    }
    task_default_queue=settings['aws_sqs_name']
    task_create_missing_queues=False
    worker_prefetch_multiplier=1
    task_track_started=True
    task_serializer='json'
    accept_content=['json']
    result_serializer='json'
    timezone='Asia/Seoul'
    enable_utc=True
    result_expires=10

app = Celery('tasks')
app.config_from_object(CeleryConfig())

@app.task
def task_create_common_tickers(exchanges:List[str]):
    try:
        start_time = time.time()
        asyncio.run(find_common_tickers(exchanges))
        logger.info(f'task success!')
        return { 'status' : 'ok' }
    except Exception as e:
        logger.error(f'error : {traceback.format_exc()}')
        return { 'status' : 'fail' }
    finally:
        logger.info(f'elapsed time : {time.time() - start_time}')

async def find_common_tickers(exchanges:List[str]):
    try:
        tasks = [load_market(exchange) for exchange in exchanges]
        results = await asyncio.gather(*tasks)

        data_with_exchanges_label = list(zip(exchanges, results))

        pairs = itertools.combinations(data_with_exchanges_label,2)

        redis_client = await aioredis.from_url(settings['redis_url'])

        sub_tasks = []
        for (ex1, d1), (ex2, d2) in pairs:
            ex1_config = settings['supported_exchanges'][ex1]
            ex2_config = settings['supported_exchanges'][ex2]
            ret = extract_common_tickers(ex1_config, d1, ex2_config, d2)
            key = f'{ex1}_{ex2}'
            sub_tasks.append(asyncio.create_task(redis_client.set(key, json.dumps(ret[key], indent=4)))) # manually saving 

        await asyncio.gather(*sub_tasks)
        
    except Exception as e:
        raise e

    finally:
        if redis_client:
            await redis_client.close()

@app.task
def process_data(data):
    logger.info(f'process : {data}')
    return {'status' : 'ok'}

def register_tasks(data):
    for item in data:
        process_data.delay(item)

@app.task
def schedule_task():
    data = ['a','b','c']
    register_tasks(data)
    
app.conf.beat_schedule = {
    # 'task_create_common_tickers' : {
    #     'task': 'tasks.worker.task_create_common_tickers',
    #     'schedule': 100,
    #     'args': [['binance', 'upbit', 'bithumb', 'bybit']]
    # },
    'schedule_task' : {
        'task' : 'tasks.worker.schedule_task',
        'schedule' : 100,
    }
}
