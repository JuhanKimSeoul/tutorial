import sys
from pathlib import Path

root_directory = str(Path(__file__).parent.parent)
sys.path.append(root_directory)

from celery import Celery
# from config import settings
import ccxt.async_support as ccxt
import asyncio
from exchange.exchange import *
import itertools
from typing import List
# import aioredis
import logging
import json
import time
import traceback
import os
from dotenv import load_dotenv
from kombu.utils.url import safequote
import boto3

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

load_dotenv('dev.env')

AWS_REDIS_SERVER_ENDPOINT = os.getenv('AWS_REDIS_SERVER_ENDPOINT')
AWS_REDIS_SERVER_PORT = os.getenv('AWS_REDIS_SERVER_PORT')
REDIS_URL = f'redis://{AWS_REDIS_SERVER_ENDPOINT}:{AWS_REDIS_SERVER_PORT}'
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_SQS_URL = os.getenv('AWS_SQS_URL')
BROKER_URL = f'sqs://{safequote(AWS_ACCESS_KEY_ID.encode())}:{safequote(AWS_SECRET_ACCESS_KEY.encode())}@'
AWS_SQS_NAME = os.getenv('AWS_SQS_NAME')

class CeleryConfig:
    result_backend = REDIS_URL
    broker_url = BROKER_URL
    broker_transport_options = {
            'region' : 'ap-northeast-2',
            'predefined_queues' : {
                AWS_SQS_NAME : {
                    'url' : AWS_SQS_URL,
                    'access_key_id' : AWS_ACCESS_KEY_ID,
                    'secret_access_key' : AWS_SECRET_ACCESS_KEY
                    }
                }
            }
    task_default_queue = AWS_SQS_NAME
    worker_prefetch_multiplier=1
    task_track_started=True
    task_create_missing_queues = False
    task_serializer='json'
    accept_content=['json']
    result_serializer='json'
    timezone='Asia/Seoul'
    enable_utc=True
    result_expires=10

app = Celery('tasks')
app.config_from_object(CeleryConfig())

# @app.task
# def task_create_common_tickers(exchanges:List[str]):
#     try:
#         start_time = time.time()
#         asyncio.run(find_common_tickers(exchanges))
#         logger.info(f'task success!')
#         return { 'status' : 'ok' }
#     except Exception as e:
#         logger.error(f'error : {traceback.format_exc()}')
#         return { 'status' : 'fail' }
#     finally:
#         logger.info(f'elapsed time : {time.time() - start_time}')

# async def find_common_tickers(exchanges:List[str]):
#     try:
#         tasks = [load_market(exchange) for exchange in exchanges]
#         results = await asyncio.gather(*tasks)

#         data_with_exchanges_label = list(zip(exchanges, results))

#         pairs = itertools.combinations(data_with_exchanges_label,2)

#         redis_client = await aioredis.from_url(settings['redis_url'])

#         sub_tasks = []
#         for (ex1, d1), (ex2, d2) in pairs:
#             ex1_config = settings['supported_exchanges'][ex1]
#             ex2_config = settings['supported_exchanges'][ex2]
#             ret = extract_common_tickers(ex1_config, d1, ex2_config, d2)
#             key = f'{ex1}_{ex2}'
#             sub_tasks.append(asyncio.create_task(redis_client.set(key, json.dumps(ret[key], indent=4)))) # manually saving 

#         await asyncio.gather(*sub_tasks)
        
#     except Exception as e:
#         raise e

#     finally:
#         if redis_client:
#             await redis_client.close()

@app.task
def process_data():
    time.sleep(3)
    return {'status' : 'ok'}

# def register_tasks(data):
#     for item in data:
#         process_data.delay(item)

# @app.task
# def schedule_task():
#     data = ['a','b','c']
#     register_tasks(data)
    
app.conf.beat_schedule = {
    # 'task_create_common_tickers' : {
    #     'task': 'tasks.worker.task_create_common_tickers',
    #     'schedule': 100,
    #     'args': [['binance', 'upbit', 'bithumb', 'bybit']]
    # },
    # 'schedule_task' : {
    #     'task' : 'jobs.worker.schedule_task',
    #     'schedule' : 3,
    # },
    'worker' : {
        'task' : 'jobs.worker.process_data',
        'schedule' : 1
    }
}

def send_message(sqs, queue_url):
    # Sending a message to the SQS queue
    response = sqs.send_message(
        QueueUrl=queue_url,
        MessageBody='test'
    )
    print("Message sent with ID : ", response['MessageId'])

def receive_message(sqs, queue_url):
    # Receiving messages from the SQS queue
    response = sqs.receive_message(
        QueueUrl=queue_url,
        MaxNumberOfMessages=1,  # Maximum number of messages to retrieve (adjust as needed)
        WaitTimeSeconds=10  # Long polling to wait for a message
    )
    
    # If no message is received
    if 'Messages' not in response:
        print("No messages to receive.")
        return
    
    # Get the message details
    message = response['Messages'][0]
    receipt_handle = message['ReceiptHandle']
    body = message['Body']

    print("Message received:", body)

    # Processing the message (you can add your logic here)

    # Delete the message after processing it
    sqs.delete_message(
        QueueUrl=queue_url,
        ReceiptHandle=receipt_handle
    )
    print("Message deleted from the queue.")

if __name__ == '__main__':
    session = boto3.Session()
    credentials = session.get_credentials()
    
    # Printing AWS credentials (useful for debugging)
    print(f'Access Key : {credentials.access_key}')
    print(f'Secret Key : {credentials.secret_key}')
    print(f'Token : {credentials.token}')

    # Creating SQS client
    sqs = boto3.client('sqs', region_name='ap-northeast-2')

    # Queue URL
    queue_url = 'https://sqs.ap-northeast-2.amazonaws.com/905418113774/TaskQueue'

    # Sending a message
    send_message(sqs, queue_url)

    # Receiving and processing a message
    receive_message(sqs, queue_url)