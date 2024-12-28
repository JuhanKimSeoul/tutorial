from celery import Celery
import os
from dotenv import load_dotenv
import time
from kombu.utils.url import safequote

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
    task_create_missing_queues = False
    task_serializer='json'
    accept_content=['json']
    result_serializer='json'
    timezone='Asia/Seoul'
    enable_utc=True
    result_expires=60
    

app = Celery('tasks')
app.config_from_object(CeleryConfig())
app.conf.beat_schedule = {
    'test': {
        'task': 'tests.test_celery.test',
        'schedule': 10,
        'args': (1,2)
    },
}

app.conf.update(
    task_ignore_result=False,
    task_acks_on_failure_or_timeout=True,
    worker_prefetch_multiplier=1,
    task_track_started=True,
)

@app.task
def test(x,y):
    time.sleep(1)
    z = x + y
    return z