from celery import Celery
from main import *
import asyncio
import redis
import os
from dotenv import load_dotenv

load_dotenv()

# Celery 인스턴스 생성
app = Celery('consumer')
app.config_from_object('celeryconfig')

@app.task(name='producer.alarm_big_vol_tickers_task')
def work_task(data, multiplier: int, usdt_price: float, binance_threshold: int):
    '''
        data = [(ex, ticker)...]
    '''
    k = KimpManager()
    res = asyncio.run(k.celery_monitor_big_volume_batch(data, multiplier, usdt_price, binance_threshold))

    if res:
        # Redis 클라이언트 생성
        redis_client = redis.StrictRedis(host=f'{os.getenv('REDIS_HOST')}', port=6379, db=0)

        # 결과를 Redis에 저장
        for item in res:
            redis_client.set(item['ticker'], item)
            redis_client.publish('big_volume_tickers', item)

if __name__ == "__main__":
    app.worker_main()