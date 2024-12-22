from celery import Celery
from main import *
import asyncio
import redis
import os
from dotenv import load_dotenv
from redis.exceptions import ConnectionError, TimeoutError
import time

load_dotenv()

# Redis 클라이언트 생성 (글로벌 네임스페이스)
redis_client = redis.StrictRedis(
    host=os.getenv('REDIS_HOST'),
    port=6379,
    db=0,
    socket_connect_timeout=5,  # 연결 타임아웃 설정
    retry_on_timeout=True      # 타임아웃 시 재시도 설정
)

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
        for item in res:
            try:
                # 결과를 Redis에 저장
                redis_client.set(item['ticker'], item)
                redis_client.publish('big_volume_tickers', item)
            except (ConnectionError, TimeoutError) as e:
                print(f"Redis connection error: {e}")
                # 재시도 로직 추가
                reconnect_redis()
                redis_client.set(item['ticker'], item)
                redis_client.publish('big_volume_tickers', item)
    
    time.sleep(0.5)

def reconnect_redis():
    global redis_client
    while True:
        try:
            redis_client = redis.StrictRedis(
                host=os.getenv('REDIS_HOST'),
                port=6379,
                db=0,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )
            # 연결 테스트
            redis_client.ping()
            print("Reconnected to Redis")
            break
        except (ConnectionError, TimeoutError) as e:
            print(f"Retrying Redis connection: {e}")
            time.sleep(5)

if __name__ == "__main__":
    app.worker_main()