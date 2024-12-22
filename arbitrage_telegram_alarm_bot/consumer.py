from celery import Celery
from main import *
import asyncio

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

if __name__ == "__main__":
    app.worker_main()