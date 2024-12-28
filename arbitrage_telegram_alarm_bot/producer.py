import threading
from celery import Celery, group
import time
import redis
from main import *
import asyncio
from apscheduler.schedulers.background import BackgroundScheduler

# from unittest.mock import MagicMock

# # Mock Update 객체 생성
# mock_update = MagicMock(spec=Update)
# mock_update.callback_query.message.reply_text = MagicMock()

# # Mock Context 객체 생성
# mock_context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
# mock_context.user_data = {
#     'event_caller': 'test_event',
#     'stop_event': asyncio.Event(),
#     'test_event': {
#         'multiplier': 5
#     }
# }

# Celery 인스턴스 생성
app = Celery('producer')
app.config_from_object('celeryconfig')

def handle_message(message):
    if message['type'] == 'message':
        data = json.loads(message['data'])
        # 결과 처리 로직 추가
        k = KimpManager()
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            loop.run_until_complete(k.send_telegram(message['data']))
        except RuntimeError as e:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(k.send_telegram(message['data']))
            

def subscribe_to_redis():
    redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
    pubsub = redis_client.pubsub()
    pubsub.subscribe('big_volume_tickers')

    print("Subscribed to big_volume_tickers. Waiting for messages...")
    while True:
        message = pubsub.get_message()
        if message:
            handle_message(message)
        time.sleep(0.01)

@app.task
def alarm_big_vol_tickers_task(data, multiplier: int, usdt_price: float, binance_threshold: int):
    '''
        data = [(ex, ticker)...]
    '''
    pass

def schedule_tasks():
    # k = KimpManager()
    # res = asyncio.run(k.get_all_tickers())

    # upbit = res['upbit']
    # bithumb = res['bithumb']
    # combined_tickers = set(res['upbit']).union(set(res['bithumb']))
    # batch_size = 10
    # for i in range(0, len(combined_tickers), batch_size):
    #     batch = list(combined_tickers)[i:i + batch_size]
    #     data = [('upbit' if ticker in upbit else 'bithumb', ticker) for ticker in batch]
    #     tasks = [alarm_big_vol_tickers_task.s(data, 5, 1500, 100_000_000)]
    #     group(tasks).apply_async()

    test_data = [('upbit', 'BTC'), ('upbit', 'ETH'), ('upbit', 'XRP'), ('upbit', 'ADA'), ('upbit', 'DOGE')]
    tasks = [alarm_big_vol_tickers_task.s(item, 5, 1500, 100_000_000) for item in test_data]
    group(tasks).apply_async()

if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    scheduler.add_job(schedule_tasks, 'cron', minute='*')  # 매분 실행
    scheduler.start()

    print("Scheduler started. Press Ctrl+C to exit.")

    # Redis Pub/Sub 구독을 별도의 스레드에서 실행
    pubsub_thread = threading.Thread(target=subscribe_to_redis)
    pubsub_thread.start()

    try:
        while True:
            pass
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
