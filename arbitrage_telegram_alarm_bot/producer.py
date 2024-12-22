from celery import Celery, group
import time
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

@app.task
def celery_monitor_big_volume_batch(data, multiplier: int, usdt_price: float, binance_threshold: int):
    '''
        data = [(ex, ticker)...]
    '''
    pass

def schedule_tasks():
    k = KimpManager()
    res = asyncio.run(k.get_all_tickers())

    upbit = res['upbit']
    bithumb = res['bithumb']
    combined_tickers = set(res['upbit']).union(set(res['bithumb']))
    batch_size = 30
    for i in range(0, len(combined_tickers), batch_size):
        batch = list(combined_tickers)[i:i + batch_size]
        data = [('upbit' if ticker in upbit else 'bithumb', ticker) for ticker in batch]
        tasks = [celery_monitor_big_volume_batch(data, 5, 1500, 100_000_000)]
        group(tasks).apply_async()

if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    scheduler.add_job(schedule_tasks, 'cron', minute='*')  # 매분 실행
    scheduler.start()

    print("Scheduler started. Press Ctrl+C to exit.")

    try:
        while True:
            pass
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
