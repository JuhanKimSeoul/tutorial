import threading
from celery import Celery, group
import time
import redis
from main import *
import asyncio
from apscheduler.schedulers.background import BackgroundScheduler
from pytz import timezone  # 추가

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

async def order_handler(data):
    t = TradingDataManager(data.get('exchange'))

    if t.config.name != 'bybit':
        return False

    balance, minOrderQty, price = await asyncio.gather(
        t.get_balance(),
        t.get_min_order_qty(data.get('ticker')),
        t.get_single_ticker_price(data.get('ticker')),
        t.set_leverage(data.get('ticker'), '2')
    )
    tp = price * (1 + 0.02)
    sl = price * (1 - 0.02)

    if float(balance) > float(minOrderQty) * float(price):
        order = PositionEntryIn(
            symbol=data.get('ticker'),
            side='bid' if data.get('candle_type') == '-' else 'ask',
            order_type='market',
            qty=minOrderQty,
            tp=tp,
            sl=sl
        )
        return await TradingBroker('bybit').send_order(order)
    return False

def handle_message(message):
    if message['type'] == 'message':
        data = json.loads(message['data'])

        if data.get('exchange') == 'bybit':
            asyncio.run(order_handler(data))

        # 결과 처리 로직 추가
        k = KimpManager()
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            loop.run_until_complete(asyncio.gather(
                k.send_telegram(message['data']),
                order_handler(data)
            ))
        except RuntimeError as e:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(asyncio.gather(
                k.send_telegram(message['data']),
                order_handler(data)
            ))
            

def subscribe_to_redis():
    redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
    pubsub = redis_client.pubsub()
    pubsub.subscribe('big_volume_tickers')

    print("Subscribed to big_volume_tickers. Waiting for messages...")
    while True:
        message = pubsub.get_message()
        if message:
            handle_message(message)
        time.sleep(1)  # 대기 시간을 추가하여 CPU 사용량 줄이기

@app.task
def alarm_big_vol_tickers_task(data, multiplier: int, usdt_price: float, binance_threshold: int):
    '''
        data = [(ex, ticker)...]
    '''
    pass

def schedule_tasks():
    k = KimpManager()
    res = asyncio.run(k.get_all_tickers())
    res2 = asyncio.run(UpbitManager().get_single_ticker_price('USDT'))
    usdt_price = res2[0]['trade_price']

    upbit = [ {'exchange' : 'upbit', 'ticker' : ticker} for ticker in res['upbit']]
    bithumb = [ {'exchange' : 'bithumb', 'ticker' : ticker} for ticker in res['bithumb']]
    bybit = [ {'exchange' : 'bybit', 'ticker' : ticker} for ticker in res['bybit']]
    combined = bybit + bithumb + upbit
    union_combined = list({v['ticker']:v for v in combined}.values())
    batch_size = 10
    tasks = []
    for i in range(0, len(union_combined), batch_size):
        batch = union_combined[i:i + batch_size]
        tasks.append(alarm_big_vol_tickers_task.s(batch, 5, usdt_price, 100_000_000))
    group(tasks).apply_async()

if __name__ == "__main__":
    kst = timezone('Asia/Seoul')  # 한국 시간대 설정
    scheduler = BackgroundScheduler(timezone=kst)  # 명시적으로 한국 시간대 설정
    scheduler.add_job(schedule_tasks, 'cron', minute='*/5')  # 5분마다 실행
    scheduler.start()

    print("Scheduler started. Press Ctrl+C to exit.")

    # Redis Pub/Sub 구독을 별도의 스레드에서 실행
    pubsub_thread = threading.Thread(target=subscribe_to_redis)
    pubsub_thread.start()

    try:
        while True:
            time.sleep(1)  # 대기 시간을 추가하여 CPU 사용량 줄이기
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
