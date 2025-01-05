import threading
from celery import Celery, group
import time
import redis
from main import *
import asyncio
from apscheduler.schedulers.background import BackgroundScheduler
from pytz import timezone  # 추가
import sqlite3

logger = logging.getLogger(__name__)

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

# RDBMS 연결 함수
def connect_to_database(db_name="orders.db"):
    try:
        conn = sqlite3.connect(db_name)
        return conn
    except sqlite3.Error as e:
        logger.info(f"Database connection failed: {e}")
        return None

def create_table():
    try:
        conn = connect_to_database()
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS "order" (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            orderId TEXT NOT NULL
        )
        """)
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        logger.info(f"Table creation failed: {e}")
    
# 데이터베이스 작업 함수
def insert(ticker, orderId):
    conn = None
    try:
        # 연결 시도
        conn = connect_to_database()
        cursor = conn.cursor()
        
        # 데이터 삽입
        try:
            cursor.execute("INSERT INTO order (ticker, orderId) VALUES (?, ?)", (ticker, orderId))
        except sqlite3.IntegrityError as e:
            logger.info(f"Data insertion error: {e}")
        
        # 커밋
        conn.commit()
        
    except sqlite3.OperationalError as e:
        logger.info(f"Operational error: {e}")
        logger.info("Attempting to reconnect...")
        time.sleep(1)  # 잠시 대기 후 재시도
        insert()
    finally:
        if conn:
            conn.close()

async def order_handler(data):
    t = TradingDataManager(data.get('exchange'))

    if data.get('exchange') != 'bybit':
        return False

    balance, minOrderQty, price, _ = await asyncio.gather(
        t.get_balance(),
        t.get_min_order_qty(data.get('ticker')),
        t.get_single_ticker_price(data.get('ticker')),
        t.set_leverage(data.get('ticker'), '5')
    )

    # 음봉이면, 매수주문이므로 TP는 높게, SL은 낮게
    if data.get('candle_type') == '-':
        tp = float(price) * (1 + 0.01 / 5)
        sl = float(price) * (1 - 0.01 / 5)
    else:
        tp = float(price) * (1 - 0.01 / 5)
        sl = float(price) * (1 + 0.01 / 5)

    if float(balance) > float(minOrderQty) * float(price) * 2:
        # 최소주문금액이 5USDT가 안되면, 5USDT로 맞춤
        if float(minOrderQty) * float(price) < 5:
            minOrderQty = 5 / float(price)

        order = PositionEntryIn(
            symbol=data.get('ticker'),
            side='bid' if data.get('candle_type') == '-' else 'ask',
            order_type='market',
            qty=int(minOrderQty*10),
            tp=tp,
            sl=sl
        )
        res = await TradingBroker('bybit').send_order(order)

        if res:
            insert(data.get('ticker'), res)
            return True

    return False

def handle_message(message):
    if message['type'] == 'message':
        data = json.loads(message['data'])

        if data.get('exchange') != 'bybit':
            return False
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # 결과 처리 로직 추가
        k = KimpManager()
        try:
            loop = asyncio.get_event_loop()
            loop.run_until_complete(order_handler(data))
            loop.run_until_complete(k.send_telegram(message['data']))
        except RuntimeError as e:
            logger.info(f"Runtime error: {e}")
        except Exception as e:
            logger.info(e)
        finally:
            loop.close()
        
def subscribe_to_redis():
    redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
    pubsub = redis_client.pubsub()
    pubsub.subscribe('big_volume_tickers')

    logger.info("Redis Pub/Sub subscriber started")
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
    create_table()

    kst = timezone('Asia/Seoul')  # 한국 시간대 설정
    scheduler = BackgroundScheduler(timezone=kst)  # 명시적으로 한국 시간대 설정
    scheduler.add_job(schedule_tasks, 'cron', minute='*/5')  # 5분마다 실행
    scheduler.start()

    logger.info(f'Scheduler started: {scheduler.get_jobs()}')

    # Redis Pub/Sub 구독을 별도의 스레드에서 실행
    pubsub_thread = threading.Thread(target=subscribe_to_redis)
    pubsub_thread.start()

    try:
        while True:
            time.sleep(1)  # 대기 시간을 추가하여 CPU 사용량 줄이기
    except (KeyboardInterrupt, SystemExit):
        scheduler.shutdown()
