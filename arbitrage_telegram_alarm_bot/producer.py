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

# 텔레그램을 위한 쓰레드 로컬 객체 생성
thread_local = threading.local()

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
            exchange TEXT NOT NULL,
            ticker TEXT NOT NULL,
            quoteVolume REAL NOT NULL,
            orderId TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            position TEXT NOT NULL,  -- 'long' or 'short'
        )
        """)
        conn.commit()
        conn.close()
    except sqlite3.Error as e:
        logger.info(f"Table creation failed: {e}")

def migrate_table():
    try:
        conn = connect_to_database()
        cursor = conn.cursor()
        
        # Backup existing data
        cursor.execute("CREATE TABLE IF NOT EXISTS order_backup AS SELECT * FROM 'order'")
        
        # Drop existing table
        cursor.execute("DROP TABLE IF EXISTS 'order'")
        
        # Create new table with timestamp, long/short status, and average price
        cursor.execute("""
        CREATE TABLE "order" (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exchange TEXT NOT NULL,
            ticker TEXT NOT NULL,
            quoteVolume REAL NOT NULL,
            orderId TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            position TEXT NOT NULL,  -- 'long' or 'short'
        )
        """)
        
        # Restore data with current timestamp, default position as 'long', and avgPrice as 0
        cursor.execute("""
        INSERT INTO 'order' (exchange, quoteVolume, ticker, orderId, timestamp, position)
        SELECT 'bybit', 0, ticker, orderId, timestamp, 'undefined', 0, 'open'
        FROM order_backup
        """)
        
        # Drop backup table
        cursor.execute("DROP TABLE IF EXISTS order_backup")
        
        conn.commit()
        conn.close()
        logger.info("Table migration completed successfully")
    except sqlite3.Error as e:
        logger.error(f"Table migration failed: {e}")
    
# 데이터베이스 작업 함수
def insert(**kwargs):
    conn = None
    try:
        # 연결 시도
        conn = connect_to_database()
        cursor = conn.cursor()
        
        exchange = kwargs.get('exchange')
        ticker = kwargs.get('ticker')
        orderId = kwargs.get('orderId')
        quoteVolume = kwargs.get('quoteVolume')
        position = kwargs.get('position')

        # 데이터 삽입
        try:
            cursor.execute("INSERT INTO 'order' (exchange, ticker, quoteVolume, orderId, timestamp, position) VALUES (?, ?, ?, ?, datetime('now', '+9 hours'), ?, ?, ?)", (exchange, ticker, quoteVolume, orderId, position))
        except sqlite3.IntegrityError as e:
            logger.info(f"Data insertion error: {e}")
        
        # 커밋
        conn.commit()
        
    except sqlite3.OperationalError as e:
        logger.info(f"Operational error: {e}")
        logger.info("Attempting to reconnect...")
        time.sleep(1)  # 잠시 대기 후 재시도
        insert(**kwargs)
    finally:
        if conn:
            conn.close()

async def upbit_order_handler(data):
    t = TradingDataManager(data.get('exchange'))

    balance, minOrderQty, price = await asyncio.gather(
        t.get_all_position_info(),
        t.get_min_order_qty(data.get('ticker')),
        t.get_single_ticker_price(data.get('ticker'))
    )

    # 업비트는 선물이 없으므로, 양봉일 때에만 진입
    if data.get('candle_type') == '-':
        return
    
    # 10억 이하 거래량은 제외
    if data.get('quote_volume') < 1_000_000_000:
        return
    
    bef_size = None
    bef_avg_price = None
    for pos in balance:
        if pos.get('symbol') == data.get('ticker'):
            bef_size = pos.get('size')
            bef_avg_price = pos.get('avg_buy_price')
            break
    
    logger.info(f"ticker: {data.get('ticker')}, \
                  bef_size: {bef_size}, \
                  bef_avg_price: {bef_avg_price}, \
                  MinOrderQty: {minOrderQty}, \
                  Price: {price}")
    
    # 테스트용 최소주문금액
    minorder_amt = 5100
    
    order = PositionEntryIn(
        symbol=data.get('ticker'),
        side='bid',
        order_type='market',
        qty=minorder_amt,
    )

    return await TradingBroker('upbit').send_order(order)

async def bybit_order_handler(data):
    t = TradingDataManager(data.get('exchange'))

    balance, minOrderQty, price, _ = await asyncio.gather(
        t.get_balance(),
        t.get_min_order_qty(data.get('ticker')),
        t.get_single_ticker_price(data.get('ticker')),
        t.set_leverage(data.get('ticker'), '5')
    )

    pyramiding = False
    avg_price = 0
    size = 0
    # 피라미딩은 가능하나, 다른 방향으로 포지션 진입 불가
    positions = await t.get_all_position()
    for position in positions:
        if position.get('symbol') == data.get('ticker') + 'USDT':
            if position.get('side') == 'Buy' and data.get('candle_type') == '+':
                size = position.get('size')
                avg_price = position.get('avgPrice')
                pyramiding = True
                return False
            elif position.get('side') == 'Sell' and data.get('candle_type') == '-':
                pyramiding = True
                size = position.get('size')
                avg_price = position.get('avgPrice')
                return False
            
    if float(balance) > float(minOrderQty) * float(price) * 2:
        # 최소주문금액이 5USDT가 안되면, 5USDT로 맞춤
        if float(minOrderQty) * float(price) < 5:
            minOrderQty = 5 / float(price)

        # 현재봉이 양봉이면, short진입이므로 TP가 -, SL가 +
        # 피라미딩이면, 기존 포지션의 size와 avg_price를 가져와서 tp, sl을 조정
        if data.get('candle_type') == '+':
            if pyramiding:
                tp = (size * avg_price + minOrderQty * price) / (size + int(minOrderQty*10)) * (1 - 0.02 / 5)
                sl = (size * avg_price + minOrderQty * price) / (size + int(minOrderQty*10)) * (1 + 0.1 / 5)
            else:
                tp = float(price) * (1 - 0.05 / 5)
                sl = float(price) * (1 + 0.1 / 5)
        else:
            if pyramiding:
                tp = (size * avg_price + minOrderQty * price) / (size + int(minOrderQty*10)) * (1 + 0.02 / 5)
                sl = (size * avg_price + minOrderQty * price) / (size + int(minOrderQty*10)) * (1 - 0.1 / 5)
            else:
                tp = float(price) * (1 + 0.05 / 5)
                sl = float(price) * (1 - 0.1 / 5)

        logger.debug(f"ticker: {data.get('ticker')}, Balance: {balance}, MinOrderQty: {minOrderQty}, Price: {price}")

        order = PositionEntryIn(
            symbol=data.get('ticker'),
            side='bid' if data.get('candle_type') == '-' else 'ask',
            order_type='market',
            qty=int(minOrderQty*10),
            tp=tp,
            sl=sl
        )

        logger.info(f"Order: {json.dumps(order.to_dict(), indent=4)}")

    return await TradingBroker('bybit').send_order(order)

async def order_handler(data):
    if data.get('exchange') == 'bybit':
        res = await bybit_order_handler(data)

    if data.get('exchange') == 'upbit':
        res = await upbit_order_handler(data)
    
    if res:
        insert(exchange=data.get('exchange'), \
               ticker=data.get('ticker'), \
               quoteVolume=data.get('quote_volume'), \
               orderId=res, \
               position='long' if data.get('candle_type') == '-' else 'short')
        return True

    return False

async def handle_message_async(**kwargs):
    k = KimpManager()
    await asyncio.gather(
        order_handler(kwargs.get('data1')),
        k.send_telegram(kwargs.get('data2'))
    )

def get_event_loop():
    if not hasattr(thread_local, 'loop'):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        thread_local.loop = loop
    return thread_local.loop

def handle_message(message):
    if message['type'] == 'message':
        data = json.loads(message['data'])
        logger.info(data)

        try:
            loop = get_event_loop()
        
            if data.get('exchange') not in ['bybit', 'upbit']:
                k = KimpManager()
                return loop.run_until_complete(k.send_telegram(message['data']))
        
            loop.run_until_complete(handle_message_async(data1=data, data2=message['data']))
        except RuntimeError as e:   
            logger.info(f"Runtime error: {e}")
        except Exception as e:
            logger.info(f"Error: {e}")
        
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
