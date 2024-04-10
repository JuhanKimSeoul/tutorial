# samplepackage/main.py
import argparse
import asyncio
from datetime import datetime, timezone
from enum import Enum
import json
import time
from schedule import Scheduler
import ccxt
import logging
import pytz
import functools
import requests

'''Constants'''
korean_tz = pytz.timezone('Asia/Seoul')

'''Logger set'''

class Formatter(logging.Formatter):
    """override logging.Formatter to use an aware datetime object"""
    def converter(self, timestamp):
        dt = datetime.fromtimestamp(timestamp)
        return korean_tz.localize(dt)
         
    def formatTime(self, record, datefmt=None):
        dt = self.converter(record.created)
        if datefmt:
            s = dt.strftime(datefmt)
        else:
            try:
                s = dt.isoformat(timespec='milliseconds')
            except TypeError:
                s = dt.isoformat()
        return s

# create logger(root logger)
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# initialize the log file when running again
f = logging.FileHandler('log.txt', 'w')
f.setLevel(logging.INFO)

# create formatter
formatter = Formatter('[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s','%m-%d %H:%M:%S')

# add formatter to ch
ch.setFormatter(formatter)
f.setFormatter(formatter)

# add handlers to logger
logger.addHandler(ch)
logger.addHandler(f)

def debugger(log_prefix=""):
    """decorators for debugging executing time"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                ret = func(*args, **kwargs)
            except Exception as e:
                logging.error(f'{log_prefix} An error occurred: {e}')
            finally:
                elapsed_time = time.time() - start_time
                logging.info(f'{log_prefix} Execution Time : {elapsed_time}')
                logging.debug(f'{log_prefix} ret : {ret}') if ret == None else None
            return ret
        return wrapper
    return decorator

def main():
    commands = {
        'verbose': function_for_verbose,
        # 'monitor' : function_for_monitor
        # Add more commands here
    }

    parser = argparse.ArgumentParser()
    for command in commands:
        parser.add_argument('--' + command, action='store_true')
    args = parser.parse_args()

    for command, function in commands.items():
        if getattr(args, command):
            function(getattr(args, command))

def function_for_verbose(verbose=False):
    if verbose:
        print("This is a verbose message.")
    else:
        print("Hello from SamplePackage!")

class CandleType(str, Enum):
    '''Enum to represent candle type'''
    SPOT = 'spot'
    FUTURES = 'futures'

binance_candletype_host_mapping_dict = {
    CandleType.SPOT: 'https://api.binance.com/api/v3',
    CandleType.FUTURES: 'https://fapi.binance.com/fapi/v1'
}

@debugger("fetching all ticker from binance api...")
def fetch_all_ticker_binance(candle_type: CandleType)->list:
    url = binance_candletype_host_mapping_dict[candle_type] + '/exchangeInfo'
    response = requests.get(url)

    usdt_tickers = [x['symbol'] for x in response.json()['symbols'] if x['symbol'].endswith('USDT')]

    logger.info(f'X-MBX-USED-WEIGHT : {response.headers.get("X-MBX-USED-WEIGHT")}')
    return usdt_tickers

@debugger("fetching windowsize data from binance api...")
def fetch_all_ticker_binance_by_windowsize(candle_type: CandleType, params: dict):
    '''
        params should be like below
        {
            'symbols': ['BTCUSDT', 'ETHUSDT'...]
            'windowSize': 15m
        }
    '''
    url = binance_candletype_host_mapping_dict[candle_type] + '/ticker?' + 'symbols=[' + '"' + '","'.join(params["symbols"]) + '"' + ']&windowSize=' + params['windowSize']
    response = requests.get(url)
    logger.error(response.json() if response.status_code != 200 else response.status_code)
    logger.debug(response.json())

    insert_tickers(response.json())

    logger.info(f'X-MBX-USED-WEIGHT : {response.headers.get("X-MBX-USED-WEIGHT")}')
    return None

from sqlalchemy import DateTime, create_engine, Column, Integer, String, Float, BigInteger
from sqlalchemy.orm import sessionmaker, declarative_base

Base = declarative_base()

class Ticker(Base):
    __tablename__ = 'tickers'

    symbol = Column(String(50), primary_key=True)
    closeTime = Column(DateTime, primary_key=True)
    openTime = Column(DateTime)
    priceChange = Column(Float)
    priceChangePercent = Column(Float)
    weightedAvgPrice = Column(Float)
    openPrice = Column(Float)
    highPrice = Column(Float)
    lowPrice = Column(Float)
    lastPrice = Column(Float)
    volume = Column(Float)
    quoteVolume = Column(Float)
    firstId = Column(BigInteger)
    lastId = Column(BigInteger)
    count = Column(Integer)

engine = create_engine('mysql+mysqldb://testuser:1111@localhost:3306/testdb', echo=True)
Session = sessionmaker(bind=engine)

# Create all tables in the database
Base.metadata.create_all(engine)

def insert_tickers(tickers):
    session = Session()

    for ticker in tickers:
        # Convert epoch time to UTC time
        openTime = datetime.fromtimestamp(ticker['openTime'] / 1000, tz=timezone.utc)
        closeTime = datetime.fromtimestamp(ticker['closeTime'] / 1000, tz=timezone.utc)

        print(openTime)
        # Convert UTC time to Korean time
        openTime = openTime.astimezone(korean_tz)
        closeTime = closeTime.astimezone(korean_tz)
        print(openTime)

        ticker_obj = Ticker(
            symbol=ticker['symbol'],
            closeTime=ticker['closeTime'],
            openTime=ticker['openTime'],
            priceChange=ticker['priceChange'],
            priceChangePercent=ticker['priceChangePercent'],
            weightedAvgPrice=ticker['weightedAvgPrice'],
            openPrice=ticker['openPrice'],
            highPrice=ticker['highPrice'],
            lowPrice=ticker['lowPrice'],
            lastPrice=ticker['lastPrice'],
            volume=ticker['volume'],
            quoteVolume=ticker['quoteVolume'],
            firstId=ticker['firstId'],
            lastId=ticker['lastId'],
            count=ticker['count']
        )
        session.add(ticker_obj)

    session.commit()

def interval_string_to_int(interval: str)->int:
    if interval[-1:] == 'm':
        return int(interval[:-1]) * 60
    elif interval[-1:] == 'h':
        return int(interval[:-1]) * 60 * 60 
    else:
        return int(interval[:-1])

def test(interval: str):
    '''
    test function
    '''
    while True:
        tickers:list = list(set(fetch_all_ticker_binance(CandleType.SPOT)))
        ticker_batches = [tickers[i:i + 100] for i in range(0, len(tickers), 100)]
        
        for ticker_batch in ticker_batches:
            fetch_all_ticker_binance_by_windowsize(CandleType.SPOT, {'symbols': ticker_batch, 'windowSize': interval})
        
        time.sleep(interval_string_to_int(interval))
        
if __name__ == "__main__":
    # asyncio.run(function_for_monitor())
    test('15m')