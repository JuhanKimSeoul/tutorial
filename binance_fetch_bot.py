# samplepackage/main.py
import argparse
import asyncio
from datetime import datetime, timedelta, timezone
from enum import Enum
import json
import time
from schedule import Scheduler
import ccxt
import logging
import pytz
import functools
import requests
import io
from os import getenv
from dotenv import load_dotenv
from aiogram import Bot, types
from aiohttp import ClientError
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import asyncio
from sqlalchemy import DateTime, create_engine, Column, Integer, String, Float, BigInteger, func
from sqlalchemy.orm import sessionmaker, declarative_base
from typing import List, Dict, Any
import traceback

#region Logger
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

#logging.basicConfig(level=logging.INFO)

# diable other loggers for develop phase 
sql_logger = logging.getLogger('sqlalchemy.engine.Engine')
sql_logger.disabled = True

# create root logger
logger = logging.getLogger()
logger.setLevel(logging.INFO) # default level : WARNING

# create console handler and set level to error
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# initialize the log file when running again
fh = logging.FileHandler('log.txt', 'w')
fh.setLevel(logging.INFO)

# create formatter and apply this only to file handler
formatter = Formatter('[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s','%m-%d %H:%M:%S')

# add formatter to ch
ch.setFormatter(formatter)
fh.setFormatter(formatter)

# add handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)
#endregion

#region Database
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
#endregion

#region Constants
korean_tz = pytz.timezone('Asia/Seoul')
load_dotenv()
BOT_ID = getenv('TELEGRAM_BOT_TOKEN')
CHAT_ID = getenv('TELEGRAM_CHAT_ID')

assert BOT_ID != None, "Invalid Bot ID"
assert CHAT_ID != None, "Invalid Chat ID"

# Create a Telegram bot
bot = Bot(token=BOT_ID)

class CandleType(str, Enum):
    '''Enum to represent candle type'''
    SPOT = 'spot'
    FUTURES = 'futures'

binance_candletype_host_mapping_dict = {
    CandleType.SPOT: 'https://api.binance.com/api/v3',
    CandleType.FUTURES: 'https://fapi.binance.com/fapi/v1'
}
#endregion

#region Decorators
def debugger(log_prefix=""):
    """decorators for debugging executing time"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            ret = None
            try:
                ret = func(*args, **kwargs)
            except Exception as e:
                logging.error(f'{log_prefix} An error occurred: {e}')
            finally:
                elapsed_time = time.time() - start_time
                logging.info(f'{log_prefix} Execution Time : {elapsed_time}')
                if ret is not None:
                    logging.debug(f'{log_prefix} ret : {ret}')
                    # if type(ret) == pd.DataFrame:
                    #     logging.debug(f'{log_prefix} ret : {ret.head()}')
                    # else:
            return ret
        return wrapper
    return decorator
#endregion

#region ArgumentParser
# def main():
#     commands = {
#         'verbose': function_for_verbose,
#         # 'monitor' : function_for_monitor
#         # Add more commands here
#     }

#     parser = argparse.ArgumentParser()
#     for command in commands:
#         parser.add_argument('--' + command, action='store_true')
#     args = parser.parse_args()

#     for command, function in commands.items():
#         if getattr(args, command):
#             function(getattr(args, command))

# def function_for_verbose(verbose=False):
#     if verbose:
#         print("This is a verbose message.")
#     else:
#         print("Hello from SamplePackage!")
#endregion

#region API
@debugger("fetching all ticker from binance api...")
def fetch_all_ticker_binance(candle_type: CandleType)->list:
    url = binance_candletype_host_mapping_dict[candle_type] + '/exchangeInfo'
    response = requests.get(url)

    usdt_tickers = [x['symbol'] for x in response.json()['symbols'] if x['symbol'].endswith('USDT')]

    logger.info(f'X-MBX-USED-WEIGHT : {response.headers.get("X-MBX-USED-WEIGHT")}')
    return usdt_tickers

@debugger("fetching windowsize data from binance api...")
def fetch_all_ticker_binance_by_windowsize(candle_type: CandleType, params: dict)->List[Dict[str, Any]]:
    '''
        params should be like below
        {
            'symbols': ['BTCUSDT', 'ETHUSDT'...]
            'windowSize': 15m
        }
    '''
    url = binance_candletype_host_mapping_dict[candle_type] + '/ticker?symbols=[' + '"' + '","'.join(params["symbols"]) + '"' + ']&windowSize=' + params['windowSize']
    response = requests.get(url)
    if response.status_code == 200:
        logger.info(response.status_code)
        logger.debug(response.json())
    else:
        logger.error(response.json())

    logger.info(f'X-MBX-USED-WEIGHT : {response.headers.get("X-MBX-USED-WEIGHT")}')
    return response.json()

@debugger("fetching kline data from binance api...")
def fetch_kline_binance(symbol: str, interval: str, limit: int = 500):
    url = f'https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}'
    response = requests.get(url)
    if response.status_code == 200:
        logger.info(response.status_code)
        logger.debug(response.json())
    else:
        logger.error(response.json())

    logger.info(f'X-MBX-USED-WEIGHT : {response.headers.get("X-MBX-USED-WEIGHT")}')
    return response.json()

#endregion

#region Model
@debugger("inserting tickers into database...")
def insert_ticker(ticker: Dict[str, Any]):
    with Session() as session:
        # Convert epoch time to UTC time
        openTime = datetime.fromtimestamp(ticker['openTime'] / 1000, tz=timezone.utc)
        closeTime = datetime.fromtimestamp(ticker['closeTime'] / 1000, tz=timezone.utc)

        # Convert UTC time to Korean time
        openTime = openTime.astimezone(korean_tz)
        closeTime = closeTime.astimezone(korean_tz)

        ticker_obj = Ticker(
            symbol=ticker['symbol'],
            closeTime=closeTime,
            openTime=openTime,
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

        try:
            session.commit()
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            session.rollback()
            raise e

@debugger("getting average quote volume...")
def get_average_quote_volume(standard:str)->List:

    with Session() as session:
        if standard[-1:] == 'd':
            st_dt = datetime.now() - timedelta(days=int(standard[:-1]))
        elif standard[-1:] == 'h':
            st_dt = datetime.now() - timedelta(hours=int(standard[:-1]))
        elif standard[-1:] == 'm':
            st_dt = datetime.now() - timedelta(minutes=int(standard[:-1]))

        assert st_dt != None, "Invalid standard"

        # Query the average quoteVolume for the past day
        avg_quote_volume = session.query(
            Ticker.symbol,
            func.avg(Ticker.quoteVolume)
        ).filter(
            Ticker.openTime >= st_dt
        ).group_by(
            Ticker.symbol
        ).all()
        
    return avg_quote_volume

@debugger("getting tickers as dataframe...")
def get_tickers_as_dataframe(symbol: str):
    with Session() as session:
        ticker = session.query(Ticker).filter(Ticker.symbol == symbol).all()
    
    return pd.DataFrame([t.__dict__ for t in ticker])
#endregion

#region Util Functions
def interval_string_to_int(interval: str)->int:
    if interval[-1:] == 'm':
        return int(interval[:-1]) * 60
    elif interval[-1:] == 'h':
        return int(interval[:-1]) * 60 * 60 
    else:
        return int(interval[:-1])
    
@debugger("comparing quote volume with average...")
async def compare_quote_volume_with_average(ticker, standard:str, rate:float)->bool:
    # Convert the average quoteVolumes to a dictionary for easy lookup
    avg_quote_volumes_dict = {symbol: avg for symbol, avg in get_average_quote_volume(standard)}

    symbol = ticker['symbol']
    quoteVolume = float(ticker['quoteVolume'])

    # Get the average quoteVolume for this symbol
    avg_quote_volume = float(avg_quote_volumes_dict.get(symbol, 0))  # Add default value of 0

    if avg_quote_volume == 0:
        logger.warning(f"No average quoteVolume found for symbol {symbol}")
        return False

    # main logic is here!
    # Compare the ticker's quoteVolume with the average and quoteVolume is greater than 100,000$
    if quoteVolume >= avg_quote_volume * rate and quoteVolume > 100_000:
        logger.info(f"symbol : {symbol}, quoteVolume : {quoteVolume}, avg_quote_volume :  {avg_quote_volume}, rate : {rate} ")
        return True
    
    return False
        
async def send_telegram(buffer):
    result = None
    # Send the image to the Telegram chat
    try:
        result = await bot.send_photo(chat_id=CHAT_ID, photo=buffer)
    except ClientError as e: 
        logging.error(f'Connection error: {e}')
    except Exception as e:
        logging.error(f'error info: {e}')
        logging.error(traceback.format_exc())
    return result

@debugger("transforming into dataframe...")
def transform_into_dataframe(*args, data:List[List[Any]])->pd.DataFrame:
    '''
    transform data into dataframe
    ex) transform_into_dataframe((4,'closePrice'), (6,'closeTime'), (10,'quoteVolume'), data=data)
    '''
    extracted_data = []
    for el in data:
        tmp = []
        for arg in args:
            tmp.append(el[arg[0]])
        extracted_data.append(tmp)

    df = pd.DataFrame(extracted_data, columns=[arg[1] for arg in args])

    if 'closeTime' in df.columns:
        df['closeTime'] = pd.to_datetime(df['closeTime'], unit='ms')
        df['closeTime'] = df['closeTime'].dt.tz_localize(pytz.utc)
        df['closeTime'] = df['closeTime'].dt.tz_convert('Asia/Seoul')
        df['closeTime'] = df['closeTime'].dt.tz_localize(None)

    for col in df.columns:
        if 'Price' in col or 'Volume' in col:
            df[col] = df[col].astype(float)

    if 'closePrice' in df.columns:
        df['closePrice'] = df['closePrice'] / df['closePrice'].iloc[0] * 100

    return df
    
@debugger("plot_diagram_into_buffer...")
def plot_diagram_into_buffer(*args, **kwargs):
    '''
    plot one figure with multiple subplots by vertically
    ex) plot_diagram(('closeTime', 'closePrice'), ('closeTime', 'quoteVolume'), BTCUSDT=df1, ETHUSDT=df2)
    '''
    fig, ax = plt.subplots(len(args), 1, figsize=(12, 12))

    i = 0
    for x, y in args:
        for key, dataFrame in kwargs.items():
            # if x == 'closeTime' then format date
            if x == 'closeTime':
                locator = mdates.AutoDateLocator()
                ax[i].xaxis.set_major_locator(locator)
                ax[i].xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))
                # Rotate x-axis labels for better readability
                plt.xticks(rotation=45)

            if (y == 'quoteVolume' or y == 'takerBuyQuoteVolume') and key != 'BTCUSDT':
                ax2 = ax[i].twinx()
                ax2.plot(dataFrame[x], dataFrame[y], label=key, color='r')
                ax2.legend()
            else:
                # Plot the price value
                ax[i].plot(dataFrame[x], dataFrame[y], label=key)

            ax[i].legend()

        i += 1

    # Create a BytesIO buffer to save the Matplotlib plot image in memory
    buffer = io.BytesIO()
    plt.savefig(buffer)

    # moves the file pointer to the beginning of the buffer
    buffer.seek(0)

    # upload from buffer
    buf_trs = types.BufferedInputFile(buffer.getvalue(), filename=f'plot.png')

    # plt.show()
    plt.close()
    return buf_trs

@debugger("plotting ticker quote volume graph...")
async def plot_ticker_quote_volume_graph(symbol:str, **userData):
    try:
        df = get_tickers_as_dataframe(symbol)
        df['closeTime'] = pd.to_datetime(df['closeTime'])
        # autodatelocator will interpret the timezone unless setting localization.
        df['closeTime'] = df['closeTime'].dt.tz_localize(None) 

        fig, ax = plt.subplots()

        # Plot the data
        ax.plot(df['closeTime'], df['quoteVolume'], label=symbol)
        
        for key, dataFrame in userData.items():
            ax.plot(dataFrame['closeTime'], dataFrame['quoteVolume'])

        # format date 
        locator = mdates.AutoDateLocator()
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(locator))

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.title(f'{symbol} QuoteVolume')
        plt.xlabel('Timestamp')
        plt.ylabel('Volume')

        # Display legend
        plt.legend()
        
        # Create a BytesIO buffer to save the Matplotlib plot image in memory
        buffer = io.BytesIO()
        plt.savefig(buffer)

        # moves the file pointer to the beginning of the buffer
        buffer.seek(0)

        # upload from buffer
        buf_trs = types.BufferedInputFile(buffer.getvalue(), filename=f'{symbol}.png')
        
        res = await send_telegram(buf_trs)

        plt.close()

    except Exception as e:
        logging.error(f'error info: {e}')
        logging.error(traceback.format_exc())
#endregion

#region Strategy Functions
async def strategy1(fetch_interval: str, compare_interval: str, rate: float):
    '''
    strategy1 function
    '''
    while True:
        tickers:list = list(set(fetch_all_ticker_binance(CandleType.SPOT)))
        ticker_batches = [tickers[i:i + 100] for i in range(0, len(tickers), 100)]
        
        for ticker_batch in ticker_batches:
            res = fetch_all_ticker_binance_by_windowsize(CandleType.SPOT, {'symbols': ticker_batch, 'windowSize': fetch_interval})

            for ticker in res:
                alarm_yn = await compare_quote_volume_with_average(ticker, compare_interval, rate)
                insert_ticker(ticker)
                
                if alarm_yn:
                    df = transform_into_dataframe((4,'closePrice'), (6,'closeTime'), (7,'quoteVolume'), (10,'takerBuyQuoteVolume'), data=fetch_kline_binance(ticker['symbol'], '5m', 500))
                    if ticker['symbol'] == 'BTCUSDT':
                        comp_symbol = 'ETHUSDT'
                    else:
                        comp_symbol = 'BTCUSDT'
                    comp_df = transform_into_dataframe((4,'closePrice'), (6,'closeTime'), (10,'quoteVolume'), (10,'takerBuyQuoteVolume'), data=fetch_kline_binance(comp_symbol, '5m', 500))
                    
                    if ticker['symbol'] == 'BTCUSDT':
                        buf = plot_diagram_into_buffer(('closeTime', 'closePrice'), ('closeTime', 'quoteVolume'), ('closeTime', 'takerBuyQuoteVolume'), **{'ETHUSDT': comp_df, ticker['symbol']: df})
                    else:
                        buf = plot_diagram_into_buffer(('closeTime', 'closePrice'), ('closeTime', 'quoteVolume'), ('closeTime', 'takerBuyQuoteVolume'), **{'BTCUSDT': comp_df, ticker['symbol']: df})
                    await send_telegram(buf)
                    # await plot_ticker_quote_volume_graph(ticker['symbol'])

        time.sleep(interval_string_to_int(fetch_interval))
#endregion

#region UnitTest
@debugger("UnitTest...")
async def UnitTest():
    #await plot_two_graph_test('BTCUSDT', 'ETHUSDT')
    res = fetch_kline_binance('BTCUSDT', '5m', 500)
    data = [[kline[4], kline[6], kline[10]] for kline in res]

    df = pd.DataFrame(data, columns=['closePrice', 'closeTime', 'quoteVolume'])
    df['closeTime'] = pd.to_datetime(df['closeTime'], unit='ms')
    df['closeTime'] = df['closeTime'].dt.tz_localize(pytz.utc)
    df['closeTime'] = df['closeTime'].dt.tz_convert('Asia/Seoul')
    df['closeTime'] = df['closeTime'].dt.tz_localize(None)
    df['quoteVolume'] = df['quoteVolume'].astype(float)
    df['closePrice'] = df['closePrice'].astype(float)
    # Normalize the 'quoteVolume' series to its initial value
    df['closePrice'] = df['closePrice'] / df['closePrice'].iloc[0] * 100
    await plot_ticker_quote_volume_graph('BTCUSDT', userData1=df)

def UnitTest2():
    res = fetch_kline_binance('BTCUSDT', '5m', 500)
    data = [[kline[4], kline[6], kline[10]] for kline in res]

    df = pd.DataFrame(data, columns=['closePrice', 'closeTime', 'quoteVolume'])
    df['closeTime'] = pd.to_datetime(df['closeTime'], unit='ms')
    df['closeTime'] = df['closeTime'].dt.tz_localize(pytz.utc)
    df['closeTime'] = df['closeTime'].dt.tz_convert('Asia/Seoul')
    df['closeTime'] = df['closeTime'].dt.tz_localize(None)
    df['quoteVolume'] = df['quoteVolume'].astype(float)
    df['closePrice'] = df['closePrice'].astype(float)
    # Normalize the 'quoteVolume' series to its initial value
    df['closePrice'] = df['closePrice'] / df['closePrice'].iloc[0] * 100

    res2 = fetch_kline_binance('ETHUSDT', '5m', 500)
    data2 = [[kline[4], kline[6], kline[10]] for kline in res2]

    df2 = pd.DataFrame(data2, columns=['closePrice', 'closeTime', 'quoteVolume'])
    df2['closeTime'] = pd.to_datetime(df2['closeTime'], unit='ms')
    df2['closeTime'] = df2['closeTime'].dt.tz_localize(pytz.utc)
    df2['closeTime'] = df2['closeTime'].dt.tz_convert('Asia/Seoul')
    df2['closeTime'] = df2['closeTime'].dt.tz_localize(None)
    df2['quoteVolume'] = df2['quoteVolume'].astype(float)
    df2['closePrice'] = df2['closePrice'].astype(float)
    # Normalize the 'quoteVolume' series to its initial value
    df2['closePrice'] = df2['closePrice'] / df2['closePrice'].iloc[0] * 100

async def UnitTest3():
    res = fetch_all_ticker_binance_by_windowsize(CandleType.SPOT, {'symbols': ['HIFIUSDT'], 'windowSize': '5m'})

    for ticker in res:
        alarm_yn = await compare_quote_volume_with_average(ticker, '1d', 5)
        insert_ticker(ticker)
        # if alarm_yn:
        df = transform_into_dataframe((4,'closePrice'), (6,'closeTime'), (7,'quoteVolume'), (10,'takerBuyQuoteVolume'), data=fetch_kline_binance(ticker['symbol'], '5m', 500))
        if ticker['symbol'] == 'BTCUSDT':
            comp_symbol = 'ETHUSDT'
        else:
            comp_symbol = 'BTCUSDT'
        comp_df = transform_into_dataframe((4,'closePrice'), (6,'closeTime'), (10,'quoteVolume'), (10,'takerBuyQuoteVolume'), data=fetch_kline_binance(comp_symbol, '5m', 500))
        
        if ticker['symbol'] == 'BTCUSDT':
            buf = plot_diagram_into_buffer(('closeTime', 'closePrice'), ('closeTime', 'quoteVolume'), ('closeTime', 'takerBuyQuoteVolume'), **{'ETHUSDT': comp_df, ticker['symbol']: df})
        else:
            buf = plot_diagram_into_buffer(('closeTime', 'closePrice'), ('closeTime', 'quoteVolume'), ('closeTime', 'takerBuyQuoteVolume'), **{'BTCUSDT': comp_df, ticker['symbol']: df})
        await send_telegram(buf)
#endregion

async def main():
    await strategy1('5m', '1d', 5)

if __name__ == "__main__":
    asyncio.run(main())
    # asyncio.run(UnitTest3())
    # UnitTest2()    
    