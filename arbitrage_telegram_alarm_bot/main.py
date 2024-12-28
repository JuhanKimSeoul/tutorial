import asyncio
from asyncio import Event
import json
import pandas as pd
import matplotlib.pyplot as plt
from constants import *
import ccxt.async_support as ccxt
import jwt
import hashlib
import os
import requests
import uuid
import time
import aiohttp
from urllib.parse import urlencode, unquote
from dotenv import load_dotenv
import hmac
import logging
import traceback
from aiogram import Bot, types
from aiogram.enums import ParseMode
import io
import re
import itertools
from telegram import ReplyKeyboardMarkup, KeyboardButton, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler, CallbackQueryHandler, CallbackContext
from functools import wraps
from itertools import product
from bs4 import BeautifulSoup
from abc import ABC, abstractmethod
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import sqlite3
import copy
from cachetools import TTLCache

cache = TTLCache(maxsize=100, ttl=300)

load_dotenv('dev.env')

upbit_access_key = os.getenv('UPBIT_API_KEY')
upbit_secret_key = os.getenv('UPBIT_SECRET_KEY')

bithumb_access_key = os.getenv('BITHUMB_API_KEY')
bithumb_secret_key = os.getenv('BITHUMB_SECRET_KEY')

bybit_access_key = os.getenv('BYBIT_SUBACC_API_KEY')
bybit_secret_key = os.getenv('BYBIT_SUBACC_SECRET_KEY')

binance_access_key = os.getenv('BINANCE_API_KEY')
binance_secret_key = os.getenv('BINANCE_SECRET_KEY')

bot_id = os.getenv('TELEGRAM_BOT_TOKEN')
chat_id = os.getenv('TELEGRAM_CHAT_ID')

bot = Bot(token=bot_id)

# Basic configuration for the logging
logging.basicConfig(
    level=logging.INFO,                      # Log level
    format='%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s',  # Format
    datefmt='%Y-%m-%d %H:%M:%S'               # Date format
)

# Create a logger object
logger = logging.getLogger(__name__)

def ttl_cache(cache):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            if key in cache:
                logger.debug(f"Cache hit for key: {key}")
                return cache[key]
            logger.debug(f"Cache miss for key: {key}")
            result = await func(*args, **kwargs)
            cache[key] = result
            return result
        return wrapper
    return decorator

class ExchangeAPIConfig(ABC):
    """A base class to store API-specific configurations for each exchange."""

    @abstractmethod
    def get_params(self, url, **kwargs):
        """Override in subclasses to provide the specific parameters for each endpoint."""
        pass

class UpbitAPIConfig(ExchangeAPIConfig):
    name                = 'upbit'
    server_url          = 'https://api.upbit.com'
    orderbook_url       = "/v1/orderbook"
    kline_url           = "/v1/candles/"
    all_tickers_url     = "/v1/ticker/all"
    single_ticker_url   = "/v1/ticker"
    balance_url         = "/v1/accounts"
    order_url           = "/v1/orders"
    withdraw_info_url   = "/v1/withdraws/chance"
    interval_enum       = [1,3,5,10,15,30,60,240]
    limit               = 200
    fee_rate            = Fees.UPBIT.value
    
    def get_params(self, url, **kwargs):
        if url == self.orderbook_url:
            return {
                'markets': kwargs.get('symbol'),
                'level': kwargs.get('level', 0)
            }
        elif url == self.kline_url:
            return {
                'market': kwargs.get('symbol'),
                'count': kwargs.get('limit', self.limit)
            }
        elif url == self.all_tickers_url:
            return {
                'quote_currencies': kwargs.get('quote_currencies')
            }
        elif url == self.single_ticker_url:
            return {
                'markets': kwargs.get('symbol')
            }
        elif url == self.order_url:
            return {
                'market': kwargs.get('market'),
                'side': kwargs.get('side'),
                'volume': kwargs.get('volume'),
                'price': kwargs.get('price'),
                'ord_type': kwargs.get('ord_type')
            }
        elif url == self.withdraw_info_url:
            return {
                'currency': kwargs.get('currency'),
                'net_type': kwargs.get('net_type')
            }    
        else:
            return {}
    
    def get_kline_endpoint(self, interval):
        if int(interval[:-1]) not in self.interval_enum:
            raise ValueError

        endpoint = '/v1/candles/'

        if interval[-1] == 'h':
            endpoint += 'minutes/'
            interval = int(interval[:-1]) * 60
            endpoint += str(interval)

        elif interval[-1] == 'm':
            endpoint += 'minutes/'
            interval = int(interval[:-1])
            endpoint += str(interval)

        elif interval[-1] == 'd':
            endpoint += 'days'

        elif interval[-1] == 'w':  
            endpoint += 'weeks'
        
        return endpoint

class BithumbAPIConfig(ExchangeAPIConfig):
    name                = 'bithumb'
    server_url          = 'https://api.bithumb.com'
    orderbook_url       = "/v1/orderbook"
    kline_url           = "/v1/candles/"
    all_tickers_url     = "/v1/ticker"
    single_ticker_url   = "/v1/ticker"
    balance_url         = "/v1/accounts"
    order_url           = "/v1/orders"
    withdraw_info_url   = "/v1/withdraws/chance"  
    interval_enum       = [1,3,5,10,15,30,60,240]
    limit               = 200
    fee_rate            = Fees.BITHUMB.value

    def get_params(self, url, **kwargs):
        if url == self.orderbook_url:
            return {
                'markets': kwargs.get('symbol')
            }
        elif url == self.kline_url:
            return {
                'market': kwargs.get('symbol'),
                'count': kwargs.get('limit', self.limit)
            }
        elif url == self.all_tickers_url:
            return {
                'markets': kwargs.get('markets')
            }
        elif url == self.single_ticker_url:
            return {
                'markets': kwargs.get('markets')
            }
        elif url == self.order_url:
            return {
                'market': f"KRW-{kwargs.get('market')}",
                'side': kwargs.get('side'),
                'volume': kwargs.get('volume'),
                'price': kwargs.get('price'),
                'ord_type': kwargs.get('ord_type')
            }
        elif url == self.withdraw_info_url:
            return {
                'currency': kwargs.get('currency'),
                'net_type': kwargs.get('net_type')
            }
        else:
            return {}

    def get_kline_endpoint(self, interval):
        if int(interval[:-1]) not in self.interval_enum:
            raise ValueError

        endpoint = '/v1/candles/'

        if interval[-1] == 'h':
            endpoint += 'minutes/'
            interval = int(interval[:-1]) * 60
            endpoint += str(interval)

        elif interval[-1] == 'm':
            endpoint += 'minutes/'
            interval = int(interval[:-1])
            endpoint += str(interval)

        elif interval[-1] == 'd':
            endpoint += 'days'

        elif interval[-1] == 'w':  
            endpoint += 'weeks'
        
        return endpoint

class BybitAPIConfig(ExchangeAPIConfig):
    name                = 'bybit'
    server_url          = 'https://api.bybit.com'
    orderbook_url       = "/v5/market/orderbook"
    kline_url           = "/v5/market/kline"
    all_tickers_url     = "/v5/market/tickers"
    balance_url         = "/v5/asset/transfer/query-account-coins-balance"
    order_url           = "/v5/order/create"
    leverage_set_url    = "/v5/position/set-leverage"
    ticker_info_url     = "/v5/market/instruments-info?category=linear"
    position_info_url   = "/v5/position/list"
    recent_trade_url    = "/v5/market/recent-trade"
    interval_enum       = [1,3,5,15,30,60,120,240,360,720,'D','W','M']
    orderbook_limit     = 500
    kline_limit         = 1000
    fee_rate            = Fees.BYBIT.value

    def get_params(self, url, **kwargs):
        if url == self.orderbook_url:
            return {
                "category": "linear",
                'symbol': kwargs.get('symbol'),
                'limit': self.orderbook_limit
            }
        elif url == self.kline_url:
            # 입력값 검증
            interval = kwargs.get('interval')

            if interval[-1] in ['d', 'w']:
                interval = interval[-1].upper()
            elif interval[-1] == 'h':
                interval = int(interval[:-1]) * 60
            elif interval[-1] == 'm':
                interval = int(interval[:-1])

            if interval not in self.interval_enum:
                raise ValueError

            return {
                'category' : 'linear',
                'symbol': kwargs.get('symbol'),
                'interval': interval,
                'limit': kwargs.get('limit', self.kline_limit)
            }
        elif url == self.all_tickers_url:
            return {
                'category': kwargs.get('category')
            }
        elif url == self.order_url:
            return {
                'category': kwargs.get('category'),
                'symbol': kwargs.get('symbol'),
                'side': kwargs.get('side'),
                'orderType': kwargs.get('orderType'),
                'qty': kwargs.get('qty')
            }
        elif url == self.leverage_set_url:
            return {
                'category': kwargs.get('category'),
                'symbol': kwargs.get('symbol'),
                'buyLeverage': kwargs.get('buyLeverage'),
                'sellLeverage': kwargs.get('sellLeverage')
            }
        elif url == self.ticker_info_url:
            return {
                'symbol': kwargs.get('symbol')
            }
        elif url == self.position_info_url:
            return {
                'symbol': kwargs.get('symbol')
            }
        else:
            return {}

class BinanceAPIConfig(ExchangeAPIConfig):
    name                = 'binance'
    server_url          = 'https://fapi.binance.com'
    orderbook_url       = '/fapi/v1/depth'
    kline_url           = '/fapi/v1/klines'
    all_tickers_url     = '/fapi/v2/ticker/price'
    balance_url         = '/fapi/v2/balance'
    order_url           = '/fapi/v1/order'
    leverage_set_url    = '/fapi/v1/leverage'
    ticker_info_url     = '/fapi/v1/exchangeInfo'
    position_info_url   = '/fapi/v3/positionRisk'
    recent_trade_url    = '/fapi/v1/trades'
    interval_enum       = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
    orderbook_limit     = 1000
    fee_rate            = Fees.BINANCE.value

    def get_params(self, url, **kwargs):
        if url == self.orderbook_url:
            return {
                'symbol': kwargs.get('symbol'),
                'limit': self.orderbook_limit
            }
        elif url == self.kline_url:
            return {
                'symbol': kwargs.get('symbol'),
                'interval': kwargs.get('interval'),
                'limit': kwargs.get('limit', 500)
            }
        elif url == self.all_tickers_url:
            return {}
        elif url == self.balance_url:
            return {
                'recvWindow': kwargs.get('recvWindow', 5000),
                'timestamp': round(time.time() * 1000)
            }
        elif url == self.order_url:
            return {
                'symbol': kwargs.get('symbol'),
                'side': kwargs.get('side'),
                'type': kwargs.get('order_type'),
                'quantity': kwargs.get('quantity'),
                'positionSide': kwargs.get('positionSide'),
                # 'timeInForce': kwargs.get('timeInForce', 'GTC'),
                'recvWindow': kwargs.get('recvWindow', 5000),
                'timestamp': round(time.time() * 1000)
            }
        elif url == self.leverage_set_url:
            return {
                'symbol': kwargs.get('symbol'),
                'leverage': kwargs.get('leverage'),
                'recvWindow': kwargs.get('recvWindow', 5000),
                'timestamp': round(time.time() * 1000)
            }
        elif url == self.ticker_info_url:
            return {}
        elif url == self.position_info_url:
            return {
                'recvWindow': kwargs.get('recvWindow', 5000),
                'timestamp': round(time.time() * 1000)
            }
        else:
            return {}

class ExchangeManager(ABC):
    def __init__(self):
        self.config = None

    @abstractmethod
    def get_config(self):
        """Factory method to retrieve the appropriate configuration based on the exchange."""
        pass
    
    @abstractmethod
    def ticker_mapper(self, symbol):
        pass
        
    async def request(self, method, endpoint, headers: dict = None, params: dict = None, contentType: str = 'json', retries: int = 3):
        try:
            async with aiohttp.ClientSession() as session:
                if method == 'get':
                    async with session.get(self.config.server_url + endpoint, params=params, headers=headers) as response:
                        if response.status == 429:
                            await self.handle_rate_limit(response)
                            return await self.request(method, endpoint, headers, params, contentType)  # Retry the request
                        return await response.json()
                elif method == 'post':
                    if contentType != 'json':
                        async with session.post(self.config.server_url + endpoint, data=params, headers=headers) as response:
                            if response.status == 429:
                                await self.handle_rate_limit(response)
                                return await self.request(method, endpoint, headers, params, contentType)  # Retry the request
                            return await response.json()
                    else:
                        async with session.post(self.config.server_url + endpoint, json=params, headers=headers) as response:
                            if response.status == 429:
                                await self.handle_rate_limit(response)
                                return await self.request(method, endpoint, headers, params, contentType)  # Retry the request
                            return await response.json()
                elif method == 'delete':
                    pass
        except aiohttp.ClientConnectionError as e:
            logger.error(f'Connection error: {e}')
            if retries > 0:
                logger.info(f'Retrying... ({retries} retries left)')
                await asyncio.sleep(1)  # Wait before retrying
                return await self.request(method, endpoint, headers, params, contentType, retries - 1)
        except asyncio.TimeoutError:
            logger.error('Request timed out')
        except Exception as e:
            logger.error(f'Unexpected error: {e}')

    async def handle_rate_limit(self, response):
        logger.info(response.headers)
        retry_after = response.headers.get('Retry-After')
        if retry_after:
            logger.warning(f"Rate limit exceeded. Retrying after {retry_after} seconds...")
            await asyncio.sleep(int(retry_after))
        else:
            logger.error("Rate limit exceeded, but no 'Retry-After' header found.")
            await asyncio.sleep(1)

    async def get_ticker_withdraw_info(self):
        pass

    @abstractmethod
    async def get_orderbook(self, symbol):
        raise NotImplementedError

    @abstractmethod
    def format_orderbook_response(self, response)->dict:
        pass

    @abstractmethod
    async def get_kline(self, symbol, interval):
        pass 
    
    @abstractmethod
    async def get_all_ticker_price(self):
        pass
        
    @abstractmethod
    async def get_single_ticker_price(self, symbol):
        pass

    @abstractmethod
    async def get_balance(self, symbol):
        pass

    @abstractmethod
    async def post_order(self, PositionEntryIn: PositionEntryIn):
        pass

    def interval_to_second(self, interval:str):
        if interval[-1] == 'm':
            return int(interval[:-1]) * 60
        elif interval[-1] == 'h':
            return int(interval[:-1]) * 3600
        elif interval[-1] == 'd':
            return int(interval[:-1]) * 86400
        elif interval[-1] == 'w':
            return int(interval[:-1]) * 604800

class UpbitManager(ExchangeManager):
    def __init__(self):
        super().__init__()
        self.config = self.get_config()

    def get_config(self):
        return UpbitAPIConfig()

    def ticker_mapper(self, symbol):
        return f'KRW-{symbol}'

    def format_orderbook_response(self, response)->dict:
        ret = {
            'asks' : [],
            'bids' : []
        }
        
        for i in response[0]['orderbook_units']:
            ret['asks'].append([i['ask_price'], i['ask_size']])
            ret['bids'].append([i['bid_price'], i['bid_size']])
    
        ret['asks'] = list(map(lambda x : [float(x[0]), float(x[1])], ret['asks']))
        ret['bids'] = list(map(lambda x : [float(x[0]), float(x[1])], ret['bids']))
        
        return ret

    async def get_all_ticker_price(self):
        endpoint = self.config.all_tickers_url
        params = self.config.get_params(endpoint, quote_currencies='KRW')
        headers = {"accept" : "application/json"}

        return await self.request('get', endpoint, headers, params)

    async def get_single_ticker_price(self, symbol):
        endpoint = self.config.single_ticker_url
        params = self.config.get_params(endpoint, symbol=self.ticker_mapper(symbol))
        headers = {"accept" : "application/json"}

        return await self.request('get', endpoint, headers, params)

    @ttl_cache(cache)
    async def get_cached_single_ticker_price(self, symbol):
        return await self.get_single_ticker_price(symbol)

    async def get_balance(self, symbol):
        payload = {
            'access_key': upbit_access_key,
            'nonce': str(uuid.uuid4()),
        }

        jwt_token = jwt.encode(payload, upbit_secret_key)
        authorization = 'Bearer {}'.format(jwt_token)
        headers = {
            'Authorization': authorization,
        }

        res = await self.request('get', self.config.balance_url, headers)
        logger.info(res)
        return float([ i.get('balance', 0) for i in res if i.get('currency') == symbol][0])

    async def get_kline(self, symbol, interval, limit: int = 200):
        endpoint = self.config.get_kline_endpoint(interval)
        params = self.config.get_params(self.config.kline_url, symbol=self.ticker_mapper(symbol), limit=limit)
        headers = {"accept" : "application/json"}

        return await self.request('get', endpoint, headers, params)

    async def get_orderbook(self, symbol):
        endpoint = self.config.orderbook_url
        params = self.config.get_params(endpoint, symbol=self.ticker_mapper(symbol))
        headers = {"accept" : "application/json"}

        return await self.request('get', endpoint, headers, params)

    async def post_order(self, PositionEntryIn: PositionEntryIn):
        '''
            시장가 매수시 params ex)
                {
                    'market' : 'KRW-XRP',
                    'side' : 'bid',
                    'price' : '10000',
                    'ord_type' : 'price'
                }
            시장가 매도시 params ex)
                {
                    'market' : 'KRW-XRP',
                    'side' : 'ask',
                    'volume' : '10',
                    'ord_type' : 'market'
                }
        '''
        endpoint = self.config.order_url
        params = PositionEntryMapper.to_exchange(PositionEntryIn, self.config.name).not_None_to_dict()

        query_string = unquote(urlencode(params, doseq=True)).encode("utf-8")

        m = hashlib.sha512()
        m.update(query_string)
        query_hash = m.hexdigest()

        payload = {
            'access_key': upbit_access_key,
            'nonce': str(uuid.uuid4()),
            'query_hash': query_hash,
            'query_hash_alg': 'SHA512', 
        }

        jwt_token = jwt.encode(payload, upbit_secret_key)
        authorization = 'Bearer {}'.format(jwt_token)
        headers = {
            'Authorization': authorization,
        }

        return await self.request('post', endpoint, headers, params)

    async def get_ticker_withdraw_info(self, currency, net_type):
        endpoint = self.config.withdraw_info_url
        params = self.config.get_params(endpoint, currency=currency, net_type=net_type)
        query_string = unquote(urlencode(params, doseq=True)).encode("utf-8")

        m = hashlib.sha512()
        m.update(query_string)
        query_hash = m.hexdigest()

        payload = {
            'access_key': upbit_access_key,
            'nonce': str(uuid.uuid4()),
            'query_hash': query_hash,
            'query_hash_alg': 'SHA512',
        }

        jwt_token = jwt.encode(payload, upbit_secret_key)
        authorization = 'Bearer {}'.format(jwt_token)
        headers = {
            'Authorization': authorization,
        }

        return await self.request('get', endpoint, headers, params)

    def get_depwith_status(self, market=None):
        url = "https://upbit.com/service_center/wallet_status?is_retargeting=true&source_caller=ui&shortlink=gd2ruhr7&c=wallet_status&pid=wallet_status&af_xp=custom"
        
        # options = Options()
        # options.headless = True
        # service = Service('/usr/local/bin/chromedriver')  # ChromeDriver 경로 설정
        # driver = webdriver.Chrome(service=service, options=options)

        # WebDriverManager를 사용해 ChromeDriver 설정
        options = Options()
        options.add_argument('--headless')  # 창 없이 실행
        options.add_argument('--no-sandbox')  # 샌드박스 모드 비활성화 (Linux 환경)
        options.add_argument('--disable-blink-features=AutomationControlled')  # 자동화 탐지 방지
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-extensions')
        options.add_argument('--remote-debugging-port=9222')  # 디버깅 포트 추가
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.6778.108 Safari/537.36')

        # ChromeDriverManager를 통해 ChromeDriver 경로 설정
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

        # 웹 페이지 로드
        driver.get(url)

        # 특정 요소가 로드될 때까지 대기
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'table.css-8atqhb'))
        )

        time.sleep(1)

        html = driver.page_source
        print(html)
        
        # 웹 페이지 닫기
        driver.quit()

        soup = BeautifulSoup(html, 'html.parser')

        data = []
        table = soup.find('table', {'class': 'css-8atqhb'})
        rows = table.find_all('tr')

        for row in rows[1:]:
            cols = row.find_all('td')
            deposit_withdraw_status = cols[0].text.strip()

            depwith_status = None
            if deposit_withdraw_status == '입출금':
                depwith_status = 'O/O'
            elif deposit_withdraw_status == '출금가능':
                depwith_status = 'X/O'
            elif deposit_withdraw_status == '입금가능':
                depwith_status = 'O/X'
            elif deposit_withdraw_status == '일시중단':
                depwith_status = 'X/X'
            else:
                depwith_status = 'X/X'

            symbol = cols[1].text.strip()
            network = cols[2].text.strip()
            data.append([symbol, network, depwith_status])

        df = pd.DataFrame(data, columns=['market', 'network', 'status'])
        return df if market is None else df[df['market'] == market]

    async def get_trade_data(self, symbol, window: int = 1000, interval: str = '1m')->list:
        try:
            # Get recent trades from Upbit
            trades = await self.request(
                'get',
                f'/v1/trades/ticks?market=KRW-{symbol}&count={window}',
                headers={"accept": "application/json"}
            )

            if not trades:
                return None

            # filter trade data by interval
            # data is in reverse chronological order
            trades = [trade for trade in trades if trade['timestamp'] > trades[0]['timestamp'] - self.interval_to_second(interval) * 1000]
            return {
                'timestamp': [int(trade['timestamp']) for trade in trades],
                'price': [float(trade['trade_price']) for trade in trades],
                'volume': [float(trade['trade_volume']) for trade in trades],
                'ask_bid': [trade['ask_bid'] for trade in trades]
            }
        
        except Exception as e:
            logger.error(f"Error calculating trade strength: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    async def get_trade_strength(self, symbol: str, window: int = 1000, interval: str = '1m') -> float:
        """
        Calculate trading strength (체결강도) for a given symbol
        체결강도 = (매수 체결량 / 매도 체결량) * 100
        
        Args:
            symbol: Trading pair symbol (e.g. 'BTC')
            window: Number of recent trades to analyze (default 100)
        
        Returns:
            float: Trading strength percentage
        """
        try:
            trades = self.get_trade_data(symbol, window, interval)

            # Calculate buy and sell volumes
            buy_volume = sum(float(trade['trade_volume']) for trade in trades if trade['ask_bid'] == 'BID')
            sell_volume = sum(float(trade['trade_volume']) for trade in trades if trade['ask_bid'] == 'ASK')

            # Calculate strength
            if sell_volume == 0:
                return 0.0
                
            strength = (buy_volume / sell_volume) * 100
            return int(strength)

        except Exception as e:
            logger.error(f"Error calculating trade strength: {str(e)}")
            logger.error(traceback.format_exc())
            return 0.0

    async def get_trade_volume(self, symbol: str, window: int = 1000, interval: str = '1m') -> float:
        """
        Calculate trading volume for a given symbol
        체결량 = 매수 체결량 + 매도 체결량
        
        Args:
            symbol: Trading pair symbol (e.g. 'BTC')
            window: Number of recent trades to analyze (default 100)
        
        Returns:
            float: Trading volume
        """
        try:
            trades = await self.get_trade_data(symbol, window, interval)
            # Calculate total volume
            return sum(float(trade['volume']) for trade in trades)

        except Exception as e:
            logger.error(f"Error calculating trade volume: {str(e)}")
            logger.error(traceback.format_exc())
            return 0.0
        
    async def get_quote_volume(self, symbol: str, window: int = 1000, interval: str = '1m', to: str = None) -> int:
        """
        Calculate trading volume for a given symbol
        체결가격 * 체결량(매수체결량 + 매도체결량)
        
        Args:
            symbol: Trading pair symbol (e.g. 'BTC')
            window: Number of recent trades to analyze (default 100)
        
        Returns:
            float: Trading quote volume
        """
        try:
            trades = await self.get_trade_data(symbol, window, interval)
            # Calculate total volume
            return int(sum(float(trade['price']) * float(trade['volume']) for trade in trades))
        
        except Exception as e:
            logger.error(f"Error calculating quote volume: {str(e)}")
            logger.error(traceback.format_exc())
            return 0

    async def get_all_position_info(self):
        payload = {
            'access_key': upbit_access_key,
            'nonce': str(uuid.uuid4()),
        }

        jwt_token = jwt.encode(payload, upbit_secret_key)
        authorization = 'Bearer {}'.format(jwt_token)
        headers = {
            'Authorization': authorization,
        }

        res = await self.request('get', self.config.balance_url, headers)

        return [{'symbol': item['currency'], \
                 'side': 'bid', \
                 'size': item['balance']} \
                 for item in res]
    
class BithumbManager(ExchangeManager):
    def __init__(self):
        super().__init__()
        self.config = self.get_config()
    
    def get_config(self):
        return BithumbAPIConfig()

    def ticker_mapper(self, symbol):
        return f'KRW-{symbol}'

    def format_orderbook_response(self, response)->dict:
        ret = {
            'asks' : [],
            'bids' : []
        }
        
        for i in response[0]['orderbook_units']:
            ret['asks'].append([i['ask_price'], i['ask_size']])
            ret['bids'].append([i['bid_price'], i['bid_size']])
    
        ret['asks'] = list(map(lambda x : [float(x[0]), float(x[1])], ret['asks']))
        ret['bids'] = list(map(lambda x : [float(x[0]), float(x[1])], ret['bids']))
        
        return ret

    async def get_all_ticker_price(self):
        # fetch all markets first
        tickers = await self.request('get', '/v1/market/all', {"accept" : "application/json"})
        krw_tickers = [ x.get('market') for x in tickers if x.get('market').startswith('KRW-')]

        endpoint = self.config.all_tickers_url
        params = self.config.get_params(endpoint, markets= ", ".join(krw_tickers))
        headers = {"accept" : "application/json"}

        return await self.request('get', endpoint, headers, params)

    async def get_single_ticker_price(self, symbol):
        endpoint = self.config.all_tickers_url
        params = self.config.get_params(endpoint, markets=self.ticker_mapper(symbol))
        headers = {"accept" : "application/json"}

        return await self.request('get', endpoint, headers, params)

    async def get_balance(self, symbol):
        payload = {
            'access_key': bithumb_access_key,
            'nonce': str(uuid.uuid4()),
            'timestamp': round(time.time() * 1000)
        }
        jwt_token = jwt.encode(payload, bithumb_secret_key)
        authorization_token = 'Bearer {}'.format(jwt_token)
        headers = {
            'Authorization': authorization_token
        }

        res = await self.request('get', self.config.balance_url, headers)
        return float([ i.get('balance', 0) for i in res if i.get('currency') == symbol][0])

    async def get_kline(self, symbol, interval, limit: int = 200):
        endpoint = self.config.get_kline_endpoint(interval)
        params = self.config.get_params(self.config.kline_url, symbol=self.ticker_mapper(symbol), limit=limit)
        headers = {"accept" : "application/json"}

        return await self.request('get', endpoint, headers, params)

    async def get_orderbook(self, symbol):
        endpoint = self.config.orderbook_url
        params = self.config.get_params(endpoint, symbol=self.ticker_mapper(symbol))
        headers = {"accept" : "application/json"}

        return await self.request('get', endpoint, headers, params)

    async def post_order(self, PositionEntryIn: PositionEntryIn):
        '''
            시장가 매수시 params ex)
                {
                    'market' : 'KRW-XRP',
                    'side' : 'bid',
                    'price' : '10000',
                    'ord_type' : 'price'
                }
            시장가 매도시 params ex)
                {
                    'market' : 'KRW-XRP',
                    'side' : 'ask',
                    'volume' : '10',
                    'ord_type' : 'market'
                }
        '''
        endpoint = self.config.order_url
        params = PositionEntryMapper.to_exchange(PositionEntryIn, self.config.name).not_None_to_dict()

        # Generate access token
        query = urlencode(params).encode()
        hash = hashlib.sha512()
        hash.update(query)
        query_hash = hash.hexdigest()
        payload = {
            'access_key': bithumb_access_key,
            'nonce': str(uuid.uuid4()),
            'timestamp': round(time.time() * 1000), 
            'query_hash': query_hash,
            'query_hash_alg': 'SHA512',
        }   
        jwt_token = jwt.encode(payload, bithumb_secret_key)
        authorization_token = 'Bearer {}'.format(jwt_token)
        headers = {
            'Authorization': authorization_token,
            'Content-Type': 'application/json'
        }

        return await self.request('post', endpoint, headers, params, 'x-www-form-urlencoded')

    async def get_ticker_withdraw_info(self, currency, net_type):
        endpoint = self.config.withdraw_info_url
        params = self.config.get_params(endpoint, currency=currency, net_type=net_type)

        query_string = urlencode(params).encode()
        m = hashlib.sha512()
        m.update(query_string)
        query_hash = m.hexdigest()

        payload = {
            'access_key': bithumb_access_key,
            'nonce': str(uuid.uuid4()),
            'timestamp': round(time.time() * 1000), 
            'query_hash': query_hash,
            'query_hash_alg': 'SHA512',
        }

        jwt_token = jwt.encode(payload, bithumb_secret_key)
        authorization = 'Bearer {}'.format(jwt_token)
        headers = {
            'Authorization': authorization,
        }

        return await self.request('get', endpoint, headers, params)

    def get_depwith_status(self, market=None):
        url = "https://www.bithumb.com/react/info/inout-condition"
        
        # options = Options()
        # options.headless = True
        # service = Service('/usr/local/bin/chromedriver')  # ChromeDriver 경로 설정
        # driver = webdriver.Chrome(service=service, options=options)

        # WebDriverManager를 사용해 ChromeDriver 설정
        options = Options()
        options.add_argument('--headless')  # 창 없이 실행
        options.add_argument('--no-sandbox')  # 샌드박스 모드 비활성화 (Linux 환경)
        options.add_argument('--disable-blink-features=AutomationControlled')  # 자동화 탐지 방지
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-extensions')
        options.add_argument('--remote-debugging-port=9222')  # 디버깅 포트 추가
        options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.6778.108 Safari/537.36')

        # ChromeDriverManager를 통해 ChromeDriver 경로 설정
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        
        driver.get(url)
        
        # 특정 요소가 로드될 때까지 최대 10초 동안 대기
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, 'table.InoutCondition_inout-condition-table__tWVtr'))
        )

        time.sleep(1)
        
        html = driver.page_source
        driver.quit()
        
        soup = BeautifulSoup(html, 'html.parser')

        data = []
        table = soup.find('table', {'class': 'InoutCondition_inout-condition-table__tWVtr'})
        rows = table.find_all('tr')

        for row in rows[1:]:
            cols = row.find_all('td')

            if 'InoutCondition_text-weight--500__9Vz3g' not in cols[0].get('class', []):
                continue

            symbol = cols[0].text.strip()
             # re 패키지를 사용하여 괄호 안의 내용을 추출
            match = re.search(r'\((.*?)\)', symbol)
            if match:
                symbol = match.group(1)

            depwith_status = None
            network = cols[1].text.strip()
            deposit_status = cols[3].text.strip()
            if deposit_status == '정상':
                deposit_status = True
                depwith_status = 'O'
            else:
                deposit_status = False
                depwith_status = 'X'

            withdraw_status = cols[4].text.strip()
            if withdraw_status == '정상':
                withdraw_status = True
                depwith_status += '/O'
            else:
                withdraw_status = False
                depwith_status += '/X'

            data.append([symbol, network, depwith_status])

        df = pd.DataFrame(data, columns=['market', 'network', 'status'])
        return df if market is None else df[df['market'] == market]
    
    async def get_trade_data(self, symbol, window: int = 1000, interval: str = '1m')->list:
        try:
            # Get recent trades from Upbit
            trades = await self.request(
                'get',
                f'/v1/trades/ticks?market=KRW-{symbol}&count={window}',
                headers={"accept": "application/json"}
            )

            if not trades:
                return None

            # filter trade data by interval
            # data is in reverse chronological order
            trades = [trade for trade in trades if trade['timestamp'] > trades[0]['timestamp'] - self.interval_to_second(interval) * 1000]
            return {
                'timestamp': [int(trade['timestamp']) for trade in trades],
                'price': [float(trade['trade_price']) for trade in trades],
                'volume': [float(trade['trade_volume']) for trade in trades],
                'ask_bid': [trade['ask_bid'] for trade in trades]
            }
        
        except Exception as e:
            logger.error(f"Error calculating trade strength: {str(e)}")
            logger.error(traceback.format_exc())
            return None

    async def get_trade_strength(self, symbol: str, window: int = 1000, interval: str = '1m') -> float:
        """
        Calculate trading strength (체결강도) for a given symbol
        체결강도 = (매수 체결량 / 매도 체결량) * 100
        
        Args:
            symbol: Trading pair symbol (e.g. 'BTC')
            window: Number of recent trades to analyze (default 100)
        
        Returns:
            float: Trading strength percentage
        """
        try:
            trades = await self.get_trade_data(symbol, window, interval)

            # Calculate buy and sell volumes
            buy_volume = sum(float(trade['trade_volume']) for trade in trades if trade['ask_bid'] == 'BID')
            sell_volume = sum(float(trade['trade_volume']) for trade in trades if trade['ask_bid'] == 'ASK')

            # Calculate strength
            if sell_volume == 0:
                return 0.0
                
            strength = (buy_volume / sell_volume) * 100
            return int(strength)

        except Exception as e:
            logger.error(f"Error calculating trade strength: {str(e)}")
            logger.error(traceback.format_exc())
            return 0.0

    async def get_trade_volume(self, symbol:str, window: int = 1000, interval: str = '1m') -> float:
        """
        Calculate trading volume for a given symbol
        체결량 = 매수 체결량 + 매도 체결량
        
        Args:
            symbol: Trading pair symbol (e.g. 'BTC')
            window: Number of recent trades to analyze (default 100)
        
        Returns:
            float: Trading volume
        """
        try:
            trades = await self.get_trade_data(symbol, window, interval)
            # Calculate total volume
            return sum(float(trade['trade_volume']) for trade in trades)

        except Exception as e:
            logger.error(f"Error calculating trade volume: {str(e)}")
            logger.error(traceback.format_exc())
            return 0.0

    async def get_quote_volume(self, symbol: str, window: int = 1000, interval: str = '1m') -> int:  
        """
        Calculate trading volume for a given symbol
        체결가격 * 체결량(매수체결량 + 매도체결량)
        
        Args:
            symbol: Trading pair symbol (e.g. 'BTC')
            window: Number of recent trades to analyze (default 100)
        
        Returns:
            float: Trading quote volume
        """
        try:
            trades = await self.get_trade_data(symbol, window, interval)
            # Calculate total volume
            return int(sum(float(trade['trade_price']) * float(trade['trade_volume']) for trade in trades))
        
        except Exception as e:
            logger.error(f"Error calculating quote volume: {str(e)}")
            logger.error(traceback.format_exc())
            return 0

    async def get_all_position_info(self):
        payload = {
            'access_key': bithumb_access_key,
            'nonce': str(uuid.uuid4()),
            'timestamp': round(time.time() * 1000)
        }
        jwt_token = jwt.encode(payload, bithumb_secret_key)
        authorization_token = 'Bearer {}'.format(jwt_token)
        headers = {
            'Authorization': authorization_token
        }

        res = await self.request('get', self.config.balance_url, headers)

        return [{'symbol': item['currency'], \
                 'side': 'bid', \
                 'size': item['balance']} \
                 for item in res]
    
class BybitManager(ExchangeManager):
    def __init__(self):
        super().__init__()
        self.config = self.get_config()
    
    def get_config(self):
        return BybitAPIConfig()

    def ticker_mapper(self, symbol):   
        if symbol in BybitSymbols.__members__:
            return BybitSymbols[symbol].value + 'USDT'
        return f'{symbol}USDT'

    def format_orderbook_response(self, response)->dict:
        ret = {
            'asks' : response['result']['a'],
            'bids' : response['result']['b']
        }
        
        ret['asks'] = list(map(lambda x : [float(x[0]), float(x[1])], ret['asks']))
        ret['bids'] = list(map(lambda x : [float(x[0]), float(x[1])], ret['bids']))
        
        return ret

    async def get_all_ticker_price(self):
        endpoint = self.config.all_tickers_url
        params = self.config.get_params(endpoint, category='linear')
        headers = {"accept" : "application/json"}

        return await self.request('get', endpoint, headers, params)

    async def get_balance(self, symbol):
        timestamp = str(int(time.time() * 1000))  # Current timestamp in milliseconds
        recv_window = '5000'

        if symbol == 'USDT':
            params = {
                'accountType': 'UNIFIED',
                'coin': symbol
            }

            endpoint = self.config.balance_url

        else:
            params = {
                'category': 'linear',
                'symbol': self.ticker_mapper(symbol)
            }

            endpoint = self.config.position_info_url

        param_str = '&'.join(f'{key}={value}' for key, value in params.items())
        pre_sign = f'{timestamp}{bybit_access_key}{recv_window}{param_str}'

        signature = hmac.new(
            bybit_secret_key.encode('utf-8'),
            pre_sign.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        # Prepare headers
        headers = {
            'X-BAPI-API-KEY': bybit_access_key,
            'X-BAPI-SIGN': signature,
            'X-BAPI-TIMESTAMP': timestamp,
            'X-BAPI-RECV-WINDOW': recv_window
        }

        res = await self.request('get', endpoint, headers, params)
        if symbol == 'USDT':
            if res['retMsg'] == 'success':
                return res['result']['balance'][0]['transferBalance']
        else:
            if res['retMsg'] == 'OK':
                return res['result']['list'][0]['size']

    async def post_order(self, PositionEntryIn: PositionEntryIn):
        '''
            시장가 매수시 params ex)
                payload = {
                    "category": "linear",
                    "symbol": "BTCUSDT",
                    "side": "Buy",
                    "orderType": "Market",
                    "qty": "1",
                }

            시장가 매도시 params ex)
                payload = {
                    "category": "linear",
                    "symbol": "BTCUSDT",
                    "side": "Sell",
                    "orderType": "Market",
                    "qty": "1",
                }

            스탑로스 시장가 매수시 params ex)
                payload = {
                    "category": "linear",
                    "symbol": "BTCUSDT",
                    "side": "Buy",
                    "orderType": "Market",
                    "qty": "1",
                    "tpslMode": "Full",
                    "tpOrderType": "Market", # default
                    "slOrderType": "Market", # default
                    "slLimitPrice": "10000",
                }
        ''' 
        endpoint = self.config.order_url
        params: BybitPositionEntryIn = PositionEntryMapper.to_exchange(PositionEntryIn, self.config.name).not_None_to_dict()

        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"

        json_payload = json.dumps(params)
        # Convert payload to JSON and create the pre-sign string
        param_str = f"{timestamp}{bybit_access_key}{recv_window}{json_payload}"

        # Create the signature
        signature = hmac.new(
            bytes(bybit_secret_key, "utf-8"),
            param_str.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        # Define headers
        headers = {
            "X-BAPI-SIGN": signature,
            "X-BAPI-API-KEY": bybit_access_key,
            'X-BAPI-SIGN-TYPE': '2',
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "Content-Type": "application/json"
        }

        return await self.request('post', endpoint, headers, json_payload, 'x-www-form-urlencoded')

    async def set_leverage(self, category, symbol, leverage:str):
        # Generate timestamp
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"

        endpoint = self.config.leverage_set_url
        payload = self.config.get_params(endpoint, \
                                         category=category, \
                                         symbol=self.ticker_mapper(symbol), \
                                         buyLeverage=leverage, \
                                         sellLeverage=leverage)

        # Convert payload to JSON and create the pre-sign string
        json_payload = json.dumps(payload)
        pre_sign = f"{timestamp}{bybit_access_key}{recv_window}{json_payload}"
        
        # Create the signature
        signature = hmac.new(
            bytes(bybit_secret_key, "utf-8"),
            pre_sign.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        # Define headers
        headers = {
            "X-BAPI-SIGN": signature,
            "X-BAPI-API-KEY": bybit_access_key,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-RECV-WINDOW": recv_window,
            "Content-Type": "application/json"
        }

        res = await self.request('post', endpoint, headers, json_payload, 'x-www-form-urlencoded')
        logger.debug(res)
        if res['retMsg'] == 'leverage not modified' or res['retMsg'].lower() == 'ok':
            return True
        return False

    async def get_kline(self, symbol, interval, limit: int = 1000):
        endpoint = self.config.kline_url
        params = self.config.get_params(endpoint, symbol=self.ticker_mapper(symbol), interval=interval, limit=limit)
        headers = {"accept" : "application/json"}

        res = await self.request('get', endpoint, headers, params)
        if not res['result'].get('list', False):
            raise ValueError(f"{__class__} {__name__} Symbol {symbol} not found")
        return res['result']['list']

    async def get_orderbook(self, symbol):
        endpoint = self.config.orderbook_url
        params = self.config.get_params(endpoint, symbol=self.ticker_mapper(symbol))
        headers = {"accept" : "application/json"}

        return await self.request('get', endpoint, headers, params)
        
    async def get_min_order_qty(self, symbol):
        endpoint = self.config.ticker_info_url
        params = self.config.get_params(endpoint, symbol=self.ticker_mapper(symbol))
        headers = {"accept" : "application/json"}

        res = await self.request('get', endpoint, headers, params)
        return res['result']['list'][0]['lotSizeFilter']['minOrderQty']
        
    async def get_single_ticker_price(self, symbol):
        endpoint = self.config.all_tickers_url
        params = self.config.get_params(endpoint, category='linear')
        if symbol:
            params['symbol'] = self.ticker_mapper(symbol)
        headers = {"accept": "application/json"}

        res = await self.request('get', endpoint, headers, params)

        if res['retMsg'] == 'OK':
            for ticker in res['result']['list']:
                return ticker['lastPrice']
                    
        raise ValueError(f"Symbol {symbol} not found")
    
    async def get_numeric_tickers(self):
        """Get tickers containing numbers from Bybit exchange"""
        try:
            # Get all tickers
            result = await self.get_all_ticker_price()
            
            # Extract tickers containing numbers
            numeric_tickers = []
            for item in result['result']['list']:
                symbol = item['symbol']
                # Find tickers ending with USDT and containing numbers
                if symbol.endswith('USDT'):
                    ticker = symbol[:symbol.find('USDT')]
                    if re.search(r'\d', ticker):
                        numeric_tickers.append(ticker)
            
            return numeric_tickers
            
        except Exception as e:
            logger.error(f"Error getting numeric tickers: {str(e)}")
            return []
    
    async def get_all_position_info(self):
        timestamp = str(int(time.time() * 1000))
        recv_window = '5000'

        params = {
            'category': 'linear',
            'settleCoin': 'USDT'
        }

        endpoint = self.config.position_info_url

        param_str = '&'.join(f'{key}={value}' for key, value in params.items())
        pre_sign = f'{timestamp}{bybit_access_key}{recv_window}{param_str}'

        signature = hmac.new(
            bybit_secret_key.encode('utf-8'),
            pre_sign.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

        headers = {
            'X-BAPI-API-KEY': bybit_access_key,
            'X-BAPI-SIGN': signature,
            'X-BAPI-TIMESTAMP': timestamp,
            'X-BAPI-RECV-WINDOW': recv_window
        }

        raw_res = await self.request('get', endpoint, headers, params)

        return [{'symbol': item['symbol'], \
                 'side': 'bid' if item['side'] == 'Buy' else 'ask', \
                 'size': item['size'], \
                 'unrealisedPnl': item['unrealisedPnl']} \
                 for item in raw_res.get('result', {}).get('list', [])]

    async def get_quote_volume(self, symbol: str, window: int = 1000, interval: str = '1m') -> int:
        """
        Calculate trading volume for a given symbol
        체결가격 * 체결량(매수체결량 + 매도체결량)
        
        Args:
            symbol: Trading pair symbol (e.g. 'BTC')
            window: Number of recent trades to analyze (default 100)
        
        Returns:
            float: Trading quote volume
        """
        try:
            trades = await self.get_trade_data(symbol, window, interval)
            # Calculate total volume
            return int(sum(float(trade['price']) * float(trade['size']) for trade in trades))
        
        except Exception as e:
            logger.error(f"Error calculating quote volume: {str(e)}")
            logger.error(traceback.format_exc())
            return 0

    async def get_trade_data(self, symbol, window: int = 1000, interval: str = '1m')->list:
        try:
            endpoint = self.config.recent_trade_url
            params = {
                'category': 'linear',
                'symbol': self.ticker_mapper(symbol),
                'limit': window
            }
            # Get recent trades from Bybit
            trades = await self.request(
                'get',
                endpoint,
                headers={"accept": "application/json"},
                params=params
            )

            if not trades.get('result'):
                return 0

            trades = trades['result']['list']

            # filter trade data by interval
            # data is in reverse chronological order
            trades = [trade for trade in trades if int(trade['time']) > int(trades[0]['time']) - self.interval_to_second(interval) * 1000]
            return {
                'timestamp': [int(trade['time']) for trade in trades],
                'price': [float(trade['price']) for trade in trades],
                'volume': [float(trade['size']) for trade in trades],
            }
        
        except Exception as e:
            logger.error(f"Error calculating trade strength: {str(e)}")
            logger.error(traceback.format_exc())
            return None

class BinanceManager(ExchangeManager):
    def __init__(self):
        super().__init__()
        self.config = self.get_config()
    
    def get_config(self):
        return BinanceAPIConfig()
    
    def ticker_mapper(self, symbol):
        if symbol in BinanceSymbols.__members__:
            return BinanceSymbols[symbol].value + 'USDT'
        return f'{symbol}USDT'

    def format_orderbook_response(self, response)->dict:
        ret = {
            'asks' : response['bids'],
            'bids' : response['asks']
        }
        
        ret['asks'] = list(map(lambda x : [float(x[0]), float(x[1])], ret['asks']))
        ret['bids'] = list(map(lambda x : [float(x[0]), float(x[1])], ret['bids']))
        
        return ret

    async def get_all_ticker_price(self):
        endpoint = self.config.all_tickers_url
        params = self.config.get_params(endpoint)
        headers = {"accept": "application/json"}

        return await self.request('get', endpoint, headers, params)

    async def get_single_ticker_price(self, symbol):
        endpoint = self.config.all_tickers_url
        params = self.config.get_params(endpoint)
        params['symbol'] = self.ticker_mapper(symbol)
        headers = {"accept": "application/json"}

        return await self.request('get', endpoint, headers, params)

    async def get_balance(self, symbol):
        if symbol == 'USDT':
            endpoint = self.config.balance_url
            params = self.config.get_params(endpoint)
            query_string = urlencode(params)
            signature = hmac.new(
                binance_secret_key.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            params['signature'] = signature
            headers = {
                'X-MBX-APIKEY': binance_access_key,
            }

            res = await self.request('get', endpoint, headers, params)
            balances = {balance['asset']: balance['balance'] for balance in res}
            return balances['USDT']
        
        else:
            endpoint = self.config.position_info_url
            params = self.config.get_params(endpoint, symbol=self.ticker_mapper(symbol))
            query_string = urlencode(params)
            signature = hmac.new(
                binance_secret_key.encode('utf-8'),
                query_string.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            params['signature'] = signature
            headers = {
                'X-MBX-APIKEY': binance_access_key,
            }

            res = await self.request('get', endpoint, headers, params)
            for position in res:
                if position['symbol'] == self.ticker_mapper(symbol):
                    return position['positionAmt']
            return 0
    
    async def get_kline(self, symbol, interval, limit: int = 1000):
        endpoint = self.config.kline_url
        params = self.config.get_params(endpoint, symbol=self.ticker_mapper(symbol), interval=interval, limit=limit)
        headers = {"accept": "application/json"}

        return await self.request('get', endpoint, headers, params)

    async def get_orderbook(self, symbol):
        endpoint = self.config.orderbook_url
        params = self.config.get_params(endpoint, symbol=self.ticker_mapper(symbol))
        headers = {"accept": "application/json"}

        return await self.request('get', endpoint, headers, params)

    async def post_order(self, PositionEntryIn: PositionEntryIn):
        endpoint = self.config.order_url
        params = PositionEntryMapper.to_exchange(PositionEntryIn, self.config.name).not_None_to_dict()
        params['timestamp'] = int(time.time() * 1000)

        query_string = urlencode(params)
        signature = hmac.new(
            binance_secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        params['signature'] = signature
        headers = {
            'X-MBX-APIKEY': binance_access_key,
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        params = urlencode(params)

        return await self.request('post', endpoint, headers, params, contentType='x-www-form-urlencoded')
    
    async def get_min_order_qty(self, symbol):
        endpoint = self.config.ticker_info_url
        params = self.config.get_params(endpoint)
        headers = {"accept": "application/json"}

        res = await self.request('get', endpoint, headers, params)
        for symbol_info in res['symbols']:
            if symbol_info['symbol'] == self.ticker_mapper(symbol):
                return list(filter(lambda x: x['filterType'] == 'MARKET_LOT_SIZE', symbol_info['filters']))[0]['minQty']
        raise ValueError(f"Symbol {symbol} not found")

    async def set_leverage(self, symbol, leverage):
        endpoint = self.config.leverage_set_url
        params = self.config.get_params(endpoint, symbol=self.ticker_mapper(symbol), leverage=leverage)
        query_string = urlencode(params)
        signature = hmac.new(
            binance_secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        params['signature'] = signature
        headers = {
            'X-MBX-APIKEY': binance_access_key,
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        params = urlencode(params)

        res = await self.request('post', endpoint, headers, params, contentType='x-www-form-urlencoded')
        logger.debug(res)
        if res.get('leverage', 0) == int(leverage):
            return True
        return False

    async def get_all_position_info(self):
        endpoint = self.config.position_info_url
        params = self.config.get_params(endpoint)
        query_string = urlencode(params)
        signature = hmac.new(
            binance_secret_key.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        params['signature'] = signature
        headers = {
            'X-MBX-APIKEY': binance_access_key,
            'Content-Type': 'application/x-www-form-urlencoded'
        }

        raw_res = await self.request('get', endpoint, headers, params)
        logger.info(f'Position info: {raw_res}')

        return [{'symbol': item['symbol'], \
                 'side': 'bid' if item['positionSide'] == 'LONG' else 'ask', \
                 'size': item['positionAmt'], \
                 'unrealisedPnl': item['unRealizedProfit']} \
                 for item in raw_res]
    
    async def get_numeric_tickers(self):
        """Get Binance tickers containing numbers"""
        try:
            # Get all tickers
            tickers = await self.get_all_ticker_price()
            
            # Extract and filter tickers with numbers
            numeric_tickers = []
            for item in tickers:
                symbol = item['symbol']
                if symbol.endswith('USDT'):
                    ticker = symbol[:symbol.find('USDT')]
                    # Find tickers containing digits
                    if re.search(r'\d', ticker):
                        numeric_tickers.append(ticker)
            
            return numeric_tickers
                
        except Exception as e:
            logger.error(f"Error getting numeric tickers: {str(e)}")
            return []

    async def get_quote_volume(self, symbol: str, window: int = 1000, interval: str = '1m') -> int:
        """
        Calculate trading volume for a given symbol
        체결가격 * 체결량(매수체결량 + 매도체결량)
        
        Args:
            symbol: Trading pair symbol (e.g. 'BTC')
            window: Number of recent trades to analyze (default 100)
        
        Returns:
            float: Trading quote volume
        """
        try:
            trades = await self.get_trade_data(symbol, window, interval)
            # Calculate total volume
            return int(sum(float(trade['price']) * float(trade['qty']) for trade in trades))
        
        except Exception as e:
            logger.error(f"Error calculating quote volume: {str(e)}")
            logger.error(traceback.format_exc())
            return 0

    async def get_trade_data(self, symbol, window: int = 1000, interval: str = '1m')->list:
        try:
            endpoint = self.config.recent_trade_url
            params = {
                'symbol': self.ticker_mapper(symbol),
                'limit': window
            }
            # Get recent trades from Binance
            trades = await self.request(
                'get',
                endpoint,
                headers={"accept": "application/json"},
                params=params
            )

            if not trades:
                return 0

            # filter trade data by interval
            # data is in reverse chronological order
            trades = [trade for trade in trades if trade['time'] > trades[0]['time'] - self.interval_to_second(interval) * 1000]
            return {
                'timestamp': [int(trade['time']) for trade in trades],
                'price': [float(trade['price']) for trade in trades],
                'volume': [float(trade['qty']) for trade in trades],
            }
        
        except Exception as e:
            logger.error(f"Error calculating trade strength: {str(e)}")
            logger.error(traceback.format_exc())
            return None

def truncate_number(a, b)->str:
    """
    a: 입력 숫자 (소수 또는 정수)
    b: 기준 값 (소수 또는 정수)
    반환값: b 기준으로 버림된 a
    """
    import math


    if isinstance(b, int) or b.is_integer():  # b가 정수인 경우
        result = math.floor(a / b) * b
    else:  # b가 소수인 경우
        decimal_places = len(str(b).split(".")[1])  # 소수 자릿수 계산
        multiplier = 10 ** decimal_places
        result = math.floor(a * multiplier) / multiplier
    
    return str(result)

class KimpManager:
    def __init__(self):
        self.krw_exchanges = ['upbit', 'bithumb']
        self.foreign_exchanges = ['bybit', 'binance']
        self.ex_lists = ['upbit', 'bithumb', 'bybit', 'binance']
        self.managers = {
            'upbit': UpbitManager(),
            'bithumb': BithumbManager(),
            'bybit': BybitManager(),
            'binance': BinanceManager()
        }
        self.depwith_status_cache = None
        self.depwith_last_cache_time = 0
        self.usdt_price_last_cache_time = 0

    def get_exchange_manager(self, ex):
        return self.managers.get(ex)
    
    @staticmethod
    def get_exclude_symbols():
        return ['TON', 'STRAX']

    async def get_single_ticker_kimp_by_seed(self, symbol:str, krw_ex:str, for_ex:str, seed:float=None)->pd.DataFrame:
        '''
            symbol: 코인 심볼 (예: SHIB)
            krw_ex: KRW 거래소 (예: upbit)
            for_ex: 외화 거래소 (예: binance)
        '''
        try:
            # 거래소 유효성 검사
            if krw_ex not in self.ex_lists or for_ex not in self.ex_lists:
                raise ValueError(f'krw_ex 또는 for_ex는 {self.ex_lists} 중 하나여야 합니다.')
        
            krw_ex_manager = self.get_exchange_manager(krw_ex)
            for_ex_manager = self.get_exchange_manager(for_ex)

            # krw_ex에서 주문서 응답 가져오기
            krw_orderbook = krw_ex_manager.get_orderbook(symbol)

            # for_ex에서 주문서 응답 가져오기
            for_orderbook = for_ex_manager.get_orderbook(symbol)

            results = await asyncio.gather(krw_orderbook, for_orderbook)

            # 주문서 응답 형식 지정
            krw_orderbook = krw_ex_manager.format_orderbook_response(results[0])
            for_orderbook = for_ex_manager.format_orderbook_response(results[1])

            ret = []
            if seed:
                buy_amt, buy_fee = KimpManager.calculate_market_order_amount(symbol, krw_orderbook, seed, is_buy=True, fee_rate=krw_ex_manager.config.fee_rate['maker'])
                if buy_amt == 0:
                    logger.info(f'buy_amt is 0')
                    return pd.DataFrame(ret) 
                
                usdt_revenue, sell_fee = KimpManager.calculate_market_order_amount(symbol, for_orderbook, buy_amt, is_buy=False, fee_rate=for_ex_manager.config.fee_rate['maker'])
                if usdt_revenue == 0:
                    logger.info(f'usdt_revenue is 0')
                    return pd.DataFrame(ret) 

                ret.append({
                    'seed' : seed,
                    'buy_amt' : buy_amt,
                    'buy_fee' : str(buy_fee) + 'KRW',
                    'usdt_revenue' : usdt_revenue,
                    'sell_fee' : str(round(sell_fee,2)) + 'USDT',
                    'exrate' : round(seed / (usdt_revenue + sell_fee), 4),
                    'market' : symbol,
                    'krw_ex' : krw_ex, 
                    'for_ex' : for_ex
                })
            else:
                for seed in range(1_000_000, 100_000_000, 1_000_000):
                    buy_amt, buy_fee = KimpManager.calculate_market_order_amount(symbol, krw_orderbook, seed, is_buy=True, fee_rate=krw_ex_manager.config.fee_rate['maker'])

                    if buy_amt == 0:
                        break

                    usdt_revenue, sell_fee = KimpManager.calculate_market_order_amount(symbol, for_orderbook, buy_amt, is_buy=False, fee_rate=for_ex_manager.config.fee_rate['maker'])

                    if usdt_revenue == 0:
                        break
                        
                    ret.append({
                        'seed' : seed,
                        'buy_amt' : buy_amt,
                        'buy_fee' : str(buy_fee) + 'KRW',
                        'usdt_revenue' : usdt_revenue,
                        'sell_fee' : str(round(sell_fee,2)) + 'USDT',
                        'exrate' : round(seed / (usdt_revenue + sell_fee), 4),
                        'market' : symbol,
                        'krw_ex' : krw_ex, 
                        'for_ex' : for_ex
                    })

            return pd.DataFrame(ret)
        
        except Exception as e:
            logger.info(traceback.format_exc())
            logger.info(f'\n symbol: {symbol} \
                          \n krw_ex: {krw_ex} \
                          \n for_ex: {for_ex}')

    async def run_arbitrage_bot(self, update: Update, context: ContextTypes.DEFAULT_TYPE, event_caller):
        """
        사용자가 입력한 매개변수와 사전 정의된 매개변수를 기반으로 차익 거래 봇 로직을 실행합니다.

        이 함수는 Telegram 봇 대화 중에 트리거되며, `context.user_data` 사전에 저장된 사용자 정의 거래 매개변수를 가져와 차익 거래 전략을 실행합니다. 
        봇은 환율을 모니터링하고, 지정된 레버리지를 적용하며, 사용자의 KRW 예산 내에서 적절한 거래가 이루어지도록 합니다.

        매개변수:
            self: 이 메서드를 포함하는 클래스의 인스턴스 참조.
            update (Update): 메타데이터와 사용자 상호작용 데이터를 포함하는 Telegram 업데이트 객체.
            context (ContextTypes.DEFAULT_TYPE): 다음을 포함하는 컨텍스트 객체:
            - user_data (dict): 사용자 정의 거래 매개변수를 포함하는 사전:
                - 'symbol' (str): 거래 심볼 (예: "BTC").
                - 'entry_exrate' (float): 포지션을 진입할 환율.
                - 'close_exrate' (float): 포지션을 종료할 환율.
                - 'krw_budget' (float): 이 차익 거래 작업에 할당된 KRW 예산.
                - 'krw_ex' (str): KRW 기반 거래소 이름.
                - 'for_ex' (str): 외화 거래소 이름.
                - 'leverage' (str): 적용할 레버리지 비율 (해당되는 경우).
                - 'stop_event' (asyncio.Event): 봇을 중지할 신호를 보내는 스레딩 이벤트.

        예상 동작:
            1. 제공된 거래 매개변수를 검증합니다.
            2. 지정된 거래소에서 환율을 모니터링합니다.
            3. `entry_exrate` 및 `close_exrate`에 따라 진입 및 종료 거래를 실행합니다.
            4. 할당된 `krw_budget`을 준수하여 거래를 보장합니다.
            5. 지정된 경우 레버리지를 적용하여 마진 요구 사항을 준수합니다.
            6. `stop_event`를 지속적으로 확인하여 봇을 정상적으로 종료합니다.

        오류 처리:
            - `context.user_data`에서 잘못된 또는 누락된 매개변수를 처리합니다.
            - 거래소 모니터링 또는 거래 실행 중 발생하는 예기치 않은 오류를 기록합니다.

        반환값:
            없음. 비동기적으로 작업을 수행하고 외부 API와 상호작용합니다.

        사용 예:
            # Telegram 봇 상호작용을 가정
            context.user_data = {
                'symbol': 'BTC',
                'entry_exrate': 1390,
                'close_exrate': 1420,
                'krw_budget': 1000000,
                'krw_ex': 'upbit',
                'for_ex': 'binance',
                'leverage': 2,
                'stop_event': asyncio.Event()
            }
            await bot.run_arbitrage_bot(update, context)
        """
        try:
            symbol          = context.user_data[event_caller]['symbol']
            entry_exrate    = float(context.user_data[event_caller]['entry_exrate'])
            close_exrate    = float(context.user_data[event_caller]['close_exrate'])
            krw_budget      = float(context.user_data[event_caller]['budget'])
            leverage        = context.user_data[event_caller]['leverage']
            krw_ex          = self.get_exchange_manager(context.user_data[event_caller]['krw_ex'])
            for_ex          = self.get_exchange_manager(context.user_data[event_caller]['for_ex'])
        
            krw_balance, usdt_balance, leverage_res, min_order_qty = await asyncio.gather(krw_ex.get_balance('KRW'), \
                                                                                          for_ex.get_balance('USDT'), \
                                                                                          for_ex.set_leverage('linear', \
                                                                                                                symbol, \
                                                                                                                leverage, \
                                                                                                                leverage), \
                                                                                          for_ex.get_min_order_qty(symbol))
            if leverage_res['retMsg'] == 'OK':
                await update.callback_query.message.reply_text(f'레버리지가 {leverage}배로 설정되었습니다.')

            krw_balance, usdt_balance, min_order_qty = float(krw_balance), float(usdt_balance), float(min_order_qty)
            
            logger.debug(f'\n Initial Balance \
                          \n - KRW Balance : {krw_balance} \
                          \n - USDT Balance : {usdt_balance} \
                          \n - Total Balance : {krw_balance} ') 
            
            await update.callback_query.message.reply_text(f'Initial Balance\n'
                                                           f'- KRW Balance : {krw_balance}\n'
                                                           f'- USDT Balance : {usdt_balance}\n'
                                                           f'- Total Balance : {krw_balance}')
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(e)

        position_yn = 0
        buy_position = 0
        short_position = 0
        real_entry_exrate = None
        
        while not context.user_data['stop_event'].is_set():
            try:
                result = await asyncio.gather(self.get_single_ticker_kimp_by_seed(symbol, \
                                                                                  context.user_data[event_caller]['krw_ex'], \
                                                                                  context.user_data[event_caller]['for_ex']), \
                                              krw_ex.get_single_ticker_price('USDT'))
                
                df = result[0]
                df.seed = df.seed.astype(float)
                df.exrate = df.exrate.astype(float)

                krw_budget, now_exrate = self.get_max_exrate_for_budget(df, krw_budget)
                krw_budget, now_exrate = float(krw_budget), float(now_exrate)
                
                usdt_price = result[1][0]['trade_price']

                logger.info(f'\n krw_balance : {krw_balance}, \
                              \n for_balance : {usdt_balance}, \
                              \n krw_budget : {krw_budget}, \
                              \n buy_position : {buy_position}, \
                              \n short_position : {short_position}, \
                              \n usdt_price : {usdt_price}, \
                              \n now_exrate : {now_exrate}, \
                              \n entry_exrate : {entry_exrate} \
                              \n close_exrate : {close_exrate}')

                if position_yn == 0 and now_exrate <= entry_exrate:
                    pre_entry_krw_balance = krw_balance
                    pre_entry_for_balance = usdt_balance

                    res = await krw_ex.post_order(symbol, 'bid', str(krw_budget * (1-0.0025)), None, 'price')
                    logger.info(res)

                    await asyncio.sleep(0.1)

                    buy_position = await krw_ex.get_balance(symbol)
                    logger.info(buy_position)

                    res = await for_ex.post_order('linear', symbol, 'Sell', 'Market', truncate_number(float(buy_position), min_order_qty))
                    logger.info(res)

                    # res = await krw_ex.post_order(symbol, 'ask', None, truncate_number(buy_position, min_order_qty), 'market')
                    # logger.info(res)

                    buy_position = await krw_ex.get_balance(symbol)
                    logger.info(buy_position)

                    short_position = await for_ex.get_balance(symbol)
                    logger.info(short_position)

                    position_yn = 1

                    new_krw_balance, new_usdt_balance = await asyncio.gather(krw_ex.get_balance('KRW'), for_ex.get_balance('USDT'))
                    new_krw_balance, new_usdt_balance = float(new_krw_balance), float(new_usdt_balance)

                    real_entry_exrate = round((krw_balance - new_krw_balance) / (usdt_balance - new_usdt_balance), 3)
                    krw_balance, usdt_balance = new_krw_balance, new_usdt_balance

                    logger.info(f"\n buy position : {buy_position} \
                                  \n short position : {short_position} \
                                  \n exrate : {real_entry_exrate}")

                    logger.info('Entry Kimp Position')

                    await update.callback_query.message.reply_text(f"buy_position : {buy_position}\n"
                                                                   f"short_position : {short_position}\n"
                                                                   f"exrate : {real_entry_exrate}", \
                                                                    parse_mode='Markdown')

                elif position_yn == 1 and now_exrate >= close_exrate:
                    order1 = krw_ex.post_order(symbol, 'ask', None, buy_position, 'market')
                    order2 = for_ex.post_order('linear', symbol, 'Buy', 'Market', short_position)

                    res = await asyncio.gather(order1, order2)
                    logger.info(res[0])
                    logger.info(res[1])
                    
                    new_krw_balance, new_usdt_balance = await asyncio.gather(krw_ex.get_balance('KRW'), for_ex.get_balance('USDT'))
                    new_krw_balance, new_usdt_balance = float(new_krw_balance), float(new_usdt_balance)

                    close_exrate = round((new_krw_balance - krw_balance) / (new_usdt_balance - usdt_balance), 3)
                    krw_balance, usdt_balance = new_krw_balance, new_usdt_balance

                    # calculate real profit
                    krw_balance_chg = krw_balance - pre_entry_krw_balance
                    for_balance_chg = usdt_balance - pre_entry_for_balance
                    tether_price = await krw_ex.get_single_ticker_price('USDT')
                    real_profit = krw_balance_chg + tether_price * for_balance_chg
                    
                    logger.info(f'\n krw_balance_chg : {krw_balance_chg} \
                                  \n for_balance_chg : {for_balance_chg} \
                                  \n close_exrate : {close_exrate} \
                                  \n real_profit(KRW): {real_profit}')
                    
                    await update.callback_query.message.reply_text(f"krw_balance_chg : {krw_balance_chg}\n"
                                                                   f"for_balance_chg : {for_balance_chg}\n"
                                                                   f"close_exrate : {close_exrate}\n"
                                                                   f"real_profit(KRW) : {real_profit}", \
                                                                    parse_mode='Markdown')

                    position_yn = 0
                    buy_position = 0 
                    short_position = 0
                    real_entry_exrate = None

                    logger.info('Close Kimp Position')

            except Exception as e:
                logger.info(traceback.format_exc())
                logger.error(e)

            await asyncio.sleep(1)

        if position_yn:
            order1 = krw_ex.post_order(symbol, 'ask', None, buy_position, 'market')
            order2 = for_ex.post_order('linear', symbol, 'Buy', 'Market', short_position)

            res = await asyncio.gather(order1, order2)
            logger.info(res[0])
            logger.info(res[1])
            
            new_krw_balance, new_usdt_balance = await asyncio.gather(krw_ex.get_balance('KRW'), for_ex.get_balance('USDT'))
            new_krw_balance, new_usdt_balance = float(new_krw_balance), float(new_usdt_balance)

            close_exrate = round((new_krw_balance - krw_balance) / (new_usdt_balance - usdt_balance), 3)
            krw_balance, usdt_balance = new_krw_balance, new_usdt_balance

            # calculate real profit
            krw_balance_chg = krw_balance - pre_entry_krw_balance
            for_balance_chg = usdt_balance - pre_entry_for_balance
            tether_price = await krw_ex.get_single_ticker_price('USDT')
            real_profit = krw_balance_chg + tether_price * for_balance_chg
            
            logger.info(f'\n krw_balance_chg : {krw_balance_chg} \
                          \n for_balance_chg : {for_balance_chg} \
                          \n close_exrate : {close_exrate} \
                          \n real_profit : {real_profit}')

            logger.info('Close Kimp Position')
            
    def get_max_exrate_for_budget(self, df, budget)->float:
        if budget <= df.iloc[0].seed:
            max_exrate = df.iloc[0].exrate
        elif budget >= df.iloc[-1].seed:
            max_exrate = df.iloc[-1].exrate
            budget = df.iloc[-1].seed
        else:
            max_exrate = df.loc[df.seed <= budget].iloc[-1].exrate
            budget = df.loc[df.seed <= budget].iloc[-1].seed
        return budget, max_exrate

    async def get_all_ticker_exrate(self, budget):
        exrate_df = await self.monitor_all_ticker_exrate()
        exrate_df = exrate_df.loc['bybit'] # temporary til binance api development

        rows = exrate_df.iterrows()
        
        market_col = []
        exrate_col = []
        budget_col = []
        exchange_col = []

        while True:
            batch = list(itertools.islice(rows, 5))

            if not batch:
                break

            tasks = []

            for idx, _ in batch:
                for col in exrate_df.columns:
                    logger.debug(f'{idx} {col} "bybit"')
                    tasks.append(self.get_single_ticker_kimp_by_seed(idx, col, 'bybit'))

            results = await asyncio.gather(*tasks)

            for r in results:
                if len(r) == 0:
                    continue
                budget, max_exrate = self.get_max_exrate_for_budget(r, budget)
                market_col.append(r['market'].iloc[0])
                exrate_col.append(max_exrate)
                budget_col.append(budget)
                exchange_col.append(r['krw_ex'].iloc[0])
            
            await asyncio.sleep(0.2)

        df = pd.DataFrame({'market': market_col, 'exrate': exrate_col, 'budget': budget_col, 'krw_ex': exchange_col}).set_index('market')
        df.loc[df.index.isin(['SHIB', 'BONK', 'PEPE', 'FLOKI', 'XEC', 'BTT']), 'exrate'] *= 1000
        df.sort_values(by='exrate', inplace=True)

        return df

    async def monitor_all_ticker_exrate(self, up_threshold=None, down_threshold=None):
        tasks = [self.get_exchange_manager(ex).get_all_ticker_price() for ex in self.ex_lists]
        results = await asyncio.gather(*tasks)
        results_map = dict(zip(self.ex_lists, results))

        # Process the results
        upbit_data = [{'exchange' : 'upbit', 'market' : i.get('market').split('-')[1], 'price' : i.get('trade_price')} for i in results[0] if i.get('market').startswith('KRW-')]
        bithumb_data = [{'exchange' : 'bithumb', 'market' : i.get('market').split('-')[1], 'price' : i.get('trade_price')} for i in results[1] if i.get('market').startswith('KRW-')]
        bybit_data = [{'exchange' : 'bybit', 'market' : i.get('symbol')[re.search(r'[a-zA-Z]', i.get('symbol')).start() : i.get('symbol').find('USDT')], 'price' : i.get('lastPrice')} for i in results[2]['result']['list'] if i.get('symbol').endswith('USDT')]
        binance_data = [{'exchange': 'binance', 'market': i.get('symbol')[re.search(r'[a-zA-Z]', i.get('symbol')).start():i.get('symbol').find('USDT')], 'price': i.get('price')} for i in results_map['binance'] if i.get('symbol').endswith('USDT')]
        
        upbit_df = pd.DataFrame(upbit_data)
        bithumb_df = pd.DataFrame(bithumb_data)
        bybit_df = pd.DataFrame(bybit_data)
        binance_df = pd.DataFrame(binance_data)
        
        current_time = time.time()
        if self.depwith_status_cache is None or current_time - self.depwith_last_cache_time > 300:
            tasks = [self.get_exchange_manager('upbit').get_depwith_status(), \
                     self.get_exchange_manager('bithumb').get_depwith_status()]
            self.depwith_status_cache = tasks
            self.depwith_last_cache_time = current_time

        upbit_data = self.depwith_status_cache[0]
        upbit_data['krw_ex'] = 'upbit'
        bithumb_data = self.depwith_status_cache[1]
        bithumb_data['krw_ex'] = 'bithumb'

        deposit_withdraw_df = pd.concat([pd.DataFrame(upbit_data), pd.DataFrame(bithumb_data)], axis=0)

        # List to store all merged DataFrames
        merged_dfs = []

        # Generate dynamic combinations of KRW and foreign exchanges
        combinations = product(self.krw_exchanges, self.foreign_exchanges)

        # Merge each combination and store the result
        for krw_ex, for_ex in combinations:
            krw_df = locals()[f'{krw_ex}_df']
            for_df = locals()[f'{for_ex}_df']
            merged_df = pd.merge(for_df, krw_df, on='market', suffixes=('', f'_{krw_ex}'))
            merged_df['for_ex'] = for_ex
            merged_df['krw_ex'] = krw_ex
            # Convert numeric columns to float type
            numeric_columns = ['price', f'price_{krw_ex}']
            merged_df[numeric_columns] = merged_df[numeric_columns].astype(float)
            merged_df['exrate'] = round(merged_df[f'price_{krw_ex}'] / merged_df.price, 1)
            merged_df['exrate'] = merged_df.exrate.map(lambda x : x * 1000 if x < 1000 else x) # 1000SHIB, 1000PEPE...
            merged_dfs.append(merged_df)

        # Concatenate all merged DataFrames vertically
        result_df = pd.concat(merged_dfs, axis=0)

        result_df = pd.merge(result_df, deposit_withdraw_df, on=['krw_ex', 'market'], how='left')

        # Set multi index
        result_df.set_index(['for_ex', 'krw_ex'], inplace=True)

        markets_to_remove = KimpManager.get_exclude_symbols()
        result_df = result_df[~result_df.market.isin(markets_to_remove)]

        # Filter by threshold
        if up_threshold:
            df1 = result_df[result_df.exrate >= up_threshold]
            df1.sort_values(by='exrate', ascending=False, inplace=True)
        if down_threshold:
            df2 = result_df[result_df.exrate <= down_threshold]
            df2.sort_values(by='exrate', ascending=True, inplace=True)

        if up_threshold and down_threshold:
            return df1.loc[:, ['market', 'exrate', 'status']], \
                   df2.loc[:, ['market', 'exrate', 'status']]
        elif up_threshold:
            return df1.loc[:, ['market', 'exrate', 'status']], \
                   pd.DataFrame(columns=['market', 'exrate', 'status'])
        elif down_threshold:
            return pd.DataFrame(columns=['market', 'exrate', 'status']), \
                   df2.loc[:, ['market', 'exrate', 'status']]

    async def run_monitoring_exrate(self, update: Update, context: ContextTypes.DEFAULT_TYPE, event_caller):
        while not context.user_data['stop_event'].is_set():
            df1, df2 = await self.monitor_all_ticker_exrate(context.user_data[event_caller]['up_threshold'], context.user_data[event_caller]['down_threshold'])
            await update.message.reply_text(f"```\n{df1.head(10).to_string()}\n```", parse_mode='Markdown')
            await update.message.reply_text(f"```\n{df2.tail(10).to_string()}\n```", parse_mode='Markdown')
            await asyncio.sleep(10)

    @ttl_cache(cache)
    async def get_all_tickers(self):
        tasks = [self.get_exchange_manager(ex).get_all_ticker_price() for ex in self.ex_lists]
        results = await asyncio.gather(*tasks)

        # Process the results
        upbit_data = [ i.get('market').split('-')[1] for i in results[0] if i.get('market').startswith('KRW-')]
        bithumb_data = [ i.get('market').split('-')[1] for i in results[1] if i.get('market').startswith('KRW-')]
        bybit_data = [ i.get('symbol')[ : i.get('symbol').find('USDT')] for i in results[2]['result']['list'] if i.get('symbol').endswith('USDT')]
        binance_data = [ i.get('symbol')[ : i.get('symbol').find('USDT')] for i in results[3] if i.get('symbol').endswith('USDT')]
        
        return {
            'upbit': upbit_data,
            'bithumb': bithumb_data,
            'bybit': bybit_data,
            'binance': binance_data
        }

    async def celery_monitor_big_volume_batch(self, data, multiplier: int, usdt_price: float, binance_threshold: int):
        '''
            :data: [{'exchange': 'upbit', 'ticker': 'BTC'}, {'exchange': 'bithumb', 'ticker': 'BTC'}, ...]
        '''
        tasks = [self.get_exchange_manager(item['exchange']).get_kline(item['ticker'], interval='5m', limit=3) for item in data]
        res = await asyncio.gather(*tasks)

        target = []
        for item, res in zip(data, res):
            ex = item['exchange']
            ticker = item['ticker']

            if ex in ['upbit', 'bithumb']:
                df = pd.DataFrame(res)
                df.rename(columns={'candle_acc_trade_price': 'quote_volume', 'candle_date_time_kst': 'datetime'}, inplace=True)
                df['timestamp'] = pd.to_datetime(df.datetime)
                now_candle_timestamp = df.iloc[1].timestamp

            elif ex in ['bybit', 'binance']:
                df = pd.DataFrame(res, columns=ex_kline_col_map[ex])
                df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
                df['timestamp'] = pd.to_datetime(df.timestamp, unit='ms')
                df.timestamp = df.timestamp.dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul')
                now_candle_timestamp = df.iloc[1].timestamp

            if ex != 'binance': # binance를 제외하고는 시간 내림차순으로 데이터를 준다.
                now_candle_quote_volume = int(float(df.iloc[1].quote_volume)) # 직전 5분봉
                bef_candle_quote_volume = int(float(df.iloc[2].quote_volume)) # 2번째 직전 5분봉
            else: # binance는 시간 오름차순으로 데이터를 준다.
                now_candle_quote_volume = int(float(df.iloc[2].quote_volume))
                bef_candle_quote_volume = int(float(df.iloc[1].quote_volume))
            
            # 1차 필터링 : 거래량이 0인 경우
            if now_candle_quote_volume == 0 or bef_candle_quote_volume == 0:
                continue

            # 2차 필터링 : 직전 1분간 거래량이 2분간 거래량 * volume_multiplier 보다 작은 경우
            if now_candle_quote_volume < bef_candle_quote_volume * multiplier:
                continue

            # 3차 필터링
            if ex in ['upbit', 'bithumb'] and now_candle_quote_volume > 300_000_000: # 3억원
                target.append(
                    {
                        'exchange': ex,
                        'ticker': ticker,
                        'timestamp': now_candle_timestamp,
                        'quote_volume': now_candle_quote_volume
                    }
                )
            elif ex == 'bybit' and now_candle_quote_volume * usdt_price > 300_000_000:
                target.append(
                    {
                        'exchange': ex,
                        'ticker': ticker,
                        'timestamp': now_candle_timestamp,
                        'quote_volume': int(now_candle_quote_volume * usdt_price)
                    }
                )
            elif ex == 'binance' and now_candle_quote_volume * usdt_price > binance_threshold:
                target.append(
                    {
                        'exchange': ex,
                        'ticker': ticker,
                        'timestamp': now_candle_timestamp,
                        'quote_volume': int(now_candle_quote_volume * usdt_price)
                    }
                )
        return target

    async def run_big_volume_alarm(self, update, context):
        event_caller = context.user_data['event_caller']
        volume_multiplier = context.user_data[event_caller]['multiplier']
        
        try:
            upbit_target_tickers = []
            bithumb_target_tickers = []
            bybit_target_tickers = []
            binance_target_tickers = []

            batch_size = 5
            i, j, p, q = 0, 0, 0, 0
            start_time = time.time()
            while not context.user_data['stop_event'].is_set():
                
                # USDT 가격 구하기 - 캐시사용
                usdt_price = (await self.get_exchange_manager('upbit').get_cached_single_ticker_price('USDT'))[0]['trade_price']

                # 모든 거래소의 티커 가져오기 - 캐시사용
                res = await self.get_all_tickers()
                upbit_tickers = res['upbit']
                bithumb_tickers = res['bithumb']
                bybit_tickers = res['bybit']
                binance_tickers = res['binance']

                tickers_data = []
                if upbit_tickers[i:i+batch_size]:
                    tickers_data += [('upbit', ticker) for ticker in upbit_tickers[i:i+batch_size]]

                if bithumb_tickers[j:j+batch_size]:
                    tickers_data += [('bithumb', ticker) for ticker in bithumb_tickers[j:j+batch_size]]

                if bybit_tickers[p:p+batch_size]:
                    tickers_data += [('bybit', ticker) for ticker in bybit_tickers[p:p+batch_size]]

                if binance_tickers[q:q+batch_size]:
                    tickers_data += [('binance', ticker) for ticker in binance_tickers[q:q+batch_size]]

                tasks = [self.get_exchange_manager(ex).get_kline(ticker, interval='1m', limit=2) for ex, ticker in tickers_data]

                results = await asyncio.gather(*tasks)

                for ticker, res in zip(tickers_data, results):
                    if ticker[0] in ['upbit', 'bithumb']:
                        df = pd.DataFrame(res)
                        df.rename(columns={'candle_acc_trade_price': 'quote_volume', 'candle_date_time_kst': 'datetime'}, inplace=True)
                        df['timestamp'] = pd.to_datetime(df.datetime)
                        now_candle_timestamp = df.iloc[0].timestamp

                    elif ticker[0] in ['bybit', 'binance']:
                        df = pd.DataFrame(res, columns=ex_kline_col_map[ticker[0]])
                        df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
                        df['timestamp'] = pd.to_datetime(df.timestamp, unit='ms')
                        df.timestamp = df.timestamp.dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul')
                        now_candle_timestamp = df.iloc[0].timestamp

                    if ticker[0] != 'binance':
                        now_candle_quote_volume = int(float(df.iloc[0].quote_volume))
                        bef_candle_quote_volume = int(float(df.iloc[1].quote_volume))
                    else: # binance는 시간역순으로 데이터를 준다.
                        now_candle_quote_volume = int(float(df.iloc[1].quote_volume))
                        bef_candle_quote_volume = int(float(df.iloc[0].quote_volume))
                    
                    # 1차 필터링 : 거래량이 0인 경우
                    if now_candle_quote_volume == 0 or bef_candle_quote_volume == 0:
                        continue

                    # 2차 필터링 : 직전 1분간 거래량이 2분간 거래량 * volume_multiplier 보다 작은 경우
                    if now_candle_quote_volume < bef_candle_quote_volume * volume_multiplier:
                        continue

                    # 3차 필터링
                    if ticker[0] == 'upbit' and now_candle_quote_volume > 100_000_000:
                        upbit_target_tickers.append((ticker[1], now_candle_timestamp, now_candle_quote_volume))
                    elif ticker[0] == 'bithumb' and now_candle_quote_volume > 100_000_000:
                        bithumb_target_tickers.append((ticker[1], now_candle_timestamp, now_candle_quote_volume))
                    elif ticker[0] == 'bybit' and now_candle_quote_volume * usdt_price > 100_000_000:
                        bybit_target_tickers.append((ticker[1], now_candle_timestamp, int(now_candle_quote_volume * usdt_price)))
                    elif ticker[0] == 'binance' and now_candle_quote_volume * usdt_price > 500_000_000:
                        binance_target_tickers.append((ticker[1], now_candle_timestamp, int(now_candle_quote_volume * usdt_price)))

                i += batch_size
                j += batch_size 
                p += batch_size
                q += batch_size

                if i >= len(upbit_tickers):
                    if upbit_target_tickers:
                        messages = ""
                        for ticker, timestamp, res in upbit_target_tickers:
                            messages += f"upbit [{timestamp}] {ticker}거래량 : {res:,}\n"
                        await update.message.reply_text(messages)

                    upbit_target_tickers.clear()
                    i = 0

                if j >= len(bithumb_tickers):
                    if bithumb_target_tickers:
                        messages = ""
                        for ticker, timestamp, res in bithumb_target_tickers:
                            messages += f"bithumb [{timestamp}] {ticker}거래량 : {res:,}\n"
                        await update.message.reply_text(messages)

                    bithumb_target_tickers.clear()
                    j = 0

                if p >= len(bybit_tickers):
                    if bybit_target_tickers:
                        messages = ""
                        for ticker, timestamp, res in bybit_target_tickers:
                            messages += f"bybit [{timestamp}] {ticker}거래량 : {res:,}\n"
                        await update.message.reply_text(messages)

                    bybit_target_tickers.clear()
                    p = 0

                if q >= len(binance_tickers):
                    logger.info(f'elapsed time : {time.time() - start_time}')
                    if binance_target_tickers:
                        messages = ""
                        for ticker, timestamp, res in binance_target_tickers:
                            messages += f"binance [{timestamp}] {ticker}거래량 : {res:,}\n"
                        await update.message.reply_text(messages)

                    binance_target_tickers.clear()
                    q = 0

                await asyncio.sleep(0.3)

        except Exception as e:
            logger.error(f"Error in run_big_volume_alarm")
            logger.error(traceback.format_exc())
        
        finally:
            self.usdt_price_last_cache_time = None
            del context.user_data[event_caller]
            del context.user_data['event_caller']
            del context.user_data['stop_event']
            del context.user_data['task']
            return ConversationHandler.END

    async def run_kimp_alarm(self, update, context):
        """Monitor exchange rates and send Telegram alerts with trading strength"""
        event_caller = context.user_data['event_caller']
        upkimp = context.user_data[event_caller]['up_threshold']
        downkimp = context.user_data[event_caller]['down_threshold']
        seed = context.user_data[event_caller]['seed']
        
        upbit = self.get_exchange_manager('upbit')
        try:
            while not context.user_data['stop_event'].is_set():
                # Get current USDT price
                usdt_price = (await upbit.get_single_ticker_price('USDT'))[0]['trade_price']
                
                # Calculate thresholds (±0.5% of USDT price)
                up_threshold = usdt_price * (1 + upkimp/100)
                down_threshold = usdt_price * (1 - downkimp/100)
                
                # 1차 필터링 : 현재가로 환율 계산
                df1, df2 = await self.monitor_all_ticker_exrate(up_threshold, down_threshold)
                
                # Add trading strength for df1 markets using asyncio.gather
                if not df1.empty:
                    df1 = df1.head(10)

                    # Dictionary to store real exrates
                    real_exrates = {}

                    # 2차 필터링 : 호가반영 환율 계산
                    for (for_ex, krw_ex), row in df1.groupby(level=['for_ex', 'krw_ex']):
                        for market in row['market']:
                            tmp_df = await self.get_single_ticker_kimp_by_seed(market, krw_ex, for_ex, seed)
                            
                            if tmp_df.empty:
                                continue
                            
                            real_exrates[market] = round(tmp_df.iloc[0].exrate,1)

                    df1['real_exrates'] = df1['market'].map(real_exrates)

                    strength_tasks = []
                    for (for_ex, krw_ex), row in df1.groupby(level=['for_ex', 'krw_ex']):
                        exchange = self.get_exchange_manager(krw_ex)
                        for market in row['market']:
                            strength_tasks.append(exchange.get_trade_strength(market, interval='1m'))

                    # Execute all tasks concurrently
                    strengths = await asyncio.gather(*strength_tasks)
                    
                    # Add strength column to df1
                    df1['strength'] = pd.Series(strengths, index=df1.index)
                    
                    # Filter and clean up DataFrame
                    filtered_df = df1[
                        (df1['status'] == 'O/O') & 
                        (df1['strength'] > 100)
                    ].drop('status', axis=1)

                    # Rename both index levels and columns
                    filtered_df = filtered_df.rename_axis(['해외', '한국']).rename(
                        columns={'market': '티커', 'exrate': '환율', 'strength': '체결강도', 'real_exrates': '실환율'}
                    )

                    # Replace index values
                    filtered_df.index = filtered_df.index.set_levels(
                        filtered_df.index.levels[0].str.replace('binance', '바낸'),
                        level='해외'
                    )
                    filtered_df.index = filtered_df.index.set_levels(
                        filtered_df.index.levels[0].str.replace('bybit', '바빗'),
                        level='해외'
                    )
                    filtered_df.index = filtered_df.index.set_levels(
                        filtered_df.index.levels[1].str.replace('upbit', '업빗'),
                        level='한국'
                    )
                    filtered_df.index = filtered_df.index.set_levels(
                        filtered_df.index.levels[1].str.replace('bithumb', '빗썸'),
                        level='한국'
                    )
                    
                    if not filtered_df.empty:
                        filtered_df = filtered_df.sort_values('환율', ascending=False)
                        await update.message.reply_text(f"🔔 테더 가격: {usdt_price:.2f}\n"
                                                        f"🔔 현재 환율 >= {up_threshold:.0f}\n"
                                                        f"🔔 체결 강도 >= 100\n"
                                                        f"🔔 입출금 가능 티커만\n"
                                                        f"\n{filtered_df.to_string()}\n")

                if not df2.empty:
                    # Dictionary to store real exrates
                    real_exrates = {}
                    # 2차 필터링 : 호가반영 환율 계산
                    for (for_ex, krw_ex), row in df2.groupby(level=['for_ex', 'krw_ex']):
                        for market in row['market']:
                            tmp_df = await self.get_single_ticker_kimp_by_seed(market, krw_ex, for_ex, seed)
                            if tmp_df.empty:
                                continue

                            df2['real_exrates'] = df2['market'].map(real_exrates)

                    df2['real_exrates'] = df2['market'].map(real_exrates)

                    df2.sort_values('real_exrates', ascending=True, inplace=True)
                    df2 = df2.head(10)

                    df2 = df2.loc[df2.real_exrates <= down_threshold]

                    # Rename both index levels and columns
                    df2 = df2.rename_axis(['해외', '한국']).rename(
                        columns={'market': '티커', 'exrate': '환율', 'real_exrates': '실환율'}
                    )

                    # Replace index values
                    df2.index = df2.index.set_levels(
                        df2.index.levels[0].str.replace('binance', '바낸'),
                        level='해외'
                    )
                    df2.index = df2.index.set_levels(
                        df2.index.levels[0].str.replace('bybit', '바빗'),
                        level='해외'
                    )
                    df2.index = df2.index.set_levels(
                        df2.index.levels[1].str.replace('upbit', '업빗'),
                        level='한국'
                    )
                    df2.index = df2.index.set_levels(
                        df2.index.levels[1].str.replace('bithumb', '빗썸'),
                        level='한국'
                    )

                    if not df2.empty:
                        df2 = df2.sort_values('실환율', ascending=True)
                        await update.message.reply_text(f"🔔 테더 가격: {usdt_price:.2f}\n"
                                                        f"🔔 현재 환율 <= {down_threshold:.0f}\n"
                                                        f"\n{df2.to_string()}\n")

                await asyncio.sleep(10)
                
        except Exception as e:
            logger.error(f"Error in kimp_alarm: {str(e)}")
            logger.error(traceback.format_exc())
            await update.message.reply_text(f"Error in kimp_alarm: {str(e)}")
        finally:
            del context.user_data[event_caller]
            del context.user_data['event_caller']
            del context.user_data['stop_event']
            del context.user_data['task']

    async def get_exrate_kline_dataframe(self, symbol, interval, krw_ex, for_ex):
        krw_ex_manager = self.get_exchange_manager(krw_ex)
        for_ex_manager = self.get_exchange_manager(for_ex)

        # interval validation check
        # bithumb 4h interval disallowed
        if krw_ex_manager.config.name == 'bithumb' and (interval == '4h' or interval == '1d' or interval == '1w'):
            raise ValueError(f'can compare {interval} interval of bithumb data to others')

        krw_task = krw_ex_manager.get_kline(symbol, interval)
        usdt_task = for_ex_manager.get_kline(symbol, interval)
        tether_task = krw_ex_manager.get_kline('USDT', interval)

        krw_raw_data, usdt_raw_data, tether_raw_data = await asyncio.gather(krw_task, usdt_task, tether_task)

        cols_to_convert = ['datetime', 'open', 'high', 'low', 'close', 'target_volume', 'quote_volume']

        krw_df = pd.DataFrame(krw_raw_data)

        if krw_ex_manager.config.name in ['upbit', 'bithumb']:
            krw_df.rename(columns={'candle_date_time_kst' : 'datetime'}, inplace=True)
            krw_df.datetime = pd.to_datetime(krw_df.datetime)
            krw_df.datetime = krw_df.datetime.dt.tz_localize('Asia/Seoul')

        elif krw_ex_manager.config.name == 'coinone':
            krw_df['datetime'] = pd.to_datetime(krw_df.timestamp, unit='ms')
            krw_df.datetime = krw_df.datetime.dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul')

        elif krw_ex_manager.config.name == 'korbit':
            krw_df['datetime'] = pd.to_datetime(krw_df.timestamp, unit='ms')
            krw_df.datetime = krw_df.datetime.dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul')
            krw_df.datetime = krw_df.datetime - pd.Timedelta(hours=3)
            krw_df.sort_values('datetime', inplace=True, ascending=False)

        krw_df[ex_col_map[krw_ex_manager.config.name]] = krw_df[ex_col_map[krw_ex_manager.config.name]].apply(pd.to_numeric)
        krw_df = krw_df[['datetime'] + ex_col_map[krw_ex_manager.config.name]]

        if krw_ex_manager.config.name == 'korbit':
            cols_to_convert = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        krw_df.columns = cols_to_convert

        # print(krw_df.loc[krw_df.datetime.dt.date == pd.to_datetime('2024-10-11').date()])
        usdt_df = pd.DataFrame(usdt_raw_data, columns=ex_col_map[for_ex_manager.config.name])

        if for_ex_manager.config.name == 'gateio':
            usdt_df.rename(columns={'t' : 'timestamp', 'o' : 'open', 'h' : 'high', 'l' : 'low', 'c' : 'close', 'v' : 'target_volume', 'sum' : 'quote_volume'}, inplace=True)
        
        filtered_cols = ['timestamp', 'open', 'high', 'low', 'close', 'target_volume', 'quote_volume']

        if for_ex_manager.config.name == 'okx':
            filtered_cols = ['timestamp', 'open', 'high', 'low', 'close']

        usdt_df = usdt_df[filtered_cols]

        if for_ex_manager.config.name == 'gateio':
            usdt_df['datetime'] = pd.to_datetime(usdt_df.timestamp, unit='s')
        else:
            usdt_df['datetime'] = pd.to_datetime(usdt_df.timestamp, unit='ms')

        usdt_df.datetime = usdt_df.datetime.dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul')
        usdt_df[filtered_cols] = usdt_df[filtered_cols].apply(pd.to_numeric)

        tether_df = pd.DataFrame(tether_raw_data)
        tether_df.rename(columns={'candle_date_time_kst' : 'datetime'}, inplace=True)
        tether_df.datetime = pd.to_datetime(tether_df.datetime)
        tether_df.datetime = tether_df.datetime.dt.tz_localize('Asia/Seoul')
        tether_df[ex_col_map[krw_ex_manager.config.name]] = tether_df[ex_col_map[krw_ex_manager.config.name]].apply(pd.to_numeric)
        tether_df = tether_df[['datetime'] + ex_col_map[krw_ex_manager.config.name]]
        tether_df.columns = cols_to_convert

        # print(usdt_df.loc[usdt_df.datetime.dt.date == pd.to_datetime('2024-10-11').date(), ['datetime', 'open', 'high', 'low', 'close']]) 

        merged_df = pd.merge(krw_df, usdt_df, on='datetime', how='inner')
        merged_df['exrate'] = merged_df.apply(lambda row: round(max(row['high_x'] / row['high_y'], row['low_x'] / row['low_y']), 3), axis=1)
        merged_df = pd.merge(merged_df, tether_df, on='datetime', how='inner')
        merged_df.rename(columns={'close' : 'usdt_close'}, inplace=True)
        merged_df['kimp'] = (merged_df.exrate - merged_df.usdt_close) / merged_df.usdt_close * 100

        # merged_df.datetime = merged_df.datetime.dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul')
        # lower_quantile = merged_df.spread.quantile(0.25)
        # higher_quantile = merged_df.spread.quantile(0.75)

        # print(merged_df.loc[merged_df.datetime.dt.date == pd.to_datetime('2024-10-21').date(), ['datetime', 'close_x', 'close_y', 'kimp']])

        return merged_df[['datetime', 'close_x', 'close_y', 'exrate', 'kimp', 'usdt_close']]

    async def get_tickers_by_sub_indicators(self, update, context):
        """Find tickers with price above weekly Bollinger Band middle line"""
        event_caller = context.user_data['event_caller']
        interval = context.user_data[event_caller]['interval']
        try:
            tasks = [self.get_exchange_manager(ex).get_all_ticker_price() for ex in self.ex_lists]
            results = await asyncio.gather(*tasks)
            results_map = dict(zip(self.ex_lists, results))

            # Process the results
            upbit_data = [{'exchange': 'upbit', 'market': i.get('market').split('-')[1]} for i in results[0] if i.get('market').startswith('KRW-')]
            bithumb_data = [{'exchange': 'bithumb', 'market': i.get('market').split('-')[1]} for i in results[1] if i.get('market').startswith('KRW-')]
            bybit_data = [{'exchange': 'bybit', 'market': i.get('symbol')[re.search(r'[a-zA-Z]', i.get('symbol')).start(): i.get('symbol').find('USDT')]} for i in results[2]['result']['list'] if i.get('symbol').endswith('USDT')]
            binance_data = [{'exchange': 'binance', 'market': i.get('symbol')[re.search(r'[a-zA-Z]', i.get('symbol')).start(): i.get('symbol').find('USDT')]} for i in results_map['binance'] if i.get('symbol').endswith('USDT')]

        except Exception as e:
            logger.error(f"Error in get_tickers_by_sub_indicators: {str(e)}")
            logger.error(traceback.format_exc())

        from itertools import groupby
        from operator import itemgetter



        # Combine and deduplicate by market
        all_data = upbit_data + bithumb_data + bybit_data + binance_data
        sorted_data = sorted(all_data, key=itemgetter('market'))
        tickers = [next(g) for _, g in groupby(sorted_data, key=itemgetter('market'))]
        tickers_copy_buf = copy.deepcopy(tickers)

        macd_golden_cross = []
        macd_line_golden_cross = []
        signal_line_golden_cross = []
        batch_size = 5
        cnt = 0
        # Process tickers in batches of 5
        while len(tickers_copy_buf) > 0:
            try:
                if cnt % 100 == 0:
                    await update.callback_query.message.reply_text(f"계산중입니다... {len(tickers_copy_buf)}개 남음")

                # Get next batch of 5 tickers
                batch = list(itertools.islice(tickers_copy_buf, batch_size))
                tickers_copy_buf = tickers_copy_buf[5:]  # Remove processed tickers

                # Create tasks for batch
                batch_tasks = []
                for ticker in batch:
                    ex = self.get_exchange_manager(ticker['exchange'])
                    batch_tasks.append(ex.get_kline(ticker['market'], interval))

                # Process batch concurrently
                batch_results = await asyncio.gather(*batch_tasks)

                # Process results for each ticker in batch
                for ticker, result in zip(batch, batch_results):
                    # Modify the code
                    if ticker['exchange'] in ['upbit', 'bithumb']:
                        df = pd.DataFrame(result)
                    elif ticker['exchange'] == 'bybit':
                        df = pd.DataFrame(result, columns=['timestamp', 'open', 'high', 'low', 'close', 'target_volume', 'quote_volume'])
                    elif ticker['exchange'] == 'binance':
                        df = pd.DataFrame(result, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])

                    # Standardize columns
                    df = df.rename(columns=KLINE_COLUMN_MAP[ticker['exchange']])
                    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
                    price_col = 'close'
                    df = df.apply(pd.to_numeric)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    df.sort_values('timestamp', inplace=True)

                    # Calculate SMA
                    df['sma20'] = df[price_col].rolling(window=20).mean()
                    current_price = float(df[price_col].iloc[-1])
                    current_sma = float(df['sma20'].iloc[-1])

                    # Calculate MACD
                    exp1 = df[price_col].ewm(span=12, adjust=False).mean()
                    exp2 = df[price_col].ewm(span=26, adjust=False).mean()
                    df['macd'] = exp1 - exp2
                    df['signal'] = df['macd'].ewm(span=9, adjust=False).mean()

                # insert_kline_data(df, ticker['market'], ticker['exchange'], interval)
                
                # Check conditions:
                # 1. Bollinger Band middle line
                # 2. MACD golden cross
                # 3. MACD line golden cross
                # 4. signal line golden cross
                is_golden_cross = (df['macd'].iloc[-1] > df['signal'].iloc[-1]) and (df['macd'].iloc[-2] <= df['signal'].iloc[-2])
                is_macd_line_golden_cross = (df['macd'].iloc[-1] > 0) and (df['macd'].iloc[-2] <= 0)
                is_signal_line_golden_cross = (df['signal'].iloc[-1] > 0) and (df['signal'].iloc[-2] <= 0)

                if current_price > current_sma and is_golden_cross:
                    logger.info(f"{ticker['market']} is above BB middle line with MACD golden cross")
                    macd_golden_cross.append(ticker['market'])
                
                if current_price > current_sma and is_macd_line_golden_cross:
                    logger.info(f"{ticker['market']} is above BB middle line with MACD line golden cross")
                    macd_line_golden_cross.append(ticker['market'])

                if current_price > current_sma and is_signal_line_golden_cross:
                    logger.info(f"{ticker['market']} is above BB middle line with signal line golden cross")
                    signal_line_golden_cross.append(ticker['market'])

                cnt += batch_size

                await asyncio.sleep(0.5)

            except IndexError as e:
                logger.error(f'{ticker}')
                logger.error(f'{df}')
                logger.error(f"Error in get_tickers_by_sub_indicators: {str(e)}")
                logger.error(traceback.format_exc())

            except Exception as e:
                logger.error(f'{ticker}')
                logger.error(f'{df}')
                logger.error(f"Error in get_tickers_by_sub_indicators: {str(e)}")
                logger.error(traceback.format_exc())
                
        str_translate = {
            '1w' : '주봉',
            '1d' : '일봉',
            '1h' : '1시간봉',
        }

        tmp_buffer = []
        for ticker in tickers:
            if ticker['market'] in macd_golden_cross and \
                ticker['market'] in macd_line_golden_cross and \
                ticker['market'] in signal_line_golden_cross:
                tmp_buffer.append(f'{ticker['market']} : {ticker['exchange']}')
                tickers.remove(ticker)

        try:
        
            if tmp_buffer:
                await update.callback_query.message.reply_text(f"🔔 {str_translate[interval]} MACD 골든크로스\n"
                                                            f"🔔 {str_translate[interval]} MACD 라인 0선 돌파\n"
                                                            f"🔔 {str_translate[interval]} 시그널 라인 0선 돌파\n"
                                                            f"{'\n'.join(tmp_buffer)}")
            
            tmp_buffer = []
            for ticker in tickers:
                if ticker['market'] in macd_golden_cross and \
                ticker['market'] in macd_line_golden_cross:
                    tmp_buffer.append(f'{ticker['market']} : {ticker['exchange']}')
                    tickers.remove(ticker)
            
            if tmp_buffer:
                await update.callback_query.message.reply_text(f"🔔 {str_translate[interval]} MACD 골든크로스\n"
                                                            f"🔔 {str_translate[interval]} MACD 라인 0선 돌파\n"
                                                            f"{'\n'.join(tmp_buffer)}")
            tmp_buffer = []
            for ticker in tickers:
                if ticker['market'] in macd_golden_cross and \
                ticker['market'] in signal_line_golden_cross:
                    tmp_buffer.append(f'{ticker['market']} : {ticker['exchange']}')
                    tickers.remove(ticker)
                
            if tmp_buffer:
                await update.callback_query.message.reply_text(f"🔔 {str_translate[interval]} MACD 골든크로스\n"
                                                            f"🔔 {str_translate[interval]} 시그널 라인 0선 돌파\n"
                                                            f"{'\n'.join(tmp_buffer)}")
            tmp_buffer = []
            for ticker in tickers:
                if ticker['market'] in macd_line_golden_cross and \
                ticker['market'] in signal_line_golden_cross:
                    tmp_buffer.append(f'{ticker['market']} : {ticker['exchange']}')
                    tickers.remove(ticker)
            
            if tmp_buffer:
                await update.callback_query.message.reply_text(f"🔔 {str_translate[interval]} MACD 라인 0선 돌파\n"
                                                            f"🔔 {str_translate[interval]} 시그널 라인 0선 돌파\n"
                                                            f"{'\n'.join(tmp_buffer)}")
            tmp_buffer = []
            for ticker in tickers:
                if ticker['market'] in macd_golden_cross:
                    tmp_buffer.append(f'{ticker['market']} : {ticker['exchange']}')
                    tickers.remove(ticker)

            if tmp_buffer:
                await update.callback_query.message.reply_text(f"🔔 {str_translate[interval]} MACD 골든크로스\n"
                                                                f"{'\n'.join(tmp_buffer)}")
            tmp_buffer = []
            for ticker in tickers:
                if ticker['market'] in macd_line_golden_cross:
                    tmp_buffer.append(f'{ticker['market']} : {ticker['exchange']}')
                    tickers.remove(ticker)
            
            if tmp_buffer:
                await update.callback_query.message.reply_text(f"🔔 {str_translate[interval]} MACD 라인 0선 돌파\n"
                                                                f"{'\n'.join(tmp_buffer)}")
            tmp_buffer = []
            for ticker in tickers:
                if ticker['market'] in signal_line_golden_cross:
                    tmp_buffer.append(f'{ticker['market']} : {ticker['exchange']}')
                    tickers.remove(ticker)
            
            if tmp_buffer:
                await update.callback_query.message.reply_text(f"🔔 {str_translate[interval]} 시그널 라인 0선 돌파\n"
                                                                f"{'\n'.join(tmp_buffer)}")
                
        except Exception as e:
            logger.error(f"Error in get_tickers_by_sub_indicators: {str(e)}")
            logger.error(traceback.format_exc())
            await update.callback_query.message.reply_text(f"Error in get_tickers_by_sub_indicators: {str(e)}")
        finally:
            del context.user_data[event_caller]
            del context.user_data['event_caller']
            del context.user_data['task']

    def plot_img(self, data:list, title, xlabel, ylabel, twinx_yn=False):
        '''
        :dict : [
            {
                df:pd.DataFrame : df,
                x_column: x_column, 
                y_column: y_column,
                label: label
            },
            ...
        ]
        '''
        # Create a new figure and axis
        fig, ax = plt.subplots()

        if not twinx_yn:
            for item in data:
                ax.plot(item['df'][item['x_column']], item['df'][item['y_column']], label=item['label'])
        else:
            for idx, item in enumerate(data):
                if idx == 0:
                    ax.plot(item['df'][item['x_column']], item['df'][item['y_column']], label=item['label'])
                else:
                    tmpx = ax.twinx()
                    tmpx.plot(item['df'][item['x_column']], item['df'][item['y_column']], label=item['label'])

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        plt.title(title)
        plt.legend()

        fig.autofmt_xdate() # 없으면 날짜 겹침 

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        
        buffer.seek(0)

        buf_trs = types.BufferedInputFile(buffer.getvalue(), filename=title)

        plt.close()

        return buf_trs

    async def send_telegram(self, message, message_type='text'):
        result = None
        try:
            if message_type == 'text':
                result = await bot.send_message(chat_id=chat_id, text=message)
            elif message_type == 'photo':
                result = await bot.send_photo(chat_id=chat_id, photo=message)
        except aiohttp.ClientError as e:
            logger.error(f"Error occurred while sending telegram : {e}")
        except Exception as e:
            logger.error(f"Error occurred while sending telegram : {e}")
            logger.error(traceback.format_exc())
        return result
            
    async def request_exrate_picture(self, symbol, interval, krw_ex, for_ex):
        try:
            df = await self.get_exrate_kline_dataframe(symbol, interval, krw_ex, for_ex)
            data = [
                {
                    'df' : df,
                    'x_column' : 'datetime',
                    'y_column' : 'exrate',
                    'label' : f'{symbol} exrate'
                },
                {
                    'df' : df,
                    'x_column' : 'datetime',
                    'y_column' : 'usdt_close',
                    'label' : 'USDT'
                }
            ]
            buf = self.plot_img(data, 'exrate_kline', 'datetime', 'exrate')
            await self.send_telegram(buf, message_type='photo')
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(e);

    @staticmethod
    def calculate_market_order_amount(symbol, orderbook, seed, is_buy, fee_rate):
        price_level = orderbook['asks'] if is_buy else orderbook['bids']
        logger.debug(f'\n orderbook : {price_level} \
                       \n length : {len(price_level)} \
                       \n seed : {seed} \
                       \n is_buy : {is_buy} \
                       \n fee_rate : {fee_rate}')
        end_flag = False
        if is_buy:
            # Reduce seed money by the fee
            effective_money = seed * (1 - fee_rate)
            total_fee = seed * fee_rate
            total_bought_coin = 0
            remaining_money = effective_money

            try:
                if len(price_level[0]) == 2:
                    # Iterate through the ask prices and calculate the amount of coins you can buy
                    for price, volume in price_level:
                        if remaining_money >= price * volume:
                            total_bought_coin += volume
                            remaining_money -= price * volume
                        else:
                            total_bought_coin += remaining_money / price
                            end_flag = True
                            break
                    if not end_flag:
                        logger.debug(f'{seed} overflow the orderbook of {symbol} {"buy" if is_buy else "sell"}')
                        return 0,0
                elif len(price_level[0]) == 3:
                    for price, volume, _ in price_level:
                        if remaining_money >= price * volume:
                            total_bought_coin += volume
                            remaining_money -= price * volume
                        else:
                            total_bought_coin += remaining_money / price
                            end_flag = True
                            break
                    if not end_flag:
                        logger.debug(f'{seed} overflow the orderbook of {symbol} {"buy" if is_buy else "sell"}')
                        return 0,0
                else:
                    raise ValueError
                
                return total_bought_coin, total_fee
            
            except Exception as e:
                logger.error(e)
                logger.info(f'\n orderbook : {price_level} \
                              \n length : {len(price_level)} \
                              \n seed : {seed} \
                              \n is_buy : {is_buy} \
                              \n fee_rate : {fee_rate}')


        else:
            total_revenue = 0
            remaining_coins = seed

            # Iterate through the bids to calculate revenue
            try:
                if len(price_level[0]) == 2:
                    for price, volume in price_level:
                        if remaining_coins >= volume:
                            total_revenue += price * volume
                            remaining_coins -= volume
                        else:
                            total_revenue += price * remaining_coins
                            end_flag = True
                            break
                    if not end_flag:
                        logger.debug(f'{seed} overflow the orderbook of {symbol} {"buy" if is_buy else "sell"}')
                        return 0,0
                elif len(price_level[0]) == 3:
                    for price, volume, _ in price_level:
                        if remaining_coins >= volume:
                            total_revenue += price * volume
                            remaining_coins -= volume
                        else:
                            total_revenue += price * remaining_coins
                            end_flag = True
                            break
                    if not end_flag:
                        logger.debug(f'{seed} overflow the orderbook of {symbol} {"buy" if is_buy else "sell"}')
                        return 0,0
                else:
                    raise ValueError

                # Reduce revenue by the selling fee
                return total_revenue * (1 - fee_rate), total_revenue * fee_rate
            
            except Exception as e:
                logger.error(e)
                logger.info(f'\n orderbook : {price_level} \
                              \n length : {len(price_level)} \
                              \n seed : {seed} \
                              \n is_buy : {is_buy} \
                              \n fee_rate : {fee_rate}')
        
    async def run_future_auto_trading(self, update, context):
        event_caller = context.user_data['event_caller']
        symbol = context.user_data[event_caller]['symbol']
        budget = context.user_data[event_caller]['budget']
        leverage = context.user_data[event_caller]['leverage']
        stop_loss = context.user_data[event_caller]['stop_loss']
        strategy = context.user_data[event_caller]['strategy']
        for_ex = context.user_data[event_caller]['for_ex']
        interval = context.user_data[event_caller]['interval']

        if strategy == 'bb':
            bb_strategy = BollingerBandStrategy({
                'symbol' : symbol,
                'budget' : budget,
                'leverage' : leverage,
                'stop_loss' : stop_loss,
                'for_ex' : for_ex,
                'interval' : interval, 
                'kline_data' : df, 
                'context' : context
            })
            bb_strategy.run()
        
        elif strategy == 'rsi':
            pass

class TradingDataManager:
    def __init__(self, ex):
        self.ex_name = ex
        self.ex = globals()[f'{ex.capitalize()}Manager']()

    async def get_kline_data(self, symbol, interval)->pd.DataFrame:
        kline_raw_data = await self.ex.get_kline(symbol, interval)
        df = pd.DataFrame(kline_raw_data, columns=KLINE_RESPONSE_MAP[self.ex_name])

        if self.ex_name == 'gateio':
            df.rename(columns={'t' : 'timestamp', 'o' : 'open', 'h' : 'high', 'l' : 'low', 'c' : 'close', 'v' : 'target_volume', 'sum' : 'quote_volume'}, inplace=True)

        filtered_cols = ['timestamp', 'open', 'high', 'low', 'close', 'target_volume', 'quote_volume']

        if self.ex_name == 'okx':
            filtered_cols = ['timestamp', 'open', 'high', 'low', 'close']
        
        df = df[filtered_cols]

        if self.ex_name == 'gateio':
            df['datetime'] = pd.to_datetime(df.timestamp, unit='s')
        else:
            df['datetime'] = pd.to_datetime(df.timestamp, unit='ms')

        df.datetime = df.datetime.dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul')
        df[filtered_cols].apply(pd.to_numeric, inplace=True)
        df.sort_values('timestamp', inplace=True)

        return df

    async def get_position(self)->dict:
        res_list = await self.ex.get_all_positions()
        for item in res_list:
            if item['symbol'] == self.symbol:
                return item
        return None

    async def get_balance(self):
        if self.ex_name in ['upbit', 'bithumb']:
            return await self.ex.get_balance('KRW')
        return await self.ex.get_balance('USDT')
    
class TradingBroker:
    def __init__(self, ex):
        self.ex_name = ex
        self.ex = globals()[f'{ex.capitalize()}Manager']()

    async def position_entry(self, PositionEntryIn: PositionEntryIn):
        params = PositionEntryIn.to_exchange(PositionEntryIn, self.ex_name)
        res = await self.ex.post_order(params)

        if self.ex_name == 'bybit':
            res = await self.ex.post_order(params)
            if res.get('ret_msg') == 'ok':
                return res
            raise ValueError(res)
        
        elif self.ex_name == 'binance':
            res = await self.ex.post_order(params)
            if res.get('status') == 'NEW':
                return res
            raise ValueError(res)
        
        elif self.ex_name == 'upbit':
            res = await self.ex.post_order(params)
            if res.get('uuid'):
                return res
            raise ValueError(res)
        
        elif self.ex_name == 'bithumb':
            res = await self.ex.post_order(params)
            if res.get('uuid'):
                return res
            raise ValueError(res)
    
        return ValueError('Invalid exchange name')
    
class AutoTradingStrategies(ABC):
    def init(self, *args, **kwargs):
        self.symbol = kwargs.get('symbol')
        self.ex = kwargs.get('ex')
        self.budget = 'all' if kwargs.get('budget') == 'all' else kwargs.get('budget')
        self.pyramiding = kwargs.get('pyramiding')
        self.entry_cnt = 0
        self.leverage = kwargs.get('leverage')
        self.stop_loss = kwargs.get('stop_loss')
        self.interval = kwargs.get('interval')
        self.kline_data = kwargs.get('kline_data')
        self.context = kwargs.get('context')
        self.update = kwargs.get('update')

    @abstractmethod
    def run(self, *args, **kwargs):
        pass
    
    async def position_entry(self, signal):
        tdMgr = TradingDataManager(self.for_ex)
        position = await tdMgr.get_position()

        if self.budget == 'all':
            self.budget = await tdMgr.get_balance()

        if signal == 'bid':
            if position:
                # check the position side
                if position['side'] == 'bid':
                    logger.info(f"Already in a long position for {self.symbol}")
                    return False
                else:
                    t = TradingBroker(self.ex)
                    res = await t.position_entry(PositionEntryIn(symbol=self.symbol, side='ask', order_type='market', qty=position['size']))
                    logger.info(res)

                    if self.budget == 'all':
                        self.budget = await tdMgr.get_balance()

                    res = await t.position_entry(PositionEntryIn(symbol=self.symbol, side='bid', order_type='market', qty=self.budget))
                    logger.info(res)
            else:
                res = await TradingBroker(self.ex).position_entry(PositionEntryIn(symbol=self.symbol, side='bid', order_type='market', qty=self.budget))
                logger.info(res)

        elif signal == 'ask':
            if position:
                # check the position side
                if position['side'] == 'ask':
                    logger.info(f"Already in a short position for {self.symbol}")
                    return False
                else:
                    t = TradingBroker(self.ex)
                    res = await t.position_entry(PositionEntryIn(symbol=self.symbol, side='bid', order_type='market', qty=position['size']))
                    logger.info(res)

                    if self.budget == 'all':
                        self.budget = await tdMgr.get_balance()

                    res = await t.position_entry(PositionEntryIn(symbol=self.symbol, side='ask', order_type='market', qty=self.budget))
                    logger.info(res)
            else:
                res = await TradingBroker(self.ex).position_entry(PositionEntryIn(symbol=self.symbol, side='ask', order_type='market', qty=self.budget))
                logger.info(res)

        return {'symbol' : self.symbol, 'ex' : self.ex, 'size' : self.budget, 'leverage' : self.leverage}

class BollingerBandStrategy(AutoTradingStrategies):
    async def run_strategy_loop(self):
        try:
            while True:
                now = pd.Timestamp.now(tz='Asia/Seoul')
                minutes_to_next = interval_minutes[self.interval] - (now.minute % interval_minutes[self.interval])
                seconds_to_next = minutes_to_next * 60 - now.second - 3 # 시간차이로 인한 3초 보정

                # Sleep until next candle close
                await asyncio.sleep(seconds_to_next)

                tdmgr = TradingDataManager(self.ex)
                kline_df = await tdmgr.get_kline_data()

                # Process signals
                entry_signal = self.process_signals(kline_df)
                if entry_signal == 0:
                    logger.info(f"No signal for {self.symbol}")
                    await self.update.message.reply_text(f"No signal for {self.symbol}")
                    continue

                # Entry position
                res = await self.position_entry(entry_signal)
                if not res:
                    logger.info(f"Failed to enter position for {self.symbol}")
                    return
                
                await self.update.message.reply_text(f"티커        : {res['symbol']}\n"
                                                     f"거래소       : {res['ex']}\n"
                                                     f"레버리지     : {res['leverage']}\n"
                                                     f"체결량       : {res['size']}\n",
                                                     parse_mode='Markdown')


        except Exception as e:
            logger.error(f"Error in BollingerBandStrategy: {str(e)}")
            logger.error(traceback.format_exc())

    def process_signals(self, kline_data:pd.DataFrame):
        # Calculate Bollinger Bands
        kline_data['sma'] = kline_data['close'].rolling(window=20).mean()
        kline_data['std'] = kline_data['close'].rolling(window=20).std()
        kline_data['upper_band'] = kline_data['sma'] + (kline_data['std'] * 2)
        kline_data['lower_band'] = kline_data['sma'] - (kline_data['std'] * 2)

        last_row = kline_data.iloc[-1]
        
        # Generate signal
        if last_row['close'] > last_row['upper_band']:
            signal = 'ask'  # Sell signal
        elif last_row['close'] < last_row['lower_band']:
            signal = 'bid'   # Buy signal
        else:
            signal = 0   # No signal

        return signal
            
# Define states

# 티커 환율 그래프
ASK_KLINE_INTERVAL, ASK_KLINE_KRW_EXCHANGE, ASK_KLINE_FOR_EXCHANGE, HANDLE_KLINE = range(1, 5)

# 티커 실시간 환율
ASK_EXRATE_BUDGET, ASK_EXRATE_KRW_EXCHANGE, ASK_EXRATE_FOR_EXCHANGE, HANDLE_EXRATE = range(5, 9)

# 아비트리지 감시
ASK_MONITORING_DOWN_THRESHOLD, ASK_MONITORING_SEED, HANDLE_MONITORING = range(16, 19)

# 김프포지션 진입
ASK_POSITION_ENTRY_EXRATE, ASK_POSITION_ENTRY_BUDGET, ASK_POSITION_ENTRY_LEVERAGE, ASK_POSITION_KRW_EX, ASK_POSITION_FOR_EX, HANDLE_ENTRY_POSITION = range(19, 25)

# 김프포지션 종료
ASK_POSITION_CLOSE_EXRATE, HANDLE_CLOSE_POSITION = range(25, 27)

# 선물자동매매
ASK_FUTURE_AUTOTRADING_BUDGET, ASK_FUTURE_AUTOTRADING_LEVERAGE, ASK_FUTURE_AUTOTRADING_STOP_LOSS, ASK_FUTURE_AUTOTRADING_STRATEGY, HANDLE_FUTURE_AUTOTRADING_BB, ASK_FUTURE_AUTOTRADING_FOR_EX, HANDLE_FUTURE_AUTOTRADING = range(27, 34)

# 보조지표로 종목찾기
HANDLE_SUB_INDICATOR = 34

# 실거래량 감시
ASK_REAL_VOLUME_MULTIPLIER = 35

# Dictionary to map options to their states and handlers
conversation_options = {
    "티커 환율 그래프": {
        "entry_action": "ask_kline_symbol",
        "state": [ASK_KLINE_INTERVAL, ASK_KLINE_KRW_EXCHANGE, ASK_KLINE_FOR_EXCHANGE, HANDLE_KLINE],
        "handler": ["ask_kline_interval", "ask_kline_krw_exchange", "ask_kline_for_exchange", "handle_kline"],
        "handler_type": ["MessageHandler", "CallbackQueryHandler", "CallbackQueryHandler", "CallbackQueryHandler"]
    },
    "티커 실시간 환율": {
        "entry_action": "ask_symbol",
        "state": [ASK_EXRATE_BUDGET, ASK_EXRATE_KRW_EXCHANGE, ASK_EXRATE_FOR_EXCHANGE, HANDLE_EXRATE],
        "handler": ["ask_exrate_budget", "ask_exrate_krw_exchange", "ask_exrate_for_exchange", "handle_exrate"],
        "handler_type": ["MessageHandler", "MessageHandler", "CallbackQueryHandler", "CallbackQueryHandler"]
    },
    "보조지표로 종목추천": {
        "entry_action": "ask_interval",
        "state": [HANDLE_SUB_INDICATOR],
        "handler": ['handle_sub_indicator'],
        "handler_type": ["CallbackQueryHandler"]
    },
    "아비트리지 감시": {
        "entry_action": "ask_monitoring_up_threshold",
        "state": [ASK_MONITORING_DOWN_THRESHOLD, ASK_MONITORING_SEED, HANDLE_MONITORING],
        "handler": ['ask_monitoring_down_threshold', 'ask_monitoring_seed', 'handle_monitoring'],
        "handler_type": ["MessageHandler", "MessageHandler", "MessageHandler"]
    },
    "실거래량 감시": {
        "entry_action": "ask_real_volume_threshold",
        "state": [ASK_REAL_VOLUME_MULTIPLIER],
        "handler": ['ask_real_volume_multiplier'],
        "handler_type": ["MessageHandler"]
    },  
    "선물자동매매": {
        "entry_action": "ask_future_autotrading_symbol",
        "state": [ASK_FUTURE_AUTOTRADING_BUDGET, 
                  ASK_FUTURE_AUTOTRADING_LEVERAGE, 
                  ASK_FUTURE_AUTOTRADING_FOR_EX, 
                  ASK_FUTURE_AUTOTRADING_STRATEGY, 
                  HANDLE_FUTURE_AUTOTRADING,
                  HANDLE_FUTURE_AUTOTRADING_BB],
        "handler": ['ask_future_autotrading_budget', 
                    'ask_future_autotrading_leverage', 
                    'ask_future_autotrading_for_ex', 
                    'ask_future_autotrading_strategy', 
                    'handle_future_autotrading',
                    'handle_future_autotrading_bb'],
        "handler_type": ["MessageHandler", 
                         "MessageHandler", 
                         "MessageHandler", 
                         "CallbackQueryHandler", 
                         "CallbackQueryHandler", 
                         "CallbackQueryHandler"]
    },
    "김프포지션 진입": {
        "entry_action": "ask_position_symbol",
        "state": [ASK_POSITION_ENTRY_EXRATE, ASK_POSITION_ENTRY_BUDGET, ASK_POSITION_ENTRY_LEVERAGE, ASK_POSITION_KRW_EX, ASK_POSITION_FOR_EX, HANDLE_ENTRY_POSITION],
        "handler": ['ask_position_entry_exrate', 'ask_position_entry_budget', 'ask_position_entry_leverage', 'ask_position_krw_ex', 'ask_position_for_ex', 'handle_entry_position'],
        "handler_type": ["MessageHandler", "MessageHandler", "MessageHandler", "MessageHandler", "CallbackQueryHandler", "CallbackQueryHandler"]
    },
    "김프포지션 종료": {
        "entry_action": "ask_position_close_symbol",
        "state": [ASK_POSITION_CLOSE_EXRATE, HANDLE_CLOSE_POSITION],
        "handler": ['ask_position_close_exrate', 'handle_close_position'],
        "handler_type": ["MessageHandler", "MessageHandler"]
    },
}

# Function to handle the /start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Create a reply keyboard
    keyboard = [
        [KeyboardButton("보조지표로 종목추천")],
        [KeyboardButton("선물자동매매")],
        [KeyboardButton("아비트리지 감시")],
        [KeyboardButton("실거래량 감시")],
        [KeyboardButton("티커 실시간 환율"), KeyboardButton("티커 환율 그래프")],
        [KeyboardButton("김프포지션 진입"), KeyboardButton("김프포지션 종료")],
        [KeyboardButton("봇 종료"), KeyboardButton("취소")]
    ]
    reply_markup = ReplyKeyboardMarkup(
        keyboard, 
        resize_keyboard=True,  # Adjust button size to fit screen width
        one_time_keyboard=True  # Hide keyboard after user clicks a button
    )

    # Send the message with the keyboard
    await update.message.reply_text(
        "Choose an option from the keyboard below:", 
        reply_markup=reply_markup
    )

def safe_handler(func):
    @wraps(func)
    async def wrapper(update, context, *args, **kwargs):
        try:
            # If user sends "Cancel" or "Stop", stop processing further
            if update.message and update.message.text.lower() in ["cancel", "stop"]:
                await update.message.reply_text("Action canceled.")
            
            # Call the original handler function
            return await func(update, context, *args, **kwargs)
        except Exception as e:
            # Handle any exceptions in the handlers
            logger.error(f"Error in handler {func.__name__}: {str(e)}\nTraceback:\n{traceback.format_exc()}")

            if update.message:
                await update.message.reply_text(f"An error occurred: {str(e)}")

            if update.callback_query:
                await update.callback_query.message.reply_text(f"An error occurred: {str(e)}")

            return ConversationHandler.END
    return wrapper

# 티커 환율 그래프 버튼 핸들러
@safe_handler
async def ask_kline_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    option = update.message.text
    context.user_data['event_caller'] = option
    context.user_data[option] = {}
    await update.message.reply_text("티커를 입력해주세요 : (예: XRP)")
    return ASK_KLINE_INTERVAL

@safe_handler
async def ask_kline_interval(update: Update, context: ContextTypes.DEFAULT_TYPE):
    event_caller = context.user_data['event_caller']
    context.user_data[event_caller]["symbol"] = update.message.text
    
    # Define the buttons
    keyboard = [
        [
            InlineKeyboardButton("1m", callback_data="1m"),
            InlineKeyboardButton("5m", callback_data="5m"),
            InlineKeyboardButton("15m", callback_data="15m"),
            InlineKeyboardButton("30m", callback_data="30m"),
            InlineKeyboardButton("1h", callback_data="1h"),
        ],
    ]
    
    # Create an inline keyboard markup
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Send the message with the inline keyboard
    await update.message.reply_text("봉을 선택해주세요", reply_markup=reply_markup)

    return ASK_KLINE_KRW_EXCHANGE

@safe_handler
async def ask_kline_krw_exchange(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()  # Acknowledge the callback

    event_caller = context.user_data['event_caller']
    context.user_data[event_caller]["interval"] = query.data

    # Define the buttons
    keyboard = [
        [
            InlineKeyboardButton("upbit", callback_data="upbit"),
            InlineKeyboardButton("bithumb", callback_data="bithumb"),
        ],
    ]
    
    # Create an inline keyboard markup
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Send the message with the inline keyboard
    await query.edit_message_text("한국 거래소를 선택해주세요:", reply_markup=reply_markup)

    return ASK_KLINE_FOR_EXCHANGE

@safe_handler
async def ask_kline_for_exchange(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()  # Acknowledge the callback

    event_caller = context.user_data['event_caller']
    context.user_data[event_caller]["krw_ex"] = query.data

    # Define the buttons
    keyboard = [
        [
            InlineKeyboardButton("bybit", callback_data="bybit"),
            InlineKeyboardButton("binance", callback_data="binance"),
        ],
    ]
    
    # Create an inline keyboard markup
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Send the message with the inline keyboard
    await query.edit_message_text("해외 거래소를 선택해주세요", reply_markup=reply_markup)

    return HANDLE_KLINE

@safe_handler
async def handle_kline(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()  # Acknowledge the callback

    event_caller = context.user_data['event_caller']
    context.user_data[event_caller]["for_ex"] = query.data
    
    try:
        k = KimpManager()
        await k.request_exrate_picture(context.user_data[event_caller]["symbol"], \
                                       context.user_data[event_caller]["interval"], \
                                       context.user_data[event_caller]["krw_ex"], \
                                       context.user_data[event_caller]["for_ex"])
    except Exception as e:
        logger.error(e)
        await update.message.reply_text(str(e))
    finally:
        del context.user_data[event_caller]
        del context.user_data['event_caller']
        return ConversationHandler.END


# 티커 실시간환율 버튼 핸들러
@safe_handler
async def ask_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    option = update.message.text
    context.user_data['event_caller'] = option
    context.user_data[option] = {}
    await update.message.reply_text("티커를 입력해주세요 : (예: XRP)")
    return ASK_EXRATE_BUDGET

@safe_handler
async def ask_exrate_budget(update: Update, context: ContextTypes.DEFAULT_TYPE):
    event_caller = context.user_data['event_caller']
    context.user_data[event_caller]["symbol"] = update.message.text
    await update.message.reply_text("진입예산을 입력해주세요 : (예: 1000000)")
    return ASK_EXRATE_KRW_EXCHANGE

@safe_handler
async def ask_exrate_krw_exchange(update: Update, context: ContextTypes.DEFAULT_TYPE):
    event_caller = context.user_data['event_caller']
    context.user_data[event_caller]["budget"] = float(update.message.text)

    # Define the buttons
    keyboard = [
        [
            InlineKeyboardButton("upbit", callback_data="upbit"),
            InlineKeyboardButton("bithumb", callback_data="bithumb"),
        ],
    ]
    
    # Create an inline keyboard markup
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Send the message with the inline keyboard
    await update.message.reply_text("한국 거래소를 선택하세요", reply_markup=reply_markup)

    return ASK_EXRATE_FOR_EXCHANGE

@safe_handler
async def ask_exrate_for_exchange(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()  # Acknowledge the callback

    event_caller = context.user_data['event_caller']
    context.user_data[event_caller]["krw_ex"] = query.data

    # Define the buttons
    keyboard = [
        [
            InlineKeyboardButton("bybit", callback_data="bybit"),
            InlineKeyboardButton("binance", callback_data="binance"),
        ],
    ]
    
    # Create an inline keyboard markup
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Send the message with the inline keyboard
    await query.edit_message_text("해외 거래소를 선택하세요", reply_markup=reply_markup)

    return HANDLE_EXRATE

@safe_handler
async def exrate_task(update: Update, context: ContextTypes.DEFAULT_TYPE):
    event_caller = context.user_data['event_caller']
    query = update.callback_query
    await query.answer()  # Acknowledge the callback

    context.user_data[event_caller]["for_ex"] = query.data
    
    try:
        k = KimpManager()
        while not context.user_data['stop_event'].is_set():
            task = []
            task.append(k.get_single_ticker_kimp_by_seed(context.user_data[event_caller]["symbol"], \
                                                         context.user_data[event_caller]["krw_ex"], \
                                                         context.user_data[event_caller]["for_ex"], \
                                                         context.user_data[event_caller]["budget"]))

            upbit = k.get_exchange_manager('upbit')
            task.append(upbit.get_single_ticker_price('USDT'))

            result = await asyncio.gather(*task)
            df = result[0]
            usdt_price = result[1][0]['trade_price']

            await update.callback_query.message.reply_text(f"진입예산       : {int(df.seed[0])}\n"
                                                           f"테더가격       : {usdt_price}\n"
                                                           f"현재환율       : {int(df.exrate[0])}", \
                                                           parse_mode='Markdown')
            
            await asyncio.sleep(0.5)
    except Exception as e:
        logger.error(e)
        logger.error(traceback.format_exc())
        await update.callback_query.message.reply_text(str(e))
    finally:
        del context.user_data[event_caller]
        del context.user_data['event_caller']
        del context.user_data['stop_event']
        del context.user_data['task']

@safe_handler
async def handle_exrate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()  # Acknowledge the callback

    event_caller = context.user_data['event_caller']
    context.user_data[event_caller]["for_ex"] = query.data

    # Initialize the stop event
    context.user_data["stop_event"] = asyncio.Event()

    task = asyncio.create_task(exrate_task(update, context))
    context.user_data["task"] = task
    return ConversationHandler.END # conversational handler의 마지막에 이게 들어가지 않고, 다른 핸들러가 수행되면, 예상치 못한 이슈가 발생할 수 있음.


# 아비트리지 감시 버튼 핸들러
async def ask_monitoring_up_threshold(update: Update, context: ContextTypes.DEFAULT_TYPE):
    option = update.message.text
    context.user_data['event_caller'] = option
    context.user_data[option] = {}
    await update.message.reply_text("감시할 상한 김프를 입력해주세요:")
    return ASK_MONITORING_DOWN_THRESHOLD

async def ask_monitoring_down_threshold(update: Update, context: ContextTypes.DEFAULT_TYPE):
    event_caller = context.user_data['event_caller']
    context.user_data[event_caller]["up_threshold"] = float(update.message.text)
    await update.message.reply_text("감시할 하한 김프를 입력해주세요:")
    return ASK_MONITORING_SEED

@safe_handler
async def ask_monitoring_seed(update: Update, context: ContextTypes.DEFAULT_TYPE):
    event_caller = context.user_data['event_caller']
    context.user_data[event_caller]["down_threshold"] = float(update.message.text)
    await update.message.reply_text("진입예산을 입력해주세요")
    return HANDLE_MONITORING

@safe_handler
async def handle_monitoring(update: Update, context: ContextTypes.DEFAULT_TYPE):
    event_caller = context.user_data['event_caller']
    context.user_data[event_caller]["seed"] = float(update.message.text)

    # Initialize the stop event
    context.user_data["stop_event"] = asyncio.Event()
    
    k = KimpManager()
    task = asyncio.create_task(k.run_kimp_alarm(update, context))
    context.user_data["task"] = task
    return ConversationHandler.END


# 실거래량 감시 버튼 핸들러
@safe_handler
async def ask_real_volume_threshold(update: Update, context: ContextTypes.DEFAULT_TYPE):
    option = update.message.text
    context.user_data['event_caller'] = option
    context.user_data[option] = {}
    await update.message.reply_text("실거래량의 배수를 입력해주세요 : (예: 10)")
    return ASK_REAL_VOLUME_MULTIPLIER

@safe_handler
async def ask_real_volume_multiplier(update: Update, context: ContextTypes.DEFAULT_TYPE):
    event_caller = context.user_data['event_caller']
    context.user_data[event_caller]["multiplier"] = float(update.message.text)

    # Initialize the stop event
    context.user_data["stop_event"] = asyncio.Event()

    k = KimpManager()
    task = asyncio.create_task(k.run_big_volume_alarm(update, context))
    context.user_data["task"] = task
    return ConversationHandler.END


# 김프포지션 진입 버튼 핸들러
@safe_handler
async def ask_position_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    option = update.message.text
    context.user_data['event_caller'] = option
    context.user_data[option] = {}

    await update.message.reply_text("티커를 입력해주세요 : (예: XRP)")
    return ASK_POSITION_ENTRY_EXRATE

@safe_handler
async def ask_position_entry_exrate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    event_caller = context.user_data['event_caller']
    context.user_data[event_caller]["symbol"] = (update.message.text).upper()
    
    await update.message.reply_text("진입환율을 입력해주세요(단, 0을 입력하면 즉시진입)")

    return ASK_POSITION_ENTRY_BUDGET

@safe_handler
async def ask_position_entry_budget(update: Update, context: ContextTypes.DEFAULT_TYPE):
    event_caller = context.user_data['event_caller']
    context.user_data[event_caller]["entry_exrate"] = float(update.message.text)

    await update.message.reply_text("진입예산을 입력해주세요 :")
    return ASK_POSITION_ENTRY_LEVERAGE

@safe_handler
async def ask_position_entry_leverage(update: Update, context: ContextTypes.DEFAULT_TYPE):
    event_caller = context.user_data['event_caller']
    context.user_data[event_caller]["entry_budget"] = float(update.message.text)

    await update.message.reply_text("진입할 숏포지션 레버리지를 입력해주세요 :")

    return ASK_POSITION_KRW_EX

@safe_handler
async def ask_position_krw_ex(update: Update, context: ContextTypes.DEFAULT_TYPE):
    event_caller = context.user_data['event_caller']
    context.user_data[event_caller]["leverage"] = update.message.text

    # Define the buttons
    keyboard = [
        [
            InlineKeyboardButton("upbit", callback_data="upbit"),
            InlineKeyboardButton("bithumb", callback_data="bithumb"),
        ],
    ]
    
    # Create an inline keyboard markup
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Send the message with the inline keyboard
    await update.message.reply_text("한국 거래소를 입력하세요 :", reply_markup=reply_markup)

    return ASK_POSITION_FOR_EX

@safe_handler
async def ask_position_for_ex(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()  # Acknowledge the callback

    event_caller = context.user_data['event_caller']
    context.user_data[event_caller]["krw_ex"] = query.data

    # Define the buttons
    keyboard = [
        [
            InlineKeyboardButton("bybit", callback_data="bybit"),
            InlineKeyboardButton("binance", callback_data="binance"),
        ],
    ]
    
    # Create an inline keyboard markup
    reply_markup = InlineKeyboardMarkup(keyboard)

    # Send the message with the inline keyboard
    await query.edit_message_text("해외 거래소를 입력하세요 :", reply_markup=reply_markup)

    return HANDLE_ENTRY_POSITION

@safe_handler
async def handle_entry_position(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()  # Acknowledge the callback

    event_caller = context.user_data['event_caller']
    context.user_data[event_caller]["for_ex"] = query.data
    context.user_data["stop_event"] = asyncio.Event()

    task = asyncio.create_task(position_task(update, context, 'entry'))
    context.user_data["task"] = task

    return ConversationHandler.END

async def position_task(update: Update, context: ContextTypes.DEFAULT_TYPE, action:str):
    event_caller    = context.user_data['event_caller']
    symbol          = context.user_data[event_caller]["symbol"]

    if action == 'entry':
        entry_exrate    = context.user_data[event_caller]["entry_exrate"]
        krw_budget      = context.user_data[event_caller]["entry_budget"]
        leverage        = context.user_data[event_caller]["leverage"]        
        krw_ex          = context.user_data[event_caller]["krw_ex"]
        for_ex          = context.user_data[event_caller]["for_ex"]

    elif action == 'close':
        close_exrate    = context.user_data[event_caller]["close_exrate"]
    else:
        raise ValueError('Invalid action')

    try:
        if action == 'entry':
            k = KimpManager()
            krw_ex = k.get_exchange_manager(krw_ex)
            for_ex = k.get_exchange_manager(for_ex)

            krw_balance, usdt_balance, min_order_qty, leverage_set_yn = await asyncio.gather(krw_ex.get_balance('KRW'), \
                                                                                             for_ex.get_balance('USDT'), \
                                                                                             for_ex.get_min_order_qty(symbol), \
                                                                                             for_ex.set_leverage('linear', symbol, leverage))
            
            if leverage_set_yn:
                await update.callback_query.message.reply_text(f'레버리지가 {leverage}배로 설정되었습니다.')
            else:
                raise ValueError('레버리지 설정에 실패했습니다')

            await update.callback_query.message.reply_text(f"{krw_ex.config.name}거래소 잔액  : {krw_balance}\n"
                                                           f"{for_ex.config.name}거래소 잔액  : {usdt_balance}\n"
                                                           f"티커           : {symbol}\n"
                                                           f"최소주문량       : {min_order_qty}\n"
                                                           f"진입설정환율     : {entry_exrate}", \
                                                           parse_mode='Markdown')
            
            krw_balance, usdt_balance, min_order_qty = float(krw_balance), float(usdt_balance), float(min_order_qty)
            real_entry_exrate = None

            if entry_exrate == 0:
                res = await krw_ex.post_order(symbol, 'bid', str(krw_budget * (1-0.0025)), None, 'price')
                logger.debug(res)

                await asyncio.sleep(0.1)

                buy_position = await krw_ex.get_balance(symbol)
                logger.info(buy_position)

                res = await for_ex.post_order('linear', symbol, 'Sell', 'Market', truncate_number(float(buy_position), min_order_qty))
                logger.info(res)

                buy_position = await krw_ex.get_balance(symbol)
                logger.info(buy_position)

                short_position = await for_ex.get_balance(symbol)
                logger.info(short_position)

                res = await krw_ex.post_order(symbol, 'ask', None, str(buy_position - short_position), 'market')

                await asyncio.sleep(0.1)

                new_krw_balance, new_usdt_balance = await asyncio.gather(krw_ex.get_balance('KRW'), for_ex.get_balance('USDT'))
                new_krw_balance, new_usdt_balance = float(new_krw_balance), float(new_usdt_balance)

                real_entry_exrate = round((krw_balance - new_krw_balance) / (usdt_balance - new_usdt_balance), 3)
                krw_balance, usdt_balance = new_krw_balance, new_usdt_balance

                insert_position(symbol, krw_ex.config.name, for_ex.config.name, float(buy_position), float(short_position), float(leverage), float(real_entry_exrate), float(krw_budget))

                logger.info('Entry Kimp Position')

            else:
                start_time = time.time()
                while not context.user_data['stop_event'].is_set():
                    result = await asyncio.gather(k.get_single_ticker_kimp_by_seed(symbol, \
                                                                                    context.user_data[event_caller]['krw_ex'], \
                                                                                    context.user_data[event_caller]['for_ex'], \
                                                                                    krw_budget), \
                                                krw_ex.get_single_ticker_price('USDT'))
                    
                    df = result[0]
                    krw_budget, now_exrate = float(df.seed), float(df.exrate)
                    usdt_price = result[1][0]['trade_price']

                    if now_exrate <= entry_exrate and now_exrate <= usdt_price:
                        res = await krw_ex.post_order(symbol, 'bid', str(krw_budget * (1-0.0025)), None, 'price')
                        logger.debug(res)

                        await asyncio.sleep(0.1)

                        buy_position = await krw_ex.get_balance(symbol)
                        logger.debug(buy_position)

                        res = await for_ex.post_order('linear', symbol, 'Sell', 'Market', truncate_number(float(buy_position), min_order_qty))
                        logger.debug(res)

                        await asyncio.sleep(0.1)

                        buy_position = await krw_ex.get_balance(symbol)
                        logger.debug(buy_position)

                        short_position = await for_ex.get_balance(symbol)
                        logger.debug(short_position)

                        new_krw_balance, new_usdt_balance = await asyncio.gather(krw_ex.get_balance('KRW'), for_ex.get_balance('USDT'))
                        new_krw_balance, new_usdt_balance = float(new_krw_balance), float(new_usdt_balance)

                        real_entry_exrate = round((krw_balance - new_krw_balance) / (usdt_balance - new_usdt_balance), 3)
                        krw_balance, usdt_balance = new_krw_balance, new_usdt_balance

                        insert_position(symbol, krw_ex, for_ex, buy_position, short_position, leverage, real_entry_exrate, krw_budget)

                        logger.info('Entry Kimp Position')

                        break
                
                    if time.time() - start_time >= 60:
                        await update.callback_query.message.reply_text(f"진입예산     : {krw_budget}\n"
                                                                       f"티커        : {symbol}\n"
                                                                       f"현재환율     : {now_exrate}\n"
                                                                       f"테더가격     : {usdt_price}", \
                                                                        parse_mode='Markdown')
                        start_time = time.time()

                    await asyncio.sleep(0.5)
            
            if not context.user_data['stop_event'].is_set():
                await update.callback_query.message.reply_text(f"티커       : {symbol}\n"
                                                               f"한국거래소  : {krw_ex.config.name}\n"
                                                               f"외국거래소  : {for_ex.config.name}\n"
                                                               f"레버리지    : {leverage}\n"
                                                               f"진입금액    : {krw_budget}\n"
                                                               f"진입환율    : {real_entry_exrate}\n"
                                                               f"롱포지션개수  : {buy_position}\n" 
                                                               f"숏포지션개수  : {short_position}\n", \
                                                               parse_mode='Markdown')

        elif action == 'close':
            result = inquire_position(symbol)

            if not result:
                raise ValueError('종료할 포지션이 없습니다.')
            
            k = KimpManager()
            krw_ex = k.get_exchange_manager(result[2])
            for_ex = k.get_exchange_manager(result[3])
            buy_position = result[4]
            short_position = result[5]
            leverage = result[6]
            entry_exrate = result[7]
            close_exrate = result[8]
            krw_budget = result[9]
            krw_profit = result[10]

            await update.message.reply_text(f"티커        : {symbol}\n"
                                            f"한국거래소   : {krw_ex.config.name}\n"
                                            f"외국거래소   : {for_ex.config.name}\n"
                                            f"레버리지     : {leverage}\n"
                                            f"포지션금액    : {krw_budget}\n"
                                            f"진입환율     : {entry_exrate}\n"
                                            f"롱포지션개수  : {buy_position}\n"
                                            f"숏포지션개수  : {short_position}\n", \
                                            parse_mode='Markdown')

            # 미체결약정
            if close_exrate == None:
                res = await krw_ex.post_order(symbol, 'ask', price=None, volume=str(buy_position), ord_type='market')
                logger.info(res)

                res = await for_ex.post_order('linear', symbol, 'Buy', 'Market', str(short_position))
                logger.info(res)

                await asyncio.sleep(0.1)

                new_krw_balance, new_usdt_balance, res_usdt = await asyncio.gather(krw_ex.get_balance('KRW'), for_ex.get_balance('USDT'), krw_ex.get_single_ticker_price('USDT'))
                new_krw_balance, new_usdt_balance = float(new_krw_balance), float(new_usdt_balance)

                close_exrate = round((new_krw_balance - krw_balance) / (new_usdt_balance - usdt_balance), 3)
                usdt_price = res_usdt[0]['trade_price']
                krw_profit = round((new_krw_balance - krw_balance) + (new_usdt_balance - usdt_balance) * usdt_price, 3)
                krw_balance, usdt_balance = new_krw_balance, new_usdt_balance

                update_position(symbol, close_exrate, krw_profit)

                logger.info('Close Kimp Position')

            else:
                start_time = time.time()
                while not context.user_data['stop_event'].is_set():
                    result = await asyncio.gather(k.get_single_ticker_kimp_by_seed(symbol, krw_ex.config.name, for_ex.config.name), \
                                                  krw_ex.get_single_ticker_price('USDT'))
                    
                    df = result[0]
                    df.seed = df.seed.astype(float)
                    df.exrate = df.exrate.astype(float)

                    krw_budget, now_exrate = k.get_max_exrate_for_budget(df, krw_budget)
                    krw_budget, now_exrate = float(krw_budget), float(now_exrate)
                    
                    usdt_price = result[1][0]['trade_price']

                    if now_exrate >= close_exrate and now_exrate >= entry_exrate:
                        res = await krw_ex.post_order(symbol, 'ask', price=None, volume=str(buy_position), ord_type='price')
                        logger.debug(res)
                        res = await for_ex.post_order('linear', symbol, 'Buy', 'Market', str(short_position))
                        logger.debug(res)

                        await asyncio.sleep(0.1)

                        new_krw_balance, new_usdt_balance = await asyncio.gather(krw_ex.get_balance('KRW'), for_ex.get_balance('USDT'))
                        new_krw_balance, new_usdt_balance = float(new_krw_balance), float(new_usdt_balance)

                        close_exrate = round((new_krw_balance - krw_balance) / (new_usdt_balance - usdt_balance), 3)
                        krw_profit = round((new_krw_balance - krw_balance) + (new_usdt_balance - usdt_balance) * usdt_price, 3)
                        krw_balance, usdt_balance = new_krw_balance, new_usdt_balance

                        update_position(symbol, close_exrate, krw_profit)

                        logger.info('Close Kimp Position')

                        break
                
                    if time.time() - start_time >= 60:
                        await update.message.reply_text(f"진입예산     : {krw_budget}\n"
                                                        f"티커        : {symbol}\n"
                                                        f"현재환율     : {now_exrate}\n"
                                                        f"테더가격     : {usdt_price}", \
                                                        parse_mode='Markdown')

                    await asyncio.sleep(0.5)

            await update.message.reply_text(f"티커        : {symbol}\n"
                                            f"한국거래소   : {krw_ex.config.name}\n"
                                            f"외국거래소   : {for_ex.config.name}\n"
                                            f"레버리지     : {leverage}\n"
                                            f"진입금액     : {krw_budget}\n" 
                                            f"진입환율     : {entry_exrate}\n"
                                            f"종료환율     : {close_exrate}\n"
                                            f"롱포지션개수  : {buy_position}\n"
                                            f"숏포지션개수  : {short_position}\n"
                                            f"수익금       : {krw_profit}", \
                                            parse_mode='Markdown')

    except Exception as e:
        logger.error(traceback.format_exc())
        logger.error(e)
        if action == 'entry':
            await update.callback_query.message.reply_text(str(e))
        elif action == 'close':
            await update.message.reply_text(str(e))
    finally:
        del context.user_data[event_caller]
        del context.user_data['event_caller']
        del context.user_data['stop_event']
        del context.user_data['task']
        return ConversationHandler.END


# 김프포지션 종료 버튼 핸들러
@safe_handler
async def ask_position_close_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    option = update.message.text
    context.user_data['event_caller'] = option
    context.user_data[option] = {}

    await update.message.reply_text("Please enter the symbol to close the position:")
    return ASK_POSITION_CLOSE_EXRATE

@safe_handler
async def ask_position_close_exrate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    event_caller = context.user_data['event_caller']
    context.user_data[event_caller]["symbol"] = (update.message.text).upper()
    
    await update.message.reply_text("Please enter the close exrate (enter 0 for immediate close):")
    return HANDLE_CLOSE_POSITION

@safe_handler
async def handle_close_position(update: Update, context: ContextTypes.DEFAULT_TYPE):
    event_caller = context.user_data['event_caller']
    context.user_data[event_caller]["close_exrate"] = float(update.message.text)
    context.user_data["stop_event"] = asyncio.Event()

    task = asyncio.create_task(position_task(update, context, 'close'))
    context.user_data["task"] = task

    return ConversationHandler.END


# 보조지표로 종목추천 버튼 핸들러
@safe_handler
async def ask_interval(update: Update, context: ContextTypes.DEFAULT_TYPE):
    option = update.message.text
    context.user_data['event_caller'] = option
    context.user_data[option] = {}

    keyboard = [
        [
            InlineKeyboardButton("1h", callback_data="1h"),
            InlineKeyboardButton("1d", callback_data="1d"),
            InlineKeyboardButton("1w", callback_data="1w"),
        ]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("참조할 보조지표 봉을 선택하세요", reply_markup=reply_markup)
    return HANDLE_SUB_INDICATOR

@safe_handler
async def handle_sub_indicator(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    event_caller = context.user_data['event_caller']
    context.user_data[event_caller]["interval"] = query.data

    k = KimpManager()
    task = asyncio.create_task(k.get_tickers_by_sub_indicators(update, context))
    context.user_data["task"] = task
    return ConversationHandler.END


# 볼밴 선물자동매매 버튼 핸들러
@safe_handler
async def ask_future_autotrading_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    option = update.message.text
    context.user_data['event_caller'] = option
    context.user_data[option] = {}
    await update.message.reply_text("티커를 입력해주세요 : (예: XRP)")
    return ASK_FUTURE_AUTOTRADING_BUDGET

@safe_handler
async def ask_future_autotrading_budget(update: Update, context: ContextTypes.DEFAULT_TYPE):
    event_caller = context.user_data['event_caller']

    context.user_data[event_caller]["symbol"] = update.message.text.upper()
    await update.message.reply_text("진입마진을 입력해주세요 : (예: 1000, 단위는 USDT)")
    return ASK_FUTURE_AUTOTRADING_LEVERAGE

@safe_handler
async def ask_future_autotrading_leverage(update: Update, context: ContextTypes.DEFAULT_TYPE):
    event_caller = context.user_data['event_caller']

    # Validate the input to ensure it's a valid number
    if not update.message.text.isdigit():
        await update.message.reply_text("올바른 숫자를 입력해주세요")
        return ASK_FUTURE_AUTOTRADING_LEVERAGE

    context.user_data[event_caller]["margin"] = float(update.message.text)
    await update.message.reply_text("진입레버리지를 입력해주세요 : (예: 10)")
    return 

@safe_handler
async def ask_future_autotrading_stop_loss(update: Update, context: ContextTypes.DEFAULT_TYPE):
    event_caller = context.user_data['event_caller']

    # Validate the input to ensure it's a valid number
    if not update.message.text.isdigit():
        await update.message.reply_text("올바른 숫자를 입력해주세요")
        return ASK_FUTURE_AUTOTRADING_LEVERAGE

    context.user_data[event_caller]["leverage"] = float(update.message.text)
    await update.message.reply_text("스탑로스를 입력해주세요 : (예: 5, 단위: %)")
    return ASK_FUTURE_AUTOTRADING_FOR_EX

@safe_handler
async def ask_future_autotrading_for_ex(update: Update, context: ContextTypes.DEFAULT_TYPE):
    event_caller = context.user_data['event_caller']

    # Validate the input to ensure it's a valid number
    if not update.message.text.isdigit():
        await update.message.reply_text("올바른 숫자를 입력해주세요")
        return ASK_FUTURE_AUTOTRADING_STOP_LOSS

    context.user_data[event_caller]["stop_loss"] = float(update.message.text)

    # Define the buttons
    keyboard = [
        [
            InlineKeyboardButton("bybit", callback_data="bybit"),
            InlineKeyboardButton("binance", callback_data="binance"),
        ],
    ]
    
    # Create an inline keyboard markup
    reply_markup = InlineKeyboardMarkup(keyboard)

    # Send the message with the inline keyboard
    await update.message.reply_text("해외 거래소를 입력하세요 :", reply_markup=reply_markup)

    return ASK_FUTURE_AUTOTRADING_STRATEGY

@safe_handler
async def ask_future_autotrading_strategy(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    event_caller = context.user_data['event_caller']
    context.user_data[event_caller]["for_ex"] = query.data

    keyboard = [
        [
            InlineKeyboardButton("볼린저밴드전략", callback_data="bb"),
            InlineKeyboardButton("RSI전략", callback_data="rsi"),
        ]
    ]

    reply_markup = InlineKeyboardMarkup(keyboard)
    await query.edit_message_text("자동매매 전략을 선택하세요 :", reply_markup=reply_markup)

    return HANDLE_FUTURE_AUTOTRADING

@safe_handler
async def handle_future_autotrading(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    event_caller = context.user_data['event_caller']
    context.user_data[event_caller]["strategy"] = query.data

    if query.data == 'bb':
        keyboard = [
            [
                InlineKeyboardButton("1m", callback_data="1h"),
                InlineKeyboardButton("5m", callback_data="1h"),
                InlineKeyboardButton("15m", callback_data="1h"),
                InlineKeyboardButton("1h", callback_data="1h"),
                InlineKeyboardButton("4h", callback_data="1d"),
                InlineKeyboardButton("1d", callback_data="1w"),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await query.edit_message_text("참조할 보조지표 봉을 선택하세요", reply_markup=reply_markup)
        return HANDLE_FUTURE_AUTOTRADING_BB
    elif query.data == 'rsi':
        pass

@safe_handler
async def handle_future_autotrading_bb(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    event_caller = context.user_data['event_caller']
    context.user_data[event_caller]["interval"] = query.data

    k = KimpManager()
    task = asyncio.create_task(k.run_future_autotrading(update, context))
    context.user_data["task"] = task
    return ConversationHandler.END


# 봇 종료 버튼 핸들러
@safe_handler
async def break_loop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stop_event = context.user_data.get('stop_event')
    task = context.user_data.get('task')

    if not stop_event or not task:
        await update.message.reply_text('종료할 이벤트가 없습니다.')
        return
    
    stop_event.set()
    await task

    if not task.done():
        await update.message.reply_text(f'{context.user_data["event_caller"]}가 종료되지 않았습니다')
        raise ValueError(f'{context.user_data["event_caller"]}가 종료되지 않았습니다')

    await update.message.reply_text(f'{context.user_data["event_caller"]}가 정상적으로 종료되었습니다.')
    
    if context.user_data.get('stop_event'):
        del context.user_data['stop_event']
    if context.user_data.get('task'):
        del context.user_data['task']
    if context.user_data.get('event_caller'):
        del context.user_data['event_caller']

    return ConversationHandler.END

# 취소 버튼 핸들러
async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info('conversation cancel')
    if context.user_data.get('stop_event'):
        del context.user_data['stop_event']
    if context.user_data.get('task'):
        del context.user_data['task']
    if context.user_data.get('event_caller'):
        del context.user_data['event_caller']
    await update.message.reply_text("정상적으로 취소되었습니다.")
    return ConversationHandler.END



def inquire_position(symbol):
    conn = sqlite3.connect('positions.db')
    cursor = conn.cursor()
    cursor.execute('''
        SELECT * 
        FROM positions 
        WHERE symbol = ? and close_exrate is NULL
    ''', (symbol,))
    result = cursor.fetchone()
    conn.close()
    return result

def update_position(symbol, close_exrate, profit):
    conn = sqlite3.connect('positions.db')
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE positions
        SET close_exrate = ? and profit = ?
        WHERE symbol = ? and close_exrate is NULL
    ''', (close_exrate, profit, symbol))
    conn.commit()
    conn.close()

def insert_position(symbol, krw_ex, for_ex, buy_position, short_position, short_leverage, entry_exrate, krw_budget):
    conn = sqlite3.connect('positions.db')
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO positions (
            symbol, 
            krw_ex, 
            for_ex, 
            buy_position, 
            short_position, 
            short_leverage, 
            entry_exrate,
            close_exrate, 
            krw_budget
        ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?)
    ''', (symbol, krw_ex, for_ex, buy_position, short_position, short_leverage, entry_exrate, krw_budget))
    conn.commit()
    conn.close()

def initialize_db():
    conn = sqlite3.connect('positions.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS positions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol TEXT,
            krw_ex TEXT,
            for_ex TEXT,
            buy_position REAL,
            short_position REAL,
            short_leverage REAL,
            entry_exrate REAL,
            close_exrate REAL,
            krw_budget REAL,
            krw_profit REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            UNIQUE (symbol, krw_ex, for_ex)
        )
    ''')
    conn.commit()
    conn.close()

def initialize_kline_db():
    conn = sqlite3.connect('klines.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS kline (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market TEXT,
            interval TEXT,
            exchange TEXT,
            timestamp TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            quote_volume REAL,
            macd REAL,
            signal REAL,
            sma20 REAL,
            UNIQUE(market, interval, exchange, timestamp)
        )
    ''')
    conn.commit()
    conn.close()

def insert_kline_data(df, market, exchange, interval):
    conn = sqlite3.connect('klines.db')
    try:
        # Prepare data for insertion
        df['market'] = market
        df['exchange'] = exchange
        df['interval'] = interval
        
        # Convert DataFrame to SQLite
        df.to_sql('kline', conn, if_exists='append', index=False)
        
    except Exception as e:
        logger.error(f"Error inserting kline data: {str(e)}")
    finally:
        conn.close()


def main():
    initialize_db()

    application = Application.builder().token(bot_id).build()

    # Add the /start command handler
    application.add_handler(CommandHandler("start", start))

    # Dynamically create and add conversation handlers
    for option, details in conversation_options.items():
        # Check if it's a multi-step conversation
        if isinstance(details["state"], list) and isinstance(details["handler"], list):
            # Build the states dynamically for multi-step conversations
            states = {}
            for i, state in enumerate(details["state"]):
                handler_type = details.get("handler_type", ["MessageHandler"])[i]  # Default to MessageHandler
                handler_function = globals()[details["handler"][i]]
                
                if handler_type == "MessageHandler":
                    handler = MessageHandler(filters.TEXT & ~filters.COMMAND & ~filters.Regex("^취소$"), handler_function)
                elif handler_type == "CallbackQueryHandler":
                    handler = CallbackQueryHandler(handler_function)
                else:
                    raise ValueError(f"Unsupported handler type: {handler_type}")
                
                states[state] = [handler]

            entry_points = [MessageHandler(filters.Regex(f"^{option}$"), globals()[details["entry_action"]])]
            # Create the ConversationHandler
            conv_handler = ConversationHandler(
                entry_points=entry_points,
                states=states,
                fallbacks=[MessageHandler(filters.Regex("^취소$"), cancel)],
            )

        else:
            # Single-step conversation
            handler_type = details.get("handler_type", "MessageHandler")  # Default to MessageHandler
            handler_function = globals()[details["handler"]]
            
            if handler_type == "MessageHandler":
                handler = MessageHandler(filters.TEXT & ~filters.COMMAND & ~filters.Regex("^취소$"), handler_function)
            elif handler_type == "CallbackQueryHandler":
                handler = CallbackQueryHandler(handler_function)
            else:
                raise ValueError(f"Unsupported handler type: {handler_type}")
            
            conv_handler = ConversationHandler(
                entry_points=[
                    MessageHandler(filters.Regex(f"^{option}$"), globals()[details["entry_action"]])
                ],
                states={
                    details["state"]: [handler]
                },
                fallbacks=[MessageHandler(filters.Regex("^취소$"), cancel)],
            )

        # Add the conversation handler to the application
        application.add_handler(conv_handler)

    application.add_handler(MessageHandler(filters.Regex("^봇 종료$"), break_loop))

    # Run the bot
    application.run_polling()

if __name__ == "__main__":
    main()

    # b = BinanceManager()
    # print(asyncio.run(b.get_numeric_tickers()))

    # b = BybitManager()
    # print(asyncio.run(b.get_numeric_tickers()))
    # initialize_kline_db()
    # k = KimpManager()
    # asyncio.run(k.run_kimp_alarm(1, 1, 10000000))
