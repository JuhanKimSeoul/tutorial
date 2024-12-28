import asyncio
from asyncio import Event
import json
import pandas as pd
import matplotlib.pyplot as plt
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
import io
import re
import itertools
from telegram import ReplyKeyboardMarkup, KeyboardButton, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes, ConversationHandler, CallbackQueryHandler
from functools import wraps

load_dotenv('dev.env')

upbit_access_key = os.getenv('UPBIT_API_KEY')
upbit_secret_key = os.getenv('UPBIT_SECRET_KEY')
upbit_server_url = 'https://api.upbit.com'

bithumb_access_key = os.getenv('BITHUMB_API_KEY')
bithumb_secret_key = os.getenv('BITHUMB_SECRET_KEY')
bithumb_server_url = 'https://api.bithumb.com'

bybit_access_key = os.getenv('BYBIT_SUBACC_API_KEY')
bybit_secret_key = os.getenv('BYBIT_SUBACC_SECRET_KEY')
bybit_server_url = 'https://api.bybit.com'

bot_id = os.getenv('TELEGRAM_BOT_TOKEN')
chat_id = os.getenv('TELEGRAM_CHAT_ID')

bot = Bot(token=bot_id)

fee_rate = {
    'coinone' : 0.002,
    'bithumb' : 0.004,
    'upbit' : 0.0005,
    'binance' : 0.0005,
    'bybit' : 0.00055,
    'okx' : 0.0005,
    'bitget' : 0.0006,
    'gateio' : 0.00075
}

# Basic configuration for the logging
logging.basicConfig(
    level=logging.INFO,                      # Log level
    format='%(asctime)s - %(name)s - %(lineno)d - %(levelname)s - %(message)s',  # Format
    datefmt='%Y-%m-%d %H:%M:%S'               # Date format
)

# Create a logger object
logger = logging.getLogger(__name__)

class ExchangeAPIConfig:
    """A base class to store API-specific configurations for each exchange."""

    def get_orderbook_params(self, symbol):
        """Override in subclasses to define the orderbook parameters for each exchange."""
        raise NotImplementedError

    @property
    def orderbook_url(self):
        """Override in subclasses to provide the specific orderbook endpoint."""
        raise NotImplementedError

class UpbitAPIConfig(ExchangeAPIConfig):
    name = 'upbit'
    orderbook_url = "/v1/orderbook"
    kline_url = "/v1/candles/"
    all_tickers_url = "/v1/ticker/all"
    single_ticker_url = "/v1/ticker"
    balance_url = "/v1/accounts"
    order_url = "/v1/orders"
    interval_enum = [1,3,5,10,15,30,60,240]
    limit = 200
    
    def get_orderbook_params(self, symbol):
        return {"markets": f"KRW-{symbol}", "level": 0}

    def get_all_tickers_params(self, quote_currencies):
        return {
            'quote_currencies' : quote_currencies
        }
    
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

    def get_kline_params(self, symbol, interval):
        return {
            'market' : 'KRW-' + symbol,
            'count' : self.limit
        }

    def get_single_ticker_param(self, symbol):
        return {
            'markets': f'KRW-{symbol}'
        }

class BybitAPIConfig(ExchangeAPIConfig):
    name = 'bybit'
    orderbook_url = "/v5/market/orderbook"
    kline_url = "/v5/market/kline"
    all_tickers_url = "/v5/market/tickers"
    balance_url = "/v5/asset/transfer/query-account-coins-balance"
    order_url = "/v5/order/create"
    leverage_set_url = "/v5/position/set-leverage"
    ticker_info_url = "/v5/market/instruments-info?category=linear"
    position_info_url = "/v5/position/list"
    interval_enum = [1,3,5,15,30,60,120,240,360,720,'D','W','M']
    orderbook_limit = 500
    kline_limit = 1000

    def get_orderbook_params(self, symbol):
        return {"category": "linear", "symbol": f"{symbol}USDT", "limit": self.orderbook_limit}

    def get_all_tickers_params(self, category):
        return {
            'category' : category
        }

    def get_kline_endpoint(self, interval):
        return self.kline_url

    def get_kline_params(self, symbol, interval):
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
            'symbol' : f"{symbol}USDT",
            'interval' : interval,
            'limit' : self.kline_limit
        }

class BithumbAPIConfig(ExchangeAPIConfig):
    name = 'bithumb'
    orderbook_url = "/v1/orderbook"
    kline_url = "/v1/candles/"
    all_tickers_url = "/v1/ticker"
    balance_url = "/v1/accounts"
    order_url = "/v1/orders"
    interval_enum = [1,3,5,10,15,30,60,240]
    limit = 200

    def get_orderbook_params(self, symbol):
        return {"markets": f"KRW-{symbol}"}

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
    
    def get_kline_params(self, symbol, interval):
        return {
            'market' : 'KRW-' + symbol,
            'count' : self.limit
        }

class BinanceAPIConfig(ExchangeAPIConfig):
    name = 'binance'
    orderbook_url = '/fapi/v1/depth'
    kline_url = '/fapi/v1/klines'
    all_tickers_url = '/fapi/v2/ticker/price'
    interval_enum = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']

    def get_orderbook_params(self, symbol):
        return {"symbol": f"{symbol}USDT"}

    def get_kline_params(self, symbol, interval):
        return {
            'symbol' : symbol + 'USDT',
            'interval' : interval
        }

class ExchangeManager:
    def __init__(self, exchange, fee_rate, server_url):
        self.exchange = exchange
        self.fee_rate = fee_rate
        self.server_url = server_url
        self.config = self.get_config()

    def get_config(self):
        """Factory method to retrieve the appropriate configuration based on the exchange."""
        configs = {
            "upbit": UpbitAPIConfig(),
            "bybit": BybitAPIConfig(),
            "bithumb": BithumbAPIConfig(),
            "binance": BinanceAPIConfig()
        }
        return configs.get(self.exchange)
    
    async def request(self, method, endpoint, headers: dict = None, params: dict = None, contentType: str = 'json'):
        try:
            async with aiohttp.ClientSession() as session:
                if method == 'get':
                    async with session.get(self.server_url + endpoint, params=params, headers=headers) as response:
                        if response.status == 429:
                            await self.handle_rate_limit(response)
                        return await response.json()
                elif method == 'post':
                    if contentType != 'json':
                        async with session.post(self.server_url + endpoint, data=json.dumps(params), headers=headers) as response:
                            if response.status == 429:
                                await self.handle_rate_limit(response)
                            return await response.json()
                    else:
                        async with session.post(self.server_url + endpoint, json=params, headers=headers) as response:
                            if response.status == 429:
                                await self.handle_rate_limit(response)
                            return await response.json()
                elif method == 'delete':
                    pass
        except aiohttp.ClientConnectionError as e:
            logger.error(f'Connection error: {e}')
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

    async def get_orderbook(self, symbol):
        endpoint = self.config.orderbook_url
        params = self.config.get_orderbook_params(symbol)
        headers = {"accept" : "application/json"}

        return await self.request('get', endpoint, headers, params)

    def format_orderbook_response(self, response)->dict:
        raise NotImplementedError

    async def get_kline(self, symbol, interval):
        endpoint = self.config.get_kline_endpoint(interval)
        params = self.config.get_kline_params(symbol, interval)
        headers = {"accept" : "application/json"}

        return await self.request('get', endpoint, headers, params)
    
    async def get_all_ticker_price(self):
        pass
        
    async def get_single_ticker_price(self, symbol):
        pass

    async def get_balance(self):
        pass

    async def post_order(self):
        pass

class UpbitManager(ExchangeManager):
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
        params = self.config.get_all_tickers_params(quote_currencies='KRW')
        headers = {"accept" : "application/json"}

        return await self.request('get', endpoint, headers, params)

    async def get_single_ticker_price(self, symbol):
        endpoint = self.config.single_ticker_url
        params = self.config.get_single_ticker_param(symbol)
        headers = {"accept" : "application/json"}

        return await self.request('get', endpoint, headers, params)

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
        return float([ i.get('balance', 0) for i in res if i.get('currency') == symbol][0])

    async def post_order(self, market, side, price, volume, ord_type):
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
        params = {
            'market': 'KRW-'+market,
            'side': side,
            'volume': volume,
            'price': price,
            'ord_type': ord_type
        }

        if side == 'bid':
            params.pop('volume')
        elif side == 'ask':
            params.pop('price')

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

        return await self.request('post', self.config.order_url, headers, params)

class BithumbManager(ExchangeManager):
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
        params = {
            'markets' : ", ".join(krw_tickers)
        }
        headers = {"accept" : "application/json"}

        return await self.request('get', endpoint, headers, params)

    async def get_single_ticker_price(self, symbol):
        endpoint = self.config.all_tickers_url
        params = {
            'markets' : f'KRW-{symbol}'
        }
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

    async def post_order(self, market, side, price, volume, ord_type):
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
        params = {
            'market': 'KRW-'+market,
            'side': side,
            'volume': volume,
            'price': price,
            'ord_type': ord_type
        }

        if side == 'bid':
            params.pop('volume')
        elif side == 'ask':
            params.pop('price')

        logger.info(params)
        
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

        logger.info(headers)

        return await self.request('post', self.config.order_url, headers, params, 'x-www-form-urlencoded')

class BybitManager(ExchangeManager):
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
        params = self.config.get_all_tickers_params(category='linear')
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
                'symbol': KimpManager.symbol_exception_handler_for_for_ex('bybit', symbol) + 'USDT'
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

    async def post_order(self, category, symbol, side, orderType, qty):
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
        '''

        # Generate timestamp
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"

        # Define the payload
        payload = {
            "category": category,
            "symbol": symbol+"USDT",
            "side": side,
            "orderType": orderType,
            "qty": qty,
        }

        # Convert payload to JSON and create the pre-sign string
        json_payload = json.dumps(payload)
        pre_sign = f"{timestamp}{bybit_access_key}{recv_window}{json_payload}"

        # Create the signature
        signature = hmac.new(
            bybit_secret_key.encode('utf-8'),
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

        return await self.request('post', self.config.order_url, headers, payload, 'x-www-form-urlencoded')

    async def set_leverage(self, category, symbol, buyLeverage, sellLeverage):
        # Generate timestamp
        timestamp = str(int(time.time() * 1000))
        recv_window = "5000"

        # Define the payload
        payload = {
            "category": category,
            "symbol": symbol+"USDT",
            "buyLeverage": buyLeverage,
            "sellLeverage": sellLeverage,
        }

        # Convert payload to JSON and create the pre-sign string
        json_payload = json.dumps(payload)
        pre_sign = f"{timestamp}{bybit_access_key}{recv_window}{json_payload}"

        # Create the signature
        signature = hmac.new(
            bybit_secret_key.encode('utf-8'),
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

        return await self.request('post', self.config.leverage_set_url, headers, payload, 'x-www-form-urlencoded')

    async def get_kline(self, symbol, interval):
        endpoint = self.config.get_kline_endpoint(interval)
        params = self.config.get_kline_params(symbol, interval)
        headers = {"accept" : "application/json"}

        res = await self.request('get', endpoint, headers, params)
        return res['result']['list']

    async def get_min_order_qty(self, symbol):
        endpoint = self.config.ticker_info_url
        params = {
            "symbol": symbol + 'USDT'
        }
        headers = {"accept" : "application/json"}

        res = await self.request('get', endpoint, headers, params)
        return res['result']['list'][0]['lotSizeFilter']['minOrderQty']
        
class BinanceManager(ExchangeManager):
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
        headers = {"accept" : "application/json"}

        return await self.request('get', endpoint, headers)

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
    def __init__(self, krw_ex, for_ex):
        self.krw_ex_manager:ExchangeManager = self.get_exchange_manager(krw_ex)
        self.for_ex_manager:ExchangeManager = self.get_exchange_manager(for_ex)

    def get_exchange_manager(self, ex):
        managers = {
            'upbit' : UpbitManager('upbit', 0.0005, 'https://api.upbit.com'),
            'bithumb' : BithumbManager('bithumb', 0.0004, 'https://api.bithumb.com'),
            'bybit' : BybitManager('bybit', 0.00055, 'https://api.bybit.com'),
            'binance' : BinanceManager('binance', 0.0005, 'https://fapi.binance.com')
        }
        return managers.get(ex)

    @staticmethod
    def symbol_exception_handler_for_for_ex(ex, symbol):
        data = {
            'bybit' : {
                'SHIB' : 'SHIB1000',
                'PEPE' : '1000PEPE',
                'FLOKI' : '1000FLOKI',
                'BONK' : '1000BONK',
                'XEC' : '1000XEC',
                'BTT' : '1000BTT'
            },

            'binance' : {
                'SHIB' : '1000SHIB',
                'PEPE' : '1000PEPE',
                'FLOKI' : '1000FLOKI',
                'BONK' : '1000BONK',
                'XEC' : '1000XEC'
            }
        }

        return data[ex].get(symbol, symbol)

    async def get_single_ticker_kimp_by_seed(self, symbol, krw_ex, for_ex):
        try:
            krw_ex_manager = None
            for_ex_manager = None

            if krw_ex == 'upbit':
                krw_ex_manager = UpbitManager('upbit', 0.0005, 'https://api.upbit.com')
            elif krw_ex == 'bithumb':
                krw_ex_manager = BithumbManager('bithumb', 0.0004, 'https://api.bithumb.com')
            else:
                raise ValueError(f'input : {krw_ex} \n krw_ex should be upbit or bithumb')

            if for_ex == 'binance':
                for_ex_manager = BinanceManager('binance', 0.0005, 'https://fapi.binance.com')
            elif for_ex == 'bybit':
                for_ex_manager = BybitManager('bybit', 0.00055, 'https://api.bybit.com')
            else:
                raise ValueError(f'input : {for_ex} \n for_ex should be binance or bybit')

            krw_orderbook = krw_ex_manager.get_orderbook(symbol)

            trans_symbol = KimpManager.symbol_exception_handler_for_for_ex(for_ex, symbol)

            for_orderbook = for_ex_manager.get_orderbook(trans_symbol)
            results = await asyncio.gather(krw_orderbook, for_orderbook)

            krw_orderbook = krw_ex_manager.format_orderbook_response(results[0])
            for_orderbook = for_ex_manager.format_orderbook_response(results[1])

            ret = []
            for seed in range(1_000_000, 100_000_000, 1_000_000):
                buy_amt, buy_fee = KimpManager.calculate_market_order_amount(symbol, krw_orderbook, seed, is_buy=True, fee_rate=krw_ex_manager.fee_rate)

                if buy_amt == 0:
                    break

                usdt_revenue, sell_fee = KimpManager.calculate_market_order_amount(symbol, for_orderbook, buy_amt, is_buy=False, fee_rate=for_ex_manager.fee_rate)

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
        Executes the arbitrage bot logic based on the user's input and predefined parameters.

        This function is triggered during a Telegram bot conversation, taking user-defined trading parameters 
        from the `context.user_data` dictionary and executing an arbitrage strategy. The bot monitors exchange rates, 
        applies leverage if specified, and ensures proper trade execution within the user's KRW budget.

        Parameters:
            self: Reference to the instance of the class containing this method.
            update (Update): Telegram update object containing metadata and user interaction data.
            context (ContextTypes.DEFAULT_TYPE): Context object that holds:
                - user_data (dict): A dictionary containing user-defined trading parameters:
                    - 'symbol' (str): The trading symbol (e.g., "BTC").
                    - 'entry_exrate' (float): The exchange rate at which to enter the position.
                    - 'close_exrate' (float): The exchange rate at which to close the position.
                    - 'krw_budget' (float): The budget allocated in KRW for this arbitrage operation.
                    - 'krw_ex' (str): Name of the KRW-based exchange.
                    - 'for_ex' (str): Name of the foreign exchange.
                    - 'leverage' (str): The leverage ratio to be applied (if applicable).
                    - 'stop_event' (asyncio.Event): A threading event to signal when to stop the bot.

        Expected Behavior:
            1. Validate the provided trading parameters.
            2. Monitor exchange rates on the specified exchanges.
            3. Execute entry and exit trades based on the `entry_exrate` and `close_exrate`.
            4. Ensure trades respect the allocated `krw_budget`.
            5. Apply leverage if specified, ensuring compliance with margin requirements.
            6. Continuously check the `stop_event` to gracefully terminate the bot.

        Error Handling:
            - Handles invalid or missing parameters in `context.user_data`.
            - Logs any unexpected errors during exchange monitoring or trade execution.

        Returns:
            None. Performs operations asynchronously and interacts with external APIs.

        Example Usage:
            # Assuming a Telegram bot interaction
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
        
        symbol          = context.user_data[event_caller]['symbol']
        entry_exrate    = float(context.user_data[event_caller]['entry_exrate'])
        close_exrate    = float(context.user_data[event_caller]['close_exrate'])
        krw_budget      = float(context.user_data[event_caller]['budget'])
        leverage        = context.user_data[event_caller]['leverage']
        krw_ex          = self.get_exchange_manager(context.user_data[event_caller]['krw_ex'])
        for_ex          = self.get_exchange_manager(context.user_data[event_caller]['for_ex'])

        try:
            krw_balance, usdt_balance, leverage_res, min_order_qty = await asyncio.gather(krw_ex.get_balance('KRW'), \
                                                                                          for_ex.get_balance('USDT'), \
                                                                                          for_ex.set_leverage('linear', \
                                                                                                                symbol, \
                                                                                                                leverage, \
                                                                                                                leverage), \
                                                                                          for_ex.get_min_order_qty(symbol))
            logger.info(leverage_res)
            krw_balance, usdt_balance, min_order_qty = float(krw_balance), float(usdt_balance), float(min_order_qty)
            logger.info(f'\n Initial Balance \
                          \n - KRW Balance : {krw_balance} \
                          \n - USDT Balance : {usdt_balance} \
                          \n - Total Balance : {krw_balance} ') 
        except Exception as e:
            logger.info(traceback.format_exc())
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

                krw_budget, now_exrate = self.iter_row_make_real_exrate_column(df, krw_budget)
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

                elif position_yn == 1 and (now_exrate >= close_exrate or (real_entry_exrate is not None and (now_exrate - real_entry_exrate) / real_entry_exrate * 100 >= 1)):
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
            
    def iter_row_make_real_exrate_column(self, df, budget)->float:
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
                budget, max_exrate = self.iter_row_make_real_exrate_column(r, budget)
                market_col.append(r['market'].iloc[0])
                exrate_col.append(max_exrate)
                budget_col.append(budget)
                exchange_col.append(r['krw_ex'].iloc[0])
            
            await asyncio.sleep(0.2)

        df = pd.DataFrame({'market': market_col, 'exrate': exrate_col, 'budget': budget_col, 'krw_ex': exchange_col}).set_index('market')
        df.loc[df.index.isin(['SHIB', 'BONK', 'PEPE', 'FLOKI', 'XEC', 'BTT']), 'exrate'] *= 1000
        df.sort_values(by='exrate', inplace=True)

        return df

    async def monitor_all_ticker_exrate(self):
        upbitMgr = UpbitManager('upbit', 0.0005, 'https://api.upbit.com')
        bithumbMgr = BithumbManager('bithumb', 0, 'https://api.bithumb.com')
        bybitMgr = BybitManager('bybit', 0.00055, 'https://api.bybit.com')
        # binanceMgr = BinanceManager('binance', 0.0005, 'https://fapi.binance.com')

        # tasks = [upbitMgr.get_all_ticker_price(), bithumbMgr.get_all_ticker_price(), bybitMgr.get_all_ticker_price(), binanceMgr.get_all_ticker_price()]
        tasks = [upbitMgr.get_all_ticker_price(), bithumbMgr.get_all_ticker_price(), bybitMgr.get_all_ticker_price()]
        results = await asyncio.gather(*tasks)

        results[0] = [{'exchange' : 'upbit', 'market' : i.get('market').split('-')[1], 'price' : i.get('trade_price')} for i in results[0] if i.get('market').startswith('KRW-')]
        results[1] = [{'exchange' : 'bithumb', 'market' : i.get('market').split('-')[1], 'price' : i.get('trade_price')} for i in results[1] if i.get('market').startswith('KRW-')]
        results[2] = [{'exchange' : 'bybit', 'market' : i.get('symbol')[re.search(r'[a-zA-Z]', i.get('symbol')).start() : i.get('symbol').find('USDT')], 'price' : i.get('lastPrice')} for i in results[2]['result']['list'] if i.get('symbol').endswith('USDT')]
        # results[3] = [{'exchange' : 'binance', 'market' : i.get('symbol')[re.search(r'[a-zA-Z]', i.get('symbol')).start() : i.get('symbol').find('USDT')], 'price' : i.get('price')} for i in results[3] if i.get('symbol').endswith('USDT')]
        
        upbit_df = pd.DataFrame(results[0])
        bithumb_df = pd.DataFrame(results[1])
        bybit_df = pd.DataFrame(results[2])
        # binance_df = pd.DataFrame(results[3])

        usdt_price = bithumb_df.loc[bithumb_df.market == 'USDT', 'price'].values[0]

        # merged_df = pd.concat([bybit_df, binance_df]).set_index(['exchange', 'market'])
        merged_df = pd.concat([bybit_df]).set_index(['exchange', 'market'])
        merged_df = merged_df.join(upbit_df.set_index('market'), on='market', rsuffix='_upbit')
        merged_df = merged_df.join(bithumb_df.set_index('market'), on='market', rsuffix='_bithumb')
        valid_df = merged_df.dropna()
        valid_df.loc[:, 'price'] = valid_df.loc[:, 'price'].map(float)

        valid_df['upbit'] = round(valid_df.price_upbit / valid_df.price, 3)
        valid_df['bithumb'] = round(valid_df.price_bithumb / valid_df.price, 3)
        valid_df.loc[:, ['upbit', 'bithumb']] = valid_df.loc[:, ['upbit', 'bithumb']].map(lambda x : x * 1000 if x < 1000 else x) # 1000SHIB, 1000PEPE...

        # valid_df.loc[:, 'kimp_upbit'] = round((valid_df.exrate_upbit - usdt_price) / usdt_price * 100, 3)
        # valid_df.loc[:, 'kimp_bithumb'] = round((valid_df.exrate_bithumb - usdt_price) / usdt_price * 100, 3)
        
        return valid_df.loc[:, ['upbit', 'bithumb']]

    async def get_exrate_kline_dataframe(self, symbol, interval, krw_ex, for_ex):
        self.krw_ex_manager = self.get_exchange_manager(krw_ex)
        self.for_ex_manager = self.get_exchange_manager(for_ex)

        # interval validation check
        # bithumb 4h interval disallowed
        if self.krw_ex_manager.config.name == 'bithumb' and (interval == '4h' or interval == '1d' or interval == '1w'):
            raise ValueError(f'can compare {interval} interval of bithumb data to others')

        krw_task = self.krw_ex_manager.get_kline(symbol, interval)
        usdt_task = self.for_ex_manager.get_kline(symbol, interval)
        tether_task = self.krw_ex_manager.get_kline('USDT', interval)

        krw_raw_data, usdt_raw_data, tether_raw_data = await asyncio.gather(krw_task, usdt_task, tether_task)

        ex_col_map = {
            'upbit' : ['opening_price', 'high_price', 'low_price', 'trade_price', 'candle_acc_trade_volume', 'candle_acc_trade_price'],
            'bithumb' : ['opening_price', 'high_price', 'low_price', 'trade_price', 'candle_acc_trade_volume', 'candle_acc_trade_price'],
            'coinone' : ['open', 'high', 'low', 'close', 'target_volume', 'quote_volume'],
            'korbit' : ['open', 'high', 'low', 'close', 'volume'],
            'bybit' : ['timestamp', 'open', 'high', 'low', 'close', 'target_volume', 'quote_volume'],
            'bitget' : ['timestamp', 'open', 'high', 'low', 'close', 'target_volume', 'quote_volume'],
            'binance' : ['timestamp', 'open', 'high', 'low', 'close', 'target_volume', 'close_time', 'quote_volume', 'Number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'],
            'okx' : ['timestamp', 'open', 'high', 'low', 'close', 'confirm'],
            'gateio' : ['t', 'v', 'c', 'h', 'l', 'o', 'sum']
        }

        ex_datetime_col_map = {
            'upbit' : 'candle_date_time_kst',
            'bithumb' : 'candle_date_time_kst',
            'coinone' : 'timestamp',
            'korbit' : 'timestamp',
            'bybit' : 'timestamp',
            'bitget' : 'timestamp',
            'binance' : 'timestamp',
            'okx' : 'timestamp',
            'gateio' : 't'
        }

        cols_to_convert = ['datetime', 'open', 'high', 'low', 'close', 'target_volume', 'quote_volume']

        krw_df = pd.DataFrame(krw_raw_data)

        if self.krw_ex_manager.config.name in ['upbit', 'bithumb']:
            krw_df.rename(columns={'candle_date_time_kst' : 'datetime'}, inplace=True)
            krw_df.datetime = pd.to_datetime(krw_df.datetime)
            krw_df.datetime = krw_df.datetime.dt.tz_localize('Asia/Seoul')

        elif self.krw_ex_manager.config.name == 'coinone':
            krw_df['datetime'] = pd.to_datetime(krw_df.timestamp, unit='ms')
            krw_df.datetime = krw_df.datetime.dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul')

        elif self.krw_ex_manager.config.name == 'korbit':
            krw_df['datetime'] = pd.to_datetime(krw_df.timestamp, unit='ms')
            krw_df.datetime = krw_df.datetime.dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul')
            krw_df.datetime = krw_df.datetime - pd.Timedelta(hours=3)
            krw_df.sort_values('datetime', inplace=True, ascending=False)

        krw_df[ex_col_map[self.krw_ex_manager.config.name]] = krw_df[ex_col_map[self.krw_ex_manager.config.name]].apply(pd.to_numeric)
        krw_df = krw_df[['datetime'] + ex_col_map[self.krw_ex_manager.config.name]]

        if self.krw_ex_manager.config.name == 'korbit':
            cols_to_convert = ['datetime', 'open', 'high', 'low', 'close', 'volume']
        krw_df.columns = cols_to_convert

        # print(krw_df.loc[krw_df.datetime.dt.date == pd.to_datetime('2024-10-11').date()])
        usdt_df = pd.DataFrame(usdt_raw_data, columns=ex_col_map[self.for_ex_manager.config.name])

        if self.for_ex_manager.config.name == 'gateio':
            usdt_df.rename(columns={'t' : 'timestamp', 'o' : 'open', 'h' : 'high', 'l' : 'low', 'c' : 'close', 'v' : 'target_volume', 'sum' : 'quote_volume'}, inplace=True)
        
        filtered_cols = ['timestamp', 'open', 'high', 'low', 'close', 'target_volume', 'quote_volume']

        if self.for_ex_manager.config.name == 'okx':
            filtered_cols = ['timestamp', 'open', 'high', 'low', 'close']

        usdt_df = usdt_df[filtered_cols]

        if self.for_ex_manager.config.name == 'gateio':
            usdt_df['datetime'] = pd.to_datetime(usdt_df.timestamp, unit='s')
        else:
            usdt_df['datetime'] = pd.to_datetime(usdt_df.timestamp, unit='ms')

        usdt_df.datetime = usdt_df.datetime.dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul')
        usdt_df[filtered_cols] = usdt_df[filtered_cols].apply(pd.to_numeric)

        tether_df = pd.DataFrame(tether_raw_data)
        tether_df.rename(columns={'candle_date_time_kst' : 'datetime'}, inplace=True)
        tether_df.datetime = pd.to_datetime(tether_df.datetime)
        tether_df.datetime = tether_df.datetime.dt.tz_localize('Asia/Seoul')
        tether_df[ex_col_map[self.krw_ex_manager.config.name]] = tether_df[ex_col_map[self.krw_ex_manager.config.name]].apply(pd.to_numeric)
        tether_df = tether_df[['datetime'] + ex_col_map[self.krw_ex_manager.config.name]]
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
        except ClientError as e:
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
        


# Define states

# Get Low Exrate Tickers Or Get High Exrate Tickers
ASK_BUDGET = 0

# Get Exrate Kline Plot
ASK_KLINE_INTERVAL, ASK_KLINE_KRW_EXCHANGE, ASK_KLINE_FOR_EXCHANGE, HANDLE_KLINE = range(1, 5)

# Get Real Time Exrate
ASK_EXRATE_BUDGET, ASK_EXRATE_KRW_EXCHANGE, ASK_EXRATE_FOR_EXCHANGE, HANDLE_EXRATE = range(5, 9)

# Run Exrate Arbitrage Bot
ASK_ARBITRAGE_ENTRY_EXRATE, ASK_ARBITRAGE_CLOSE_EXRATE, ASK_ARBITRAGE_BUDGET, ASK_ARBITRAGE_SHORT_LEVERAGE, ASK_ARBITRAGE_KRW_EX, ASK_ARBITRAGE_FOR_EX, HANDLE_ARBIRTAGE = range(9, 16)

# Dictionary to map options to their states and handlers
conversation_options = {
    "Get Low Exrate Tickers": {
        "entry_action": "ask_budget",
        "state": ASK_BUDGET,
        "handler": "handle_exrate_tickers",
        "params": ["low"]
    },
    "Get High Exrate Tickers": {
        "entry_action": "ask_budget",
        "state": ASK_BUDGET,
        "handler": "handle_exrate_tickers",
        "params": ["high"]
    },
    "Get Exrate Kline Plot": {
        "entry_action": "ask_kline_symbol",
        "state": [ASK_KLINE_INTERVAL, ASK_KLINE_KRW_EXCHANGE, ASK_KLINE_FOR_EXCHANGE, HANDLE_KLINE],
        "handler": ["ask_kline_interval", "ask_kline_krw_exchange", "ask_kline_for_exchange", "handle_kline"],
        "handler_type": ["MessageHandler", "CallbackQueryHandler", "CallbackQueryHandler", "CallbackQueryHandler"]
    },
    "Get Real Time Exrate": {
        "entry_action": "ask_symbol",
        "state": [ASK_EXRATE_BUDGET, ASK_EXRATE_KRW_EXCHANGE, ASK_EXRATE_FOR_EXCHANGE, HANDLE_EXRATE],
        "handler": ["ask_exrate_budget", "ask_exrate_krw_exchange", "ask_exrate_for_exchange", "handle_exrate"],
        "handler_type": ["MessageHandler", "MessageHandler", "CallbackQueryHandler", "CallbackQueryHandler"]
    },
    "Run Exrate Arbitrage Bot": {
        "entry_action": "ask_arbitrage_symbol",
        "state": [ASK_ARBITRAGE_ENTRY_EXRATE, ASK_ARBITRAGE_CLOSE_EXRATE, ASK_ARBITRAGE_BUDGET, ASK_ARBITRAGE_SHORT_LEVERAGE, ASK_ARBITRAGE_KRW_EX, ASK_ARBITRAGE_FOR_EX, HANDLE_ARBIRTAGE],
        "handler": ['ask_arbitrage_entry_exrate', 'ask_arbitrage_close_exrate', 'ask_arbitrage_budget', 'ask_arbitrage_short_leverage', 'ask_arbitrage_krw_ex', 'ask_arbitrage_for_ex', 'handle_arbitrage'],
        "handler_type": ["MessageHandler", "MessageHandler", "MessageHandler", "MessageHandler", "MessageHandler", "CallbackQueryHandler", "CallbackQueryHandler"]
    },
}

# Function to handle the /start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # Create a reply keyboard
    keyboard = [
        [KeyboardButton("Get Low Exrate Tickers"), KeyboardButton("Get High Exrate Tickers")],
        [KeyboardButton("Get Exrate Kline Plot"), KeyboardButton("Get Real Time Exrate")],
        [KeyboardButton("Run Exrate Arbitrage Bot"), KeyboardButton("Change Close Exrate")],
        [KeyboardButton("Stop"), KeyboardButton("Cancel")]
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
            # If user sends "Cancel", stop processing further
            if update.message and update.message.text.lower() == "cancel":
                await update.message.reply_text("Action canceled.")
                return ConversationHandler.END
            
            # Call the original handler function
            return await func(update, context, *args, **kwargs)
        except Exception as e:
            # Handle any exceptions in the handlers
            logger.error(f"Error in handler {func.__name__}: {e}")
            await update.message.reply_text(f"An error occurred: {str(e)}")
            return ConversationHandler.END
    return wrapper

# Get Low Exrate Tickers & Get High Exrate Tickers buttons
@safe_handler
async def ask_budget(update: Update, context: ContextTypes.DEFAULT_TYPE):
    option = update.message.text
    context.user_data['event_caller'] = option
    context.user_data[option] = {}
    context.user_data[option]['param'] = conversation_options.get(option, {}).get('params')[0]
    await update.message.reply_text("Please enter your budget (numeric value):")
    return ASK_BUDGET

@safe_handler
async def handle_exrate_tickers(update: Update, context: ContextTypes.DEFAULT_TYPE):
    event_caller = context.user_data.get('event_caller')
    try:
        budget = float(update.message.text)
        param = context.user_data[event_caller].get('param')

        k = KimpManager('bithumb', 'bybit')
        df = await k.get_all_ticker_exrate(budget)

        result = await k.krw_ex_manager.get_single_ticker_price('USDT')
        usdt_price = result[0]['trade_price']

        if param == 'low':
            tmp_df = df.loc[(df.exrate < usdt_price) & (~df.index.isin(['TON']))].sort_values(by='exrate')
            tmp_df = tmp_df.head(10)

        elif param == 'high':
            tmp_df = df.loc[(df.exrate >= usdt_price) & (~df.index.isin(['TON']))].sort_values(by='exrate')
            tmp_df = tmp_df.tail(10)

        df_str = tmp_df.to_string()
        await update.message.reply_text(f'USDT PRICE : {usdt_price}')
        await update.message.reply_text(f"```\n{df_str}\n```", parse_mode='Markdown')
    
    except Exception as e:
        logger.error(e)
        await update.message.reply_text(str(e))
    finally:
        del context.user_data[event_caller]
        return ConversationHandler.END


# Get Exrate Kline Plot button
@safe_handler
async def ask_kline_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    option = update.message.text
    context.user_data['event_caller'] = option
    context.user_data[option] = {}
    await update.message.reply_text("Please enter symbol, interval : (e.g. XRP)")
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
    await update.message.reply_text("Please choose an option:", reply_markup=reply_markup)

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
    await query.edit_message_text("Please choose an option:", reply_markup=reply_markup)

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
    await query.edit_message_text("Please choose an option:", reply_markup=reply_markup)

    return HANDLE_KLINE

@safe_handler
async def handle_kline(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()  # Acknowledge the callback

    event_caller = context.user_data['event_caller']
    context.user_data[event_caller]["for_ex"] = query.data
    
    try:
        k = KimpManager('bithumb', 'bybit')
        await k.request_exrate_picture(context.user_data[event_caller]["symbol"], \
                                       context.user_data[event_caller]["interval"], \
                                       context.user_data[event_caller]["krw_ex"], \
                                       context.user_data[event_caller]["for_ex"])
    except Exception as e:
        logger.error(e)
        await update.message.reply_text(str(e))
    finally:
        del context.user_data[event_caller]
        return ConversationHandler.END


# Get Real Time Exrate button
@safe_handler
async def ask_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    option = update.message.text
    context.user_data['event_caller'] = option
    context.user_data[option] = {}
    await update.message.reply_text("Please enter symbol : (e.g. XRP)")
    return ASK_EXRATE_BUDGET

@safe_handler
async def ask_exrate_budget(update: Update, context: ContextTypes.DEFAULT_TYPE):
    event_caller = context.user_data['event_caller']
    context.user_data[event_caller]["symbol"] = update.message.text
    await update.message.reply_text("Please enter your budget (numeric value):")
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
    await update.message.reply_text("Please choose an option:", reply_markup=reply_markup)

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
    await query.edit_message_text("Please choose an option:", reply_markup=reply_markup)

    return HANDLE_EXRATE

@safe_handler
async def exrate_task(update: Update, context: ContextTypes.DEFAULT_TYPE):
    event_caller = context.user_data['event_caller']
    
    try:
        k = KimpManager('bithumb', 'bybit')
        while not context.user_data['stop_event'].is_set():
            task = []
            task.append(k.get_single_ticker_kimp_by_seed(context.user_data[event_caller]["symbol"], \
                                                         context.user_data[event_caller]["krw_ex"], \
                                                         context.user_data[event_caller]["for_ex"]))
            task.append(k.krw_ex_manager.get_single_ticker_price('USDT'))

            result = await asyncio.gather(*task)
            df = result[0]
            df = df.loc[df.seed <= context.user_data[event_caller]['budget'], ['seed', 'exrate']].iloc[-1]
            seed = int(df.seed)
            exrate = df.exrate
            usdt_price = result[1][0]['trade_price']

            await update.callback_query.message.reply_text(f"SEED: {seed}\n"
                                                           f"USDT PRICE: {usdt_price}\n"
                                                           f"EXRATE: {exrate}",
                                                           parse_mode='Markdown')
            
            await asyncio.sleep(0.5)
    except Exception as e:
        logger.error(e)
        await update.callback_query.message.reply_text(str(e))
    finally:
        del context.user_data[event_caller]
        return ConversationHandler.END   

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


# Starting point for the arbitrage bot
@safe_handler
async def ask_arbitrage_symbol(update: Update, context: ContextTypes.DEFAULT_TYPE):
    option = update.message.text
    context.user_data['event_caller'] = option
    context.user_data[option] = {}
    await update.message.reply_text("Please enter the symbol:")
    return ASK_ARBITRAGE_ENTRY_EXRATE

@safe_handler
async def ask_arbitrage_entry_exrate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    event_caller = context.user_data['event_caller']
    context.user_data[event_caller]["symbol"] = update.message.text  # Save the symbol
    await update.message.reply_text("Please enter the entry exrate:")
    return ASK_ARBITRAGE_CLOSE_EXRATE

@safe_handler
async def ask_arbitrage_close_exrate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    event_caller = context.user_data['event_caller']
    context.user_data[event_caller]["entry_exrate"] = update.message.text # Save entry exrate
    await update.message.reply_text("Please enter the close exrate:")
    return ASK_ARBITRAGE_BUDGET

@safe_handler
async def ask_arbitrage_budget(update: Update, context: ContextTypes.DEFAULT_TYPE):
    event_caller = context.user_data['event_caller']
    context.user_data[event_caller]["close_exrate"] = update.message.text  # Save close exrate
    await update.message.reply_text("Please enter the budget:")
    return ASK_ARBITRAGE_SHORT_LEVERAGE

@safe_handler
async def ask_arbitrage_short_leverage(update: Update, context: ContextTypes.DEFAULT_TYPE):
    event_caller = context.user_data['event_caller']
    context.user_data[event_caller]["budget"] = update.message.text  # Save short leverage
    await update.message.reply_text("Please enter the short leverage:")
    return ASK_ARBITRAGE_KRW_EX

@safe_handler
async def ask_arbitrage_krw_ex(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
    await update.message.reply_text("Please choose an option:", reply_markup=reply_markup)

    return ASK_ARBITRAGE_FOR_EX

@safe_handler
async def ask_arbitrage_for_ex(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
    await query.edit_message_text("Please choose an option:", reply_markup=reply_markup)

    return HANDLE_ARBIRTAGE

@safe_handler
async def handle_arbitrage(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()

    event_caller = context.user_data['event_caller']
    context.user_data[event_caller]["for_ex"] = query.data
    
    # Perform the arbitrage logic
    symbol          = context.user_data[event_caller]["symbol"]
    entry_exrate    = context.user_data[event_caller]["entry_exrate"]
    close_exrate    = context.user_data[event_caller]["close_exrate"]
    budget          = context.user_data[event_caller]["budget"]
    leverage        = context.user_data[event_caller]["leverage"]
    krw_ex          = context.user_data[event_caller]["krw_ex"]
    for_ex          = context.user_data[event_caller]["for_ex"]

    logger.info(f'\n krw_ex : {krw_ex} \
                  \n for_ex : {for_ex}')

    # Create a stop event
    context.user_data["stop_event"] = asyncio.Event()

    try:
        k = KimpManager('bithumb', 'bybit')
        task = asyncio.create_task(k.run_arbitrage_bot(update, context, event_caller)) 
        context.user_data["task"] = task
        return ConversationHandler.END
    except Exception as e:
        logger.error(e)
        await update.callback_query.message.reply_text(str(e))
        context.user_data["stop_event"] = None
        del context.user_data[event_caller]


async def break_loop(update: Update, context: ContextTypes.DEFAULT_TYPE):
    stop_event = context.user_data.get('stop_event')
    task = context.user_data.get('task')

    if not stop_event or not task or task.done():
        await update.message.reply_text('No running loop to stop...')
        return
    
    stop_event.set()
    await task
    await update.message.reply_text('Break the loop!!!')

async def change_close_exrate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if context.user_data.get('Run Exrate Arbitrage Bot', 0):
        await update.message.reply_text("please enter the new close exrate: ")
        context.user_data['Run Exrate Arbitrage Bot']['close_exrate'] = update.message.text
    await update.message.reply_text("No bot is running...")

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info('conversation cancel')
    await update.message.reply_text("Action canceled.")
    return ConversationHandler.END

def main():
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
                    handler = MessageHandler(filters.TEXT & ~filters.COMMAND, handler_function)
                elif handler_type == "CallbackQueryHandler":
                    handler = CallbackQueryHandler(handler_function)
                else:
                    raise ValueError(f"Unsupported handler type: {handler_type}")
                
                states[state] = [handler]

            # Create the ConversationHandler
            conv_handler = ConversationHandler(
                entry_points=[
                    MessageHandler(filters.Regex(f"^{option}$"), globals()[details["entry_action"]])
                ],
                states=states,
                fallbacks=[MessageHandler(filters.Regex("^Cancel$"), cancel)],
            )
        else:
            # Single-step conversation
            handler_type = details.get("handler_type", "MessageHandler")  # Default to MessageHandler
            handler_function = globals()[details["handler"]]
            
            if handler_type == "MessageHandler":
                handler = MessageHandler(filters.TEXT & ~filters.COMMAND, handler_function)
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
                fallbacks=[MessageHandler(filters.Regex("^Cancel$"), cancel)],
            )

        # Add the conversation handler to the application
        application.add_handler(conv_handler)

    application.add_handler(MessageHandler(filters.Regex("^Stop$"), break_loop))
    application.add_handler(MessageHandler(filters.Regex("^Change Close Exrate$"), change_close_exrate))

    # Run the bot
    application.run_polling()

if __name__ == "__main__":
    main()
    # k = KimpManager('upbit', 'bybit')
    # print(asyncio.run(k.krw_ex_manager.get_balance('BSV')))
    # print(json.dumps(asyncio.run(k.for_ex_manager.get_balance('PEPE')), indent=4))
    # print(asyncio.run(k.for_ex_manager.post_order('linear', 'BSV', 'Sell', 'Market', '0.09')))

    # class Context:
    #     user_data = None
    
    # context = Context()
    # context.user_data = {
    #             'symbol': 'DOGE',
    #             'entry_exrate': 1390,
    #             'close_exrate': 1420,
    #             'budget': 10000,
    #             'krw_ex': 'upbit',
    #             'for_ex': 'bybit',
    #             'leverage': 1,
    #             'stop_event': asyncio.Event()
    #         }
    # asyncio.run(k.run_arbitrage_bot({}, context)) 
    # print(asyncio.run(k.for_ex_manager.set_leverage('linear', 'ADA', '5', '5')))
    # print(asyncio.run(k.for_ex_manager.post_order('linear', 'ADA', 'Sell', 'Market', '10')))


    # print(asyncio.run(k.get_single_ticker_kimp_by_seed('ASTR', 'bithumb', 'bybit')))
    # df = asyncio.run(k.get_all_ticker_exrate(1_000_000))
    # print(df)
# asyncio.run(k.run_exrate_alarm_bot(1_000_000))
# print(json.dumps(asyncio.run(k.for_ex_manager.get_orderbook('1000PEPE')), indent=4))
# print(asyncio.run(k.get_single_ticker_kimp_by_seed('BTT', 'bithumb', 'bybit')))
# asyncio.run(k.request_exrate_picture('GMT', '5m'))
# print(asyncio.run(k.get_single_ticker_kimp_by_seed('DOGE')))


# print(asyncio.run(k.get_exrate_dataframe('XRP', '1m')))
# print(asyncio.run(post_bybit_order('linear', 'XRP', 'Sell', 'Market', '6.96092114')))
# print(asyncio.run(k.for_ex_manager.get_balance('USDT')))
# print(asyncio.run(k.for_ex_manager.post_order('linear', 'CARV', 'Sell', 'Market', '6')))
# print(asyncio.run(k.krw_ex_manager.get_single_ticker_price('XRP')))
# asyncio.run(k.run_arbitrage_bot('CARV', 1390, 1400, 10000))
# asyncio.run(k.run_monitor_all_ticker_kimp())

# t = ExchangeManager('bybit', '0.00055', 'https://api.bybit.com')
# print(asyncio.run(t.get_kline('BTC', '15m')))

# k = KimpManager('upbit', 'bybit')
# print(asyncio.run(k.get_single_ticker_kimp_by_seed('XRP')))

# class BithumbManager(CryptoManager):
    

# get_bithumb_market_data('BTC')
# asyncio.run(kimp_bot('DOGE', 'bithumb', 'bybit'))

# from pybit.unified_trading import HTTP
# session = HTTP(
#     api_key="GgFAIatE9xbOZ5Yrul",
#     api_secret="kubaMxbOddkWcJDZk0A8yv8E7kMQ2v5j5pZS",
# )
# print(session.get_positions(
#     category="linear"
# ))