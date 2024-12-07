# constants.py
from enum import Enum

class ExApiUrls(Enum):
    UPBIT = 'https://api.upbit.com'
    BITHUMB = 'https://api.bithumb.com'
    BYBIT = 'https://api.bybit.com'
    BINANCE = 'https://fapi.binance.com'

class Fees(Enum):
    UPBIT = {'maker': 0.0005, 'taker': 0.0005}
    BITHUMB = {'maker': 0.0004, 'taker': 0.0004}
    BYBIT = {'maker': 0.0002, 'taker': 0.00055}
    BINANCE = {'maker': 0.0005, 'taker': 0.0005}

class RequestConfig(Enum):
    RECV_WINDOW = "5000"

class BybitSymbols(Enum):
    AIDOGE = '10000000AIDOGE'
    BABYDOGE = '1000000BABYDOGE'
    CHEEMS = '1000000CHEEMS'
    MOG = '1000000MOG' 
    PEIPEI = '1000000PEIPEI'
    COQ = '10000COQ'
    LADYS = '10000LADYS'
    SATS = '10000SATS'
    WEN = '10000WEN'
    WHY = '10000WHY'
    APU = '1000APU'
    BONK = '1000BONK'
    BTT = '1000BTT'
    CATS = '1000CATS'
    CAT = '1000CAT'
    FLOKI = '1000FLOKI'
    LUNC = '1000LUNC'
    MUMU = '1000MUMU'
    NEIROCTO = '1000NEIROCTO'
    PEPE = '1000PEPE'
    RATS = '1000RATS'
    TURBO = '1000TURBO'
    XEC = '1000XEC'
    X = '1000X'
    INCH = '1INCH'
    A = 'A8'
    API = 'API3'
    C = 'C98'
    DOP = 'DOP1'
    HPOSI = 'HPOS10I'
    L = 'L3'
    LUNA = 'LUNA2'
    RSS = 'RSS3'
    SHIB = 'SHIB1000'
    TOSHI = '1000TOSHI'

class BinanceSymbols(Enum):
    SHIB = '1000SHIB'
    PEPE = '1000PEPE'
    FLOKI = '1000FLOKI'
    BONK = '1000BONK'
    XEC = '1000XEC'
    RATS = '1000RATS'
    MOG = '1000000MOG'
    WHY = '1000WHY'
    C = 'C98'
    SATS = '1000SATS'
    X = '1000X'
    LUNC = '1000LUNC'
    CHEEMS = '1000CHEEMS'
    INCH = '1INCH'
    CAT = '1000CAT'
    API = 'API3'
    LUNA = 'LUNA2'
    MBABYDOGE = '1MBABYDOGE'

class ExchangeSymbols(Enum):
    BYBIT = BybitSymbols
    BINANCE = BinanceSymbols

KLINE_COLUMN_MAP = {
    'upbit': {
        'opening_price': 'open',
        'high_price': 'high', 
        'low_price': 'low',
        'trade_price': 'close',
        'candle_acc_trade_volume': 'volume'
    },
    'bithumb': {
        'opening_price': 'open',
        'high_price': 'high',
        'low_price': 'low', 
        'trade_price': 'close',
        'candle_acc_trade_volume': 'volume'
    },
    'bybit': {
        'timestamp': 'timestamp',
        'open': 'open', 
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'target_volume': 'volume'
    },
    'binance': {
        'timestamp': 'timestamp',
        'open': 'open',
        'high': 'high', 
        'low': 'low',
        'close': 'close',
        'volume': 'volume'
    }
}