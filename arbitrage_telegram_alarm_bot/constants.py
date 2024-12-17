# constants.py
from enum import Enum
from dataclasses import dataclass, asdict

interval_minutes = {
                    '1m' : 1,
                    '3m' : 3,
                    '5m' : 5,
                    '15m' : 15,
                    '30m' : 30,
                    '1h' : 60,
                    '4h' : 240,
                    '1d' : 1440,
                    '1w' : 10080
                }

ex_col_map = {
    'upbit' : ['opening_price', 'high_price', 'low_price', 'trade_price', 'candle_acc_trade_volume', 'candle_acc_trade_price'],
    'bithumb' : ['opening_price', 'high_price', 'low_price', 'trade_price', 'candle_acc_trade_volume', 'candle_acc_trade_price'],
    'coinone' : ['open', 'high', 'low', 'close', 'target_volume', 'quote_volume'],
    'korbit' : ['open', 'high', 'low', 'close', 'volume'],
    'bybit' : ['timestamp', 'open', 'high', 'low', 'close', 'target_volume', 'quote_volume'],
    'bitget' : ['timestamp', 'open', 'high', 'low', 'close', 'target_volume', 'close_time', 'quote_volume', 'Number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'],
    'binance' : ['timestamp', 'open', 'high', 'low', 'close', 'target_volume', 'quote_volume'],
    'okx' : ['timestamp', 'open', 'high', 'low', 'close', 'confirm'],
    'gateio' : ['t', 'v', 'c', 'h', 'l', 'o', 'sum']
}

ex_kline_col_map = {
    'bybit': ['timestamp', 'open', 'high', 'low', 'close', 'target_volume', 'quote_volume'],
    'binance': ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'],
}

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

KLINE_RESPONSE_MAP = {
    'bybit' : ['timestamp', 'open', 'high', 'low', 'close', 'target_volume', 'quote_volume'],
    'bitget' : ['timestamp', 'open', 'high', 'low', 'close', 'target_volume', 'close_time', 'quote_volume', 'Number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'],
    'binance' : ['timestamp', 'open', 'high', 'low', 'close', 'target_volume', 'quote_volume'],
    'okx' : ['timestamp', 'open', 'high', 'low', 'close', 'confirm'],
    'gateio' : ['t', 'v', 'c', 'h', 'l', 'o', 'sum']
}

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

@dataclass
class PositionEntryIn:
    symbol: str
    side: str
    order_type: str
    qty: float
    tp: float = None
    sl: float = None

    def to_dict(self):
        return asdict(self)

    def not_None_to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}
    
@dataclass
class BybitPositionEntryIn:
    symbol: str
    side: str # Buy or Sell
    orderType: str # Market or Limit
    qty: str
    category: str = 'linear'
    tpslMode: str = None # Full or Partial
    slOrderType: str = None # Market or Limit
    slLimitPrice: str = None 

    def to_dict(self):
        return asdict(self)

    def not_None_to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}
    
@dataclass
class BinancePositionEntryIn:
    symbol: str
    side: str # BUY or SELL
    type: str # MARKET or LIMIT or STOP_MARKET or TAKE_PROFIT_MARKET
    quantity: float
    positionSide: str
    stopPrice: float = None
    closePosition: str = None # true or false
    recvWindow: int = 5000

    def to_dict(self):
        return asdict(self)

    def not_None_to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}
    
@dataclass
class UpbitPositionEntryIn:
    market: str
    side: str
    price: str
    volume: str
    ord_type: str

    def to_dict(self):
        return asdict(self)

    def not_None_to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}
    
@dataclass
class BithumbPositionEntryIn:
    market: str
    side: str
    price: str
    volume: str
    ord_type: str

    def to_dict(self):
        return asdict(self)

    def not_None_to_dict(self):
        return {k: v for k, v in asdict(self).items() if v is not None}
    
import time
from typing import Union

class PositionEntryMapper:
    # Mapping dictionaries
    BYBIT_SIDE_MAP = {
        'bid': 'Buy',
        'ask': 'Sell'
    }
    
    BINANCE_SIDE_MAP = {
        'bid': 'BUY',
        'ask': 'SELL'
    }

    ORDER_TYPE_MAP = {
        'market': 'Market',
        'limit': 'Limit',
        'stop_market': 'STOP_MARKET',
        'take_profit_market': 'TAKE_PROFIT_MARKET'
    }

    @staticmethod
    def to_bybit(entry: PositionEntryIn) -> BybitPositionEntryIn:
        """Convert PositionEntryIn to BybitPositionEntryIn"""
        return BybitPositionEntryIn(
            symbol=entry.symbol,
            side=PositionEntryMapper.BYBIT_SIDE_MAP[entry.side],
            orderType=PositionEntryMapper.ORDER_TYPE_MAP[entry.order_type],
            qty=str(entry.qty),
            tpslMode='Full' if entry.tp or entry.sl else None,
            slOrderType='Market' if entry.sl else None,
            slLimitPrice=str(entry.sl) if entry.sl else None
        )

    @staticmethod
    def to_binance(entry: PositionEntryIn) -> BinancePositionEntryIn:
        """Convert PositionEntryIn to BinancePositionEntryIn"""
        return BinancePositionEntryIn(
            symbol=entry.symbol,
            side=PositionEntryMapper.BINANCE_SIDE_MAP[entry.side],
            type=PositionEntryMapper.ORDER_TYPE_MAP[entry.order_type].upper(),
            quantity=float(entry.qty),
            positionSide='LONG' if entry.side == 'bid' else 'SHORT',
            stopPrice=entry.sl if entry.sl else None,
            closePosition='true' if entry.tp else None
        )
    
    @staticmethod
    def to_upbit(entry: PositionEntryIn) -> dict:
        """Convert PositionEntryIn to Upbit-specific entry"""
        return UpbitPositionEntryIn(
            market=entry.symbol,
            side=entry.side,
            price=str(entry.qty) if entry.side == 'bid' else None,
            volume=str(entry.qty) if entry.side == 'ask' else None,
            ord_type='price' if entry.side == 'bid' else 'market'
        )
    
    @staticmethod
    def to_bithumb(entry: PositionEntryIn) -> dict:
        """Convert PositionEntryIn to Bithumb-specific entry"""
        return BithumbPositionEntryIn(
            market=entry.symbol,
            side=entry.side,
            price=str(entry.qty) if entry.side == 'bid' else None,
            volume=str(entry.qty) if entry.side == 'ask' else None,
            ord_type='price' if entry.side == 'bid' else 'market'
        )

    @staticmethod
    def to_exchange(entry: PositionEntryIn, exchange: str) -> Union[BybitPositionEntryIn, BinancePositionEntryIn]:
        """Convert PositionEntryIn to exchange-specific entry"""
        if exchange.lower() == 'bybit':
            return PositionEntryMapper.to_bybit(entry)
        elif exchange.lower() == 'binance':
            return PositionEntryMapper.to_binance(entry)
        elif exchange.lower() == 'upbit':
            return PositionEntryMapper.to_upbit(entry)
        elif exchange.lower() == 'bithumb':
            return PositionEntryMapper.to_bithumb(entry)
        else:
            raise ValueError(f"Unsupported exchange: {exchange}")