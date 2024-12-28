import ccxt.async_support as ccxt
from typing import List
import pandas as pd
import asyncio
import json
import time

fee_rate = {
    'coinone' : 0.002,
    'bithumb' : 0.0004,
    'upbit' : 0.0005,
    'binance' : 0.0005,
    'bybit' : 0.00055,
    'okx' : 0.0005,
    'bitget' : 0.0006,
    'gateio' : 0.00075
}

async def load_market(exchange):
    exchange_class = getattr(ccxt, exchange)
    exchange = exchange_class()

    try:
        return await exchange.load_markets()
    except Exception as e:
        print(f'error : {e}')
    finally:
        await exchange.close()

def extract_common_tickers(ex1_config:str, d1:dict, ex2_config:str, d2:dict):
    d1 = { t['base'] : t for t in filter(lambda x : x['active'] == True and \
                                         x[ex1_config['type']] == True and \
                                         x['quote'] == ex1_config['quote'], 
                                         d1.values()) }

    d2 = { t['base'] : t for t in filter(lambda x : x['active'] == True and \
                                         x[ex2_config['type']] == True and \
                                         x['quote'] == ex2_config['quote'], 
                                         d2.values()) }
    
    common_tickers = d1.keys() & d2.keys()

    return { f'{ex1_config["name"]}_{ex2_config["name"]}' : [(d1[x], d2[x]) for x in common_tickers] }

async def concurrent_fetch_tickers(exchanges:List[str]):
    tasks = []
    ex_list = []
    for ex in exchanges:
        exchange_class = getattr(ccxt, ex)
        exchange = exchange_class()
        ex_list.append(exchange)

        tasks.append(exchange.fetch_tickers())
    
    results = await asyncio.gather(*tasks)

    tasks = []
    for ex in ex_list:
        tasks.append(ex.close())
    
    asyncio.gather(*tasks)

    return results

async def extract_lowest_kimp_ex_set_for_all_tickers(exchanges):
    '''
        161    NFP     132.20    coinone    0.23020      bybit       574.2832
        176    PHA     148.70    coinone    0.11534      bybit      1289.2318
        137    LIT     912.50    coinone    0.70410      bybit      1295.9807
        74    DATA      48.88    coinone    0.03744      bybit      1305.5556
        163   NTRN     545.00    coinone    0.41720      bybit      1306.3279
        ..     ...        ...        ...        ...        ...            ...
        156    MON     158.90    coinone    0.10277      bybit      1546.1711
        201    SCA     572.50    coinone    0.36570      bybit      1565.4908
        14   ALPHA     140.10    coinone    0.08775      bybit      1596.5812
        188   RDNT      96.96    bithumb    0.06055      bybit      1601.3212
        193    RON    2783.00    coinone    1.67100      bybit      1665.4698
    '''
    results = await concurrent_fetch_tickers(exchanges)

    columns = ['exchange', 'symbol', 'quote', 'price']

    # empty dataframe with columns defined
    df = pd.DataFrame(columns=columns)

    for result, ex in zip(results, exchanges):
        data = {
            'symbol' : [],
            'quote' : [],
            'price' : [],
            'spotOrfuture' : []
        }
        for key, val in result.items():
            if not isinstance(val['last'], int) and not isinstance(val['last'], float):
                continue

            try:
                quote = key.split('/')[1]
                if ':' in key.split('/')[1]:
                    quote = quote.split(':')[0]
                    data['spotOrfuture'] = 'future'
                else:
                    data['spotOrfuture'] = 'spot'
                
                symbol = key.split('/')[0]

            # [No Quote] except example : GRASSUSDT on bybit 
            except IndexError as e:
                # if there are funding rate and open interest, then this is for perpetual ticker
                if float(val['info'].get('fundingRate', 0)) > 0 or float(val['info'].get('openInterest', 0)) > 0:

                    f = key.rfind('USDT')
                    if f == -1:
                        print(key, ex)
                        continue
                    
                    data['spotOrfuture'] = 'future'
                    quote = 'USDT'
                    symbol = key[:f]
                else:
                    continue
            
            except Exception as e:
                print(e)
                continue

            data['symbol'].append(key.split('/')[0])
            data['quote'].append(quote)
            data['price'].append(val['last'])
        
        tmp_df = pd.DataFrame(data)
        tmp_df['exchange'] = ex

        df = pd.concat([df, tmp_df])

    df.reset_index(drop=True, inplace=True)

    spot_df = df.loc[(df.spotOrfuture == 'spot') & (df.quote == 'KRW')]
    spot_df2 = spot_df.groupby('symbol').agg(
        min_price=('price', 'min'),
        min_idx=('price', 'idxmin')
    ).reset_index()

    spot_df2['exchange'] = spot_df2.min_idx.map(spot_df.exchange)
    final_spot_df = spot_df2.drop(columns=['min_idx'])

    future_df = df.loc[(df.spotOrfuture == 'future') * (df.quote == 'USDT')]
    future_df2 = future_df.groupby('symbol').agg(
        max_price=('price', 'max'),
        max_idx=('price', 'idxmax')
    ).reset_index()

    future_df2['exchange'] = future_df2.max_idx.map(future_df.exchange)
    final_future_df = future_df2.drop(columns=['max_idx'])

    merged_df = pd.merge(final_spot_df, final_future_df, on='symbol', how='inner')
    merged_df['kimchipremium'] = round(merged_df.min_price / merged_df.max_price, 4)
    merged_df.sort_values('kimchipremium', inplace=True)

    print(merged_df)

    return merged_df

async def get_kline_data(exchange:str, symbol:str, interval:str, size:int):
    '''
        https://api.bybit.com/v5/market/kline?category=linear&symbol=XRPUSDT&interval=D&limit=5
        [
            [
                "1729468800000",
                "0.5481",
                "0.561",
                "0.5421",
                "0.544",
                "336702374",
                "185667278.0993"
            ],
            [
                "1729382400000",
                "0.5439",
                "0.5487",
                "0.5361",
                "0.5481",
                "176072052",
                "95427806.2162"
            ],
            [
                "1729296000000",
                "0.5465",
                "0.5499",
                "0.5398",
                "0.5439",
                "142907464",
                "77902425.1009"
            ],
            [
                "1729209600000",
                "0.5441",
                "0.553",
                "0.5394",
                "0.5465",
                "296037114",
                "161691909.3721"
            ],
            [
                "1729123200000",
                "0.5479",
                "0.5666",
                "0.5411",
                "0.5441",
                "431572776",
                "238848513.4854"
            ]
        ]
    '''
    exchanges = {
        'upbit' : {
            'url' : 'https://api.upbit.com/v1/candles/{}{}?market={}&count={}',
            'path_params' : ['minutes', 'days', 'weeks', 'months'],
            'query_params' : ['market', 'count'],
            'limit' : 200,
            'interval_enum' : [1,3,5,10,15,30,60,240]
        },
        'bithumb' : {
            'url' : 'https://api.bithumb.com/v1/candles/{}{}?market={}&count={}',
            'path_params' : ['minutes', 'days', 'weeks', 'months'],
            'query_params' : ['market', 'count'],
            'limit' : 200,
            'interval_enum' : [1,3,5,10,15,30,60,240]
        },
        'coinone' : {
            'url' : 'https://api.coinone.co.kr/public/v2/chart/KRW/{}?interval={}&size={}',
            'path_params' : ['target_currency'],
            'query_params' : ['interval', 'size'],
            'limit' : 500,
            'interval_enum' : ['1m', '3m', '5m', '10m', '15m', '30m', '1h', '2h', '4h', '6h', '1d', '1w', '1mon']
        },
        'korbit' : {
            'url' : 'https://api.korbit.co.kr/v2/candles?symbol={}&interval={}&limit={}',
            'query_params' : ['symbol', 'interval', 'limit'],
            'limit' : 200,
            'interval_enum' : [1,5,15,30,60,240,'1D','1W']
        },
        'bybit' : {
            'url' : 'https://api.bybit.com/v5/market/kline?category=linear&symbol={}&interval={}&limit={}',
            'query_params' : ['category', 'symbol', 'interval', 'limit'],
            'limit' : 1000,
            'interval_enum' : [1,3,5,15,30,60,120,240,360,720,'D','W','M']
        },
        'binance' : {
            'url' : 'https://fapi.binance.com/fapi/v1/klines?symbol={}&interval={}&limit={}',
            'query_params' : ['symbol', 'interval', 'limit'],
            'limit' : 1500,
            'interval_enum' : ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '1w', '1M']
        },
        'okx' : {
            'url' : 'https://www.okx.com/api/v5/market/mark-price-candles?instId={}&bar={}&limit={}',
            'query_params' : ['instId', 'bar', 'limit'],
            'limit' : 100,
            'interval_enum' : ['1m', '3m', '5m', '15m', '30m', '1H', '2H', '4H', '6H', '12H', '1D', '1W', '1M']
        },
        'bitget' : {
            'url' : 'https://api.bitget.com/api/v2/mix/market/candles?symbol={}&granularity={}&limit={}&productType=usdt-futures',
            'query_params' : ['symbol', 'granularity', 'limit', 'productType'],
            'limit' : 1000,
            'interval_enum' : ['1m', '3m', '5m', '15m', '30m', '1H', '2H', '4H', '6H', '12H', '1D', '1W', '1M']
        },
        'gateio' : {
            'url' : 'https://api.gateio.ws/api/v4/futures/usdt/candlesticks?contract={}&interval={}&limit={}',
            'query_params' : ['symbol', 'interval', 'limit'],
            'limit' : 2000,
            'interval_enum' : ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '1w']
        }
    }

    spec = exchanges[exchange]

    import aiohttp

    async def aiocurl(url):
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    return await response.text()
                else:
                    print(f'Error: {response.status}')
                    return None
    
    query_string = spec['url']
    args = []
    interval = str(interval).lower()

    if exchange in ['upbit', 'bithumb', 'coinone', 'korbit']:
        # print(f'exchange: {exchange} is Korean Exchange')

        if exchange in ['upbit', 'bithumb']:
            if interval[-1] == 'h':
                args.append('minutes')
                interval = int(interval[:-1]) * 60

                if interval not in spec['interval_enum']:
                    raise ValueError

                args.append('/' + str(interval))

            elif interval[-1] == 'm':
                args.append('minutes')

                if int(interval[:-1]) not in spec['interval_enum']:
                    raise ValueError

                args.append('/' + interval[:-1])

            elif interval[-1] == 'd':
                args.append('days')
                args.append('')

            elif interval[-1] == 'w':  
                args.append('weeks')
                args.append('')
            
            symbol = 'KRW-' + symbol
            args.append(symbol)
        
        elif exchange == 'korbit':
            if interval[-1] == 'h':
                interval = int(interval[:-1]) * 60

                if interval not in spec['interval_enum']:
                    raise ValueError

            elif interval[-1] == 'm':
                interval = int(interval[:-1])

                if interval not in spec['interval_enum']:
                    raise ValueError
                
            elif interval[-1] == 'd':
                if interval != '1d':
                    raise ValueError
                interval = '1D'
                
            elif interval[-1] == 'w':
                if interval != '1w':
                    raise ValueError
                interval = '1W'

            symbol = symbol.lower() + '_krw'
            args.append(symbol)
            args.append(interval)

        elif exchange == 'coinone':
            if interval not in spec['interval_enum']:
                raise ValueError

            args.append(symbol)
            args.append(interval)

    elif exchange in ['bybit', 'binance', 'okx', 'bitget', 'gateio']:
        # print(f'exchange: {exchange} is Global Exchange')

        if exchange == 'binance':
            if interval not in spec['interval_enum']:
                raise ValueError

            symbol += 'USDT'

        if exchange == 'bybit':
            if interval[-1] in ['d', 'm', 'w']:
                interval = interval[-1].upper()
            elif interval[-1] == 'h':
                interval = int(interval[:-1]) * 60
            elif interval[-1] == 'm':
                interval = int(interval[:-1])

            if interval not in spec['interval_enum']:
                raise ValueError
            
            symbol += 'USDT'
            
        if exchange == 'bitget':
            if interval[-1] in ['d', 'w', 'h']:
                interval = interval.upper()
            
            if interval not in spec['interval_enum']:
                raise ValueError
            
            symbol += 'USDT'
            
        if exchange == 'okx':
            if interval[-1] in ['d', 'w', 'h']:
                interval = interval.upper()

            if interval not in spec['interval_enum']:
                raise ValueError
            
            symbol += '-USDT-SWAP'

        if exchange == 'gateio':
            if interval not in spec['interval_enum']:
                raise ValueError
            
            symbol += '_USDT'

        args.append(symbol)
        args.append(interval)

    args.append(spec['limit'])

    query_string = spec['url'].format(*args)
    # print(query_string)

    result = await aiocurl(query_string)
    result = json.loads(result)

    # if isinstance(result, dict):
    #     print(len(result))
    #     print(result.keys())
    # elif isinstance(result, list):
    #     print(len(result))

    if exchange == 'coinone':
        result = result['chart']
    elif exchange == 'bybit':
        result = result['result']['list']
    elif exchange in ['bitget', 'okx', 'korbit']:
        result = result['data']

    # print(json.dumps(result, indent=4))
    return result

async def get_kimp_kline_data(symbol:str, ex1:str, ex2:str, interval:str, size:int):
    '''
      4h interval일 때, 거래소마다 기준이 다르다.
        upbit : candle_date_time_kst를 datetime 칼럼으로 사용.(UTC 00:00을 기준으로 4h 데이터를 제공)
        bithumb : candle_date_time_kst를 datetime 칼럼으로 사용.(UTC KST 00:00을 기준으로 4h 데이터를 제공). 따라서, upbit와 1시간 차이가 난다.
        coinone : timestamp(UTC) -> datetime(UTC+9) 변환한 칼럼 사용. (UTC 00:00을 기준으로 4h 데이터를 제공)
        korbit : timestamp(UTC) -> datetime(UTC+9) 변환한 칼럼 사용. (UTC KST 00:00을 기준으로 4h 데이터를 제공)
        해외거래소가 UTC 기준이므로, 국제표준에 맞게 bithumb, korbit datetime(UTC KST) - pd.Timedelta(hours=3)을 해준다.
    '''
    krw_task = get_kline_data(ex1, symbol, interval, size)
    usdt_task = get_kline_data(ex2, symbol, interval, size)
    krw_raw_data, usdt_raw_data = await asyncio.gather(krw_task, usdt_task)

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

    cols_to_convert = ['datetime', 'open', 'high', 'low', 'close', 'target_volume', 'quote_volume']

    krw_df = pd.DataFrame(krw_raw_data)

    if ex1 == 'upbit':
        krw_df.rename(columns={'candle_date_time_kst' : 'datetime'}, inplace=True)
        krw_df.datetime = pd.to_datetime(krw_df.datetime)
        krw_df.datetime = krw_df.datetime.dt.tz_localize('Asia/Seoul')

    elif ex1 == 'bithumb':
        krw_df.candle_date_time_kst = pd.to_datetime(krw_df.candle_date_time_kst)

        if interval == '4h':
            krw_df.candle_date_time_kst = krw_df.candle_date_time_kst - pd.Timedelta(hours=3)

        krw_df.rename(columns={'candle_date_time_kst' : 'datetime'}, inplace=True)
        krw_df.datetime = krw_df.datetime.dt.tz_localize('Asia/Seoul')
    
    elif ex1 == 'coinone':
        krw_df['datetime'] = pd.to_datetime(krw_df.timestamp, unit='ms')
        krw_df.datetime = krw_df.datetime.dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul')

    elif ex1 == 'korbit':
        krw_df['datetime'] = pd.to_datetime(krw_df.timestamp, unit='ms')
        krw_df.datetime = krw_df.datetime.dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul')
        krw_df.datetime = krw_df.datetime - pd.Timedelta(hours=3)
        krw_df.sort_values('datetime', inplace=True, ascending=False)

    krw_df[ex_col_map[ex1]] = krw_df[ex_col_map[ex1]].apply(pd.to_numeric)
    krw_df = krw_df[['datetime'] + ex_col_map[ex1]]

    if ex1 == 'korbit':
        cols_to_convert = ['datetime', 'open', 'high', 'low', 'close', 'volume']
    krw_df.columns = cols_to_convert

    # print(krw_df.loc[krw_df.datetime.dt.date == pd.to_datetime('2024-10-11').date()])
    usdt_df = pd.DataFrame(usdt_raw_data, columns=ex_col_map[ex2])

    if ex2 == 'gateio':
        usdt_df.rename(columns={'t' : 'timestamp', 'o' : 'open', 'h' : 'high', 'l' : 'low', 'c' : 'close', 'v' : 'target_volume', 'sum' : 'quote_volume'}, inplace=True)
    
    filtered_cols = ['timestamp', 'open', 'high', 'low', 'close', 'target_volume', 'quote_volume']

    if ex2 == 'okx':
        filtered_cols = ['timestamp', 'open', 'high', 'low', 'close']

    usdt_df = usdt_df[filtered_cols]

    if ex2 == 'gateio':
        usdt_df['datetime'] = pd.to_datetime(usdt_df.timestamp, unit='s')
    else:
        usdt_df['datetime'] = pd.to_datetime(usdt_df.timestamp, unit='ms')

    usdt_df.datetime = usdt_df.datetime.dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul')
    usdt_df[filtered_cols] = usdt_df[filtered_cols].apply(pd.to_numeric)

    # print(usdt_df.loc[usdt_df.datetime.dt.date == pd.to_datetime('2024-10-11').date(), ['datetime', 'open', 'high', 'low', 'close']]) 

    merged_df = pd.merge(krw_df, usdt_df, on='datetime', how='inner')
    merged_df['kimp'] = merged_df.close_x / merged_df.close_y
    # merged_df.datetime = merged_df.datetime.dt.tz_localize('UTC').dt.tz_convert('Asia/Seoul')
    lower_quantile = merged_df.kimp.quantile(0.25)
    higher_quantile = merged_df.kimp.quantile(0.75)

    # print(merged_df.loc[merged_df.datetime.dt.date == pd.to_datetime('2024-10-21').date(), ['datetime', 'close_x', 'close_y', 'kimp']])

    import seaborn as sns
    import matplotlib.pyplot as plt 

    fig, ax = plt.subplots(figsize=(10,6))
    p = sns.lineplot(x='datetime', y='kimp', data=merged_df, color='b')

    return {
        'symbol' : symbol,
        'data' : merged_df[['datetime', 'close_x', 'close_y', 'kimp']],
        'lower_quantile' : lower_quantile,
        'higher_quantile' : higher_quantile
    }

async def fetch_orderbook(exchange, symbol, quote, future=False):
    target = symbol + '/' + quote
    if future:
        target += ':' + quote
    try:
        return await exchange.fetch_order_book(target)
    except Exception as e:
        print(f"Error fetching orderbook for {target} on {exchange.id}: {e}")
        return None

def calculate_market_order_amount(orderbook, seed, is_buy, fee_rate):
    price_level = orderbook['asks'] if is_buy else orderbook['bids']

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
                        break
            elif len(price_level[0]) == 3:
                for price, volume, _ in price_level:
                    if remaining_money >= price * volume:
                        total_bought_coin += volume
                        remaining_money -= price * volume
                    else:
                        total_bought_coin += remaining_money / price
                        break
            else:
                raise ValueError
        except Exception as e:
            print(price_level)
            raise e

        return total_bought_coin, total_fee

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
                        break
            elif len(price_level[0]) == 3:
                for price, volume, _ in price_level:
                    if remaining_coins >= volume:
                        total_revenue += price * volume
                        remaining_coins -= volume
                    else:
                        total_revenue += price * remaining_coins
                        break
            else:
                raise ValueError
        except Exception as e:
            print(price_level)
            raise e

        # Reduce revenue by the selling fee
        return total_revenue * (1 - fee_rate), total_revenue * fee_rate

async def find_real_kimp(exchange1:str, exchange2:str, symbol:str, seed=1_000_000):
    exchange_class = getattr(ccxt, exchange1)
    ex1 = exchange_class()

    exchange_class = getattr(ccxt, exchange2)
    ex2 = exchange_class()

    results = await asyncio.gather(fetch_orderbook(ex1, symbol, 'KRW'), fetch_orderbook(ex2, symbol, 'USDT', True))
    buy_amt, buy_fee = calculate_market_order_amount(results[0], seed, is_buy=True, fee_rate=fee_rate[exchange1])
    usdt_revenue, sell_fee = calculate_market_order_amount(results[1], buy_amt, is_buy=False, fee_rate=fee_rate[exchange2])

    ret = {
        'buy_amt' : buy_amt,
        'buy_fee' : str(buy_fee) + 'KRW',
        'usdt_revenue' : usdt_revenue,
        'sell_fee' : str(round(sell_fee,2)) + 'USDT',
        'real_kimp' : round(seed / usdt_revenue, 4)
    }

    await asyncio.gather(ex1.close(), ex2.close())

    # print(json.dumps(ret, indent=4))
    return ret

from dotenv import load_dotenv
import boto3
import json

load_dotenv('dev.env')

session = boto3.Session()
credentials = session.get_credentials()

# Printing AWS credentials (useful for debugging)
print(f'Access Key : {credentials.access_key}')
print(f'Secret Key : {credentials.secret_key}')
print(f'Token : {credentials.token}')

# Creating SQS client
sqs = boto3.client('sqs', region_name='ap-northeast-2')

# Queue URL
queue_url = 'https://sqs.ap-northeast-2.amazonaws.com/905418113774/TaskQueue'

async def send_jobs_to_sqs(exchanges):
    while True:
        start_time = time.time()

        df:pd.DataFrame = await extract_lowest_kimp_ex_set_for_all_tickers(exchanges)

        print(df.shape)

        rows = df.iterrows()

        messages = []

        for idx, (_, r) in enumerate(rows):
            # message for find real kimp
            message = {
                'Id' : str(idx),
                'MessageBody' : json.dumps({
                    'symbol' : f'{r.symbol}',
                    'exchange_x' : f'{r.exchange_x}',
                    'exchange_y' : f'{r.exchange_y}'
                })
            }
            messages.append(message)

            if len(messages) == 10:
                sqs.send_message_batch(QueueUrl=queue_url, Entries=messages)
                messages = []

        elapsed_time = time.time() - start_time

        # 5 minutes throttle
        if elapsed_time < 300:
            print(f'elapsed_time : {elapsed_time}')
            await asyncio.sleep(300-elapsed_time)
            

if __name__ == '__main__':
    exchanges = ['upbit', 'bithumb', 'coinone', 'binance', 'bybit', 'bitget', 'okx', 'gateio']
    asyncio.run(send_jobs_to_sqs(exchanges))

    # asyncio.run(get_kimp_kline_data('AERGO', 'upbit', 'bybit', '4h', 200))
