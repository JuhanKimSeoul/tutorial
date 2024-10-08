import sys
import time
import asyncio
import traceback
import functools

def async_timer(func):
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = await func(*args, **kwargs)
            print(f'{func.__name__} executed in {time.time() - start} seconds')
            return result
        except Exception as e:
            print(f'{func.__name__} failed in {time.time() - start} seconds')
            traceback.print_exc()
            print(e)
    return wrapper

def sync_timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            print(f'{func.__name__} executed in {time.time() - start} seconds')
            return result
        except Exception as e:
            print(f'{func.__name__} failed in {time.time() - start} seconds')
            traceback.print_exc()
            print(e)
    return wrapper

def async_retrier(retries=3):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            for i in range(retries):
                try:
                    await func(*args, **kwargs)
                except Exception as e:
                    print(f'Retrying {func.__name__} {i+1} time')
                    traceback.print_exc()
                    print(e)
        return wrapper
    return decorator

def sync_retrier(retries=3):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(retries):
                try:
                    func(*args, **kwargs)
                except Exception as e:
                    print(f'Retrying {func.__name__} {i+1} time')
                    traceback.print_exc()
                    print(e)
        return wrapper
    return decorator

@async_retrier(3)
@async_timer
async def fetch_order_book_async():
    import ccxt.async_support as ccxt
    try:
        exchange_class = getattr(ccxt, 'binance')
        exchange = exchange_class({
            'enableRateLimit': False
        })

        symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
        tasks = [exchange.fetch_order_book(symbol=symbol, limit=1000) for symbol in symbols]
        results = await asyncio.gather(*tasks)
        
        for i, result in enumerate(results):
            symbol = symbols[i]
            symbol = symbols[i]
            result_size = sys.getsizeof(result)
            print(f"Symbol: {symbol}, Result Size: {result_size} bytes")

        return results
    except Exception as e:
        raise e
    finally:
        await exchange.close()  

asyncio.run(fetch_order_book_async())

@sync_retrier(3)
@sync_timer
def fetch_order_book_sync():
    import ccxt
    try:
        exchange = ccxt.binance({
            'enableRateLimit': False,
        })

        symbols = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
        return [exchange.fetch_order_book(symbol=symbol, limit=1000) for symbol in symbols]
    except Exception as e:
        raise e

# fetch_order_book_sync()