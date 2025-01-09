from main import *
import asyncio
import traceback
import logging

# 로거 설정
logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

async def monitor_positions():
    EXCLUDE_COINS = ['BTC', 'ETH', 'ARB', 'TOKAMAK', 'HUNT', 'QTUM', 'ETHW', 'ETHF', 'APENFT']
    PNL_THRESHOLD = -0.01  # -1%
    
    t = TradingDataManager('upbit')
    broker = TradingBroker('upbit')

    while True:
        try:
            # 1. Get all positions
            positions = await t.get_all_position()
            positions.pop(0)  # Remove KRW position
            
            logger.debug(f"Positions: {positions}")
            
            # 2. Filter out excluded coins
            targets = [p for p in positions if p['symbol'] not in EXCLUDE_COINS]
            
            if targets:
                # 3. Get current prices for all position coins
                res = await t.ex.get_all_ticker_price()
                current_prices = {item['market'].split('-')[1]: float(item['trade_price']) for item in res if item['market'] in ['KRW-' + p['symbol'] for p in positions]}
                
                logger.debug(f"Current prices: {current_prices}")
                
                # 4. Check PnL and place sell orders
                for target in targets:
                    symbol = target['symbol']
                    avg_price = float(target['avg_buy_price'])
                    current_price = current_prices[symbol]
                    pnl = (current_price - avg_price) / avg_price

                    logger.debug(f"Checking {symbol} - Avg price: {avg_price}, Current price: {current_price}, PnL: {pnl:.2%}")

                    if pnl < PNL_THRESHOLD:
                        # Create market sell order
                        order = PositionEntryIn(
                            symbol=symbol,
                            side='ask',
                            order_type='market',
                            qty=target['size']
                        )
                        
                        # Place sell order
                        await broker.send_order(order)
                        logger.info(f"Sold {symbol} at {current_price}, PnL: {pnl:.2%}")

            await asyncio.sleep(5)  # Wait 5 seconds before next check

        except Exception as e:
            logger.error(f"Error in monitor_positions: {str(e)}")
            logger.error(traceback.format_exc())
            await asyncio.sleep(5)

if __name__ == "__main__":
    try:
        # Start position monitoring
        asyncio.run(monitor_positions())
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        logger.error(traceback.format_exc())

