import logging

from app.arbitrage.domain.usecase import ArbitrageUseCase
from helpers.helper import apply_timer
from core.db.session import redis_pool
from redis import asyncio as aioredis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

@apply_timer
class ArbitrageService(ArbitrageUseCase):
    async def get_kimchi_premium(self):
        try:
            redis = aioredis.Redis(connection_pool=redis_pool)
            result = await redis.execute_command("get", "binance_bithumb")
            print(result)
            return result

        except RedisError as e:
            raise e
        except Exception as e:
            raise e

        
