from functools import wraps
import logging

from core.db.session import session
import traceback

logger = logging.getLogger(__name__)

class Transactional:
    def __call__(self, func):
        @wraps(func)
        async def _transactional(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)
                await session.commit()
            except Exception as e:
                logger.info(f'{traceback.format_exc()}')
                await session.rollback()
                raise e
            
            return result
        
        return _transactional