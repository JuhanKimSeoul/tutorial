from functools import wraps

from core.db import Session

class Transactional:
    def __call__(self, func):
        @wraps(func)
        async def _transactional(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)
                await Session.commit()
            except Exception as e:
                await Session.rollback()
                raise e
            
            return result
        return _transactional