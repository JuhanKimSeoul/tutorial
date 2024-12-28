import functools
import time
from typing import Dict
import logging

def deep_extend(dictA:Dict, dictB:Dict):
    for key, value in dictB.items():
        if key in dictA:
            if isinstance(dictA[key], dict) and isinstance(value, dict):
                deep_extend(dictA[key], value)
            else:
                dictA[key] = value
        else:
            dictA[key] = value
    return dictA

def timer(func):
    logger = logging.getLogger(__name__)

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            logger.info(f"{func.__module__}.{func.__name__} with args: {args}, kwargs: {kwargs}")
            result = await func(*args, **kwargs)
            logger.info(f"{func.__module__}.{func.__name__} return: {result}")
            return result
        except Exception as e:
            raise e
        finally:
            elapsed_time = time.time() - start_time
            logger.info(f'{func.__module__}.{func.__name__} took {elapsed_time :.4f} seconds')

    return wrapper

def apply_timer(cls):
    for attr_name in dir(cls):
        attr = getattr(cls, attr_name)
        if callable(attr) and not attr_name.startswith("__"):
            setattr(cls, attr_name, timer(attr))
    return cls