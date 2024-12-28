from datetime import datetime, timedelta

import jwt

from config import settings
from core.exceptions import CustomException

class DecodeTokenException(CustomException):
    code = 400
    error_code = "TOKEN_DECODE_ERR"
    message = "token decode err"

class ExpiredTokenException(CustomException):
    code = 400
    error_code = "TOKEN_EXPIRED_ERR"
    message = "token expired err"

class TokenHelper:
    @staticmethod
    def encode(payload:dict, expire_period:int=3600)->str:
        token = jwt.encode(
            payload={
                **payload,
                'exp':datetime.now() + timedelta(seconds=expire_period)
            },
            key=settings['jwt_secret_key'],
            algorithm=settings['jwt_algorithm']
        )
        return token
        
    @staticmethod
    def decode(token:str)->dict:
        try:
            return jwt.decode(
                token, 
                settings['jwt_secret_key'],
                settings['jwt_algorithm']
            )
        except DecodeTokenException as e:
            raise DecodeTokenException
        except ExpiredTokenException as e:
            raise ExpiredTokenException
        except Exception as e:
            raise e
    
    