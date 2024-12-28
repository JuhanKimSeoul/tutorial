from starlette import status

class CustomException(Exception):
    pass

class UnauthorizedException(CustomException):
    code = status.HTTP_401_UNAUTHORIZED
    error_code = "UNAUTHORIZED_ERR"
    message = "unauthorized access"