import jwt
from pydantic import BaseModel, Field
from starlette.authentication import AuthenticationBackend
from starlette.middleware.authentication import AuthenticationMiddleware as BaseAuthenticationMiddleware
from starlette.requests import HTTPConnection
from config import settings

class CurrentUser(BaseModel):
    email: int = Field(None, description="email")

class AuthBackend(AuthenticationBackend):
    async def authenticate(
        self, conn: HTTPConnection
    )->tuple[bool, CurrentUser | None]:
        current_user = CurrentUser()
        authorization:str = conn.headers.get("Authorization")
        if not authorization:
            return False, current_user
        
        try:
            scheme, credentials = authorization.split(' ')
            if scheme.lower() != "bearer":
                return False, current_user
            if not credentials:
                return False, current_user
        except ValueError:
            return False, current_user
        
        try:
            payload = jwt.decode(
                credentials,
                settings['jwt_secret_key'],
                settings['jwt_algorithm']
            )
            email = payload.get("email")
        except jwt.exceptions.PyJWTError:
            return False, current_user
        
        current_user.email = email
        return True, current_user

class AuthenticationMiddleware(BaseAuthenticationMiddleware):
    pass