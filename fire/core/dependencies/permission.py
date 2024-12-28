from abc import ABC, abstractmethod
from typing import List, Type
from fastapi import Request
from fastapi.security.base import SecurityBase
from fastapi.openapi.models import APIKey, APIKeyIn
from core.exceptions.base import CustomException, UnauthorizedException

class BasePermission(ABC):
    exception = CustomException

    @abstractmethod
    async def has_permission(self, request: Request)->bool:
        pass

class AuthenticatedUserOnly(BasePermission):
    exception = UnauthorizedException

    async def has_permission(self, request: Request)->bool:
        try:
            return getattr(request.user, 'email') is not None
        except AttributeError as e:
            raise self.exception

class PermissionDependency(SecurityBase):
    def __init__(self, permissions: List[Type[BasePermission]]):
        self.permissions = permissions
        self.model:APIKey = APIKey(**{"in": APIKeyIn.header}, name="Authorization")
        self.scheme_name = self.__class__.__name__

    async def __call__(self, request: Request):
        for permission in self.permissions:
            cls = permission()
            if not await cls.has_permission(request=request):
                raise cls.exception