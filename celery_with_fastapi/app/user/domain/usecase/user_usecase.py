from abc import ABC, abstractmethod
from celery_with_fastapi.app.user.adapter.api.v1.request import CreateUserRequest, LoginRequest
from celery_with_fastapi.app.user.domain.entity.user import User
from celery_with_fastapi.app.user.application.dto import LoginResponseDTO

class UserUseCase(ABC):
    @abstractmethod
    async def get_users(
        self,
        *,
        limit: int = 10,
        prev: int | None = None
    ) -> list[User]:
        pass

    @abstractmethod
    async def create_user(
        self,
        *,
        input : CreateUserRequest
    ) -> User:
        pass

    @abstractmethod
    async def login_user(
        self,
        *,
        input : LoginRequest
    ) -> LoginResponseDTO:
        pass