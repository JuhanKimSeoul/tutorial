from abc import ABC, abstractmethod
from app.user.domain.schemas import UserBase, UserSignUp, UserLogin, UserLoggedIn

class UserUseCase(ABC):
    @abstractmethod
    async def sign_up(
        self,
        *,
        user: UserSignUp
    )->UserBase:
        pass

    @abstractmethod
    async def login(
        self,
        *,
        user: UserLogin
    )->UserLoggedIn | None:
        pass