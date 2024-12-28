from abc import ABC, abstractmethod
from app.user.domain.entity import User
from app.user.domain.schemas import UserBase, UserUpdate
from typing import List, Optional

class UserRepo(ABC):
    @abstractmethod
    async def get_users(
        self,
        *,
        limit:int = 100,
        prev:Optional[int] = None
    )->List[User]:
        pass

    @abstractmethod
    async def get_user_by_id(
        self, 
        *,
        user_id:int
    )->Optional[User]:
        pass

    @abstractmethod
    async def get_user_by_email_or_nickname(
        self,
        *,
        email:str,
        nickname:str
    )->Optional[User]:
        pass

    @abstractmethod
    async def get_user_by_email_and_password(
        self,
        *,
        email_or_nickname:str,
        password:str
    )->Optional[User]:
        pass

    @abstractmethod
    async def save(
        self,
        *,
        user: UserBase
    )->User:
        pass

    @abstractmethod
    async def update(
        self,
        *,
        user: UserUpdate
    )->User:
        pass