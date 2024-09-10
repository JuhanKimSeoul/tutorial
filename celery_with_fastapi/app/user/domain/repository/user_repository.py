from abc import ABC, abstractmethod
from typing import List
from ..entity.user import User

class UserRepository(ABC):
    @abstractmethod
    def get_users(
        self,
        *,
        limit: int = 10,
        prev: int | None = None
    ) -> List[User]:
        pass

    @abstractmethod
    def get_user_by_id(
        self,
        *,
        user_id: int
    ) -> User | None:
        pass

    @abstractmethod
    def get_user_by_email(
        self,
        *,
        email: str
    ) -> User | None:
        pass

    @abstractmethod
    def get_user_by_email_and_password(
        self,
        *,
        email: str,
        password: str
    ) -> User | None:
        pass

    @abstractmethod
    def create_user(self, *, user: User) -> User:
        pass

    @abstractmethod
    def update_user(self, *, user: User) -> User:
        pass

    @abstractmethod
    def delete_user(self, *, user_id: int) -> None:
        pass