from typing import List
from app.user.domain.entity import User, UserRead
from app.user.domain.repository import UserRepository

class UserRepositoryAdapter:
    def __init__(self, repository: UserRepository):
        self.repository = repository

    async def get_users(
        self,
        *,
        limit: int = 10,
        prev: int | None = None
    ) -> List[User]:
        users = await self.repository.get_users(limit=limit, prev=prev)
        return [UserRead.model_validate(user) for user in users]
    
    async def get_user_by_id(
        self,
        *,
        user_id: int
    ) -> User | None:
        user = await self.repository.get_user_by_id(user_id=user_id)
        if user:
            return UserRead.model_validate(user)
        return None
    
    async def get_user_by_email(
        self,
        *,
        email: str
    ) -> User | None:
        user = await self.repository.get_user_by_email(email=email)
        if user:
            return UserRead.model_validate(user)
        return None
    
    async def get_user_by_email_and_password(
        self,
        *,
        email: str,
        password: str
    ) -> User | None:
        user = await self.repository.get_user_by_email_and_password(email=email, password=password)
        if user:
            return UserRead.model_validate(user)
        return None
    
    async def create_user(
        self,
        *,
        user: User
    ) -> User:
        await self.repository.create_user(user=user)
        
    