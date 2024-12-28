from app.user.domain.entity import User
from app.user.domain.repository import UserRepo
from app.user.domain.schemas import UserUpdate, UserBase

class UserRepoAdapter:
    def __init__(
        self, 
        *,
        user_repo:UserRepo
    ):
        self.user_repo = user_repo

    async def get_user_by_email_and_password(
        self, 
        *,
        email_or_nickname:str,
        password:str
    )->User | None:
        return await self.user_repo.get_user_by_email_and_password(
            email_or_nickname=email_or_nickname,
            password=password
        )
    
    async def get_user_by_email_or_nickname(
        self,
        *,
        email:str,
        nickname:str
    )->User | None:
        return await self.user_repo.get_user_by_email_or_nickname(
            email=email,
            nickname=nickname
        )
        
    async def save(
        self,
        *,
        user:UserBase
    )->UserBase:
        return await self.user_repo.save(
            user=user
        )

    async def update(
        self,
        *,
        user:UserUpdate
    )->User:
        return await self.user_repo.update(
            user=user
        )