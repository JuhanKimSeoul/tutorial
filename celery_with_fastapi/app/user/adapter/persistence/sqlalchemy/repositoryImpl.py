from sqlalchemy import select

from app.user.domain.entity.user import User
from app.user.domain.repository.user_repository import UserRepository
from core.db.session import session, session_factory

class UserRepositoryImpl(UserRepository):
    async def get_users(
        self,
        *,
        limit: int = 10,
        prev: int | None = None
    ) -> list[User]:
        async with session_factory() as read_session:
            query = select(User).limit(limit)
            if prev:
                result = await read_session.execute(query)
            return result.scalars().all()

    async def get_user_by_id(
        self,
        *,
        user_id: int
    ) -> User | None:
        async with session_factory() as read_session:
            query = select(User).where(User.id == user_id)
            result = await read_session.execute(query)
            return result.scalars().first()
        
    async def get_user_by_email(
        self,
        *,
        email: str
    ) -> User | None:
        async with session_factory() as read_session:
            query = select(User).where(User.email == email)
            result = await read_session.execute(query)
            return result.scalars().first()
        
    async def get_user_by_email_and_password(
        self,
        *,
        email: str,
        password: str
    ) -> User | None:
        async with session_factory() as read_session:
            query = select(User).where(User.email == email, User.password == password)
            result = await read_session.execute(query)
            return result.scalars().first()

    async def create_user(
        self,
        *,
        user: User
    ) -> User:
        session.add(user)