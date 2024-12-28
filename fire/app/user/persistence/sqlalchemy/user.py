from sqlalchemy import and_, or_, select
from sqlalchemy.exc import SQLAlchemyError
from app.user.application.exception import AlreadySignedUpException, PasswordWrongException
from app.user.domain.entity import User
from app.user.domain.schemas import UserBase, UserUpdate
from app.user.domain.repository import UserRepo
from core.db.session import session, session_factory
from typing import List
from passlib.context import CryptContext
import logging
from helpers.helper import apply_timer

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

logger = logging.getLogger(__name__)

@apply_timer
class UserSqlalchemyRepo(UserRepo):
    async def get_users(
        self,
        *,
        limit:int = 100,
        prev:int | None = None
    )->List[User]:
        query = select(User)

        if prev:
            query = query.where(User.id < prev)

        if limit > 100:
            limit = 100

        query = query.limit(limit)

        async with session_factory() as read_session:
            result = await read_session.execute(query)
        
        return result.scalars().all()
    
    async def get_user_by_id(
        self,
        *,
        user_id:int
    )->User | None:
        async with session_factory() as read_session:
            result = await read_session.execute(select(User).where(User.id == user_id))
            
        return result.scalars().first()
    
    async def get_user_by_email_or_nickname(
        self,
        *,
        email:str,
        nickname:str
    )->User | None:
        async with session_factory() as read_session:
            result = await read_session.execute(
                select(User).where(or_(User.email == email, User.nickname == nickname))
            )
            
        return result.scalars().first()
    
    async def get_user_by_email_and_password(
        self,
        *,
        email_or_nickname:str,
        password:str
    )->User | None:
        try:
            async with session_factory() as read_session:
                result = await read_session.execute(
                    select(User).where(
                        or_(
                            User.email == email_or_nickname, 
                            User.nickname == email_or_nickname
                        )
                    )
                )
                user = result.scalar_one_or_none()
            
            if user:
                input_password = user.email + password

                if pwd_context.verify(input_password, user.password):
                    return user
                else:
                    raise PasswordWrongException
        
        except PasswordWrongException as e:
            raise e
        except SQLAlchemyError as e:
            raise e
        except Exception as e:
            raise e
    
    async def save(
        self,
        *,
        user:UserBase
    )->UserBase:
        try:
            existing_user = await self.get_user_by_email_or_nickname(
                email=user.email,
                nickname=user.nickname
            )

            if existing_user:
                raise AlreadySignedUpException
            
            new_user = User(**user.model_dump())

            session.add(new_user)

            return user
        
        except AlreadySignedUpException as e:
            raise e
        except SQLAlchemyError as e:
            raise e
        except Exception as e:
            raise e

    async def update(
        self,
        *,
        user:UserUpdate
    )->User:
        existing_user = await self.get_user_by_email_or_nickname(
            email=user.email,
            nickname=user.nickname
        )
        
        existing_user = existing_user.scalar_one_or_none()

        if existing_user:
            existing_user.email = user.email
            existing_user.password = user.password

            if user.password:
                existing_user.password = pwd_context.hash(user.password, user.email)
            
            return existing_user # already tracked by session because the object is fetched by session

        raise ValueError('todo')