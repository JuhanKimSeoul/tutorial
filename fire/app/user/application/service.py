import logging
from uuid import uuid4
from app.user.application.exception import NotFoundUserException, PasswordLengthUnsatisfiedException, PasswordNotEqualsException
from app.user.domain.entity import User
from app.user.persistence.repository_adapter import UserRepoAdapter
from app.user.domain.usecase import UserUseCase
from core.db.transactional import Transactional
from app.user.domain.schemas import UserSignUp, UserLogin, UserLoggedIn, UserBase
from helpers.helper import apply_timer
from passlib.context import CryptContext

from helpers.jwt import TokenHelper

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@apply_timer
class UserService(UserUseCase):
    def __init__(self, repository:UserRepoAdapter):
        self.repository = repository

    @Transactional()
    async def sign_up(
        self,
        *,
        user:UserSignUp
    )->UserBase:
        if user.password1 != user.password2:
            raise PasswordNotEqualsException
        
        if len(user.password1) < 10:
            raise PasswordLengthUnsatisfiedException
        
        hashed_password = pwd_context.hash(user.email + user.password1)
        
        user = UserBase(**{
            'email': user.email,
            'password': hashed_password,
            'nickname': user.nickname,
            'phone': user.phone,
            'grade': user.grade
        })

        await self.repository.save(user=user)

        return user

    async def login(
        self,
        *,
        user:UserLogin
    )->UserLoggedIn | None:
        target_user:User|None = await self.repository.get_user_by_email_and_password(
            email_or_nickname=user.email,
            password=user.password
        )

        if target_user is None:
            raise NotFoundUserException

        access_token = TokenHelper.encode({'email': target_user.email, 'grade': target_user.grade}, expire_period=900)
        refresh_token = TokenHelper.encode({'uuid': str(uuid4()), 'email': target_user.email}, expire_period=3600*24)

        return UserLoggedIn(**{
            'email': target_user.email,
            'nickname': target_user.nickname,
            'grade': target_user.grade,
            'access_token': access_token,
            'refresh_token': refresh_token
        })