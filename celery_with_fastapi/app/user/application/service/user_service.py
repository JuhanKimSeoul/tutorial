from app.user.adapter.api.v1.request import CreateUserRequest, LoginRequest
from app.user.adapter.persistence.repository_adapter import UserRepositoryAdapter
from app.user.application.dto import LoginResponseDTO
from app.user.domain.entity.user import User, UserRead
from app.user.domain.usecase.user_usecase import UserUseCase
from core.db.transactional import Transactional

class UserService(UserUseCase):
    def __init__(self, repository: UserRepositoryAdapter):
        self.repository = repository

    async def get_users(
        self,
        *,
        limit: int = 10,
        prev: int | None = None
    ) -> list[UserRead]:
        return await self.repository.get_users(limit=limit, prev=prev)

    @Transactional
    async def create_user(
        self,
        *,
        input : CreateUserRequest
    ) -> User:
        if input.password != input.confirm_password:
            raise ValueError('Password and confirm password do not match')
        
        is_exist = await self.repository.get_user_by_email(email=input.email)

        if is_exist:
            raise ValueError('User already exists')
        
        user = User(
            email=input.email,
            password=input.password,
        )

        await self.repository.create_user(user=user)

    async def login_user(
        self,
        *,
        input : LoginRequest
    ) -> LoginResponseDTO:
        user = await self.repository.get_user_by_email_and_password(email=input.email, password=input.password)

        if not user:
            return ValueError('User does not exist')

        response = LoginResponseDTO(
            token='token',
            refresh_token='refresh_token',
        )

        return response