from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends

from app.container import Container
from app.user.api.v1.request_dto import LoginRequestDto, SignUpRequestDto
from app.user.api.v1.response_dto import LoginReponseDto, SignUpResponseDto

from app.user.domain.usecase import UserUseCase
from app.user.domain.schemas import UserSignUp, UserLogin

user_router = APIRouter()

@user_router.post(
    "/signup",
    response_model=SignUpResponseDto
)
@inject
async def create_user(
    request: SignUpRequestDto,
    usecase: UserUseCase = Depends(Provide[Container.user_service])
):
    user = UserSignUp(**request.model_dump())
    result = await usecase.sign_up(user=user)
    return SignUpResponseDto(**result.model_dump())

@user_router.post(
    "/login",
    response_model=LoginReponseDto
)
@inject
async def login_user(
    request: LoginRequestDto,
    usecase: UserUseCase = Depends(Provide[Container.user_service])
):
    user = UserLogin(**request.model_dump())
    result = await usecase.login(user=user)
    return LoginReponseDto(**result.model_dump())