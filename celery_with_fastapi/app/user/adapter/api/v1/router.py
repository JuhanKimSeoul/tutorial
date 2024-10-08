from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends, Query

from app.container import Container
from app.user.adapter.api.v1.request import CreateUserRequest
from app.user.adapter.api.v1.response import CreateUserResponse
from app.user.domain.usecase.user_usecase import UserUseCase

user_router = APIRouter()

@user_router.post(
    "",
    response_model=CreateUserResponse,
)
@inject
async def create_user(
    request: CreateUserRequest,
    usecase: UserUseCase = Depends(Provide[Container.user_service]),
):
    await usecase.create_user(**request.model_dump())
    return {"email": request.email, "messages": "Welcome to the club!"}