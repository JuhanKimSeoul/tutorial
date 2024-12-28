from dependency_injector.wiring import Provide, inject
from fastapi import APIRouter, Depends
from typing import Dict

from app.arbitrage.domain.usecase import ArbitrageUseCase
from app.container import Container
from core.dependencies.permission import AuthenticatedUserOnly, PermissionDependency

arbitrage_router = APIRouter()

@arbitrage_router.get(
    "/kimchi_premium"
)
@inject
async def fetch_kimchi_premium(
    current_user: Dict = Depends(PermissionDependency([AuthenticatedUserOnly])),
    usecase: ArbitrageUseCase = Depends(Provide[Container.arbitrage_service])
):
    result = await usecase.get_kimchi_premium()
    return {"status" : "ok"}