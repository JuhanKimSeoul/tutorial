from fastapi import APIRouter
from app.arbitrage.api.v1.router import arbitrage_router

router = APIRouter()
router.include_router(arbitrage_router, prefix='/api/v1/arbitrage', tags=['Arbitrage'])

__all__ = ['router']