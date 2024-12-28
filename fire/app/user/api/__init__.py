from fastapi import APIRouter
from app.user.api.v1.router import user_router

router = APIRouter()
router.include_router(user_router, prefix='/api/v1/user', tags=['User'])

__all__ = ['router']