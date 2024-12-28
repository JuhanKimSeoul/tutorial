import logging
from typing import List
from fastapi import FastAPI, Request
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.container import Container
from app.user.api import router as user_router
from app.arbitrage.api import router as arbitrage_router
from core.exceptions.base import CustomException
from core.middlewares.authentication import AuthBackend, AuthenticationMiddleware
from core.middlewares.sqlalchemy import SqlAlchemyMiddleware
from core.db.session import Base, redis_pool, sqlite_engine
from config import get_config
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

def init_routers(app_:FastAPI)->None:
    container = Container()
    container.wire(packages=["app"])
    logger.info("[Complete] dependency injection wiring configuration done")

    user_router.container = container
    arbitrage_router.container = container
    app_.include_router(user_router)
    app_.include_router(arbitrage_router)
    logger.info("[Complete] attach routers")

def init_middleware()->List[Middleware]:
    middlewares = [
        Middleware(
            CORSMiddleware,
            # allow_origins=["http://localhost", "http://127.0.0.1"],
            allow_origins=['*'],
            allow_credentials=True,
            allow_methods=['*'],
            allow_headers=['*']
        ),
        Middleware(SqlAlchemyMiddleware),
        Middleware(
            AuthenticationMiddleware,
            backend=AuthBackend(),
        )
    ]
    return middlewares

def init_exception_handler(app_:FastAPI)->None:
    @app_.exception_handler(CustomException)
    async def custom_exception_handler(request:Request, exc:CustomException):
        return JSONResponse(
            status_code=exc.code,
            content={"error_code": exc.error_code, "message": exc.message}
        )

@asynccontextmanager
async def lifespan(app_:FastAPI):
    cache_pool = redis_pool
    logger.info("[Complete] redis-cache pool assigned")

    async with sqlite_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("[Complete] sqlite3 DB tables configured")

    yield

    await cache_pool.disconnect(inuse_connections=True)
    logger.info("[Complete] redis-cache pool disconnected")
    
    await sqlite_engine.dispose()
    logger.info("[Complete] sqlite3 DB engine disposed")

def create_app()->FastAPI:
    app_ = FastAPI(
        title="Demo",
        description="Demo API",
        version="1.0.0",
        docs_url=None if get_config()['env'] == "prod" else "/docs",
        redoc_url=None if get_config()['env'] == "prod" else "/redoc",
        middleware=init_middleware(),
        lifespan=lifespan
    )
    init_routers(app_)
    init_exception_handler(app_)
    return app_

app = create_app()