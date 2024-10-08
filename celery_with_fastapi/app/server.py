from fastapi import FastAPI
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
import os
from app.user.adapter.api.v1.router import user_router
from app.container import Container

def init_routers(app: FastAPI)->None:
    container = Container()
    user_router.container = container
    app.include_router(user_router, prefix="/api/v1/user")

def init_middlewares(app: FastAPI)->None:
    middlewares = [
        Middleware(
            CORSMiddleware,
            allow_origins=["http://localhost", "http://127.0.0.1"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        ),
        Middleware(Sq)
    ]

    return middlewares

def init_exception_handlers(app: FastAPI)->None:
    pass

def init_db(app: FastAPI)->None:
    pass

def create_app()->FastAPI:
    app_ = FastAPI(
        title="FastAPI with Celery",
        description="This is a simple example of FastAPI with Celery.",
        version="0.1.0",
        docs_url=None if os.getenv("ENV") == "production" else "/docs",
        redoc_url=None if os.getenv("ENV") == "production" else "/redoc",
        dependencies=[],
        middleware=[],
    )
    init_routers(app_)
    init_middlewares(app_)
    init_exception_handlers(app_)
    init_db(app_)
    return app_

app = create_app()
