from contextlib import asynccontextmanager
from contextvars import ContextVar, Token
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_scoped_session,
    async_sessionmaker,
    create_async_engine,
)


from sqlalchemy.orm import DeclarativeBase, Session
from sqlalchemy.sql import Delete, Insert, Update

from celery_with_fastapi.config import get_config
from celery_with_fastapi.enums.constant import EngineType
config = get_config()

class Base(DeclarativeBase):
    pass

session_context: ContextVar[str] = ContextVar('session_context')

def get_session_context() -> str:
    return session_context.get()

def set_session_context(value: str) -> Token:
    return session_context.set(value)

def reset_session_context(token: Token) -> None:
    session_context.reset(token)

engines = {
    EngineType.WRITER : create_async_engine(config.WRITER_DB_URL, pool_recycle=3600),
    EngineType.READER : create_async_engine(config.READER_DB_URL, pool_recycle=3600)
}

class RoutingSession(Session):
    def get_bind(self, mapper=None, clause=None, **):
        if self._flushing or isinstance(clause, (Insert, Update, Delete)):
            return engines[EngineType.WRITER].sync_engine 
        else:
            return engines[EngineType.READER].sync_engine

_async_session_factory = async_sessionmaker(
    class_ = AsyncSession,
    sync_session_class = RoutingSession(),
    expire_on_commit=False,
)

session = async_scoped_session(
    _async_session_factory,
    scopefunc=get_session_context
)

@asynccontextmanager
async def session_factory() -> AsyncGenerator[AsyncSession, None]:
    _session = async_sessionmaker(
        class_ = AsyncSession,
        sync_session_class = RoutingSession(),
        expire_on_commit=False
    )()

    try:
        yield _session
    finally:
        await _session.close()