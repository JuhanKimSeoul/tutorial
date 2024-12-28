from sqlalchemy.orm import DeclarativeBase, Session
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_scoped_session, 
    async_sessionmaker,
    create_async_engine
)
from contextlib import asynccontextmanager
from contextvars import ContextVar, Token
from typing import AsyncGenerator
from config import get_config
from redis import asyncio as aioredis

class Base(DeclarativeBase):
    ...

session_context: ContextVar[str] = ContextVar("session_context")

def get_session_context()->str:
    return session_context.get()

def set_session_context(session_id:str)->Token:
    return session_context.set(session_id)

def reset_session_context(context:Token)->None:
    session_context.reset(context)

settings = get_config()

sqlite_engine = create_async_engine(settings['user_db_url'])

redis_pool = aioredis.ConnectionPool.from_url(
    settings['redis_url'],
    max_connections=10
)
    
_async_session_factory = async_sessionmaker(
    bind=sqlite_engine,
    class_=AsyncSession,
    expire_on_commit=False
)
session = async_scoped_session(
    session_factory=_async_session_factory,
    scopefunc=get_session_context
)

@asynccontextmanager
async def session_factory()->AsyncGenerator[AsyncSession, None]:
    _session = async_sessionmaker(
        bind=sqlite_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )()
    try:
        yield _session
    finally:
        await _session.close()