from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column, composite

from core.db.session import Base
from core.db.mixins import TimestampMixin

from enums.constant import Grade

class User(Base, TimestampMixin):
    __tablename__ = 'user'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    password: Mapped[str] = mapped_column(String(255), nullable=False)
    nickname: Mapped[str] = mapped_column(String(100), nullable=False, unique=True)
    phone: Mapped[str] = mapped_column(String(20), nullable=False, unique=True)
    grade: Mapped[str] = mapped_column(String(10), nullable=False, default=Grade.BASIC)

    @classmethod
    def create(
        cls, *, email: str, password: str, nickname: str, phone: str
    )->'User':
        return cls(
            email=email,
            password=password,
            nickname=nickname,
            phone=phone
        )
        
