from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column
from core.db import Base
from core.db.mixins.timestamp_mixin import TimestampMixin

class User(Base, TimestampMixin):
    __tablename__ = 'users'

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(255), nullable=False)
    password: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)

    @classmethod
    def from_dict(cls, data: dict) -> 'User':
        return cls(
            email=data['email'],
            password=data['password']
        )

class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int = Field(..., title='User ID')
    email: str = Field(..., title='User Email')