from sqlalchemy import Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Department(Base):
    __tablename__ = 'department'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50))
    employees = relationship("Employee", back_populates="department")