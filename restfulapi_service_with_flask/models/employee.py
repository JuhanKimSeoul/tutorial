from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class Employee(Base):
    __tablename__ = 'employee'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50))
    age = Column(Integer)
    address = Column(String(100))
    department_id = Column(Integer, ForeignKey('department.id'))
    department = relationship("Department", back_populates="employees")