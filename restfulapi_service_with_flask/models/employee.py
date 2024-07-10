from sqlalchemy import Column, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from models.base import Base
from marshmallow_sqlalchemy import SQLAlchemyAutoSchema

class Employee(Base):
    __tablename__ = 'employee'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50))
    age = Column(Integer)
    address = Column(String(100))
    department_id = Column(Integer, ForeignKey('department.id'))
    department = relationship("Department", back_populates="employees")

# class EmployeeSchema(SQLAlchemyAutoSchema):
#     class Meta:
#         model = Employee
#         load_instance = True  # Enables deserialization into model instances