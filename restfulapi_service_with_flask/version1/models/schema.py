from marshmallow_sqlalchemy import SQLAlchemyAutoSchema
from models.department import Department
from models.employee import Employee

class DepartmentSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Department
        load_instance = True  # Enables deserialization into model instances
        include_relationships = True

class EmployeeSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Employee
        load_instance = True  # Enables deserialization into model instances
        include_relationships = True

