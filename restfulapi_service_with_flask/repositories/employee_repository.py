from models.employee import Employee

class EmployeeRepository:
    def __init__(self, session, engine):
        self.session = session
        self.engine = engine

    def create_employee(self, employee_data):
        self.session.add(Employee(**employee_data))
        self.session.commit()

    def get_employee_by_id(self, employee_id):
        return self.session.query(Employee).filter_by(id=employee_id).first()
    
    def get_all_employees(self):
        return self.session.query(Employee).all()

    def update_employee(self, employee):
        employee = self.get_employee_by_id(employee.id)
        if employee:
            for key, value in employee.items():
                setattr(employee, key, value)
            self.session.commit()
        return employee

    def delete_employee(self, employee):
        employee = self.get_employee_by_id(employee.id)
        if employee:
            self.session.delete(employee)
            self.session.commit()
        return employee