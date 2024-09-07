from models.department import Department

class DepartmentRepository:
    def __init__(self, engine, session):
        self.engine = engine
        self.session = session

    def get_department_by_id(self, department_id):
        return self.session.query(Department).filter_by(id=department_id).first()

    def get_all_departments(self):
        return self.session.query(Department).all()

    def create_department(self, department_data):
        department = Department(**department_data)
        self.session.add(department)
        self.session.commit()
        return department

    def update_department(self, department_id, department_data):
        department = self.get_department_by_id(department_id)
        if department:
            for key, value in department_data.items():
                setattr(department, key, value)
            self.session.commit()
        return department

    def delete_department(self, department_id):
        department = self.get_department_by_id(department_id)
        if department:
            self.session.delete(department)
            self.session.commit()
        return department
