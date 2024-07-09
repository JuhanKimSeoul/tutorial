from flask import Flask
from controllers.departement_controller import DepartmentController
from controllers.employee_controller import EmployeeController
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session

app = Flask(__name__)

engine = create_engine('sqlite:///database.db', echo=True)

session_factory = sessionmaker(bind=engine, autocommit=False, autoflush=True)
session = scoped_session(session_factory)

# Initialize controllers
department_controller = DepartmentController(session=session, engine=engine)
employee_controller = EmployeeController(session=session, engine=engine)

# Register routes from both controllers
app.register_blueprint(department_controller.get_blueprint())
app.register_blueprint(employee_controller.get_blueprint())

if __name__ == '__main__':
    app.run(debug=True)