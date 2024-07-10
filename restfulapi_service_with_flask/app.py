from flask import Flask
import models # Importing models to define models before controllers. Because in controllers, we are using schema classes, so it won't be instantiated without models.
from controllers import DepartmentController, EmployeeController
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, scoped_session 
from models.base import Base

app = Flask(__name__)

engine = create_engine('sqlite:///database.db', echo=True)
Base.metadata.create_all(engine)

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