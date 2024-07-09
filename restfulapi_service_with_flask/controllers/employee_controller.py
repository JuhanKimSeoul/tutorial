from flask import Blueprint, jsonify, request
from repositories.employee_repository import EmployeeRepository

class EmployeeController:
    def __init__(self, session, engine):
        self.repository = EmployeeRepository(session=session, engine=engine)
        self.blueprint = Blueprint('employee', __name__)
        self.register_routes()

    def register_routes(self):
        @self.blueprint.route('/employees', methods=['GET'])
        def get_all_employees():
            employees = self.repository.get_all_employees()
            return jsonify(employees)

        @self.blueprint.route('/employees/<int:employee_id>', methods=['GET'])
        def get_employee_by_id(employee_id):
            employee = self.repository.get_employee_by_id(employee_id)
            if employee:
                return jsonify(employee)
            else:
                return jsonify({'message': 'Employee not found'}), 404

        @self.blueprint.route('/employees', methods=['POST'])
        def create_employee():
            data = request.get_json()
            employee = self.repository.create_employee(data)
            return jsonify(employee), 201

        @self.blueprint.route('/employees/<int:employee_id>', methods=['PUT'])
        def update_employee(employee_id):
            data = request.get_json()
            employee = self.repository.update_employee(employee_id, data)
            if employee:
                return jsonify(employee)
            else:
                return jsonify({'message': 'Employee not found'}), 404

        @self.blueprint.route('/employees/<int:employee_id>', methods=['DELETE'])
        def delete_employee(employee_id):
            result = self.repository.delete_employee(employee_id)
            if result:
                return jsonify({'message': 'Employee deleted'})
            else:
                return jsonify({'message': 'Employee not found'}), 404

    def get_blueprint(self):
        return self.blueprint