from flask import Blueprint, jsonify, request
from repositories.department_repository import DepartmentRepository
from models.schema import DepartmentSchema

class DepartmentController:
    def __init__(self, session, engine):
        self.repository = DepartmentRepository(session=session, engine=engine)
        self.blueprint = Blueprint('department', __name__)
        self.register_routes()

    def register_routes(self):
        @self.blueprint.route('/departments', methods=['GET'])
        def get_all_departments():
            departments = self.repository.get_all_departments()
            return jsonify(departments)

        @self.blueprint.route('/departments/<int:department_id>', methods=['GET'])
        def get_department_by_id(department_id):
            department = self.repository.get_department_by_id(department_id)
            if department:
                return jsonify(department)
            else:
                return jsonify({'message': 'Department not found'}), 404

        @self.blueprint.route('/departments', methods=['POST'])
        def create_department():
            data = request.get_json()
            department = self.repository.create_department(data)
            department_schema = DepartmentSchema()
            return jsonify(department_schema.dump(department)), 201

        @self.blueprint.route('/departments/<int:department_id>', methods=['PUT'])
        def update_department(department_id):
            data = request.get_json()
            department = self.repository.update_department(department_id, data)
            if department:
                return jsonify(department)
            else:
                return jsonify({'message': 'Department not found'}), 404

        @self.blueprint.route('/departments/<int:department_id>', methods=['DELETE'])
        def delete_department(department_id):
            result = self.repository.delete_department(department_id)
            if result:
                return jsonify({'message': 'Department deleted'})
            else:
                return jsonify({'message': 'Department not found'}), 404

    def get_blueprint(self):
        return self.blueprint