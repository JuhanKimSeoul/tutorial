import pytest
from unittest.mock import patch, MagicMock
from app import app

# Use pytest fixture to set up the Flask test client
@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_add_employee_mock(client):
    # Mock the create_employee function
    with patch('repositories.employee_repository.EmployeeRepository.create_employee') as mock_create_employee:
        mock_create_employee.return_value = {'id': 1, 'name': 'John Doe', 'age': 30, 'department': 'Engineering'}

        response = client.post('/employees', json={'name': 'John Doe', 'age': 30, 'department': 'Engineering'})
        
        assert response.status_code == 201
        assert response.json == {'id': 1, 'name': 'John Doe', 'age': 30, 'department': 'Engineering'}

def test_get_employee_mock(client):
    # Mock the read_employee function
    with patch('controllers.employee.read_employee') as mock_read_employee:
        mock_read_employee.return_value = MagicMock(id=1, name='Jane Smith', age=25, department='Marketing')

        response = client.get('/employees/1')
        
        assert response.status_code == 200
        assert response.json == {'id': 1, 'name': 'Jane Smith', 'age': 25, 'department': 'Marketing'}