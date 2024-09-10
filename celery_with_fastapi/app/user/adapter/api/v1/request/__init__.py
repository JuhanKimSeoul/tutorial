from pydantic import BaseModel, Field

class LoginRequest(BaseModel):
    email: str = Field(..., title='User Email')
    password: str = Field(..., title='User Password')

class CreateUserRequest(BaseModel):
    email: str = Field(..., title='User Email')
    password: str = Field(..., title='User Password')
    verify_password: str = Field(..., title='User Password Verification')
