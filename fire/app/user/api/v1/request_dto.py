from pydantic import BaseModel, Field

class LoginRequestDto(BaseModel):
    email:str = Field(..., description='Email or Nickname')
    password:str = Field(..., description='password')

class SignUpRequestDto(BaseModel):
    email:str = Field(..., description='Email')
    nickname:str = Field(..., description='Nickname')
    password1:str = Field(..., description='Password1')
    password2:str = Field(..., description='Password2')
    phone:str = Field(..., description='Phone')

