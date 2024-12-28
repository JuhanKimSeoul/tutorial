from typing import Optional
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.user.application.exception import EmailValidationException, PasswordValidationException

class UserBase(BaseModel):
    email:str = Field(..., title='Email')
    password:str = Field(..., title='Password')
    nickname:str = Field(..., title='Nickname')
    phone:str = Field(..., title='Phone')
    grade:str = Field(..., title='Grade')

    @field_validator('email')
    def validate_email(cls, value):
        if '@' not in value:
            raise EmailValidationException
        return value
    
    @field_validator('password')
    def validate_password(cls, value):
        if len(value) < 10:
            raise PasswordValidationException
        return value

class UserSignUp(BaseModel):
    email:str = Field(..., title='Email')
    nickname:str = Field(..., title='Nickname')
    password1:str = Field(..., title='Password1')
    password2:str = Field(..., title='Password2')
    phone:str = Field(..., title='Phone')
    grade:str = Field(title='Grade', default='BASIC')

class UserUpdate(BaseModel):
    email_or_nickname:str = Field(..., title='Email')
    password:Optional[str] = None
    phone:Optional[str] = None
    grade:Optional[str] = None

class UserLogin(BaseModel):
    email:str = Field(..., title='Email or Nickname')
    password:str = Field(..., title='Password')

class UserLoggedIn(BaseModel):
    email:str = Field(..., title='Email')
    nickname:str = Field(..., title='Nickname')
    grade:str = Field(..., title='Grade')
    access_token:str = Field(..., title='Token')
    refresh_token:str = Field(..., title='Refresh Token')