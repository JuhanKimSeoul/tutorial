from pydantic import BaseModel, Field

class LoginReponseDto(BaseModel):
    access_token:str = Field(..., description='Token')
    refresh_token:str = Field(..., description='Refresh Token')
    email:str = Field(..., description='Email')
    nickname:str = Field(..., description='Nickname')
    grade:str = Field(..., description='grade')

class SignUpResponseDto(BaseModel):
    email:str = Field(..., description='Email')
    nickname:str = Field(..., description='nickname')
    grade:str = Field(..., description='grade')

    