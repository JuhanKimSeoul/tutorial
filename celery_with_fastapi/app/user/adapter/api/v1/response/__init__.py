from pydantic import BaseModel, Field

class LoginResponse(BaseModel):
    token: str = Field(..., title='User Token')
    refresh_token: str = Field(..., title='User Refresh Token')

class CreateUserResponse(BaseModel):
    email: str = Field(..., title='User Email')
    messages: str = Field(..., title='Welcome Message')