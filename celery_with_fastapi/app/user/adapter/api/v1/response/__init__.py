from pydantic import BaseModel, Field

class LoginResponse(BaseModel):
    token: str = Field(..., title='User Token')
    refresh_token: str = Field(..., title='User Refresh Token')