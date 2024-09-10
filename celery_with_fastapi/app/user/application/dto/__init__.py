from pydantic import BaseModel, Field

class LoginResponseDTO(BaseModel):
    token: str = Field(..., description="JWT token")
    refresh_token: str = Field(..., description="JWT refresh token")