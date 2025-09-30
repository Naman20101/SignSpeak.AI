from beanie import Document
from pydantic import EmailStr, Field
from typing import Optional
from datetime import datetime

class User(Document):
    email: EmailStr = Field(unique=True)
    name: str = Field(min_length=2, max_length=50)
    hashed_password: str
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    subscription_tier: str = Field(default="free")  # free, pro, enterprise
    
    class Settings:
        name = "users"
        use_state_management = True
    
    class Config:
        schema_extra = {
            "example": {
                "email": "user@example.com",
                "name": "John Doe",
                "is_active": True,
                "subscription_tier": "pro"
            }
        }

class UserCreate:
    email: EmailStr
    name: str
    password: str

class UserResponse:
    id: str
    email: EmailStr
    name: str
    is_active: bool
    subscription_tier: str
    created_at: datetime
