from fastapi import APIRouter, HTTPException, status, Depends
from app.models.user import User
from passlib.hash import bcrypt
from pydantic import BaseModel, EmailStr
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/users", tags=["Users"])

# Pydantic models for request/response
class UserCreate(BaseModel):
    email: EmailStr
    name: str
    password: str

class UserResponse(BaseModel):
    id: str
    email: EmailStr
    name: str
    is_active: bool
    is_verified: bool
    subscription_tier: str
    created_at: str

    class Config:
        from_attributes = True

@router.post("/", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(user_data: UserCreate):
    """
    Create a new user account
    """
    try:
        # Check if user already exists
        existing_user = await User.find_one(User.email == user_data.email)
        if existing_user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User with this email already exists"
            )
        
        # Validate password strength
        if len(user_data.password) < 6:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 6 characters long"
            )
        
        # Hash password
        hashed_pw = bcrypt.hash(user_data.password)
        
        # Create user
        user = User(
            email=user_data.email,
            name=user_data.name,
            hashed_password=hashed_pw
        )
        
        await user.insert()
        
        logger.info(f"New user created: {user_data.email}")
        
        return UserResponse(
            id=str(user.id),
            email=user.email,
            name=user.name,
            is_active=user.is_active,
            is_verified=user.is_verified,
            subscription_tier=user.subscription_tier,
            created_at=user.created_at.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"User creation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user"
        )

@router.get("/", response_model=List[UserResponse])
async def list_users(limit: int = 10, skip: int = 0):
    """
    Get list of users (for admin purposes)
    """
    try:
        users = await User.find_all().skip(skip).limit(limit).to_list()
        
        return [
            UserResponse(
                id=str(user.id),
                email=user.email,
                name=user.name,
                is_active=user.is_active,
                is_verified=user.is_verified,
                subscription_tier=user.subscription_tier,
                created_at=user.created_at.isoformat()
            )
            for user in users
        ]
        
    except Exception as e:
        logger.error(f"Error fetching users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch users"
        )

@router.get("/{user_id}", response_model=UserResponse)
async def get_user(user_id: str):
    """
    Get user by ID
    """
    try:
        user = await User.get(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return UserResponse(
            id=str(user.id),
            email=user.email,
            name=user.name,
            is_active=user.is_active,
            is_verified=user.is_verified,
            subscription_tier=user.subscription_tier,
            created_at=user.created_at.isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch user"
        )

@router.get("/count/total")
async def get_total_users():
    """Get total number of users"""
    try:
        count = await User.find_all().count()
        return {"total_users": count}
    except Exception as e:
        logger.error(f"Error counting users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to count users"
        )
