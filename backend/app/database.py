import motor.motor_asyncio
from beanie import init_beanie
from app.models.user import User
import os
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "signspeak")

# Validate environment variables
if not MONGO_URI:
    logger.error("MONGO_URI environment variable is not set")
    raise ValueError("MONGO_URI environment variable is required")

client = None
db = None

async def init_db():
    """Initialize database connection"""
    global client, db
    
    try:
        client = motor.motor_asyncio.AsyncIOMotorClient(MONGO_URI)
        
        # Test connection
        await client.admin.command('ping')
        logger.info("✅ Successfully connected to MongoDB")
        
        db = client[MONGO_DB]
        
        # Initialize Beanie
        await init_beanie(
            database=db, 
            document_models=[User]
        )
        
        logger.info(f"✅ Database '{MONGO_DB}' initialized successfully")
        
        # Create indexes
        await User.get_motor_collection().create_index("email", unique=True)
        logger.info("✅ Database indexes created")
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise

async def close_db():
    """Close database connection"""
    global client
    if client:
        client.close()
        logger.info("✅ Database connection closed")

def get_database():
    """Get database instance"""
    return db

def get_client():
    """Get MongoDB client instance"""
    return client
