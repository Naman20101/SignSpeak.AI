from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from app.database import init_db, close_db
from app.routes import users, gesture
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="SignSpeak.AI Backend API",
    description="Real-time Sign Language Recognition and Translation API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
origins = os.getenv("CORS_ORIGINS", ["http://localhost:8501"]).strip('[]').replace('"', '').split(',')

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Event handlers
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    try:
        await init_db()
        logger.info("✅ SignSpeak.AI Backend started successfully")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    await close_db()
    logger.info("✅ SignSpeak.AI Backend shutdown complete")

# Include routers
app.include_router(users.router)
app.include_router(gesture.router)

# Health check and root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "🚀 SignSpeak.AI Backend API is running!",
        "version": "2.0.0",
        "status": "operational",
        "docs": "/docs",
        "endpoints": {
            "gesture_recognition": "/gesture",
            "user_management": "/users",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "SignSpeak.AI Backend",
        "timestamp": __import__('datetime').datetime.utcnow().isoformat()
    }

@app.get("/api/v1/info")
async def api_info():
    """API information endpoint"""
    return {
        "api_name": "SignSpeak.AI",
        "version": "2.0.0",
        "description": "Real-time Sign Language Recognition API",
        "features": [
            "Real-time gesture recognition",
            "Multiple sign language support (ISL, ASL, BSL)",
            "Text-to-speech integration",
            "WebSocket support for live streaming",
            "User management system"
        ],
        "contact": {
            "email": "support@signspeak.ai",
            "website": "https://signspeak.ai"
        }
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"message": "Endpoint not found", "detail": str(exc)}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"message": "Internal server error", "detail": "Please try again later"}
    )

if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("BACKEND_PORT", 8000))
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,  # Enable auto-reload in development
        log_level="info"
    )
