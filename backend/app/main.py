from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import asyncio
import json
import logging

from app.routes import recognition, tts, animation
from app.utils.connection_manager import ConnectionManager
from app.utils.error_handler import setup_exception_handlers

# Initialize FastAPI app
app = FastAPI(
    title="SignSpeak.AI API",
    description="Multilingual Sign Language Recognition & Translation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup utilities
manager = ConnectionManager()
setup_exception_handlers(app)

# Include routers
app.include_router(recognition.router, prefix="/api/v1/recognition", tags=["recognition"])
app.include_router(tts.router, prefix="/api/v1/tts", tags=["text-to-speech"])
app.include_router(animation.router, prefix="/api/v1/animation", tags=["animation"])

@app.websocket("/ws/recognize")
async def websocket_recognition(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Process real-time frame data
            recognition_result = await process_frame_data(json.loads(data))
            await manager.send_personal_message(
                json.dumps(recognition_result), 
                websocket
            )
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/")
async def root():
    return {"message": "SignSpeak.AI API Server Running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "signspeak-backend"}

async def process_frame_data(frame_data: dict):
    """Process frame data for gesture recognition"""
    # Implementation in gesture_recognition.py
    return {"gesture": "hello", "confidence": 0.95, "language": "ASL"}

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
