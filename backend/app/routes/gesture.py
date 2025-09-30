from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from app.core.gesture_recognition import recognize_gesture, gesture_recognizer
from app.core.text_to_speech import tts_engine
import base64
import cv2
import numpy as np
import logging
import json
from typing import Dict, Any
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/gesture", tags=["Gesture Recognition"])

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_message(self, message: str, client_id: str):
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(message)
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)

manager = ConnectionManager()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_id = f"client_{id(websocket)}"
    
    await manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                # Parse the incoming data
                message_data = json.loads(data)
                frame_data = message_data.get("image")
                language = message_data.get("language", "ISL")
                confidence_threshold = message_data.get("confidence_threshold", 0.7)
                
                # Decode base64 image
                if frame_data.startswith("data:image"):
                    frame_data = frame_data.split(",")[1]
                
                img_bytes = base64.b64decode(frame_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    await websocket.send_text(json.dumps({
                        "error": "Failed to decode image",
                        "timestamp": __import__('datetime').datetime.utcnow().isoformat()
                    }))
                    continue
                
                # Recognize gesture
                gesture_result = recognize_gesture(frame, language, confidence_threshold)
                
                # Send recognition result
                await websocket.send_text(json.dumps(gesture_result))
                
                # Optional: Play TTS (could be made configurable)
                if gesture_result.get("confidence", 0) > confidence_threshold:
                    tts_engine.speak(gesture_result["text"])
                
            except json.JSONDecodeError:
                await websocket.send_text(json.dumps({
                    "error": "Invalid JSON format",
                    "timestamp": __import__('datetime').datetime.utcnow().isoformat()
                }))
            except Exception as e:
                logger.error(f"WebSocket processing error: {e}")
                await websocket.send_text(json.dumps({
                    "error": f"Processing error: {str(e)}",
                    "timestamp": __import__('datetime').datetime.utcnow().isoformat()
                }))
                
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
        manager.disconnect(client_id)

@router.post("/recognize")
async def recognize_gesture_from_image(
    image_data: str,
    language: str = "ISL",
    confidence_threshold: float = 0.7
):
    """
    Recognize gesture from base64 encoded image
    """
    try:
        # Decode base64 image
        if image_data.startswith("data:image"):
            image_data = image_data.split(",")[1]
        
        img_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")
        
        # Recognize gesture
        result = recognize_gesture(frame, language, confidence_threshold)
        
        return result
        
    except Exception as e:
        logger.error(f"Gesture recognition error: {e}")
        raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")

@router.get("/languages")
async def get_supported_languages():
    """Get list of supported sign languages"""
    return {
        "supported_languages": gesture_recognizer.get_supported_languages(),
        "default_language": "ISL"
    }

@router.get("/status")
async def get_gesture_service_status():
    """Get gesture recognition service status"""
    return {
        "status": "operational",
        "mode": "demo",  # or "production" when using real models
        "last_processed": gesture_recognizer.last_processed,
        "supported_languages": gesture_recognizer.get_supported_languages()
    }
