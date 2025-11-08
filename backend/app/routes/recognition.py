from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List, Dict, Any
import cv2
import numpy as np
import asyncio

from core.gesture_recognition import GestureRecognizer

router = APIRouter()
recognizer = GestureRecognizer()

@router.post("/image")
async def recognize_from_image(file: UploadFile = File(...)):
    """Recognize gesture from uploaded image"""
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process image
        result = await recognizer.process_image(image)
        return {
            "success": True,
            "gesture": result["gesture"],
            "confidence": result["confidence"],
            "language": result["language"],
            "landmarks": result.get("landmarks", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

@router.post("/batch")
async def batch_recognition(files: List[UploadFile] = File(...)):
    """Batch process multiple images"""
    results = []
    for file in files:
        result = await recognize_from_image(file)
        results.append(result)
    return {"results": results}

@router.get("/languages")
async def get_supported_languages():
    """Get list of supported sign languages"""
    return {
        "languages": ["ASL", "BSL", "ISL"],
        "default": "ASL"
    }
