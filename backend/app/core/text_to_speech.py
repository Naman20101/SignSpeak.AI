import pyttsx3
import logging
import tempfile
import os
from typing import Optional, Dict, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextToSpeech:
    def __init__(self):
        try:
            self.engine = pyttsx3.init()
            
            # Configure voice properties
            self.engine.setProperty('rate', 150)  # Speech rate
            self.engine.setProperty('volume', 0.8)  # Volume level
            
            # Get available voices
            self.voices = self.engine.getProperty('voices')
            logger.info(f"TTS initialized with {len(self.voices)} available voices")
            
        except Exception as e:
            logger.error(f"TTS initialization failed: {e}")
            self.engine = None
    
    def get_available_voices(self) -> list:
        """Get list of available voices"""
        if not self.engine:
            return []
        return [voice.id for voice in self.voices]
    
    def speak(self, text: str, voice_id: Optional[str] = None, rate: int = 150, volume: float = 0.8) -> bool:
        """Convert text to speech and play immediately"""
        try:
            if not self.engine:
                logger.error("TTS engine not available")
                return False
            
            if not text or not text.strip():
                logger.warning("Empty text provided for TTS")
                return False
            
            # Configure properties
            self.engine.setProperty('rate', rate)
            self.engine.setProperty('volume', volume)
            
            # Set voice if specified
            if voice_id and voice_id in self.get_available_voices():
                self.engine.setProperty('voice', voice_id)
            
            # Speak the text
            self.engine.say(text)
            self.engine.runAndWait()
            
            logger.info(f"TTS played: {text[:50]}...")
            return True
            
        except Exception as e:
            logger.error(f"TTS playback error: {e}")
            return False
    
    def save_to_file(self, text: str, filename: str, voice_id: Optional[str] = None, 
                    rate: int = 150, volume: float = 0.8) -> Optional[str]:
        """Save speech to audio file"""
        try:
            if not self.engine:
                logger.error("TTS engine not available")
                return None
            
            # Ensure filename has proper extension
            if not filename.endswith('.wav'):
                filename += '.wav'
            
            # Configure properties
            self.engine.setProperty('rate', rate)
            self.engine.setProperty('volume', volume)
            
            if voice_id and voice_id in self.get_available_voices():
                self.engine.setProperty('voice', voice_id)
            
            # Save to file
            self.engine.save_to_file(text, filename)
            self.engine.runAndWait()
            
            logger.info(f"TTS saved to file: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"TTS file save error: {e}")
            return None
    
    def get_voice_info(self) -> Dict[str, Any]:
        """Get information about available voices"""
        if not self.engine:
            return {"error": "TTS engine not available"}
        
        voices_info = []
        for voice in self.voices:
            voices_info.append({
                "id": voice.id,
                "name": voice.name,
                "languages": voice.languages if hasattr(voice, 'languages') else ["en"],
                "gender": "female" if "female" in voice.name.lower() else "male"
            })
        
        return {
            "available_voices": voices_info,
            "current_rate": self.engine.getProperty('rate'),
            "current_volume": self.engine.getProperty('volume')
        }

# Global TTS instance
tts_engine = TextToSpeech()

def speak(text: str) -> bool:
    """Backward compatibility function"""
    return tts_engine.speak(text)
