import pyttsx3
import numpy as np
import soundfile as sf
import io
import logging
from typing import Optional, Union
import asyncio

try:
    from TTS.api import TTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

class TextToSpeechEngine:
    def __init__(self, engine: str = "coqui"):
        self.engine_type = engine
        self.logger = logging.getLogger(__name__)
        self.setup_engine()

    def setup_engine(self):
        """Initialize TTS engine"""
        try:
            if self.engine_type == "coqui" and TTS_AVAILABLE:
                self.tts = TTS("tts_models/en/ljspeech/tacotron2-DDC_ph")
                self.supported_engines = ["coqui"]
            else:
                self.tts = pyttsx3.init()
                self.supported_engines = ["pyttsx3"]
                
            self.logger.info(f"TTS engine initialized: {self.engine_type}")
            
        except Exception as e:
            self.logger.error(f"TTS setup failed: {str(e)}")
            self.tts = pyttsx3.init()
            self.engine_type = "pyttsx3"

    async def text_to_speech(self, text: str, output_file: str = None) -> Optional[bytes]:
        """Convert text to speech audio"""
        try:
            if self.engine_type == "coqui" and TTS_AVAILABLE:
                return await self._coqui_tts(text, output_file)
            else:
                return await self._pyttsx3_tts(text, output_file)
                
        except Exception as e:
            self.logger.error(f"TTS conversion error: {str(e)}")
            return None

    async def _coqui_tts(self, text: str, output_file: str = None) -> Optional[bytes]:
        """Use Coqui TTS for high-quality speech"""
        try:
            if output_file:
                self.tts.tts_to_file(text=text, file_path=output_file)
                return None
            else:
                # Generate audio and return as bytes
                audio = self.tts.tts(text=text)
                audio_array = np.array(audio)
                
                # Convert to bytes
                buffer = io.BytesIO()
                sf.write(buffer, audio_array, 22050, format='WAV')
                return buffer.getvalue()
                
        except Exception as e:
            self.logger.error(f"Coqui TTS error: {str(e)}")
            return await self._pyttsx3_tts(text, output_file)

    async def _pyttsx3_tts(self, text: str, output_file: str = None) -> Optional[bytes]:
        """Use pyttsx3 as fallback TTS"""
        try:
            if output_file:
                self.tts.save_to_file(text, output_file)
                self.tts.runAndWait()
                return None
            else:
                # For in-memory audio, we'd need to use a different approach
                # This is a simplified version
                self.logger.info(f"Speaking: {text}")
                self.tts.say(text)
                self.tts.runAndWait()
                return None
                
        except Exception as e:
            self.logger.error(f"pyttsx3 TTS error: {str(e)}")
            return None

    def get_available_voices(self):
        """Get available TTS voices"""
        if hasattr(self.tts, 'getProperty'):
            voices = self.tts.getProperty('voices')
            return [voice.id for voice in voices]
        return ["default"]

    def set_voice(self, voice_id: str):
        """Set TTS voice"""
        try:
            if hasattr(self.tts, 'setProperty'):
                self.tts.setProperty('voice', voice_id)
        except Exception as e:
            self.logger.error(f"Voice setting error: {str(e)}")

    def close(self):
        """Clean up TTS resources"""
        if hasattr(self.tts, 'stop') and self.engine_type == "pyttsx3":
            self.tts.stop()
