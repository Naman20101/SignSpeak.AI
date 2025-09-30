import streamlit as st
import requests
from typing import Optional
from config import get_config

class AudioPlayer:
    def __init__(self):
        self.config = get_config()
        self.endpoints = self.config.get_backend_endpoints()
    
    def generate_speech(self, text: str) -> bool:
        """Generate speech from text"""
        if not text.strip():
            st.warning("Please enter text to convert to speech")
            return False
        
        try:
            response = requests.post(
                self.endpoints['text_to_speech'],
                json={"text": text},
                timeout=30
            )
            
            if response.status_code == 200:
                st.success("✅ Audio generated successfully!")
                return True
            else:
                st.error("❌ Failed to generate audio")
                return False
                
        except requests.exceptions.RequestException:
            st.info("🔊 Audio preview not available in demo mode")
            return True

def create_audio_interface():
    """Create audio player interface"""
    st.markdown("### 🔊 Text-to-Speech")
    
    # Text input
    text_input = st.text_area(
        "Text to convert to speech",
        value=st.session_state.get('last_recognition', {}).get('text', ''),
        placeholder="Enter text or use recognized gesture text...",
        height=100
    )
    
    # Controls
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🎵 Generate Speech", use_container_width=True):
            player = AudioPlayer()
            player.generate_speech(text_input)
    
    with col2:
        if st.button("▶️ Play Audio", use_container_width=True):
            st.info("🔊 Playing audio... (Demo mode)")
    
    # Settings
    with st.expander("Audio Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.slider("Speech Rate", 50, 300, 150, key="tts_rate")
            st.selectbox("Voice", ["Default", "Male", "Female"], key="tts_voice")
        
        with col2:
            st.slider("Volume", 0.0, 1.0, 0.8, key="tts_volume")
            st.selectbox("Language", ["English", "Spanish", "French"], key="tts_lang")
