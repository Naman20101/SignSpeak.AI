import streamlit as st
import json
import requests
from typing import Dict, Any, List
from config import get_config

class AnimationDisplay:
    def __init__(self):
        self.config = get_config()
        self.endpoints = self.config.get_backend_endpoints()
    
    def get_gesture_animation(self, text: str, language: str = "ISL") -> Dict[str, Any]:
        """Get animation data for text-to-gesture conversion"""
        try:
            response = requests.post(
                self.endpoints['text_to_gesture'],
                json={"text": text, "language": language},
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return self.get_demo_animation(text, language)
                
        except requests.exceptions.RequestException:
            return self.get_demo_animation(text, language)
    
    def get_demo_animation(self, text: str, language: str) -> Dict[str, Any]:
        """Get demo animation data"""
        return {
            "text": text,
            "language": language,
            "animation_frames": [
                {"frame": 1, "landmarks": self.generate_demo_landmarks()},
                {"frame": 2, "landmarks": self.generate_demo_landmarks()},
                {"frame": 3, "landmarks": self.generate_demo_landmarks()}
            ],
            "duration": 2.0,
            "status": "demo"
        }
    
    def generate_demo_landmarks(self) -> List[Dict]:
        """Generate demo hand landmarks"""
        import random
        landmarks = []
        for i in range(21):  # 21 hand landmarks
            landmarks.append({
                "x": random.uniform(0.1, 0.9),
                "y": random.uniform(0.1, 0.9),
                "z": random.uniform(-0.5, 0.5)
            })
        return landmarks

def create_animation_interface():
    """Create animation display interface"""
    st.markdown("### 👋 Gesture Animation")
    
    # Text input for animation
    text_input = st.text_input(
        "Text to animate",
        placeholder="Enter text to see sign language animation...",
        key="animation_text"
    )
    
    if st.button("🎬 Generate Animation", use_container_width=True):
        if text_input:
            display = AnimationDisplay()
            animation_data = display.get_gesture_animation(
                text_input, 
                st.session_state.get('translator_language', 'ISL')
            )
            
            st.session_state.animation_data = animation_data
            st.success(f"Generated animation for: {text_input}")
        else:
            st.warning("Please enter text to generate animation")
    
    # Display animation
    if st.session_state.get('animation_data'):
        display_animation_preview(st.session_state.animation_data)

def display_animation_preview(animation_data: Dict[str, Any]):
    """Display animation preview"""
    st.markdown("---")
    st.markdown("#### Animation Preview")
    
    # Animation controls
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("⏮️ Previous", use_container_width=True):
            st.info("Previous frame")
    
    with col2:
        st.markdown(
            f"<div style='text-align: center; color: #06d6a0; font-weight: bold;'>"
            f"Frame 1 of {len(animation_data.get('animation_frames', []))}"
            f"</div>", 
            unsafe_allow_html=True
        )
    
    with col3:
        if st.button("⏭️ Next", use_container_width=True):
            st.info("Next frame")
    
    # Animation canvas placeholder
    st.markdown("""
    <div style='
        background: rgba(255,255,255,0.05); 
        border: 2px dashed rgba(6,214,160,0.3);
        border-radius: 10px;
        height: 300px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 1rem 0;
    '>
        <div style='text-align: center; color: #94a3b8;'>
            <div style='font-size: 3rem;'>👋</div>
            <div>Hand Animation Preview</div>
            <div style='font-size: 0.8rem; margin-top: 0.5rem;'>
                Showing: {text}<br>
                Language: {language}
            </div>
        </div>
    </div>
    """.format(
        text=animation_data.get('text', 'Unknown'),
        language=animation_data.get('language', 'ISL')
    ), unsafe_allow_html=True)
    
    # Animation info
    with st.expander("Animation Details"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Frames", len(animation_data.get('animation_frames', [])))
            st.metric("Duration", f"{animation_data.get('duration', 0)}s")
        
        with col2:
            st.metric("Language", animation_data.get('language', 'ISL'))
            st.metric("Status", animation_data.get('status', 'unknown').title())
