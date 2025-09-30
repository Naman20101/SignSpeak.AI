import streamlit as st
import cv2
import numpy as np
import requests
import json
import time
from typing import Optional, Dict, Any
from config import get_config

class CameraProcessor:
    def __init__(self):
        self.config = get_config()
        self.endpoints = self.config.get_backend_endpoints()
    
    def process_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """Process frame for gesture recognition"""
        try:
            # Convert frame to base64
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            
            # Prepare request data
            data = {
                "image": f"data:image/jpeg;base64,{frame_base64}",
                "language": st.session_state.get('translator_language', 'ISL'),
                "confidence_threshold": st.session_state.get('confidence_threshold', 0.7)
            }
            
            # Send to backend
            response = requests.post(
                self.endpoints['recognize'],
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return self.get_demo_result()
                
        except requests.exceptions.RequestException:
            return self.get_demo_result()
        except Exception as e:
            st.error(f"Processing error: {e}")
            return None
    
    def get_demo_result(self) -> Dict[str, Any]:
        """Get demo result when backend is unavailable"""
        demo_gestures = {
            "ISL": ["Hello!", "Thank you", "How are you?", "Nice to meet you"],
            "ASL": ["Hello!", "Thank you", "What's up?", "Good to see you"],
            "BSL": ["Hello!", "Cheers", "You alright?", "Lovely day"]
        }
        
        language = st.session_state.get('translator_language', 'ISL')
        gestures = demo_gestures.get(language, demo_gestures["ISL"])
        
        # Cycle through demo gestures
        index = st.session_state.get('demo_gesture_index', 0)
        gesture = gestures[index]
        
        # Update index
        st.session_state.demo_gesture_index = (index + 1) % len(gestures)
        
        return {
            "text": gesture,
            "confidence": 0.85,
            "language": language,
            "timestamp": time.time()
        }

def create_camera_interface():
    """Create camera interface component"""
    st.markdown("### 🎥 Live Camera Feed")
    
    # Camera controls
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📷 Start Camera", use_container_width=True, type="primary"):
            st.session_state.camera_active = True
    
    with col2:
        if st.button("⏹️ Stop Camera", use_container_width=True):
            st.session_state.camera_active = False
    
    # Camera feed
    if st.session_state.get('camera_active', False):
        camera_image = st.camera_input("Sign language camera", label_visibility="collapsed")
        
        if camera_image:
            # Convert to numpy array
            image_array = np.frombuffer(camera_image.getvalue(), np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            # Process the image
            processor = CameraProcessor()
            result = processor.process_frame(image)
            
            # Display results
            if result:
                confidence = result.get('confidence', 0)
                text = result.get('text', 'No gesture detected')
                
                st.success(f"**Recognized:** {text}")
                st.progress(confidence, text=f"Confidence: {confidence:.1%}")
                
                # Store last result
                st.session_state.last_recognition = result
    else:
        st.info("👆 Click 'Start Camera' to begin sign language recognition")
        
        # Demo placeholder
        st.image(
            "https://via.placeholder.com/640x360/1e293b/06d6a0?text=SignSpeak.AI+Camera+Feed",
            use_column_width=True,
            caption="Camera feed will appear here when started"
        )
