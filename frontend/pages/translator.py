import streamlit as st
from components.camera_processor import create_camera_interface
from components.audio_player import create_audio_interface
from components.animation_display import create_animation_interface

def render_translator_page():
    """Render the main translator page"""
    st.markdown("""
    <h1 style='
        background: linear-gradient(45deg, #7c3aed, #06d6a0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    '>🎥 AI Sign Language Translator</h1>
    """, unsafe_allow_html=True)
    
    # Configuration Section
    with st.expander("⚙️ Translation Settings", expanded=True):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            language = st.selectbox(
                "Sign Language",
                options=["ISL", "ASL", "BSL"],
                index=0,
                help="Select the sign language you want to translate"
            )
            st.session_state.translator_language = language
        
        with col2:
            confidence = st.slider(
                "Confidence Threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="Adjust the sensitivity of gesture recognition"
            )
            st.session_state.confidence_threshold = confidence
        
        with col3:
            frame_rate = st.selectbox(
                "Processing Speed",
                options=["Realtime", "Fast", "Balanced", "Accurate"],
                index=1,
                help="Balance between speed and accuracy"
            )
            st.session_state.frame_rate = frame_rate
    
    # Main Translation Interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Camera Interface
        create_camera_interface()
    
    with col2:
        # Translation Results
        st.markdown("### 📝 Translation Results")
        
        # Results display
        if st.session_state.get('last_recognition'):
            result = st.session_state.last_recognition
            text = result.get('text', 'No gesture detected')
            confidence = result.get('confidence', 0)
            language = result.get('language', 'Unknown')
            
            st.markdown(f"""
            <div style='
                background: rgba(6, 214, 160, 0.1);
                border: 1px solid rgba(6, 214, 160, 0.3);
                border-radius: 10px;
                padding: 1.5rem;
                margin: 1rem 0;
            '>
                <div style='color: #06d6a0; font-weight: bold; font-size: 1.2rem;'>
                    {text}
                </div>
                <div style='color: #94a3b8; font-size: 0.9rem; margin-top: 0.5rem;'>
                    Confidence: <strong>{confidence:.1%}</strong> | Language: <strong>{language}</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='
                background: rgba(255,255,255,0.05);
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: 10px;
                padding: 2rem;
                text-align: center;
                color: #94a3b8;
            '>
                <div style='font-size: 3rem; margin-bottom: 1rem;'>👋</div>
                <div>Start the camera and begin signing to see real-time translations here.</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Quick Actions
        st.markdown("### ⚡ Quick Actions")
        
        action_col1, action_col2 = st.columns(2)
        
        with action_col1:
            if st.button("🗑️ Clear Results", use_container_width=True):
                if 'last_recognition' in st.session_state:
                    del st.session_state.last_recognition
                st.rerun()
        
        with action_col2:
            if st.button("📋 Copy Text", use_container_width=True):
                if st.session_state.get('last_recognition'):
                    text = st.session_state.last_recognition.get('text', '')
                    st.success(f"Copied: {text}")
                else:
                    st.warning("No text to copy")
    
    # Additional Features
    st.markdown("---")
    
    tab1, tab2 = st.tabs(["🔊 Text-to-Speech", "👋 Gesture Animation"])
    
    with tab1:
        create_audio_interface()
    
    with tab2:
        create_animation_interface()
    
    # Demo Mode Notice
    if st.session_state.get('demo_mode', True):
        st.markdown("---")
        st.info("""
        **🔒 Demo Mode Active**  
        This is a demonstration of SignSpeak.AI capabilities. 
        In production mode, you would have:
        - Real camera processing with MediaPipe
        - Live model inference
        - Cloud-based text-to-speech
        - Gesture animation generation
        """)
