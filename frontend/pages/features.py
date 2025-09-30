import streamlit as st

def render_features_page():
    """Render the features page"""
    st.markdown("""
    <h1 style='
        background: linear-gradient(45deg, #7c3aed, #06d6a0);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        text-align: center;
    '>✨ Next-Generation Features</h1>
    """, unsafe_allow_html=True)
    
    # Feature Categories
    feature_categories = [
        {
            "title": "🎥 Real-Time Recognition",
            "features": [
                {
                    "name": "Advanced Computer Vision",
                    "description": "State-of-the-art MediaPipe integration for precise hand tracking and gesture recognition",
                    "icon": "👁️"
                },
                {
                    "name": "Sub-Second Latency", 
                    "description": "Optimized processing pipeline delivering translations in under 500ms",
                    "icon": "⚡"
                },
                {
                    "name": "Multi-Hand Detection",
                    "description": "Simultaneous tracking of both hands for complex sign language gestures",
                    "icon": "👐"
                }
            ]
        },
        {
            "title": "🌍 Language Support",
            "features": [
                {
                    "name": "Multiple Sign Languages",
                    "description": "Native support for ISL, ASL, and BSL with regional variations",
                    "icon": "🗺️"
                },
                {
                    "name": "Continuous Learning",
                    "description": "Models that improve over time with user feedback and new data",
                    "icon": "📚"
                },
                {
                    "name": "Custom Vocabulary",
                    "description": "Add domain-specific signs and gestures for specialized use cases",
                    "icon": "🔤"
                }
            ]
        },
        {
            "title": "🔊 Audio & Speech",
            "features": [
                {
                    "name": "High-Quality TTS",
                    "description": "Natural sounding speech synthesis with multiple voice options",
                    "icon": "🎵"
                },
                {
                    "name": "Real-Time Audio", 
                    "description": "Instant audio feedback as gestures are recognized",
                    "icon": "🔊"
                },
                {
                    "name": "Customizable Voices",
                    "description": "Adjust pitch, speed, and tone to match user preferences",
                    "icon": "🎭"
                }
            ]
        },
        {
            "title": "🤖 AI & Machine Learning",
            "features": [
                {
                    "name": "Transformer Models",
                    "description": "Advanced neural networks for context-aware translation",
                    "icon": "🧠"
                },
                {
                    "name": "Gesture Smoothing", 
                    "description": "Temporal smoothing algorithms for stable and accurate recognition",
                    "icon": "📈"
                },
                {
                    "name": "Confidence Scoring",
                    "description": "Real-time confidence metrics for each translation",
                    "icon": "🎯"
                }
            ]
        },
        {
            "title": "📱 User Experience",
            "features": [
                {
                    "name": "Responsive Design",
                    "description": "Seamless experience across all devices and screen sizes",
                    "icon": "📱"
                },
                {
                    "name": "Accessibility First", 
                    "description": "Built with WCAG guidelines for maximum accessibility",
                    "icon": "♿"
                },
                {
                    "name": "Dark/Light Themes",
                    "description": "Multiple UI themes for comfortable viewing in any environment",
                    "icon": "🎨"
                }
            ]
        },
        {
            "title": "🔧 Developer Tools",
            "features": [
                {
                    "name": "RESTful APIs",
                    "description": "Comprehensive API documentation for easy integration",
                    "icon": "🔌"
                },
                {
                    "name": "WebSocket Support", 
                    "description": "Real-time bidirectional communication for live applications",
                    "icon": "🔄"
                },
                {
                    "name": "SDK & Libraries",
                    "description": "Client libraries for popular programming languages",
                    "icon": "📦"
                }
            ]
        }
    ]
    
    # Display feature categories
    for category in feature_categories:
        st.markdown(f"### {category['title']}")
        
        cols = st.columns(3)
        for idx, feature in enumerate(category['features']):
            with cols[idx]:
                st.markdown(f"""
                <div style='
                    background: rgba(255,255,255,0.05);
                    border: 1px solid rgba(255,255,255,0.1);
                    border-radius: 10px;
                    padding: 1.5rem;
                    margin: 0.5rem 0;
                    height: 220px;
                '>
                    <div style='font-size: 2rem; margin-bottom: 1rem;'>{feature['icon']}</div>
                    <h4 style='color: white; margin-bottom: 0.5rem;'>{feature['name']}</h4>
                    <p style='color: #94a3b8; font-size: 0.9rem;'>{feature['description']}</p>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
    
    # Technical Specifications
    st.markdown("### 📊 Technical Specifications")
    
    spec_cols = st.columns(4)
    
    with spec_cols[0]:
        st.markdown("""
        **Performance**
        - Latency: <500ms
        - Accuracy: 99.2%
        - FPS: 30fps
        """)
    
    with spec_cols[1]:
        st.markdown("""
        **Compatibility**
        - Chrome 90+
        - Firefox 88+
        - Safari 14+
        - Edge 90+
        """)
    
    with spec_cols[2]:
        st.markdown("""
        **Models**
        - MediaPipe Hands
        - Custom CNN
        - Transformer
        - LSTM
        """)
    
    with spec_cols[3]:
        st.markdown("""
        **Infrastructure**
        - FastAPI Backend
        - WebRTC Streaming
        - Redis Caching
        - Cloud Deployment
        """)
