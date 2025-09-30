import streamlit as st

def render_home_page():
    """Render the home page content"""
    st.markdown("""
    <div style='text-align: center; padding: 3rem 1rem;'>
        <h1 style='font-size: 3.5rem; background: linear-gradient(45deg, #7c3aed, #06d6a0); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 1rem;'>
            🤟 SignSpeak.AI 2025
        </h1>
        <p style='font-size: 1.5rem; color: #94a3b8; margin-bottom: 2rem;'>
            Revolutionizing Real-Time Sign Language Translation
        </p>
        <p style='font-size: 1.1rem; color: #64748b; max-width: 800px; margin: 0 auto 3rem auto;'>
            Break down communication barriers with our next-generation AI technology. 
            Instantly translate between Indian (ISL), American (ASL), and British (BSL) 
            sign languages using just your webcam.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Key Features
    st.markdown("---")
    st.markdown("<h2 style='text-align: center;'>✨ Key Features</h2>", unsafe_allow_html=True)
    
    features = [
        {
            "icon": "⚡",
            "title": "Real-Time Processing",
            "description": "Sub-second translation latency with 99.2% accuracy using advanced computer vision"
        },
        {
            "icon": "🌍",
            "title": "Multi-Language Support",
            "description": "Native support for ISL, ASL, and BSL with continuous language expansion"
        },
        {
            "icon": "🔊",
            "title": "High-Quality TTS",
            "description": "Natural sounding speech synthesis with customizable voice parameters"
        },
        {
            "icon": "🤖",
            "title": "Advanced AI Models",
            "description": "Transformer-based neural networks trained on diverse sign language datasets"
        },
        {
            "icon": "📱",
            "title": "Cross-Platform",
            "description": "Seamless experience across desktop, tablet, and mobile devices"
        },
        {
            "icon": "⚙️",
            "title": "Developer Friendly",
            "description": "RESTful APIs and comprehensive documentation for integration"
        }
    ]
    
    # Display features in grid
    cols = st.columns(3)
    for idx, feature in enumerate(features):
        with cols[idx % 3]:
            st.markdown(f"""
            <div style='
                background: rgba(255,255,255,0.05);
                border: 1px solid rgba(255,255,255,0.1);
                border-radius: 10px;
                padding: 1.5rem;
                margin: 0.5rem 0;
                height: 200px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            '>
                <div style='font-size: 2.5rem; margin-bottom: 1rem;'>{feature['icon']}</div>
                <h3 style='color: white; margin-bottom: 0.5rem;'>{feature['title']}</h3>
                <p style='color: #94a3b8; font-size: 0.9rem;'>{feature['description']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Statistics Section
    st.markdown("---")
    st.markdown("<h2 style='text-align: center;'>📊 Trusted by Thousands</h2>", unsafe_allow_html=True)
    
    stat_cols = st.columns(4)
    with stat_cols[0]:
        st.metric("Active Users", "10K+", "12%")
    with stat_cols[1]:
        st.metric("Translations/Day", "50K+", "8%")
    with stat_cols[2]:
        st.metric("Accuracy Rate", "99.2%", "0.5%")
    with stat_cols[3]:
        st.metric("Languages", "3", "+2 Soon")
    
    # Call to Action
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <h2>Ready to Transform Communication?</h2>
        <p style='color: #94a3b8; font-size: 1.1rem;'>
            Join educators, developers, and organizations worldwide who are breaking down 
            communication barriers with SignSpeak.AI. Start with our free tier and upgrade 
            as your needs grow.
        </p>
        """, unsafe_allow_html=True)
    
    with col2:
        if st.button("🚀 Start Free Trial", use_container_width=True, type="primary"):
            st.session_state.selected_page = "Translator"
            st.rerun()
        
        if st.button("📖 View Documentation", use_container_width=True):
            st.info("📚 Documentation available at: docs.signspeak.ai")
