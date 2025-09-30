import streamlit as st
from typing import Optional, Dict, Any

def create_sidebar_navigation(user_data: Optional[Dict[str, Any]] = None):
    """Create the main navigation sidebar"""
    
    st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #0f0f23, #0a0a16);
        border-right: 1px solid #1e293b;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Logo and header
    st.sidebar.markdown(
        """
        <div style='text-align: center; padding: 1rem 0; border-bottom: 1px solid #1e293b; margin-bottom: 1rem;'>
            <h1 style='color: #06d6a0; font-size: 1.8rem; margin: 0;'>🤟 SignSpeak.AI</h1>
            <p style='color: #94a3b8; font-size: 0.9rem; margin: 0.5rem 0 0 0;'>Real-Time Translation</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Navigation items
    nav_items = [
        {"name": "Home", "icon": "🏠"},
        {"name": "Translator", "icon": "🎥"},
        {"name": "Features", "icon": "✨"},
        {"name": "Pricing", "icon": "💳"},
        {"name": "Contact", "icon": "📞"}
    ]
    
    # User info section
    if user_data:
        st.sidebar.markdown(
            f"""
            <div style='background: rgba(6, 214, 160, 0.1); padding: 1rem; border-radius: 8px; margin: 1rem 0; border: 1px solid rgba(6, 214, 160, 0.2);'>
                <div style='color: #06d6a0; font-weight: 600;'>Welcome back!</div>
                <div style='color: #cbd5e1; font-size: 0.9rem;'>{user_data.get('name', 'User')}</div>
                <div style='color: #94a3b8; font-size: 0.8rem;'>{user_data.get('email', '')}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Create navigation
    selected = st.sidebar.radio(
        "Navigation",
        [item["name"] for item in nav_items],
        label_visibility="collapsed",
        key="main_navigation"
    )
    
    # System status
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔧 System Status")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        st.metric("API Status", "Online", delta="Active")
    
    with col2:
        st.metric("Models", "3/3", delta="Loaded")
    
    # Quick actions
    st.sidebar.markdown("### ⚡ Quick Actions")
    
    if st.sidebar.button("🚀 Start Translation", use_container_width=True):
        return "Translator"
    
    if st.sidebar.button("🎬 View Demo", use_container_width=True):
        st.session_state.demo_mode = True
        return "Translator"
    
    # Auth section
    st.sidebar.markdown("---")
    if not user_data:
        auth_col1, auth_col2 = st.sidebar.columns(2)
        with auth_col1:
            if st.button("Login", use_container_width=True):
                st.session_state.show_login = True
        with auth_col2:
            if st.button("Register", use_container_width=True):
                st.session_state.show_register = True
    else:
        if st.sidebar.button("Logout", use_container_width=True):
            st.session_state.user_data = None
            st.rerun()
    
    return selected
