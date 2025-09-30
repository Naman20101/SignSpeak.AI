import streamlit as st
import sys
import os

# Add components to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'components'))

# Import components
from components.navigation import create_sidebar_navigation
from components.auth import AuthManager
from config import config

class SignSpeakApp:
    def __init__(self):
        self.auth_manager = AuthManager()
        self.setup_page_config()
        
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="SignSpeak.AI - Real-Time Sign Language Translation",
            page_icon="🤟",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/namanreddy/signspeak-ai',
                'Report a bug': "https://github.com/namanreddy/signspeak-ai/issues",
                'About': "### SignSpeak.AI\nRevolutionizing sign language translation with AI."
            }
        )
        
        # Inject custom CSS
        self.inject_custom_css()
    
    def inject_custom_css(self):
        """Inject custom CSS for styling"""
        st.markdown("""
        <style>
            .main {
                background: linear-gradient(135deg, #0a0a16, #111827);
            }
            .stApp {
                background: transparent;
            }
            /* Custom styling for better appearance */
            .feature-card {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 10px;
                padding: 1.5rem;
                border: 1px solid rgba(255, 255, 255, 0.1);
                margin: 0.5rem 0;
            }
        </style>
        """, unsafe_allow_html=True)
    
    def run(self):
        """Main application runner"""
        # Check authentication
        user_data = self.auth_manager.check_authentication()
        
        # Create navigation
        selected_page = create_sidebar_navigation(user_data)
        
        # Load selected page
        self.load_page(selected_page)
    
    def load_page(self, page_name):
        """Load the selected page"""
        try:
            if page_name == "Home":
                self.render_home_page()
            elif page_name == "Translator":
                self.render_translator_page()
            elif page_name == "Features":
                self.render_features_page()
            elif page_name == "Pricing":
                self.render_pricing_page()
            elif page_name == "Contact":
                self.render_contact_page()
            else:
                st.error(f"Page '{page_name}' not found")
                
        except Exception as e:
            st.error(f"Error loading page: {str(e)}")
    
    def render_home_page(self):
        """Render the home page"""
        st.markdown("""
        <div style='text-align: center; padding: 3rem 1rem;'>
            <h1 style='font-size: 3.5rem; background: linear-gradient(45deg, #7c3aed, #06d6a0); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 1rem;'>
                🤟 SignSpeak.AI
            </h1>
            <p style='font-size: 1.5rem; color: #94a3b8; margin-bottom: 2rem;'>
                Real-Time Sign Language Translation Powered by AI
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Features overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class='feature-card'>
                <h3>🎥 Real-Time Recognition</h3>
                <p>Instant translation using your webcam with advanced computer vision</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='feature-card'>
                <h3>🌍 Multi-Language Support</h3>
                <p>Supports ISL, ASL, and BSL with more languages coming soon</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class='feature-card'>
                <h3>🔊 Text-to-Speech</h3>
                <p>Convert recognized signs to natural sounding speech</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Call to action
        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            <h2>Ready to Get Started?</h2>
            <p>Join thousands of users breaking down communication barriers with AI-powered sign language translation.</p>
            """, unsafe_allow_html=True)
        
        with col2:
            if st.button("🚀 Start Translating Now", use_container_width=True, type="primary"):
                st.session_state.selected_page = "Translator"
                st.rerun()
    
    def render_translator_page(self):
        """Render the translator page"""
        st.markdown("<h1>🎥 Sign Language Translator</h1>", unsafe_allow_html=True)
        
        from components.camera_processor import create_camera_interface
        from components.audio_player import create_audio_interface
        
        # Language selection
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            language = st.selectbox(
                "Sign Language",
                options=["ISL", "ASL", "BSL"],
                index=0,
                key="translator_language"
            )
        
        with col2:
            confidence = st.slider(
                "Confidence Threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.7,
                step=0.05,
                key="confidence_threshold"
            )
        
        # Camera interface
        create_camera_interface()
        
        # Audio interface
        st.markdown("---")
        create_audio_interface()
    
    def render_features_page(self):
        """Render the features page"""
        st.markdown("<h1>✨ Features</h1>", unsafe_allow_html=True)
        
        features = [
            {
                "icon": "🎥",
                "title": "Real-Time Camera Processing",
                "description": "Advanced computer vision algorithms process sign language in real-time using your webcam with sub-second latency."
            },
            {
                "icon": "🌍",
                "title": "Multiple Sign Languages",
                "description": "Support for Indian (ISL), American (ASL), and British (BSL) sign languages with continuous expansion."
            },
            {
                "icon": "🔊", 
                "title": "High-Quality Text-to-Speech",
                "description": "Natural sounding speech synthesis with multiple voice options and customizable settings."
            },
            {
                "icon": "🤖",
                "title": "AI-Powered Recognition",
                "description": "State-of-the-art machine learning models trained on diverse datasets for accurate gesture recognition."
            },
            {
                "icon": "📱",
                "title": "Cross-Platform Compatibility",
                "description": "Works seamlessly on desktop, tablet, and mobile devices with responsive design."
            },
            {
                "icon": "⚡",
                "title": "Lightning Fast Processing",
                "description": "Optimized algorithms ensure smooth performance even on lower-end devices."
            }
        ]
        
        # Display features in a grid
        cols = st.columns(2)
        for idx, feature in enumerate(features):
            with cols[idx % 2]:
                st.markdown(f"""
                <div class='feature-card'>
                    <h3>{feature['icon']} {feature['title']}</h3>
                    <p>{feature['description']}</p>
                </div>
                """, unsafe_allow_html=True)
    
    def render_pricing_page(self):
        """Render the pricing page"""
        st.markdown("<h1>💳 Pricing Plans</h1>", unsafe_allow_html=True)
        
        plans = [
            {
                "name": "Free",
                "price": "$0",
                "period": "forever",
                "features": [
                    "✓ Basic gesture recognition",
                    "✓ ISL & ASL support", 
                    "✓ Standard accuracy",
                    "✗ No translation history",
                    "✗ Limited customer support"
                ],
                "button_text": "Get Started Free",
                "highlighted": False
            },
            {
                "name": "Pro",
                "price": "$19",
                "period": "per month",
                "features": [
                    "✓ Unlimited translations",
                    "✓ All sign languages (ISL, ASL, BSL)",
                    "✓ High accuracy (99%+)",
                    "✓ 6-month translation history",
                    "✓ Priority customer support"
                ],
                "button_text": "Start Free Trial",
                "highlighted": True
            },
            {
                "name": "Enterprise",
                "price": "Custom",
                "period": "tailored",
                "features": [
                    "✓ Everything in Pro",
                    "✓ Custom language training",
                    "✓ Unlimited translation history", 
                    "✓ API access",
                    "✓ Dedicated account manager"
                ],
                "button_text": "Contact Sales",
                "highlighted": False
            }
        ]
        
        # Display pricing cards
        cols = st.columns(3)
        for idx, plan in enumerate(plans):
            with cols[idx]:
                border_color = "#7c3aed" if plan["highlighted"] else "rgba(255,255,255,0.1)"
                st.markdown(f"""
                <div style='
                    background: rgba(255,255,255,0.05); 
                    border: 2px solid {border_color};
                    border-radius: 15px;
                    padding: 2rem;
                    text-align: center;
                    height: 100%;
                    transform: {'scale(1.05)' if plan['highlighted'] else 'scale(1)'};
                    transition: all 0.3s ease;
                '>
                    <h3>{plan['name']}</h3>
                    <h1 style='font-size: 3rem; margin: 1rem 0;'>{plan['price']}</h1>
                    <p style='color: #94a3b8; margin-bottom: 2rem;'>{plan['period']}</p>
                    <div style='text-align: left; margin: 2rem 0;'>
                """, unsafe_allow_html=True)
                
                for feature in plan["features"]:
                    st.markdown(f"<p>{feature}</p>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                button_type = "primary" if plan["highlighted"] else "secondary"
                if st.button(plan["button_text"], use_container_width=True, type=button_type):
                    if plan["name"] == "Free":
                        st.session_state.selected_page = "Translator"
                        st.rerun()
                    elif plan["name"] == "Pro":
                        st.info("🎉 14-day free trial started!")
                    else:
                        st.info("📞 Our sales team will contact you shortly!")
    
    def render_contact_page(self):
        """Render the contact page"""
        st.markdown("<h1>📞 Contact Us</h1>", unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            <div style='background: rgba(255,255,255,0.05); padding: 2rem; border-radius: 10px;'>
                <h3>Get In Touch</h3>
                <p>We'd love to hear from you! Reach out with any questions or feedback.</p>
                
                <div style='margin: 2rem 0;'>
                    <p>📧 <strong>Email:</strong> support@signspeak.ai</p>
                    <p>🌐 <strong>Website:</strong> signspeak.ai</p>
                    <p>🐙 <strong>GitHub:</strong> github.com/namanreddy/signspeak-ai</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            with st.form("contact_form"):
                st.subheader("Send us a Message")
                
                name = st.text_input("Your Name")
                email = st.text_input("Your Email")
                subject = st.selectbox("Subject", [
                    "General Inquiry", 
                    "Technical Support", 
                    "Feature Request",
                    "Partnership",
                    "Other"
                ])
                message = st.text_area("Message", height=150)
                
                submitted = st.form_submit_button("Send Message", type="primary")
                
                if submitted:
                    if not all([name, email, message]):
                        st.error("Please fill in all required fields")
                    else:
                        st.success("🎉 Message sent successfully! We'll get back to you within 24 hours.")

if __name__ == "__main__":
    app = SignSpeakApp()
    app.run()
