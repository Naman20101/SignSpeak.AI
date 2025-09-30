import streamlit as st
import requests
from typing import Optional, Dict, Any
from config import get_config

class AuthManager:
    def __init__(self):
        self.config = get_config()
        self.endpoints = self.config.get_backend_endpoints()
    
    def check_authentication(self) -> Optional[Dict[str, Any]]:
        """Check if user is authenticated"""
        return st.session_state.get('user_data')
    
    def login(self, email: str, password: str) -> bool:
        """Authenticate user"""
        try:
            response = requests.post(
                self.endpoints['auth_login'],
                json={"email": email, "password": password},
                timeout=10
            )
            
            if response.status_code == 200:
                st.session_state.user_data = response.json()
                return True
            else:
                st.error(f"Login failed: {response.json().get('detail', 'Unknown error')}")
                return False
                
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {e}")
            # Demo mode fallback
            st.session_state.user_data = {
                "name": "Demo User",
                "email": email,
                "subscription_tier": "pro"
            }
            st.info("🔒 Using demo mode - Backend unavailable")
            return True
    
    def register(self, name: str, email: str, password: str) -> bool:
        """Register new user"""
        try:
            response = requests.post(
                self.endpoints['auth_register'],
                json={"name": name, "email": email, "password": password},
                timeout=10
            )
            
            if response.status_code == 201:
                st.success("Registration successful! Please login.")
                return True
            else:
                error_msg = response.json().get('detail', 'Registration failed')
                st.error(f"Registration failed: {error_msg}")
                return False
                
        except requests.exceptions.RequestException as e:
            st.error(f"Connection error: {e}")
            return False
    
    def logout(self):
        """Logout user"""
        st.session_state.user_data = None
        st.success("Logged out successfully!")
