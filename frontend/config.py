import os
import yaml
from typing import Dict, Any

class Config:
    def __init__(self):
        self.config_data = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file or environment variables"""
        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        
        # Default configuration
        return {
            'app': {
                'name': 'SignSpeak.AI',
                'version': '2025.1.0',
                'debug': True
            },
            'backend': {
                'base_url': os.getenv('BACKEND_URL', 'http://localhost:8000'),
                'timeout': 30
            },
            'features': {
                'camera_enabled': True,
                'audio_enabled': True,
                'demo_mode': True
            },
            'ui': {
                'theme': 'dark',
                'primary_color': '#7c3aed',
                'secondary_color': '#06d6a0'
            }
        }
    
    def get_backend_endpoints(self) -> Dict[str, str]:
        """Get backend API endpoints"""
        base_url = self.config_data['backend']['base_url']
        return {
            'auth_login': f"{base_url}/api/auth/login",
            'auth_register': f"{base_url}/api/auth/register",
            'recognize': f"{base_url}/api/recognize",
            'text_to_speech': f"{base_url}/api/tts",
            'text_to_gesture': f"{base_url}/api/ttg",
            'health': f"{base_url}/api/health"
        }
    
    def get(self, key: str, default=None):
        """Get configuration value by key"""
        keys = key.split('.')
        value = self.config_data
        for k in keys:
            value = value.get(k, {})
        return value if value != {} else default

# Global configuration instance
_config_instance = None

def get_config():
    """Get global configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance
