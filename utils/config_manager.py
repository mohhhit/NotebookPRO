"""
Configuration manager for persistent settings (API keys, preferences).
"""

import json
from pathlib import Path
from typing import Dict, Optional
import config


class ConfigManager:
    """Manages persistent user configuration."""
    
    def __init__(self):
        self.config_file = config.DATA_DIR / "user_config.json"
        self.config_data = self._load_config()
    
    def _load_config(self) -> Dict:
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return self._default_config()
        return self._default_config()
    
    def _default_config(self) -> Dict:
        """Default configuration."""
        return {
            "api_keys": {
                "groq": "",
                "gemini": ""
            },
            "preferences": {
                "llm_provider": "groq",
                "temperature": 0.7,
                "workflow": "Auto-Detect"
            },
            "current_space": "General"
        }
    
    def save_config(self):
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(self.config_data, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def get_api_key(self, provider: str) -> str:
        """Get API key for provider."""
        return self.config_data.get("api_keys", {}).get(provider, "")
    
    def set_api_key(self, provider: str, api_key: str):
        """Save API key for provider."""
        if "api_keys" not in self.config_data:
            self.config_data["api_keys"] = {}
        self.config_data["api_keys"][provider] = api_key
        self.save_config()
    
    def get_preference(self, key: str, default=None):
        """Get user preference."""
        return self.config_data.get("preferences", {}).get(key, default)
    
    def set_preference(self, key: str, value):
        """Save user preference."""
        if "preferences" not in self.config_data:
            self.config_data["preferences"] = {}
        self.config_data["preferences"][key] = value
        self.save_config()
    
    def get_current_space(self) -> str:
        """Get current workspace."""
        return self.config_data.get("current_space", "General")
    
    def set_current_space(self, space_name: str):
        """Set current workspace."""
        self.config_data["current_space"] = space_name
        self.save_config()
