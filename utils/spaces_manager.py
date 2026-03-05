"""
Spaces (Workspaces) manager for organizing chats and files by subject.
Each space has its own vector DB and chat history.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import config


class SpacesManager:
    """Manages workspaces (Spaces) for organizing study materials by subject."""
    
    def __init__(self):
        self.spaces_file = config.DATA_DIR / "spaces.json"
        self.spaces_data = self._load_spaces()
    
    def _load_spaces(self) -> Dict:
        """Load spaces from file."""
        if self.spaces_file.exists():
            try:
                with open(self.spaces_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return self._create_default_spaces()
        return self._create_default_spaces()
    
    def _create_default_spaces(self) -> Dict:
        """Create default spaces structure."""
        return {
            "spaces": [
                {
                    "id": "general",
                    "name": "General",
                    "description": "General study materials",
                    "created_at": datetime.now().isoformat(),
                    "file_count": 0,
                    "chat_count": 0
                }
            ]
        }
    
    def save_spaces(self):
        """Save spaces to file."""
        try:
            with open(self.spaces_file, 'w', encoding='utf-8') as f:
                json.dump(self.spaces_data, f, indent=2)
        except Exception as e:
            print(f"Error saving spaces: {e}")
    
    def get_all_spaces(self) -> List[Dict]:
        """Get all spaces."""
        return self.spaces_data.get("spaces", [])
    
    def get_space(self, space_id: str) -> Optional[Dict]:
        """Get specific space by ID."""
        for space in self.spaces_data.get("spaces", []):
            if space["id"] == space_id:
                return space
        return None
    
    def create_space(self, name: str, description: str = "") -> Dict:
        """Create a new space."""
        space_id = name.lower().replace(" ", "_")
        
        # Check if space already exists
        if self.get_space(space_id):
            raise ValueError(f"Space '{name}' already exists")
        
        new_space = {
            "id": space_id,
            "name": name,
            "description": description,
            "created_at": datetime.now().isoformat(),
            "file_count": 0,
            "chat_count": 0
        }
        
        self.spaces_data["spaces"].append(new_space)
        self.save_spaces()
        
        # Create dedicated directories for this space
        space_dir = config.DATA_DIR / "spaces" / space_id
        space_dir.mkdir(parents=True, exist_ok=True)
        (space_dir / "chats").mkdir(exist_ok=True)
        (space_dir / "vector_db").mkdir(exist_ok=True)
        (space_dir / "uploads").mkdir(exist_ok=True)
        
        return new_space
    
    def delete_space(self, space_id: str):
        """Delete a space (except General)."""
        if space_id == "general":
            raise ValueError("Cannot delete General space")
        
        self.spaces_data["spaces"] = [
            s for s in self.spaces_data["spaces"] 
            if s["id"] != space_id
        ]
        self.save_spaces()
    
    def update_space_counts(self, space_id: str, file_count: int = None, chat_count: int = None):
        """Update file/chat counts for a space."""
        space = self.get_space(space_id)
        if space:
            if file_count is not None:
                space["file_count"] = file_count
            if chat_count is not None:
                space["chat_count"] = chat_count
            self.save_spaces()
    
    def get_space_chats_dir(self, space_id: str) -> Path:
        """Get chats directory for a space."""
        return config.DATA_DIR / "spaces" / space_id / "chats"
    
    def get_space_vector_db_dir(self, space_id: str) -> Path:
        """Get vector DB directory for a space."""
        return config.DATA_DIR / "spaces" / space_id / "vector_db"
    
    def get_space_uploads_dir(self, space_id: str) -> Path:
        """Get uploads directory for a space."""
        return config.DATA_DIR / "spaces" / space_id / "uploads"
