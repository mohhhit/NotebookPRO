"""
Chat management utilities for NotebookPRO.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import config


class ChatManager:
    """Manage chat sessions and history."""
    
    def __init__(self):
        self.chats_dir = config.CHATS_DIR
        self.chats_dir.mkdir(parents=True, exist_ok=True)
    
    def save_chat(self, chat_id: str, messages: List[Dict], space: Optional[str] = None) -> None:
        """
        Save a chat session.
        
        Args:
            chat_id: Unique chat identifier
            messages: List of message dictionaries
            space: Optional space/subject name
        """
        chat_data = {
            'id': chat_id,
            'messages': messages,
            'space': space,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }
        
        chat_file = self.chats_dir / f"{chat_id}.json"
        with open(chat_file, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, indent=2, ensure_ascii=False)
    
    def load_chat(self, chat_id: str) -> Optional[Dict]:
        """
        Load a chat session.
        
        Args:
            chat_id: Unique chat identifier
            
        Returns:
            Chat data dictionary or None if not found
        """
        chat_file = self.chats_dir / f"{chat_id}.json"
        
        if not chat_file.exists():
            return None
        
        with open(chat_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def list_chats(self, space: Optional[str] = None) -> List[Dict]:
        """
        List all chats, optionally filtered by space.
        
        Args:
            space: Optional space filter
            
        Returns:
            List of chat metadata dictionaries
        """
        chats = []
        
        for chat_file in self.chats_dir.glob("*.json"):
            with open(chat_file, 'r', encoding='utf-8') as f:
                chat_data = json.load(f)
                
                if space is None or chat_data.get('space') == space:
                    chats.append({
                        'id': chat_data['id'],
                        'space': chat_data.get('space'),
                        'message_count': len(chat_data['messages']),
                        'created_at': chat_data.get('created_at'),
                        'updated_at': chat_data.get('updated_at')
                    })
        
        # Sort by updated time (most recent first)
        chats.sort(key=lambda x: x.get('updated_at', ''), reverse=True)
        
        return chats
    
    def delete_chat(self, chat_id: str) -> bool:
        """
        Delete a chat session.
        
        Args:
            chat_id: Unique chat identifier
            
        Returns:
            True if deleted, False if not found
        """
        chat_file = self.chats_dir / f"{chat_id}.json"
        
        if chat_file.exists():
            chat_file.unlink()
            return True
        
        return False
    
    def get_chat_preview(self, chat_id: str, max_messages: int = 5) -> Optional[List[Dict]]:
        """
        Get a preview of recent messages from a chat.
        
        Args:
            chat_id: Unique chat identifier
            max_messages: Maximum number of messages to return
            
        Returns:
            List of recent messages or None if chat not found
        """
        chat_data = self.load_chat(chat_id)
        
        if chat_data is None:
            return None
        
        messages = chat_data['messages']
        return messages[-max_messages:] if len(messages) > max_messages else messages
