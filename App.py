import streamlit as st
from pathlib import Path
import json
from datetime import datetime
import uuid
import sys
import warnings
import logging
import os
import re

# Suppress ALL warnings globally
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

logging.getLogger().setLevel(logging.ERROR)

# Add project root to path
sys.path.append(str(Path(__file__).parent))

import config
from utils.document_processor import DocumentProcessor
from utils.vector_db import VectorDatabase
from utils.simple_generator import SimpleGenerator
from utils.hybrid_retriever import HybridRetriever
from utils.llm_generator import LLMGenerator
from utils.config_manager import ConfigManager
from utils.spaces_manager import SpacesManager

# Page configuration
st.set_page_config(
    page_title="NotebookPRO",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Exact Gemini UI - Like React Reference
st.markdown("""
<style>
    /* Hide all Streamlit branding */
    #MainMenu, footer, header {visibility: hidden;}
    [data-testid="collapsedControl"] {display: none !important;}
    iframe {display: none !important;}
    
    /* Pure Gemini dark background */
    .stApp {
        background-color: #0d0d0d !important;
        color: #e8eaed !important;
    }
    
    /* Reset container padding */
    .main .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    /* Hide all streamlit default elements */
    .stDeployButton, .stDecoration, .stStatusWidget {display: none !important;}
    
    /* === SIDEBAR === */
    [data-testid="stSidebar"] {
        background-color: #131314 !important;
        border-right: 1px solid #2d2d2d !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #e3e3e3 !important;
    }
    
    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton button {
        background-color: transparent !important;
        border: 1px solid #3c4043 !important;
        border-radius: 20px !important;
        color: #e8eaed !important;
        padding: 9px 20px !important;
        font-size: 13px !important;
        transition: all 0.2s ease !important;
    }
    
    [data-testid="stSidebar"] .stButton button:hover {
        background-color: #2a2b2e !important;
        border-color: #5f6368 !important;
    }
    
    /* Active space button */
    [data-testid="stSidebar"] button[kind="primaryFormSubmit"] {
        background: linear-gradient(135deg, #1e3a5f 0%, #2a4a6f 100%) !important;
        border: 1px solid #4285f4 !important;
        color: #8ab4f8 !important;
        font-weight: 600 !important;
        box-shadow: 0 2px 8px rgba(66, 133, 244, 0.2) !important;
    }
    
    /* Secondary space buttons */
    [data-testid="stSidebar"] button[kind="secondary"] {
        background: #1a1b1e !important;
        border: 1px solid #3c4043 !important;
        color: #9aa0a6 !important;
    }
    
    /* Sidebar section headers */
    .spaces-label {
        color: #8ab4f8;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin: 16px 0 8px 0;
        padding: 0 4px;
    }
    
    .section-header {
        color: #9aa0a6;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin: 16px 0 8px 0;
        padding: 0 8px;
    }
    
    /* Chat list items */
    .chat-item {
        padding: 10px 12px;
        margin: 4px 8px;
        border-radius: 8px;
        cursor: pointer;
        transition: background 0.2s;
        color: #e8eaed;
        font-size: 14px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    
    .chat-item:hover {
        background: #2a2b2e;
    }
    
    .chat-item.active {
        background: #2a2b2e;
        border-left: 3px solid #8ab4f8;
    }
    
    .chat-item-date {
        font-size: 11px;
        color: #9aa0a6;
        margin-top: 2px;
    }
    
    /* === MAIN CONTENT === */
    
    /* Welcome screen - Gemini centered */
    .welcome-section {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 60vh;
        text-align: center;
        padding: 2rem;
    }
    
    .welcome-title {
        font-size: 3.2rem;
        font-weight: 400;
        background: linear-gradient(90deg, #4285f4, #9b72f2, #d96570);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    
    .welcome-subtitle {
        font-size: 1.6rem;
        color: #e8eaed;
        font-weight: 300;
        margin-bottom: 2rem;
    }
    
    /* File chips - compact pills */
    .file-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #1e1f20;
        border: 1px solid #3c4043;
        border-radius: 20px;
        padding: 6px 14px;
        margin: 4px;
        color: #9aa0a6;
        font-size: 12px;
    }
    
    .file-chip-icon {
        color: #8ab4f8;
    }
    
    /* Message bubbles - clean Gemini style */
    .message-container {
        max-width: 800px;
        margin: 0 auto 2rem auto;
        display: flex;
        align-items: flex-start;
        gap: 12px;
        padding: 0 20px;
    }
    
    .message-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 16px;
        flex-shrink: 0;
    }
    
    .user-avatar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .assistant-avatar {
        background: linear-gradient(135deg, #4285f4 0%, #9b72f2 100%);
    }
    
    .message-content {
        flex-grow: 1;
        padding: 8px 0;
    }
    
    .user-message {
        color: #e8eaed;
        font-size: 15px;
        line-height: 1.6;
    }
    
    .assistant-message {
        color: #e8eaed;
        font-size: 15px;
        line-height: 1.7;
    }
    
    .assistant-message strong {
        color: #ffffff;
        font-weight: 600;
    }
    
    .assistant-message h2, .assistant-message h3 {
        color: #8ab4f8;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
        font-weight: 500;
    }
    
    .assistant-message ul, .assistant-message ol {
        margin-left: 1.5rem;
        margin-top: 0.5rem;
    }
    
    .assistant-message li {
        margin-bottom: 0.4rem;
    }
    
    /* === INPUT AREA === */
    
    /* Chat input - rounded Gemini style */
    [data-testid="stChatInput"] {
        background: #1e1f20 !important;
        border: 1px solid #3c4043 !important;
        border-radius: 26px !important;
        padding: 12px 20px !important;
        color: #e8eaed !important;
        font-size: 15px !important;
    }
    
    [data-testid="stChatInput"]:focus {
        border-color: #8ab4f8 !important;
        box-shadow: 0 0 0 2px rgba(138, 180, 248, 0.15) !important;
    }
    
    /* Paperclip upload button */
    button[aria-label="upload_from_chat"] {
        background: #1e1f20 !important;
        border: 1px solid #3c4043 !important;
        border-radius: 50% !important;
        color: #8ab4f8 !important;
        font-size: 18px !important;
        padding: 10px !important;
        width: 44px !important;
        height: 44px !important;
        transition: all 0.2s ease !important;
    }
    
    button[aria-label="upload_from_chat"]:hover {
        background: #292a2d !important;
        border-color: #5f6368 !important;
        color: #a6c5f7 !important;
    }
    
    /* Tools menu buttons */
    button[aria-label^="tool_"] {
        background: transparent !important;
        border: 1px solid #3c4043 !important;
        border-radius: 18px !important;
        color: #e8eaed !important;
        font-size: 13px !important;
        padding: 7px 14px !important;
        transition: all 0.2s ease !important;
    }
    
    button[aria-label^="tool_"]:hover {
        background: #2a2b2e !important;
        border-color: #5f6368 !important;
    }
    
    /* Active tool */
    button[aria-label^="tool_"][kind="primary"] {
        background: #1e3a5f !important;
        border-color: #4285f4 !important;
        color: #8ab4f8 !important;
        font-weight: 500 !important;
    }
    
    /* Compact tools row */
    .tools-menu {
        display: flex;
        gap: 6px;
        align-items: center;
        justify-content: center;
        padding: 8px 0;
    }
    
    /* Document chips row */
    .uploaded-docs-compact {
        display: flex;
        gap: 6px;
        align-items: center;
        flex-wrap: wrap;
        justify-content: center;
        padding: 8px 0;
    }
    
    .doc-chip {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        background: #1e1f20;
        border: 1px solid #3c4043;
        border-radius: 14px;
        padding: 4px 10px;
        color: #9aa0a6;
        font-size: 11px;
        max-width: 140px;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    
    .doc-chip-icon {
        color: #8ab4f8;
        font-size: 11px;
    }
    
    /* === SCROLLBARS === */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1a1a1a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #3c4043;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #5f6368;
    }
    
    /* === MISC === */
    
    /* File upload cards */
    .file-card {
        display: inline-block;
        background: #1e1f20;
        border: 1px solid #3c4043;
        border-radius: 14px;
        padding: 14px 18px;
        margin: 6px;
        color: #e8eaed;
        font-size: 13px;
    }
    
    .file-card-icon {
        color: #8ab4f8;
        margin-right: 6px;
    }
    
    /* Clean spacing */
    .main-content-area {
        padding: 40px 20px;
        max-width: 900px;
        margin: 0 auto;
    }

</style>
""", unsafe_allow_html=True)


def get_current_space_id():
    """Get current space ID correctly formatted."""
    if 'current_space' in st.session_state:
        return st.session_state.current_space.lower().replace(" ", "_")
    return "general"


def load_space_vector_db(space_id: str):
    """Load vector DB for specific space."""
    try:
        # Get space-specific collection name and directory
        collection_name = f"space_{space_id}"
        db_dir = st.session_state.spaces_manager.get_space_vector_db_dir(space_id)
        
        # Create directory if it doesn't exist
        db_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize vector DB with space-specific settings
        vector_db = VectorDatabase(
            collection_name=collection_name,
            persist_directory=str(db_dir)
        )
        
        return vector_db
    except Exception as e:
        print(f"Error loading vector DB for space {space_id}: {e}")
        return None


def load_processed_files_for_space(space_id: str):
    """Load processed files metadata for specific space."""
    files_path = st.session_state.spaces_manager.get_space_uploads_dir(space_id) / "processed_files.json"
    if files_path.exists():
        try:
            with open(files_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_processed_files_for_space(space_id: str, processed_files: dict):
    """Save processed files metadata for specific space."""
    uploads_dir = st.session_state.spaces_manager.get_space_uploads_dir(space_id)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    files_path = uploads_dir / "processed_files.json"
    try:
        with open(files_path, 'w', encoding='utf-8') as f:
            json.dump(processed_files, f, indent=2)
    except Exception as e:
        print(f"Error saving processed files: {e}")


def switch_space(new_space_id: str):
    """Switch to a different space and reload its data."""
    # Save current space setting
    st.session_state.config_manager.set_current_space(new_space_id)
    st.session_state.current_space = new_space_id
    
    # Reload chats for new space
    st.session_state.all_chats = load_all_chats()
    
    # Reload vector DB for new space
    st.session_state.vector_db = load_space_vector_db(get_current_space_id())
    
    # Reload processed files for new space
    st.session_state.processed_files = load_processed_files_for_space(get_current_space_id())
    
    # Rebuild hybrid retriever if we have data
    if st.session_state.vector_db and st.session_state.vector_db.get_collection_count() > 0:
        try:
            all_docs, all_metas = st.session_state.vector_db.get_all_documents()
            st.session_state.hybrid_retriever = HybridRetriever(st.session_state.vector_db)
            st.session_state.hybrid_retriever.index_documents(all_docs)
        except Exception as e:
            print(f"Error rebuilding hybrid retriever: {e}")
            st.session_state.hybrid_retriever = None
    else:
        st.session_state.hybrid_retriever = None
    
    # Start new chat for new space
    st.session_state.current_chat_id = str(uuid.uuid4())
    st.session_state.chat_history = []


def init_session_state():
    """Initialize session state variables."""
    # Initialize managers
    if 'config_manager' not in st.session_state:
        st.session_state.config_manager = ConfigManager()
    
    if 'spaces_manager' not in st.session_state:
        st.session_state.spaces_manager = SpacesManager()
    
    # Current space
    if 'current_space' not in st.session_state:
        st.session_state.current_space = st.session_state.config_manager.get_current_space()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'current_chat_id' not in st.session_state:
        st.session_state.current_chat_id = str(uuid.uuid4())
    
    if 'all_chats' not in st.session_state:
        st.session_state.all_chats = load_all_chats()
    
    if 'model' not in st.session_state:
        st.session_state.model = SimpleGenerator()
    
    # Load API key from config
    if 'llm_provider' not in st.session_state:
        st.session_state.llm_provider = st.session_state.config_manager.get_preference("llm_provider", "groq")
    
    if 'api_key' not in st.session_state:
        st.session_state.api_key = st.session_state.config_manager.get_api_key(st.session_state.llm_provider)
    
    # Initialize LLM if API key exists
    if 'llm_generator' not in st.session_state:
        if st.session_state.api_key:
            st.session_state.llm_generator = LLMGenerator(
                provider=st.session_state.llm_provider,
                api_key=st.session_state.api_key
            )
        else:
            st.session_state.llm_generator = None
    
    if 'doc_processor' not in st.session_state:
        st.session_state.doc_processor = DocumentProcessor()
    
    if 'processed_files' not in st.session_state:
        # Load processed files for current space (not global)
        space_id = get_current_space_id()
        st.session_state.processed_files = load_processed_files_for_space(space_id)
    
    # Initialize space-specific vector DB
    if 'vector_db' not in st.session_state:
        space_id = get_current_space_id()
        st.session_state.vector_db = load_space_vector_db(space_id)
        
        # Build hybrid retriever if we have data
        if st.session_state.vector_db and st.session_state.vector_db.get_collection_count() > 0:
            try:
                all_docs, all_metas = st.session_state.vector_db.get_all_documents()
                st.session_state.hybrid_retriever = HybridRetriever(st.session_state.vector_db)
                st.session_state.hybrid_retriever.index_documents(all_docs)
                doc_count = st.session_state.vector_db.get_collection_count()
                print(f"✓ Loaded space '{st.session_state.current_space}' vector DB with {doc_count} chunks")
            except Exception as e:
                print(f"Warning: Could not build hybrid retriever: {e}")
                st.session_state.hybrid_retriever = None
    
    # Ensure hybrid retriever is initialized
    if 'hybrid_retriever' not in st.session_state:
        st.session_state.hybrid_retriever = None
    
    # Workflow selection (NotebookLM-style specialized prompts)
    if 'workflow' not in st.session_state:
        st.session_state.workflow = "Auto-Detect"
    
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.3  # Lower temperature for stricter document adherence
    
    if 'show_upload' not in st.session_state:
        st.session_state.show_upload = False


def load_all_chats():
    """Load all saved chats from current space."""
    chats = []
    if 'spaces_manager' not in st.session_state or 'current_space' not in st.session_state:
        # Fallback to old location during initialization
        chats_dir = config.CHATS_DIR
    else:
        # Load from current space
        space_id = st.session_state.current_space.lower().replace(" ", "_")
        chats_dir = st.session_state.spaces_manager.get_space_chats_dir(space_id)
        
        # Create directory if it doesn't exist
        chats_dir.mkdir(parents=True, exist_ok=True)
    
    if chats_dir.exists():
        for chat_file in chats_dir.glob("*.json"):
            try:
                with open(chat_file, 'r', encoding='utf-8') as f:
                    chats.append(json.load(f))
            except Exception:
                pass
    return sorted(chats, key=lambda x: x.get('updated_at', x.get('created_at', '')), reverse=True)


def save_chat(chat_id, chat_data):
    """Save chat to current space's directory."""
    # Get current space's chats directory
    space_id = st.session_state.current_space.lower().replace(" ", "_")
    chats_dir = st.session_state.spaces_manager.get_space_chats_dir(space_id)
    chats_dir.mkdir(parents=True, exist_ok=True)
    
    chat_file = chats_dir / f"{chat_id}.json"
    try:
        with open(chat_file, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, indent=2)
    except Exception as e:
        st.error(f"Error saving chat: {e}")


def delete_chat(chat_id: str):
    """Delete a chat from current space."""
    space_id = st.session_state.current_space.lower().replace(" ", "_")
    chats_dir = st.session_state.spaces_manager.get_space_chats_dir(space_id)
    chat_file = chats_dir / f"{chat_id}.json"
    
    if chat_file.exists():
        try:
            chat_file.unlink()
            return True
        except Exception as e:
            print(f"Error deleting chat: {e}")
            return False
    return False


def delete_space_data(space_name: str):
    """Delete a space and all its data."""
    space_id = space_name.lower().replace(" ", "_")
    
    if space_id == "general":
        return False, "Cannot delete General space"
    
    try:
        # Delete from spaces manager
        st.session_state.spaces_manager.delete_space(space_id)
        
        # If we were in this space, switch to General
        if st.session_state.current_space == space_name:
            switch_space("General")
        
        return True, "Space deleted successfully"
    except Exception as e:
        return False, str(e)


def render_sidebar():
    """Render sidebar with Spaces and ChatGPT-style chat list."""
    with st.sidebar:
        st.markdown("# 📚 NotebookPRO")
        st.markdown("")
        
        # New chat button
        if st.button("✨ New Chat", key="new_chat", use_container_width=True):
            st.session_state.current_chat_id = str(uuid.uuid4())
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown("")
        
        # Spaces selector (ChatGPT-style workspaces) with distinct styling
        st.markdown("<div class='spaces-label'>🗂️ WORKSPACES</div>", unsafe_allow_html=True)
        
        spaces = st.session_state.spaces_manager.get_all_spaces()
        
        # Render space buttons in rows of 2
        for i in range(0, len(spaces), 2):
            cols = st.columns(2)
            for idx, space in enumerate(spaces[i:i+2]):
                with cols[idx]:
                    space_active = space["name"] == st.session_state.current_space
                    if st.button(
                        f"📁 {space['name']}",
                        key=f"space_{space['id']}",
                        use_container_width=True,
                        type="primary" if space_active else "secondary"
                    ):
                        if not space_active:
                            # Switch to new space (reloads vector DB, chats, files)
                            switch_space(space["name"])
                            st.rerun()
        
        # New space button with distinct style
        if st.button("➕ New Workspace", key="new_space", use_container_width=True, type="secondary"):
            st.session_state.show_new_space_form = True
        
        # New space form
        if st.session_state.get('show_new_space_form', False):
            with st.form("new_space_form"):
                space_name = st.text_input("Space Name", placeholder="e.g., Machine Learning")
                space_desc = st.text_input("Description (optional)", placeholder="Study materials for ML course")
                submitted = st.form_submit_button("Create Space")
                
                if submitted and space_name:
                    try:
                        st.session_state.spaces_manager.create_space(space_name, space_desc)
                        switch_space(space_name)
                        st.session_state.show_new_space_form = False
                        st.success(f"✓ Created space: {space_name}")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
        
        # Delete current space option (if not General)
        if st.session_state.current_space != "General":
            with st.expander("⚙️ Manage Workspace"):
                st.markdown(f"**Current:** {st.session_state.current_space}")
                if st.button("🗑️ Delete This Workspace", key="delete_space_btn", type="secondary", use_container_width=True):
                    success, message = delete_space_data(st.session_state.current_space)
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
        
        st.markdown("")
        
        # Upload files
        if st.button("📎 Upload Files", key="upload_toggle", use_container_width=True):
            st.session_state.show_upload = not st.session_state.show_upload
        
        st.markdown("")
        
        # LLM Provider
        st.markdown("<div style='color: #9aa0a6; font-size: 12px; margin-bottom: 4px;'>🤖 AI Model</div>", unsafe_allow_html=True)
        provider = st.selectbox(
            "Provider",
            options=["Groq (Llama-3.3-70B) - Recommended", "Google Gemini 1.5 Flash", "Fast Mode (No AI)"],
            index=0,
            key="provider_selector",
            label_visibility="collapsed"
        )
        
        if "Groq" in provider:
            st.session_state.llm_provider = "groq"
        elif "Gemini" in provider:
            st.session_state.llm_provider = "gemini"
        else:
            st.session_state.llm_provider = "none"
        
        # Save provider preference
        st.session_state.config_manager.set_preference("llm_provider", st.session_state.llm_provider)
        
        # API Key input
        if st.session_state.llm_provider != "none":
            st.markdown("<div style='color: #9aa0a6; font-size: 12px; margin-bottom: 4px; margin-top: 8px;'>🔑 API Key</div>", unsafe_allow_html=True)
            api_key = st.text_input(
                "API Key",
                value=st.session_state.api_key,
                type="password",
                key="api_key_input",
                label_visibility="collapsed",
                placeholder=f"Enter {st.session_state.llm_provider.upper()} API key (saved automatically)"
            )
            
            if api_key != st.session_state.api_key:
                st.session_state.api_key = api_key
                # Save API key to config
                st.session_state.config_manager.set_api_key(st.session_state.llm_provider, api_key)
                
                # Reinitialize LLM
                if api_key:
                    st.session_state.llm_generator = LLMGenerator(
                        provider=st.session_state.llm_provider,
                        api_key=api_key
                    )
                else:
                    st.session_state.llm_generator = None
            
            # LLM status
            if st.session_state.llm_generator and st.session_state.llm_generator.is_ready():
                st.markdown(f"""<div style='background: #1e3a1e; border-radius: 8px; padding: 8px 12px; margin: 8px 0; text-align: center;'>
                    <span style='color: #81c995; font-size: 12px;'>✓ {st.session_state.llm_generator.get_provider()} Ready</span>
                </div>""", unsafe_allow_html=True)
            elif api_key:
                st.markdown("""<div style='background: #3a1e1e; border-radius: 8px; padding: 8px 12px; margin: 8px 0; text-align: center;'>
                    <span style='color: #f28b82; font-size: 12px;'>⚠ Invalid API Key</span>
                </div>""", unsafe_allow_html=True)
        
        st.markdown("")
        
        # Response Mode
        st.markdown("<div style='color: #9aa0a6; font-size: 12px; margin-bottom: 4px;'>⚙️ Response Mode</div>", unsafe_allow_html=True)
        st.session_state.workflow = st.selectbox(
            "Mode",
            options=["Auto-Detect", "Explain Concept", "Summarize", "Answer Question", "Study Notes"],
            index=0,
            key="workflow_selector",
            label_visibility="collapsed"
        )
        
        # File count badge
        total_files = sum(len(files) for files in st.session_state.processed_files.values())
        if total_files > 0:
            chunk_count = 0
            if st.session_state.vector_db:
                try:
                    chunk_count = st.session_state.vector_db.get_collection_count()
                except Exception:
                    chunk_count = 0
            
            st.markdown(f"""
                <div style='background: #1e1f20; border-radius: 8px; padding: 8px 12px; margin: 8px 0; text-align: center;'>
                    <span style='color: #8ab4f8; font-size: 13px;'>📚 {total_files} files • {chunk_count} chunks</span>
                </div>
            """, unsafe_allow_html=True)
            
            # Clear files button (space-specific)
            if st.button("🗑️ Clear Files", key="clear_files", use_container_width=True):
                space_id = get_current_space_id()
                st.session_state.vector_db = load_space_vector_db(space_id)  # Reset to empty DB
                st.session_state.hybrid_retriever = None
                st.session_state.processed_files = {}
                save_processed_files_for_space(space_id, {})
                st.success("✓ Files cleared from this space!")
                st.rerun()
        
        # ChatGPT-style chat list
        st.markdown("<div class='section-header'>Chats in " + st.session_state.current_space + "</div>", unsafe_allow_html=True)
        
        # Render chats with delete buttons
        for chat in st.session_state.all_chats[:20]:
            chat_id = chat.get('id', '')
            if not chat_id:
                continue
            
            chat_title = chat['messages'][0]['content'][:40] if chat.get('messages') else "Untitled Chat"
            is_current = (chat_id == st.session_state.current_chat_id)
            
            col1, col2 = st.columns([5, 1])
            with col1:
                if st.button(
                    f"💬 {chat_title}...",
                    key=f"chat_{chat_id}",
                    use_container_width=True,
                    type="primary" if is_current else "secondary"
                ):
                    st.session_state.current_chat_id = chat_id
                    st.session_state.chat_history = chat.get('messages', [])
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"delete_chat_{chat_id}"):
                    delete_chat(chat_id)
                    st.session_state.all_chats = load_all_chats()
                    st.rerun()


def render_upload_modal():
    """Render Gemini-style file upload."""
    if st.session_state.show_upload:
        st.markdown("""
            <div style='text-align: center; padding: 2rem 0;'>
                <h3 style='color: #e8eaed; font-weight: 300;'>📎 Upload Study Materials</h3>
                <p style='color: #9aa0a6; font-size: 14px;'>PDF, DOCX, or TXT files</p>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Drop files here",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            key="file_uploader",
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            # Show uploaded files as cards
            files_html = '<div style="text-align: center; margin: 1rem 0;">'
            for f in uploaded_files:
                files_html += f'<span class="file-card"><span class="file-card-icon">📄</span>{f.name}</span>'
            files_html += '</div>'
            st.markdown(files_html, unsafe_allow_html=True)
            
            if st.button("✨ Process Files", type="primary", use_container_width=True):
                process_files(uploaded_files)
                st.session_state.show_upload = False
                st.rerun()


def process_files(uploaded_files):
    """Process uploaded files and add to vector DB (space-specific)."""
    progress = st.progress(0)
    status = st.empty()
    
    try:
        # Get current space ID
        space_id = get_current_space_id()
        
        # Initialize vector DB for current space
        if st.session_state.vector_db is None:
            status.info("Initializing database...")
            st.session_state.vector_db = load_space_vector_db(space_id)
        
        # Get space-specific uploads directory
        uploads_dir = st.session_state.spaces_manager.get_space_uploads_dir(space_id)
        uploads_dir.mkdir(parents=True, exist_ok=True)
        
        for idx, file in enumerate(uploaded_files):
            status.info(f"Processing {file.name}...")
            
            # Save file to SPACE-SPECIFIC uploads directory
            file_path = uploads_dir / file.name
            with open(file_path, 'wb') as f:
                f.write(file.getbuffer())
            
            # Extract text
            doc_data = st.session_state.doc_processor.process_file(file_path)
            
            if not doc_data['content'] or len(doc_data['content']) < 50:
                st.warning(f"⚠️ {file.name} has very little content")
                continue
            
            # Chunk text with SEMANTIC chunking (NotebookLM-style)
            chunks = st.session_state.doc_processor.chunk_text(
                doc_data['content'],
                chunk_size=config.CHUNK_SIZE,
                overlap=config.CHUNK_OVERLAP,
                semantic=True  # Smart chunking by headers/concepts
            )
            
            # Add to vector DB
            metadatas = [{'filename': doc_data['filename'], 'chunk_index': i} for i in range(len(chunks))]
            ids = [f"{doc_data['filename']}_{i}_{datetime.now().timestamp()}" for i in range(len(chunks))]
            
            st.session_state.vector_db.add_documents(chunks, metadatas, ids)
            
            # Initialize/update hybrid retriever (combines Vector + BM25)
            if st.session_state.hybrid_retriever is None:
                st.session_state.hybrid_retriever = HybridRetriever(
                    st.session_state.vector_db,
                    alpha=0.6  # 60% semantic, 40% keyword
                )
            
            # Re-index all documents for BM25
            all_docs, all_metas = st.session_state.vector_db.get_all_documents()
            st.session_state.hybrid_retriever.index_documents(all_docs, all_metas)
            
            # Track file for CURRENT SPACE
            space_name = st.session_state.current_space
            if space_name not in st.session_state.processed_files:
                st.session_state.processed_files[space_name] = []
            
            st.session_state.processed_files[space_name].append({
                'name': doc_data['filename'],
                'chunks': len(chunks),
                'uploaded_at': datetime.now().isoformat()
            })
            
            progress.progress((idx + 1) / len(uploaded_files))
        
        # Save processed files for CURRENT SPACE
        save_processed_files_for_space(space_id, st.session_state.processed_files)
        status.success(f"✅ Processed {len(uploaded_files)} files!")
        
    except Exception as e:
        status.error(f"Error: {str(e)}")


def detect_use_case(prompt: str) -> str:
    """Detect use case from prompt."""
    prompt_lower = prompt.lower()
    
    if any(word in prompt_lower for word in ['explain', 'what is', 'how does', 'why']):
        return 'explanation'
    elif any(word in prompt_lower for word in ['summarize', 'summary', 'brief']):
        return 'summary'
    elif any(word in prompt_lower for word in ['notes', 'bullet', 'list']):
        return 'notes'
    else:
        return 'qa'


def generate_response(query: str) -> str:
    """Generate NotebookLM-quality response with HYBRID retrieval and strict citations."""
    # Check if vector DB is loaded
    if st.session_state.vector_db is None:
        return ("I don't have access to your study materials right now. "
                "Please upload some documents first by clicking '📎 Upload Files' in the sidebar.")
    
    # HYBRID RETRIEVAL: Combine Vector (semantic) + BM25 (keyword)
    context = ""
    sources = []
    metadatas_list = []
    
    try:
        if st.session_state.hybrid_retriever:
            # Use HYBRID search (NotebookLM-style)
            documents, metadatas, scores = st.session_state.hybrid_retriever.retrieve(
                query, 
                n_results=7,  # Fetch more for better quality
                score_threshold=0.1
            )
            
            if documents:
                context = "\n\n".join(documents)
                metadatas_list = metadatas
                
                # Extract unique sources
                for meta in metadatas:
                    if 'filename' in meta and meta['filename'] not in sources:
                        sources.append(meta['filename'])
            else:
                return ("I couldn't find relevant information in your uploaded documents. "
                        "Try uploading more study materials or rephrasing your question.")
        else:
            # Fallback to pure vector search
            results = st.session_state.vector_db.query(query, n_results=7)
            if results and 'documents' in results and results['documents'] and len(results['documents'][0]) > 0:
                context = "\n\n".join(results['documents'][0])
                metadatas_list = results['metadatas'][0] if 'metadatas' in results else []
                
                for metadata in metadatas_list:
                    if 'filename' in metadata and metadata['filename'] not in sources:
                        sources.append(metadata['filename'])
            else:
                return ("I couldn't find relevant information in your uploaded documents. "
                        "Try uploading more study materials or rephrasing your question.")
    except Exception as e:
        return (f"I encountered an issue searching your documents. "
                f"Please try uploading your files again. (Error: {str(e)})")
    
    # Determine workflow (user selection or auto-detect)
    if st.session_state.workflow == "Auto-Detect":
        use_case = detect_use_case(query)
    else:
        workflow_map = {
            "Explain Concept": "explanation",
            "Summarize": "summary",
            "Answer Question": "qa",
            "Study Notes": "notes"
        }
        use_case = workflow_map.get(st.session_state.workflow, "explanation")
    
    try:
        # Choose generator: LLM (if available) or SimpleGenerator (fallback)
        if st.session_state.llm_generator and st.session_state.llm_generator.is_ready():
            # Use ACTUAL LLM (NotebookLM-quality)
            response = st.session_state.llm_generator.generate_response(
                prompt=query,
                context=context,
                use_case=use_case,
                metadatas=metadatas_list,
                temperature=st.session_state.temperature,
                max_tokens=config.MAX_TOKENS
            )
        else:
            # Fallback to text extraction (Fast Mode)
            response = st.session_state.model.generate_response(
                prompt=query,
                context=context,
                use_case=use_case,
                metadatas=metadatas_list,
                temperature=st.session_state.temperature,
                max_tokens=config.MAX_TOKENS
            )
        
        # Add source footer (NotebookLM style)
        if sources:
            source_text = ", ".join([f"**{s}**" for s in sources[:4]])
            response += f"\n\n<div style='color: #8ab4f8; font-size: 13px; margin-top: 1rem;'>📚 Sources: {source_text}</div>"
        
        return response
    except Exception as e:
        return f"Sorry, I encountered an error generating a response: {str(e)}"


def render_messages():
    """Render Gemini-style chat messages with proper markdown formatting."""
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(f"""
                <div class="message-container">
                    <div class="message-avatar user-avatar">👤</div>
                    <div class="message-content">
                        <div class="user-message">{message['content']}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            content = message['content']
            
            # Convert markdown to HTML for better formatting
            # Bold text
            content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
            
            # Headers (must come before line breaks)
            content = re.sub(r'### (.+?)\n', r'<h3 style="color: #e8eaed; margin-top: 1.5rem; margin-bottom: 0.5rem;">\1</h3>', content)
            content = re.sub(r'## (.+?)\n', r'<h2 style="color: #e8eaed; margin-top: 1.5rem; margin-bottom: 0.5rem;">\1</h2>', content)
            
            # Numbered lists - preserve numbering
            content = re.sub(r'\n(\d+)\.\s+', r'<br><br><strong>\1.</strong> ', content)
            
            # Bullet points
            content = re.sub(r'\n•\s+', r'<br>&nbsp;&nbsp;• ', content)
            
            # Double line breaks for paragraph spacing
            content = content.replace('\n\n', '<br><br>')
            # Single line breaks become spaces
            content = content.replace('\n', ' ')
            
            st.markdown(f"""
                <div class="message-container">
                    <div class="message-avatar assistant-avatar">🤖</div>
                    <div class="message-content">
                        <div class="assistant-message">{content}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)


def main():
    """Main application with Gemini-inspired UI."""
    init_session_state()
    render_sidebar()
    
    # Main content area
    col1, col2, col3 = st.columns([1, 8, 1])
    
    with col2:
        # Upload modal
        render_upload_modal()
        
        # Welcome screen (Gemini-style) or chat
        if not st.session_state.chat_history:
            st.markdown("""
                <div class="welcome-section">
                    <div class="welcome-title">Hi there 👋</div>
                    <div class="welcome-subtitle">What would you like to know?</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Show uploaded files if any (horizontal card layout)
            total_files = sum(len(files) for files in st.session_state.processed_files.values())
            if total_files > 0:
                st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
                files_html = '<div style="display: flex; gap: 12px; flex-wrap: wrap; justify-content: center; max-width: 800px; margin: 0 auto;">'
                for space, files in st.session_state.processed_files.items():
                    for f in files:
                        # Clean and truncate filename
                        clean_name = f["name"]
                        display_name = clean_name[:30] + "..." if len(clean_name) > 30 else clean_name
                        files_html += f'<div class="file-card" style="min-width: 200px;"><span class="file-card-icon">📄</span><span style="display: inline-block; max-width: 180px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">{display_name}</span></div>'
                files_html += '</div>'
                st.markdown(files_html, unsafe_allow_html=True)
        else:
            render_messages()
    
    # Chat input area - Gemini style at bottom
    st.markdown("<div style='height: 80px;'></div>", unsafe_allow_html=True)  # Spacer
    
    # Bottom section with tools and chat input
    col_left, col_center, col_right = st.columns([1, 8, 1])
    with col_center:
        # Tools menu (Gemini-style)
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        tools_col1, tools_col2, tools_col3, tools_col4 = st.columns([1, 1, 1, 1])
        
        with tools_col1:
            if st.button("💬 Chat", key="tool_chat", use_container_width=True, 
                        type="primary" if st.session_state.workflow == "Auto-Detect" else "secondary"):
                st.session_state.workflow = "Auto-Detect"
                st.rerun()
        with tools_col2:
            if st.button("📝 Summarize", key="tool_summarize", use_container_width=True,
                        type="primary" if st.session_state.workflow == "Summarize" else "secondary"):
                st.session_state.workflow = "Summarize"
                st.rerun()
        with tools_col3:
            if st.button("💡 Explain", key="tool_explain", use_container_width=True,
                        type="primary" if st.session_state.workflow == "Explain Concept" else "secondary"):
                st.session_state.workflow = "Explain Concept"
                st.rerun()
        with tools_col4:
            if st.button("📚 Study Notes", key="tool_notes", use_container_width=True,
                        type="primary" if st.session_state.workflow == "Study Notes" else "secondary"):
                st.session_state.workflow = "Study Notes"
                st.rerun()
        
        # Compact uploaded documents display
        total_files = sum(len(files) for files in st.session_state.processed_files.values())
        if total_files > 0:
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            docs_html = '<div style="text-align: center; display: flex; gap: 6px; flex-wrap: wrap; justify-content: center;">'
            docs_html += f'<span style="color: #9aa0a6; font-size: 11px; padding: 4px 8px;">📎 {total_files} files:</span>'
            for space, files in st.session_state.processed_files.items():
                for f in files[:5]:  # Show max 5 files
                    doc_name = f["name"][:20] + "..." if len(f["name"]) > 20 else f["name"]
                    docs_html += f'<span class="doc-chip"><span class="doc-chip-icon">📄</span>{doc_name}</span>'
            if total_files > 5:
                docs_html += f'<span class="doc-chip">+{total_files - 5} more</span>'
            docs_html += '</div>'
            st.markdown(docs_html, unsafe_allow_html=True)
        
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        
        # Chat input with paperclip upload button
        input_col1, input_col2 = st.columns([0.5, 11])
        
        with input_col1:
            # Paperclip button for file upload
            if st.button("📎", key="upload_from_chat", help="Upload files", use_container_width=True):
                st.session_state.show_upload = True
                st.rerun()
        
        with input_col2:
            user_input = st.chat_input("Ask me anything about your materials...")
    
    # Store user_input in broader scope
    if 'user_input' not in locals():
        user_input = None
    else:
        # Process the input outside the columns
        pass
    
    # Process user input (moved outside columns context)
    if user_input:
        # Add user message
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        })
        
        # Generate response
        with st.spinner("✨ Thinking..."):
            response = generate_response(user_input)
        
        # Add assistant message
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now().isoformat()
        })
        
        # Save chat
        save_chat(st.session_state.current_chat_id, {
            'id': st.session_state.current_chat_id,
            'messages': st.session_state.chat_history,
            'created_at': st.session_state.chat_history[0]['timestamp'],
            'updated_at': datetime.now().isoformat()
        })
        
        # Reload chats
        st.session_state.all_chats = load_all_chats()
        
        st.rerun()


if __name__ == "__main__":
    main()
