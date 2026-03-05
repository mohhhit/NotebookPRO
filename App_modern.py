import streamlit as st
from pathlib import Path
import json
from datetime import datetime
import uuid
import sys
import warnings
import logging
import os

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

# Page configuration
st.set_page_config(
    page_title="NotebookPRO",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern CSS styling (ChatGPT/Claude-like)
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Clean modern layout */
    .stApp {
        background-color: #f7f7f8;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #202123;
        padding-top: 1rem;
    }
    
    [data-testid="stSidebar"] * {
        color: #ececec !important;
    }
    
    /* Chat container */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem 1rem;
    }
    
    /* Message bubbles */
    .message {
        margin-bottom: 1.5rem;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        line-height: 1.6;
    }
    
    .user-message {
        background-color: #8B5CF6;
        color: white;
        margin-left: 20%;
    }
    
    .assistant-message {
        background-color: white;
        color: #1f1f1f;
        border: 1px solid #e5e5e5;
        margin-right: 20%;
    }
    
    /* Input area */
    .stChatInput {
        position: fixed;
        bottom: 0;
        max-width: 800px;
        background: white;
        padding: 1rem;
    }
    
    /* Sidebar buttons */
    .sidebar-button {
        background-color: transparent;
        border: 1px solid #4a4a4a;
        color: #ececec;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        cursor: pointer;
        margin: 0.5rem 0;
        width: 100%;
        text-align: left;
    }
    
    .sidebar-button:hover {
        background-color: #2a2b32;
    }
    
    /* Top header */
    .top-header {
        background-color: white;
        border-bottom: 1px solid #e5e5e5;
        padding: 1rem 2rem;
        position: sticky;
        top: 0;
        z-index: 100;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'current_chat_id' not in st.session_state:
        st.session_state.current_chat_id = str(uuid.uuid4())
    
    if 'all_chats' not in st.session_state:
        st.session_state.all_chats = load_all_chats()
    
    if 'model' not in st.session_state:
        st.session_state.model = SimpleGenerator()
    
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = None
    
    if 'doc_processor' not in st.session_state:
        st.session_state.doc_processor = DocumentProcessor()
    
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = load_processed_files()
    
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.7
    
    if 'show_upload' not in st.session_state:
        st.session_state.show_upload = False


def load_all_chats():
    """Load all saved chats from disk."""
    chats = []
    if config.CHATS_DIR.exists():
        for chat_file in config.CHATS_DIR.glob("*.json"):
            try:
                with open(chat_file, 'r', encoding='utf-8') as f:
                    chats.append(json.load(f))
            except Exception:
                pass
    return sorted(chats, key=lambda x: x.get('updated_at', x.get('created_at', '')), reverse=True)


def load_processed_files():
    """Load processed files from disk."""
    files_path = config.DATA_DIR / "processed_files.json"
    if files_path.exists():
        try:
            with open(files_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_processed_files(files_data):
    """Save processed files to disk."""
    files_path = config.DATA_DIR / "processed_files.json"
    try:
        with open(files_path, 'w', encoding='utf-8') as f:
            json.dump(files_data, f, indent=2)
    except Exception as e:
        st.error(f"Error saving files: {e}")


def save_chat(chat_id, chat_data):
    """Save chat to disk."""
    chat_file = config.CHATS_DIR / f"{chat_id}.json"
    try:
        with open(chat_file, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, indent=2)
    except Exception as e:
        st.error(f"Error saving chat: {e}")


def render_sidebar():
    """Render modern sidebar with chat history."""
    with st.sidebar:
        st.markdown("# 📚 NotebookPRO")
        
        # New chat button
        if st.button("➕ New Chat", key="new_chat", use_container_width=True):
            st.session_state.current_chat_id = str(uuid.uuid4())
            st.session_state.chat_history = []
            st.rerun()
        
        st.markdown("---")
        
        # Upload files toggle
        if st.button("📤 Upload Files", key="upload_toggle", use_container_width=True):
            st.session_state.show_upload = not st.session_state.show_upload
        
        # Show file count
        total_files = sum(len(files) for files in st.session_state.processed_files.values())
        st.caption(f"📁 {total_files} files uploaded")
        
        st.markdown("---")
        st.markdown("### Recent Chats")
        
        # List recent chats
        for chat in st.session_state.all_chats[:10]:
            chat_id = chat.get('id')
            messages = chat.get('messages', [])
            
            if messages:
                # Get first user message as title
                first_msg = next((m['content'][:50] + "..." for m in messages if m['role'] == 'user'), "New Chat")
                
                # Create chat button
                if st.button(
                    first_msg,
                    key=f"chat_{chat_id}",
                    use_container_width=True,
                    help=f"Created: {chat.get('created_at', '')[:16]}"
                ):
                    st.session_state.current_chat_id = chat_id
                    st.session_state.chat_history = chat['messages'].copy()
                    
                    # Reinitialize vector DB
                    try:
                        st.session_state.vector_db = VectorDatabase(collection_name="documents")
                    except Exception:
                        pass
                    
                    st.rerun()


def render_upload_modal():
    """Render file upload modal."""
    if st.session_state.show_upload:
        st.markdown("### 📤 Upload Study Materials")
        
        uploaded_files = st.file_uploader(
            "Upload PDFs, DOCX, or TXT files",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        if uploaded_files and st.button("Process Files", type="primary"):
            process_files(uploaded_files)
            st.session_state.show_upload = False
            st.rerun()


def process_files(uploaded_files):
    """Process uploaded files and add to vector DB."""
    progress = st.progress(0)
    status = st.empty()
    
    try:
        # Initialize vector DB
        if st.session_state.vector_db is None:
            status.info("Initializing database...")
            st.session_state.vector_db = VectorDatabase(collection_name="documents")
        
        for idx, file in enumerate(uploaded_files):
            status.info(f"Processing {file.name}...")
            
            # Save file
            file_path = config.UPLOADS_DIR / file.name
            with open(file_path, 'wb') as f:
                f.write(file.getbuffer())
            
            # Extract text
            doc_data = st.session_state.doc_processor.process_file(file_path)
            
            if not doc_data['content'] or len(doc_data['content']) < 50:
                st.warning(f"⚠️ {file.name} has very little content")
                continue
            
            # Chunk text
            chunks = st.session_state.doc_processor.chunk_text(
                doc_data['content'],
                chunk_size=config.CHUNK_SIZE,
                overlap=config.CHUNK_OVERLAP
            )
            
            # Add to vector DB
            metadatas = [{'filename': doc_data['filename'], 'chunk_index': i} for i in range(len(chunks))]
            ids = [f"{doc_data['filename']}_{i}_{datetime.now().timestamp()}" for i in range(len(chunks))]
            
            st.session_state.vector_db.add_documents(chunks, metadatas, ids)
            
            # Track file
            if 'General' not in st.session_state.processed_files:
                st.session_state.processed_files['General'] = []
            
            st.session_state.processed_files['General'].append({
                'name': doc_data['filename'],
                'chunks': len(chunks),
                'uploaded_at': datetime.now().isoformat()
            })
            
            progress.progress((idx + 1) / len(uploaded_files))
        
        save_processed_files(st.session_state.processed_files)
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
    """Generate response using RAG."""
    # Retrieve context
    context = ""
    if st.session_state.vector_db:
        try:
            results = st.session_state.vector_db.query(query, n_results=5)
            if results and 'documents' in results and results['documents']:
                context = "\n\n".join(results['documents'][0])
        except Exception:
            pass
    
    # Detect use case and generate
    use_case = detect_use_case(query)
    
    try:
        response = st.session_state.model.generate_response(
            prompt=query,
            context=context,
            use_case=use_case,
            temperature=st.session_state.temperature,
            max_tokens=config.MAX_TOKENS
        )
        return response
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"


def render_messages():
    """Render chat messages."""
    for message in st.session_state.chat_history:
        if message['role'] == 'user':
            st.markdown(
                f'<div class="message user-message">{message["content"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="message assistant-message">{message["content"]}</div>',
                unsafe_allow_html=True
            )


def main():
    """Main application."""
    init_session_state()
    render_sidebar()
    
    # Main content area
    col1, col2, col3 = st.columns([1, 6, 1])
    
    with col2:
        # Upload modal
        render_upload_modal()
        
        # Welcome message or chat
        if not st.session_state.chat_history:
            st.markdown("### 👋 Welcome to NotebookPRO")
            st.info("Upload your study materials and start asking questions!")
        else:
            render_messages()
    
    # Chat input
    user_input = st.chat_input("Ask me anything about your materials...")
    
    if user_input:
        # Add user message
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        })
        
        # Generate response
        with st.spinner("Thinking..."):
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
