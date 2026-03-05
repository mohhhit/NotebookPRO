import streamlit as st
from pathlib import Path
import json
from datetime import datetime
import uuid
import sys
import warnings
import logging
import os

# Suppress ALL warnings globally - MUST BE FIRST
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='.*FontBBox.*')
warnings.filterwarnings('ignore', message='.*font descriptor.*')
warnings.filterwarnings('ignore', message='.*position_ids.*')
warnings.filterwarnings('ignore', message='.*UNEXPECTED.*')

# Suppress all library logging
logging.getLogger('PyPDF2').setLevel(logging.CRITICAL)
logging.getLogger('pdfminer').setLevel(logging.CRITICAL)
logging.getLogger('transformers').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.ERROR)
logging.getLogger('chromadb').setLevel(logging.ERROR)
logging.getLogger('onnxruntime').setLevel(logging.ERROR)

# Suppress stderr warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Add project root to path
sys.path.append(str(Path(__file__).parent))

import config
from utils.document_processor import DocumentProcessor
from utils.vector_db import VectorDatabase
from utils.simple_generator import SimpleGenerator
# Heavy model import - only use if explicitly enabled
try:
    from utils.model_inference import ModelInference
    MODEL_AVAILABLE = True
except Exception:
    MODEL_AVAILABLE = False


# Page configuration
st.set_page_config(
    page_title="NotebookPRO",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Ollama-like styling
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        color: #000000 !important;
    }
    .chat-message b {
        color: #000000 !important;
    }
    .user-message {
        background-color: #e3f2fd !important;
        margin-left: 2rem;
        color: #000000 !important;
    }
    .assistant-message {
        background-color: #f5f5f5 !important;
        margin-right: 2rem;
        color: #000000 !important;
    }
    .sidebar-section {
        padding: 1rem 0;
        border-bottom: 1px solid #ddd;
    }
    .space-card {
        padding: 1rem;
        border: 1px solid #ddd;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
        cursor: pointer;
    }
    .space-card:hover {
        background-color: #f0f0f0;
    }
    /* Force dark text on all content */
    div[data-testid="stMarkdownContainer"] p {
        color: inherit;
    }
</style>
""", unsafe_allow_html=True)


# Helper function to detect use case from user prompt
def detect_use_case(prompt: str) -> str:
    """Detect the use case from user prompt keywords."""
    prompt_lower = prompt.lower()
    
    if any(word in prompt_lower for word in ['explain', 'what is', 'how does', 'why', 'elaborate']):
        return 'explanation'
    elif any(word in prompt_lower for word in ['summarize', 'summary', 'briefly', 'key points', 'overview']):
        return 'summary'
    elif any(word in prompt_lower for word in ['notes', 'bullet points', 'list', 'outline']):
        return 'notes'
    elif any(word in prompt_lower for word in ['?', 'who', 'when', 'where', 'which']):
        return 'qa'
    else:
        # Default to explanation for general queries
        return 'explanation'


# Initialize session state
def init_session_state():
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'Chats'
    
    if 'current_space' not in st.session_state:
        st.session_state.current_space = None
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'all_chats' not in st.session_state:
        st.session_state.all_chats = load_all_chats()
    
    if 'current_chat_id' not in st.session_state:
        st.session_state.current_chat_id = None
    
    if 'spaces' not in st.session_state:
        st.session_state.spaces = load_spaces()
    
    if 'model' not in st.session_state:
        # Use lightweight generator by default for instant responses
        st.session_state.model = SimpleGenerator()
    
    if 'use_advanced_model' not in st.session_state:
        st.session_state.use_advanced_model = False
    
    if 'vector_db' not in st.session_state:
        st.session_state.vector_db = None
    
    if 'doc_processor' not in st.session_state:
        st.session_state.doc_processor = DocumentProcessor()
    
    # Track processed files per space
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = load_processed_files()  # Load from disk
    
    # Show settings panel
    if 'show_settings' not in st.session_state:
        st.session_state.show_settings = False


def load_all_chats():
    """Load all saved chats from disk."""
    chats = []
    if config.CHATS_DIR.exists():
        for chat_file in config.CHATS_DIR.glob("*.json"):
            try:
                with open(chat_file, 'r', encoding='utf-8') as f:
                    chats.append(json.load(f))
            except Exception as e:
                print(f"Error loading chat {chat_file}: {e}")
    return chats


def load_processed_files():
    """Load processed files tracking from disk."""
    files_data_path = config.DATA_DIR / "processed_files.json"
    if files_data_path.exists():
        try:
            with open(files_data_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_processed_files(processed_files: dict):
    """Save processed files tracking to disk."""
    files_data_path = config.DATA_DIR / "processed_files.json"
    try:
        with open(files_data_path, 'w', encoding='utf-8') as f:
            json.dump(processed_files, f, indent=2)
    except Exception as e:
        print(f"Error saving processed files: {e}")


def remove_file_from_space(space_key: str, filename: str):
    """Remove a file from the processed files list and vector DB."""
    try:
        # Remove from processed_files tracking
        if space_key in st.session_state.processed_files:
            st.session_state.processed_files[space_key] = [
                f for f in st.session_state.processed_files[space_key] 
                if f['name'] != filename
            ]
            # Save to disk
            save_processed_files(st.session_state.processed_files)
        
        # Note: Actually removing chunks from ChromaDB would require deleting by metadata filter
        # For now, we just remove from tracking. A full implementation would need:
        # st.session_state.vector_db.collection.delete(where={"filename": filename})
        
        st.success(f"✅ Removed {filename} from tracking")
        
    except Exception as e:
        st.error(f"❌ Error removing file: {str(e)}")


def save_chat(chat_id, chat_data):
    """Save a chat to disk."""
    chat_file = config.CHATS_DIR / f"{chat_id}.json"
    with open(chat_file, 'w') as f:
        json.dump(chat_data, f, indent=2)


def load_spaces():
    """Load all spaces (subjects)."""
    spaces_file = config.DATA_DIR / "spaces.json"
    if spaces_file.exists():
        with open(spaces_file, 'r') as f:
            return json.load(f)
    return []


def save_spaces(spaces):
    """Save spaces to disk."""
    spaces_file = config.DATA_DIR / "spaces.json"
    with open(spaces_file, 'w') as f:
        json.dump(spaces, f, indent=2)


def sidebar_menu():
    """Render the sidebar hamburger menu."""
    with st.sidebar:
        st.markdown("<h1 class='main-header'>📚 NotebookPRO</h1>", unsafe_allow_html=True)
        
        # Main navigation menu
        selected = option_menu(
            menu_title=None,
            options=["Chats", "History", "Spaces"],
            icons=["chat-dots", "clock-history", "folder"],
            menu_icon="cast",
            default_index=["Chats", "History", "Spaces"].index(st.session_state.current_page),
            styles={
                "container": {"padding": "0!important"},
                "icon": {"font-size": "1.2rem"},
                "nav-link": {"font-size": "1rem", "text-align": "left", "margin": "0px"},
                "nav-link-selected": {"background-color": "#0066cc"},
            }
        )
        st.session_state.current_page = selected
        
        st.markdown("<div class='sidebar-section'></div>", unsafe_allow_html=True)
        
        # Space selector (if on Chats page)
        if selected == "Chats":
            st.subheader("Current Space")
            space_options = ["General"] + [s['name'] for s in st.session_state.spaces]
            current_space = st.selectbox(
                "Select Subject Space",
                space_options,
                index=0 if st.session_state.current_space is None else space_options.index(st.session_state.current_space)
            )
            
            if current_space != st.session_state.current_space:
                st.session_state.current_space = current_space
                st.session_state.chat_history = []
                st.rerun()
            
            # New chat button
            if st.button("➕ New Chat", use_container_width=True):
                st.session_state.current_chat_id = str(uuid.uuid4())
                st.session_state.chat_history = []
                st.rerun()


def render_chat_page():
    """Render the main chat interface."""
    st.markdown("<h2 class='main-header'>Chat</h2>", unsafe_allow_html=True)
    
    # Initialize vector DB for current space if files are processed
    space_key = st.session_state.current_space or "General"
    if space_key in st.session_state.processed_files and st.session_state.processed_files[space_key]:
        if st.session_state.vector_db is None:
            try:
                collection_name = f"space_{st.session_state.current_space}" if st.session_state.current_space else "documents"
                st.session_state.vector_db = VectorDatabase(collection_name=collection_name)
            except Exception as e:
                pass  # Vector DB will be initialized when processing files
    
    # Header with mode and settings
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        mode_emoji = "🤖" if st.session_state.use_advanced_model else "⚡"
        mode_name = "Advanced AI" if st.session_state.use_advanced_model else "Fast Mode"
        st.markdown(f"**{mode_emoji} {mode_name}**")
    
    with col2:
        # Show processed files count
        space_key = st.session_state.current_space or "General"
        file_count = len(st.session_state.processed_files.get(space_key, []))
        st.markdown(f"📚 **{file_count} files** loaded")
    
    with col3:
        if st.button("⚙️ Settings", key="settings_btn"):
            st.session_state.show_settings = not st.session_state.show_settings
    
    # Settings panel
    if st.session_state.get('show_settings', False):
        with st.expander("Model Settings", expanded=True):
            # Model mode selection
            st.subheader("AI Mode")
            
            current_mode = "Advanced AI (Phi-2)" if st.session_state.use_advanced_model else "⚡ Fast Mode (Instant)"
            st.info(f"**Current Mode:** {current_mode}")
            
            if not st.session_state.use_advanced_model:
                st.write("**⚡ Fast Mode** - Instant responses using retrieved content")
                st.write("✅ No downloads required")
                st.write("✅ Works immediately")
                st.write("✅ Perfect for most use cases")
                
                if st.button("🚀 Enable Advanced AI Model", help="Downloads 5GB model - takes 10-30 min on first run"):
                    with st.spinner("Preparing to download advanced model..."):
                        st.session_state.use_advanced_model = True
                        st.rerun()
            else:
                st.write("**🤖 Advanced AI Mode** - Deep learning model (Phi-2)")
                st.write("⚠️ 5GB download on first use")
                st.write("⚠️ Requires 10-30 minutes initial setup")
                st.write("⚠️ Uses more memory")
                
                if st.button("⚡ Switch to Fast Mode"):
                    st.session_state.use_advanced_model = False
                    st.session_state.model = SimpleGenerator()
                    st.success("Switched to Fast Mode!")
                    st.rerun()
            
            st.divider()
            
            # Generation settings
            temperature = st.slider("Temperature", 0.0, 1.0, config.TEMPERATURE, 0.1, 
                                   help="Higher = more creative, Lower = more focused")
            max_tokens = st.slider("Max Tokens", 256, 4096, config.MAX_TOKENS, 256,
                                  help="Maximum response length")
    else:
        temperature = config.TEMPERATURE
        max_tokens = config.MAX_TOKENS
    
    # File upload and management section
    with st.expander("📁 Study Materials", expanded=False):
        st.info("💡 Upload PDFs, DOCX, or TXT files. They will be embedded for instant RAG retrieval.")
        
        # Show already processed files
        space_key = st.session_state.current_space or "General"
        if space_key in st.session_state.processed_files and st.session_state.processed_files[space_key]:
            st.success(f"✅ **{len(st.session_state.processed_files[space_key])} files processed:**")
            for file_info in st.session_state.processed_files[space_key]:
                col_a, col_b = st.columns([4, 1])
                with col_a:
                    st.text(f"📄 {file_info['name']} ({file_info['chunks']} chunks)")
                with col_b:
                    if st.button("🗑️", key=f"delete_{file_info['name']}", help="Remove this file"):
                        remove_file_from_space(space_key, file_info['name'])
                        st.rerun()
            st.divider()
        
        # Upload new files
        uploaded_files = st.file_uploader(
            "Upload new files",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX, TXT",
            key="file_uploader"
        )
        
        if uploaded_files:
            if st.button("📤 Process & Add Files", type="primary", key="process_btn"):
                process_uploaded_files(uploaded_files)
    
    # Chat display area
    chat_container = st.container()
    with chat_container:
        if not st.session_state.chat_history:
            st.info("👋 Welcome! Upload some study materials and ask me anything!")
        
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(
                    f"<div class='chat-message user-message'><b>You:</b><br>{message['content']}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div class='chat-message assistant-message'><b>NotebookPRO:</b><br>{message['content']}</div>",
                    unsafe_allow_html=True
                )
    
    # Chat input with prompt suggestions
    st.markdown("---")
    st.markdown("**💬 Ask anything:** *Explain..., Summarize..., What is..., Create notes about...*")
    user_input = st.chat_input("Type your message here...")
    
    if user_input:
        try:
            # Detect use case from prompt
            use_case = detect_use_case(user_input)
            
            # Add user message to history
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_input,
                'timestamp': datetime.now().isoformat()
            })
            
            # Generate response
            with st.spinner("Thinking..."):
                response = generate_response(user_input, use_case, temperature, max_tokens)
            
            # Add assistant response to history
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response,
                'timestamp': datetime.now().isoformat()
            })
            
            # Auto-save chat
            if not st.session_state.current_chat_id:
                st.session_state.current_chat_id = str(uuid.uuid4())
            
            save_chat(st.session_state.current_chat_id, {
                'id': st.session_state.current_chat_id,
                'space': st.session_state.current_space,
                'messages': st.session_state.chat_history,
                'created_at': st.session_state.chat_history[0]['timestamp'] if st.session_state.chat_history else datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            })
            
            # Reload chats list
            st.session_state.all_chats = load_all_chats()
            
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error processing your message: {str(e)}")
            # Remove the user message if response failed
            if st.session_state.chat_history and st.session_state.chat_history[-1]['role'] == 'user':
                st.session_state.chat_history.pop()


def render_history_page():
    """Render the chat history page."""
    st.markdown("<h2 class='main-header'>Chat History</h2>", unsafe_allow_html=True)
    
    if not st.session_state.all_chats:
        st.info("No chat history yet. Start a new chat to begin!")
        return
    
    # Sort chats by date
    sorted_chats = sorted(
        st.session_state.all_chats,
        key=lambda x: x.get('created_at', ''),
        reverse=True
    )
    
    for chat in sorted_chats:
        with st.expander(
            f"Chat from {datetime.fromisoformat(chat['created_at']).strftime('%Y-%m-%d %H:%M')} - Space: {chat.get('space', 'General')}"
        ):
            for msg in chat['messages']:
                role = "You" if msg['role'] == 'user' else "NotebookPRO"
                st.markdown(f"**{role}:** {msg['content']}")
            
            col1, col2 = st.columns([1, 5])
            with col1:
                if st.button("🔄 Load", key=f"load_{chat['id']}", help="Continue this conversation"):
                    st.session_state.current_chat_id = chat['id']
                    st.session_state.chat_history = chat['messages'].copy()
                    st.session_state.current_space = chat.get('space')
                    st.session_state.current_page = 'Chats'
                    
                    # Reinitialize vector DB for this space
                    collection_name = f"space_{chat.get('space')}" if chat.get('space') else "documents"
                    try:
                        st.session_state.vector_db = VectorDatabase(collection_name=collection_name)
                    except Exception:
                        pass
                    
                    st.rerun()


def render_spaces_page():
    """Render the spaces (subjects) management page."""
    st.markdown("<h2 class='main-header'>Spaces (Subjects)</h2>", unsafe_allow_html=True)
    
    st.write("Organize your study materials by subject or topic.")
    
    # Create new space
    with st.expander("➕ Create New Space"):
        new_space_name = st.text_input("Space Name")
        new_space_desc = st.text_area("Description")
        
        if st.button("Create Space"):
            if new_space_name:
                new_space = {
                    'id': str(uuid.uuid4()),
                    'name': new_space_name,
                    'description': new_space_desc,
                    'created_at': datetime.now().isoformat()
                }
                st.session_state.spaces.append(new_space)
                save_spaces(st.session_state.spaces)
                st.success(f"Created space: {new_space_name}")
                st.rerun()
    
    # Display existing spaces
    st.subheader("Your Spaces")
    
    if not st.session_state.spaces:
        st.info("No spaces created yet. Create your first space to organize study materials!")
    else:
        for space in st.session_state.spaces:
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"### {space['name']}")
                    st.write(space.get('description', 'No description'))
                with col2:
                    if st.button("Select", key=f"select_{space['id']}"):
                        st.session_state.current_space = space['name']
                        st.session_state.current_page = 'Chats'
                        st.rerun()
                with col3:
                    if st.button("Delete", key=f"delete_{space['id']}"):
                        st.session_state.spaces = [s for s in st.session_state.spaces if s['id'] != space['id']]
                        save_spaces(st.session_state.spaces)
                        st.rerun()
                
                st.markdown("---")


def process_uploaded_files(uploaded_files):
    """Process uploaded documents and add to vector database."""
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    try:
        # Initialize vector DB if not already done
        if st.session_state.vector_db is None:
            progress_text.text("🔧 Initializing vector database (first time setup)...")
            collection_name = f"space_{st.session_state.current_space}" if st.session_state.current_space else "documents"
            st.session_state.vector_db = VectorDatabase(collection_name=collection_name)
            progress_bar.progress(10)
        
        total_files = len(uploaded_files)
        
        for idx, uploaded_file in enumerate(uploaded_files):
            progress_text.text(f"📄 Processing {uploaded_file.name} ({idx + 1}/{total_files})...")
            
            try:
                # Save file temporarily
                file_path = config.UPLOADS_DIR / uploaded_file.name
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                progress_bar.progress(20 + (idx * 30 // total_files))
                
                # Process document
                progress_text.text(f"📖 Extracting text from {uploaded_file.name}...")
                doc_data = st.session_state.doc_processor.process_file(file_path)
                
                if not doc_data['content'] or len(doc_data['content'].strip()) < 50:
                    st.warning(f"⚠️ {uploaded_file.name} contains very little text or extraction failed.")
                    continue
                
                progress_bar.progress(50 + (idx * 20 // total_files))
                
                # Chunk text
                progress_text.text(f"✂️ Chunking text from {uploaded_file.name}...")
                chunks = st.session_state.doc_processor.chunk_text(
                    doc_data['content'],
                    chunk_size=config.CHUNK_SIZE,
                    overlap=config.CHUNK_OVERLAP
                )
                
                if not chunks:
                    st.warning(f"⚠️ No content chunks created from {uploaded_file.name}")
                    continue
                
                progress_bar.progress(70 + (idx * 20 // total_files))
                
                # Add to vector database
                progress_text.text(f"🔢 Embedding {len(chunks)} chunks from {uploaded_file.name}...")
                metadatas = [
                    {
                        'filename': doc_data['filename'],
                        'chunk_index': i,
                        'space': st.session_state.current_space or 'General'
                    }
                    for i in range(len(chunks))
                ]
                ids = [f"{doc_data['filename']}_{i}_{datetime.now().timestamp()}" for i in range(len(chunks))]
                
                st.session_state.vector_db.add_documents(chunks, metadatas, ids)
                
                # Track processed file in session state
                space_key = st.session_state.current_space or "General"
                if space_key not in st.session_state.processed_files:
                    st.session_state.processed_files[space_key] = []
                
                # Add file info to tracking
                st.session_state.processed_files[space_key].append({
                    'name': doc_data['filename'],
                    'chunks': len(chunks),
                    'uploaded_at': datetime.now().isoformat()
                })
                
                # Save to disk immediately
                save_processed_files(st.session_state.processed_files)
                
                progress_bar.progress(90 + (idx * 10 // total_files))
                st.success(f"✅ Processed {uploaded_file.name} - {len(chunks)} chunks added")
                
            except Exception as e:
                st.error(f"❌ Error processing {uploaded_file.name}: {str(e)}")
                continue
        
        progress_bar.progress(100)
        progress_text.text("✅ All documents processed successfully!")
        
    except Exception as e:
        st.error(f"❌ Error during processing: {str(e)}")
    finally:
        # Clean up progress indicators after a moment
        import time
        time.sleep(1)
        progress_text.empty()
        progress_bar.empty()


def generate_response(query: str, use_case: str, temperature: float, max_tokens: int) -> str:
    """Generate a response using RAG retrieval."""
    # Retrieve relevant context from vector database
    context = ""
    if st.session_state.vector_db is not None:
        try:
            results = st.session_state.vector_db.query(query, n_results=5)
            if results and 'documents' in results and results['documents']:
                context = "\n\n".join(results['documents'][0])
        except Exception as e:
            st.warning(f"⚠️ Could not retrieve context from documents: {str(e)}")
    else:
        st.info("💡 Tip: Upload and process study materials to get context-aware answers!")
    
    # Check if user wants advanced model (downloads 5GB Phi-2)
    if st.session_state.use_advanced_model:
        # Initialize heavy model if needed
        if not isinstance(st.session_state.model, ModelInference):
            with st.spinner("🔄 Loading advanced AI model (5GB download on first run - may take 10-30 minutes)..."):
                try:
                    if MODEL_AVAILABLE:
                        st.session_state.model = ModelInference()
                        st.success("✅ Advanced model loaded!")
                    else:
                        st.error("Advanced model not available. Using fast mode.")
                        st.session_state.use_advanced_model = False
                        st.session_state.model = SimpleGenerator()
                except Exception as e:
                    st.error(f"Could not load advanced model: {e}")
                    st.info("Falling back to fast mode.")
                    st.session_state.use_advanced_model = False
                    st.session_state.model = SimpleGenerator()
    
    # Generate response
    try:
        response = st.session_state.model.generate_response(
            prompt=query,
            context=context,
            use_case=use_case,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response
    except Exception as e:
        return f"Sorry, I encountered an error while generating a response: {str(e)}"


def main():
    """Main application entry point."""
    init_session_state()
    sidebar_menu()
    
    # Render the appropriate page
    if st.session_state.current_page == 'Chats':
        render_chat_page()
    elif st.session_state.current_page == 'History':
        render_history_page()
    elif st.session_state.current_page == 'Spaces':
        render_spaces_page()


if __name__ == "__main__":
    main()
