import streamlit as st
from pathlib import Path
import json
from datetime import datetime
import uuid
import sys
import warnings
import logging
import os

# Suppress warnings
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
    page_title="NotebookPRO - Gemini Style",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Gemini-Style CSS - Clean & Modern
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu, footer, header {visibility: hidden;}
    [data-testid="collapsedControl"] {display: none;}
    
    /* Gemini Dark Background */
    .stApp {
        background-color: #0d0d0d !important;
    }
    
    /* Reset padding */
    .main .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }
    
    /* Top Header */
    .gemini-header {
        display: flex;
        align-items: center;
        justifyContent: space-between;
        padding: 12px 20px;
        border-bottom: 1px solid #1e1e1e;
        background: #0d0d0d;
    }
    
    .gemini-title {
        color: #e8eaed;
        font-size: 18px;
        font-weight: 400;
        letter-spacing: 0.2px;
    }
    
    /* Main Content - Centered */
    .gemini-main {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: calc(100vh - 180px);
        padding: 40px 20px;
    }
    
    /* Greeting Section */
    .greeting-container {
        max-width: 440px;
        width: 100%;
        margin-bottom: 24px;
    }
    
    .greeting-logo {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 4px;
        padding-left: 2px;
    }
    
    .greeting-title {
        color: #e8eaed;
        font-size: 28px;
        font-weight: 400;
        line-height: 1.3;
    }
    
    /* Input Box Container */
    .input-container {
        background: #1e1e1e;
        border-radius: 16px;
        padding: 14px;
        max-width: 440px;
        width: 100%;
    }
    
    /* Streamlit Input Styling */
    .stTextArea textarea {
        background: transparent !important;
        border: none !important;
        color: #e8eaed !important;
        font-size: 14px !important;
        min-height: 32px !important;
        padding: 0 !important;
    }
    
    .stTextArea textarea::placeholder {
        color: #9aa0a6 !important;
    }
    
    .stTextArea textarea:focus {
        outline: none !important;
        box-shadow: none !important;
    }
    
    .stTextArea [data-baseweb="base-input"] {
        background: transparent !important;
        border: none !important;
    }
    
    /* Button Styling */
    .stButton button {
        background: transparent !important;
        border: none !important;
        color: #9aa0a6 !important;
        padding: 8px !important;
        border-radius: 50% !important;
        transition: background 0.2s !important;
    }
    
    .stButton button:hover {
        background: rgba(255,255,255,0.1) !important;
    }
    
    /* Tools Button Active State */
    button[kind="primary"] {
        background: #3c3c3c !important;
        color: #e8eaed !important;
    }
    
    /* File Cards */
    .file-chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #2a2a2a;
        border: 1px solid #3c3c3c;
        border-radius: 12px;
        padding: 6px 12px;
        margin: 4px;
        color: #e8eaed;
        font-size: 13px;
    }
    
    /* Dropdown Menu */
    .stSelectbox, .stSelect box label {
        color: #e8eaed !important;
    }
    
    /* Chat Messages */
    .chat-message {
        background: #1e1e1e;
        border-radius: 16px;
        padding: 16px;
        margin: 12px 0;
        color: #e8eaed;
        font-size: 14px;
        line-height: 1.6;
    }
    
    .user-message {
        background: #2a2a2a;
    }
    
    /* Hide labels */
    .stTextArea label, .stSelectbox label {
        display: none !important;
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #1e1e1e;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #3c3c3c;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #555;
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
        collection_name = f"space_{space_id}"
        db_dir = st.session_state.spaces_manager.get_space_vector_db_dir(space_id)
        db_dir.mkdir(parents=True, exist_ok=True)
        vector_db = VectorDatabase(collection_name=collection_name, persist_directory=str(db_dir))
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
    st.session_state.config_manager.set_current_space(new_space_id)
    st.session_state.current_space = new_space_id
    st.session_state.all_chats = load_all_chats()
    st.session_state.vector_db = load_space_vector_db(get_current_space_id())
    st.session_state.processed_files = load_processed_files_for_space(get_current_space_id())
    
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
    
    st.session_state.current_chat_id = str(uuid.uuid4())
    st.session_state.chat_history = []


def load_all_chats():
    """Load all saved chats from current space."""
    chats = []
    if 'spaces_manager' not in st.session_state or 'current_space' not in st.session_state:
        chats_dir = config.CHATS_DIR
    else:
        space_id = st.session_state.current_space.lower().replace(" ", "_")
        chats_dir = st.session_state.spaces_manager.get_space_chats_dir(space_id)
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
    space_id = st.session_state.current_space.lower().replace(" ", "_")
    chats_dir = st.session_state.spaces_manager.get_space_chats_dir(space_id)
    chats_dir.mkdir(parents=True, exist_ok=True)
    
    chat_file = chats_dir / f"{chat_id}.json"
    try:
        with open(chat_file, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, indent=2)
    except Exception as e:
        st.error(f"Error saving chat: {e}")


def init_session_state():
    """Initialize session state variables."""
    if 'config_manager' not in st.session_state:
        st.session_state.config_manager = ConfigManager()
    
    if 'spaces_manager' not in st.session_state:
        st.session_state.spaces_manager = SpacesManager()
    
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
    
    if 'llm_provider' not in st.session_state:
        st.session_state.llm_provider = st.session_state.config_manager.get_preference("llm_provider", "groq")
    
    if 'api_key' not in st.session_state:
        st.session_state.api_key = st.session_state.config_manager.get_api_key(st.session_state.llm_provider)
    
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
        space_id = get_current_space_id()
        st.session_state.processed_files = load_processed_files_for_space(space_id)
    
    if 'vector_db' not in st.session_state:
        space_id = get_current_space_id()
        st.session_state.vector_db = load_space_vector_db(space_id)
        
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
    
    if 'hybrid_retriever' not in st.session_state:
        st.session_state.hybrid_retriever = None
    
    if 'workflow' not in st.session_state:
        st.session_state.workflow = "Auto-Detect"
    
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.3
    
    if 'show_upload' not in st.session_state:
        st.session_state.show_upload = False
    
    if 'show_tools_menu' not in st.session_state:
        st.session_state.show_tools_menu = False


def generate_response(query: str) -> str:
    """Generate response using hybrid retrieval and LLM."""
    if st.session_state.vector_db is None:
        return ("I don't have access to your study materials. Upload documents using the + button.")
    
    context = ""
    sources = []
    metadatas_list = []
    
    try:
        if st.session_state.hybrid_retriever:
            documents, metadatas, scores = st.session_state.hybrid_retriever.retrieve(
                query, n_results=7, score_threshold=0.1
            )
            
            if documents:
                context = "\n\n".join(documents)
                metadatas_list = metadatas
                for meta in metadatas:
                    if 'filename' in meta and meta['filename'] not in sources:
                        sources.append(meta['filename'])
            else:
                return "I couldn't find relevant information in your uploaded documents."
        else:
            results = st.session_state.vector_db.query(query, n_results=7)
            if results and 'documents' in results and results['documents'] and len(results['documents'][0]) > 0:
                context = "\n\n".join(results['documents'][0])
                metadatas_list = results['metadatas'][0] if 'metadatas' in results else []
                for metadata in metadatas_list:
                    if 'filename' in metadata and metadata['filename'] not in sources:
                        sources.append(metadata['filename'])
            else:
                return "I couldn't find relevant information in your uploaded documents."
    except Exception as e:
        return f"Error searching documents: {str(e)}"
    
    if st.session_state.workflow == "Auto-Detect":
        use_case = "qa"
    else:
        workflow_map = {
            "Explain": "explanation",
            "Summarize": "summary",
            "Chat": "qa",
            "Study Notes": "notes"
        }
        use_case = workflow_map.get(st.session_state.workflow, "explanation")
    
    try:
        if st.session_state.llm_generator and st.session_state.llm_generator.is_ready():
            response = st.session_state.llm_generator.generate_response(
                prompt=query,
                context=context,
                use_case=use_case,
                metadatas=metadatas_list,
                temperature=st.session_state.temperature,
                max_tokens=config.MAX_TOKENS
            )
        else:
            response = st.session_state.model.generate_response(
                prompt=query,
                context=context,
                use_case=use_case,
                metadatas=metadatas_list,
                temperature=st.session_state.temperature,
                max_tokens=config.MAX_TOKENS
            )
        
        if sources:
            source_text = ", ".join([f"**{s}**" for s in sources[:4]])
            response += f"\n\n📚 Sources: {source_text}"
        
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"


def process_uploaded_files(uploaded_files):
    """Process and embed uploaded files."""
    space_id = get_current_space_id()
    
    if st.session_state.vector_db is None:
        st.session_state.vector_db = load_space_vector_db(space_id)
    
    uploads_dir = st.session_state.spaces_manager.get_space_uploads_dir(space_id)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, file in enumerate(uploaded_files):
        status_text.text(f"Processing {file.name}...")
        
        file_path = uploads_dir / file.name
        with open(file_path, 'wb') as f:
            f.write(file.getbuffer())
        
        doc_data = st.session_state.doc_processor.process_file(file_path)
        
        if not doc_data['content'] or len(doc_data['content']) < 50:
            st.warning(f"⚠️ {file.name} has very little content")
            continue
        
        chunks = st.session_state.doc_processor.chunk_text(
            doc_data['content'],
            chunk_size=config.CHUNK_SIZE,
            overlap=config.CHUNK_OVERLAP,
            semantic=True
        )
        
        metadatas = [{'filename': doc_data['filename'], 'chunk_index': i} for i in range(len(chunks))]
        ids = [f"{doc_data['filename']}_{i}_{datetime.now().timestamp()}" for i in range(len(chunks))]
        
        st.session_state.vector_db.add_documents(chunks, metadatas, ids)
        
        if st.session_state.hybrid_retriever is None:
            st.session_state.hybrid_retriever = HybridRetriever(st.session_state.vector_db, alpha=0.6)
        
        all_docs, all_metas = st.session_state.vector_db.get_all_documents()
        st.session_state.hybrid_retriever.index_documents(all_docs, all_metas)
        
        space_name = st.session_state.current_space
        if space_name not in st.session_state.processed_files:
            st.session_state.processed_files[space_name] = []
        
        st.session_state.processed_files[space_name].append({
            'name': doc_data['filename'],
            'chunks': len(chunks),
            'uploaded_at': datetime.now().isoformat()
        })
        
        progress_bar.progress((idx + 1) / len(uploaded_files))
    
    save_processed_files_for_space(space_id, st.session_state.processed_files)
    status_text.success(f"✅ Processed {len(uploaded_files)} files!")


def main():
    """Main Gemini-Style Application."""
    init_session_state()
    
    # === TOP HEADER ===
    header_html = """
    <div class="gemini-header">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span class="gemini-title">✨ NotebookPRO</span>
        </div>
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="color: #9aa0a6; font-size: 11px; border: 1px solid #444; padding: 2px 8px; border-radius: 4px; letter-spacing: 1px;">PRO</span>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)
    
    # === MAIN CONTENT AREA ===
    # Show chat history OR welcome screen
    if st.session_state.chat_history:
        # Chat view
        st.markdown("<div style='padding: 20px; max-width: 800px; margin: 0 auto;'>", unsafe_allow_html=True)
        
        for msg in st.session_state.chat_history:
            if msg['role'] == 'user':
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong><br>{msg['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="chat-message">
                    <strong>✨ NotebookPRO:</strong><br>{msg['content']}
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        # Welcome screen
        greeting_html = """
        <div class="gemini-main">
            <div class="greeting-container">
                <div class="greeting-logo">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
                        <path d="M12 2C12 2 14.5 8 20 10C14.5 12 12 18 12 18C12 18 9.5 12 4 10C9.5 8 12 2 12 2Z" fill="url(#gem-grad)"/>
                        <defs>
                            <linearGradient id="gem-grad" x1="4" y1="2" x2="20" y2="18">
                                <stop stop-color="#4285F4"/>
                                <stop offset="0.5" stop-color="#9B59B6"/>
                                <stop offset="1" stop-color="#EA4335"/>
                            </linearGradient>
                        </defs>
                    </svg>
                    <span style="color: #e8eaed; font-size: 22px; font-weight: 400;">Hi there</span>
                </div>
                <h1 class="greeting-title">What would you like to know?</h1>
            </div>
        </div>
        """
        st.markdown(greeting_html, unsafe_allow_html=True)
        
        # Show uploaded files
        total_files = sum(len(files) for files in st.session_state.processed_files.values())
        if total_files > 0:
            files_html = "<div style='text-align: center; margin: 20px 0;'>"
            for space, files in st.session_state.processed_files.items():
                for f in files:
                    files_html += f'<span class="file-chip">📄 {f["name"][:30]}...</span>'
            files_html += "</div>"
            st.markdown(files_html, unsafe_allow_html=True)
    
    # === INPUT BOX (Always at bottom) ===
    st.markdown("<div style='height: 40px;'></div>", unsafe_allow_html=True)
    
    col_left, col_center, col_right = st.columns([1, 2, 1])
    with col_center:
        st.markdown("<div class='input-container'>", unsafe_allow_html=True)
        
        user_input = st.text_area(
            "input",
            placeholder="Ask me anything about your materials...",
            height=60,
            label_visibility="collapsed",
            key="main_input"
        )
        
        # Toolbar
        tcol1, tcol2, tcol3, tcol4 = st.columns([0.6, 3, 1, 0.4])
        
        with tcol1:
            if st.button("➕", help="Upload files"):
                st.session_state.show_upload = not st.session_state.show_upload
        
        with tcol3:
            tools_options = ["Tools", "Chat", "Summarize", "Explain", "Study Notes"]
            selected_tool = st.selectbox(
                "tools",
                options=tools_options,
                index=tools_options.index(st.session_state.workflow) if st.session_state.workflow in tools_options else 0,
                label_visibility="collapsed"
            )
            
            if selected_tool != "Tools":
                st.session_state.workflow = selected_tool
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Upload modal
    if st.session_state.show_upload:
        st.markdown("---")
        st.markdown("### 📎 Upload Study Materials")
        uploaded_files = st.file_uploader(
            "PDF, DOCX, or TXT files",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            key="file_uploader",
        )
        
        if uploaded_files:
            if st.button("Process Files"):
                process_uploaded_files(uploaded_files)
                st.session_state.show_upload = False
                st.rerun()
    
    # Process input
    if user_input and user_input.strip():
        st.session_state.chat_history.append({
            'role': 'user',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        })
        
        with st.spinner("✨ Thinking..."):
            response = generate_response(user_input)
        
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now().isoformat()
        })
        
        save_chat(st.session_state.current_chat_id, {
            'id': st.session_state.current_chat_id,
            'messages': st.session_state.chat_history,
            'created_at': st.session_state.chat_history[0]['timestamp'],
            'updated_at': datetime.now().isoformat()
        })
        
        st.session_state.all_chats = load_all_chats()
        st.rerun()


if __name__ == "__main__":
    main()
