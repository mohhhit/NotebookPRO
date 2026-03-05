import streamlit as st
from pathlib import Path
import json
from datetime import datetime
import uuid
import sys
import warnings
import logging
import os
import streamlit.components.v1 as components

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
logging.getLogger().setLevel(logging.ERROR)

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
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Minimal CSS - Just to hide Streamlit elements
st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    [data-testid="collapsedControl"] {display: none !important;}
    .stApp {background-color: #0d0d0d !important;}
    .main .block-container {padding: 0 !important; max-width: 100% !important;}
</style>
""", unsafe_allow_html=True)


def get_current_space_id():
    if 'current_space' in st.session_state:
        return st.session_state.current_space.lower().replace(" ", "_")
    return "general"


def load_space_vector_db(space_id: str):
    try:
        collection_name = f"space_{space_id}"
        db_dir = st.session_state.spaces_manager.get_space_vector_db_dir(space_id)
        db_dir.mkdir(parents=True, exist_ok=True)
        vector_db = VectorDatabase(collection_name=collection_name, persist_directory=str(db_dir))
        return vector_db
    except Exception as e:
        print(f"Error loading vector DB: {e}")
        return None


def load_processed_files_for_space(space_id: str):
    files_path = st.session_state.spaces_manager.get_space_uploads_dir(space_id) / "processed_files.json"
    if files_path.exists():
        try:
            with open(files_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}


def save_processed_files_for_space(space_id: str, processed_files: dict):
    uploads_dir = st.session_state.spaces_manager.get_space_uploads_dir(space_id)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    files_path = uploads_dir / "processed_files.json"
    try:
        with open(files_path, 'w', encoding='utf-8') as f:
            json.dump(processed_files, f, indent=2)
    except Exception as e:
        print(f"Error saving: {e}")


def load_all_chats():
    chats = []
    if 'spaces_manager' in st.session_state and 'current_space' in st.session_state:
        space_id = st.session_state.current_space.lower().replace(" ", "_")
        chats_dir = st.session_state.spaces_manager.get_space_chats_dir(space_id)
        chats_dir.mkdir(parents=True, exist_ok=True)
    else:
        chats_dir = config.CHATS_DIR
    
    if chats_dir.exists():
        for chat_file in chats_dir.glob("*.json"):
            try:
                with open(chat_file, 'r', encoding='utf-8') as f:
                    chats.append(json.load(f))
            except:
                pass
    return sorted(chats, key=lambda x: x.get('updated_at', x.get('created_at', '')), reverse=True)


def save_chat(chat_id, chat_data):
    space_id = st.session_state.current_space.lower().replace(" ", "_")
    chats_dir = st.session_state.spaces_manager.get_space_chats_dir(space_id)
    chats_dir.mkdir(parents=True, exist_ok=True)
    chat_file = chats_dir / f"{chat_id}.json"
    try:
        with open(chat_file, 'w', encoding='utf-8') as f:
            json.dump(chat_data, f, indent=2)
    except Exception as e:
        st.error(f"Error saving: {e}")


def init_session_state():
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
            except Exception as e:
                st.session_state.hybrid_retriever = None
    if 'hybrid_retriever' not in st.session_state:
        st.session_state.hybrid_retriever = None
    if 'workflow' not in st.session_state:
        st.session_state.workflow = "Chat"
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.3


def generate_response(query: str) -> str:
    if st.session_state.vector_db is None:
        return "Please upload documents using the + menu."
    
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
                return "I couldn't find relevant information in your documents."
        else:
            results = st.session_state.vector_db.query(query, n_results=7)
            if results and 'documents' in results and results['documents'] and len(results['documents'][0]) > 0:
                context = "\n\n".join(results['documents'][0])
                metadatas_list = results['metadatas'][0] if 'metadatas' in results else []
                for metadata in metadatas_list:
                    if 'filename' in metadata and metadata['filename'] not in sources:
                        sources.append(metadata['filename'])
            else:
                return "I couldn't find relevant information in your documents."
    except Exception as e:
        return f"Error searching documents: {str(e)}"
    
    workflow_map = {
        "Explain": "explanation",
        "Summarize": "summary",
        "Chat": "qa",
        "Study Notes": "notes"
    }
    use_case = workflow_map.get(st.session_state.workflow, "qa")
    
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
    init_session_state()
    
    # Get file names for display
    file_names = []
    for space, files in st.session_state.processed_files.items():
        file_names.extend([f['name'] for f in files])
    
    # Create Gemini UI with custom HTML/JS
    gemini_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        body {{
            background-color: #0d0d0d;
            color: #e8eaed;
            font-family: 'Google Sans', 'Segoe UI', Tahoma, sans-serif;
            overflow-x: hidden;
        }}
        .container {{
            display: flex;
            flex-direction: column;
            min-height: 100vh;
        }}
        /* Top Header */
        .top-header {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 12px 16px;
            border-bottom: 1px solid #1e1e1e;
        }}
        .logo-section {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .logo-text {{
            color: #e8eaed;
            font-size: 18px;
            font-weight: 400;
            letter-spacing: 0.2px;
        }}
        .pro-badge {{
            color: #9aa0a6;
            font-size: 11px;
            border: 1px solid #444;
            padding: 2px 8px;
            border-radius: 4px;
            letter-spacing: 1px;
        }}
        /* Main Content */
        .main-content {{
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 40px 20px 100px 20px;
        }}
        .greeting-container {{
            max-width: 440px;
            width: 100%;
            margin-bottom: 32px;
            text-align: left;
        }}
        .greeting-with-icon {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 4px;
        }}
        .gemini-icon {{
            width: 24px;
            height: 24px;
        }}
        .greeting-text {{
            color: #e8eaed;
            fontSize: 22px;
            font-weight: 400;
        }}
        .greeting-title {{
            color: #e8eaed;
font-size: 28px;
            font-weight: 400;
            line-height: 1.3;
        }}
        /* File chips */
        .file-chips {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            justify-content: center;
            margin-bottom: 24px;
            max-width: 600px;
        }}
        .file-chip {{
            background: #1e1f20;
            border: 1px solid #3c4043;
            border-radius: 20px;
            padding: 8px 16px;
            color: #e8eaed;
            font-size: 13px;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }}
        /* Input Box */
        .input-box {{
            background: #1e1e1e;
            border-radius: 26px;
            padding: 14px 18px;
            max-width: 440px;
            width: 100%;
            position: relative;
        }}
        .input-field {{
            width: 100%;
            background: transparent;
            border: none;
            outline: none;
            color: #e8eaed;
            font-size: 14px;
            font-family: inherit;
            resize: none;
            min-height: 24px;
            max-height: 200px;
        }}
        .input-field::placeholder {{
            color: #9aa0a6;
        }}
        .toolbar {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-top: 12px;
        }}
        .toolbar-left, .toolbar-right {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .icon-btn {{
            width: 32px;
            height: 32px;
            border-radius: 50%;
            background: transparent;
            border: none;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.2s;
        }}
        .icon-btn:hover {{
            background: rgba(255,255,255,0.1);
        }}
        .tools-btn {{
            display: flex;
            align-items: center;
            gap: 6px;
            padding: 6px 12px;
            border-radius: 20px;
            background: transparent;
            border: none;
            color: #9aa0a6;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .tools-btn.active {{
            background: #3c3c3c;
            color: #e8eaed;
        }}
        .tools-btn:hover {{
            background: #3c3c3c;
        }}
        /* Popup Menus */
        .popup-menu {{
            position: absolute;
            background: #2a2a2a;
            border-radius: 16px;
            padding: 8px 0;
            box-shadow: 0 8px 32px rgba(0,0,0,0.6);
            z-index: 1000;
            min-width: 200px;
            display: none;
        }}
        .popup-menu.show {{
            display: block;
        }}
        .plus-menu {{
            bottom: 50px;
            left: 0;
        }}
        .tools-menu {{
            bottom: 50px;
            right: 0;
        }}
        .menu-header {{
            color: #9aa0a6;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            padding: 8px 16px 4px 16px;
        }}
        .menu-item {{
            display: flex;
            align-items: center;
            gap: 12px;
            padding: 10px 16px;
            color: #e8eaed;
            font-size: 14px;
            cursor: pointer;
            transition: background 0.2s;
            border: none;
            background: transparent;
            width: 100%;
            text-align: left;
        }}
        .menu-item:hover {{
            background: rgba(255,255,255,0.1);
        }}
        .menu-item.active {{
            color: #8ab4f8;
            background: rgba(138,180,248,0.1);
        }}
        .checkmark {{
            margin-left: auto;
            color: #8ab4f8;
        }}
        /* SVG Icons */
        svg {{
            flex-shrink: 0;
        }}
        .hide {{
            display: none !important;
        }}
        .submit-btn {{
            background: #8ab4f8;
            color: #0d0d0d;
            border: none;
            border-radius: 50%;
            width: 32px;
            height: 32px;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: background 0.2s;
        }}
        .submit-btn:hover {{
            background: #a6c5f7;
        }}
        .submit-btn:disabled {{
            background: #555;
            cursor: not-allowed;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Top Header -->
        <header class="top-header">
            <div class="logo-section">
                <span class="logo-text">✨ NotebookPRO</span>
            </div>
            <div class="logo-section">
                <span class="pro-badge">PRO</span>
            </div>
        </header>

        <!-- Main Content -->
        <main class="main-content">
            <!-- Greeting -->
            <div class="greeting-container">
                <div class="greeting-with-icon">
                    <svg class="gemini-icon" viewBox="0 0 24 24" fill="none">
                        <path d="M12 2C12 2 14.5 8 20 10C14.5 12 12 18 12 18C12 18 9.5 12 4 10C9.5 8 12 2 12 2Z" fill="url(#gem-grad)"/>
                        <defs>
                            <linearGradient id="gem-grad" x1="4" y1="2" x2="20" y2="18">
                                <stop stop-color="#4285F4"/>
                                <stop offset="0.5" stop-color="#9B59B6"/>
                                <stop offset="1" stop-color="#EA4335"/>
                            </linearGradient>
                        </defs>
                    </svg>
                    <span class="greeting-text">Hi there</span>
                </div>
                <h1 class="greeting-title">What would you like to know?</h1>
            </div>

            <!-- File Chips -->
            <div class="file-chips" id="fileChips">
                {''.join([f'<span class="file-chip">📄 {name[:30]}...</span>' for name in file_names[:3]])}
            </div>

            <!-- Input Box -->
            <div class="input-box">
                <textarea 
                    class="input-field" 
                    id="userInput"
                    placeholder="Ask me anything about your materials..."
                    rows="1"
                ></textarea>

                <!-- Toolbar -->
                <div class="toolbar">
                    <div class="toolbar-left">
                        <div style="position: relative;">
                            <button class="icon-btn" id="plusBtn">
                                <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                                    <path d="M12 5v14M5 12h14" stroke="#9aa0a6" stroke-width="2" stroke-linecap="round"/>
                                </svg>
                            </button>
                            <!-- Plus Menu Popup -->
                            <div class="popup-menu plus-menu" id="plusMenu">
                                <button class="menu-item" onclick="triggerUpload()">
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" stroke="#9aa0a6" stroke-width="2"/>
                                        <polyline points="17 8 12 3 7 8" stroke="#9aa0a6" stroke-width="2"/>
                                        <line x1="12" y1="3" x2="12" y2="15" stroke="#9aa0a6" stroke-width="2"/>
                                    </svg>
                                    <span>Upload files</span>
                                </button>
                            </div>
                        </div>
                    </div>
                    <div class="toolbar-right">
                        <div style="position: relative;">
                            <button class="tools-btn" id="toolsBtn">
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                                    <path d="M14.7 6.3a1 1 0 0 0 0 1.4l1.6 1.6a1 1 0 0 0 1.4 0l3.77-3.77a6 6 0 0 1-7.94 7.94l-6.91 6.91a2.12 2.12 0 0 1-3-3l6.91-6.91a6 6 0 0 1 7.94-7.94l-3.76 3.76z" stroke="#9aa0a6" stroke-width="2"/>
                                </svg>
                                <span id="toolLabel">Chat</span>
                            </button>
                            <!-- Tools Menu Popup -->
                            <div class="popup-menu tools-menu" id="toolsMenu">
                                <div class="menu-header">Tools</div>
                                <button class="menu-item active" data-tool="Chat">
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                                        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" stroke="currentColor" stroke-width="2"/>
                                    </svg>
                                    <span>Chat</span>
                                    <span class="checkmark">✓</span>
                                </button>
                                <button class="menu-item" data-tool="Summarize">
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                                        <line x1="21" y1="6" x2="3" y2="6" stroke="currentColor" stroke-width="2"/>
                                        <line x1="15" y1="12" x2="3" y2="12" stroke="currentColor" stroke-width="2"/>
                                        <line x1="17" y1="18" x2="3" y2="18" stroke="currentColor" stroke-width="2"/>
                                    </svg>
                                    <span>Summarize</span>
                                </button>
                                <button class="menu-item" data-tool="Explain">
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                                        <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/>
                                        <line x1="12" y1="8" x2="12" y2="12" stroke="currentColor" stroke-width="2"/>
                                        <line x1="12" y1="16" x2="12.01" y2="16" stroke="currentColor" stroke-width="2"/>
                                    </svg>
                                    <span>Explain</span>
                                </button>
                                <button class="menu-item" data-tool="Study Notes">
                                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" stroke="currentColor" stroke-width="2"/>
                                        <polyline points="14 2 14 8 20 8" stroke="currentColor" stroke-width="2"/>
                                    </svg>
                                    <span>Study Notes</span>
                                </button>
                            </div>
                        </div>
                        <button class="submit-btn" id="submitBtn" disabled>
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                               <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                            </svg>
                        </button>
                    </div>
                </div>
            </div>
        </main>
    </div>

    <script>
        const plusBtn = document.getElementById('plusBtn');
        const plusMenu = document.getElementById('plusMenu');
        const toolsBtn = document.getElementById('toolsBtn');
        const toolsMenu = document.getElementById('toolsMenu');
        const toolLabel = document.getElementById('toolLabel');
        const userInput = document.getElementById('userInput');
        const submitBtn = document.getElementById('submitBtn');
        
        let activeTool = 'Chat';
        
        // Toggle Plus Menu
        plusBtn.addEventListener('click', (e) => {{
            e.stopPropagation();
            plusMenu.classList.toggle('show');
            toolsMenu.classList.remove('show');
        }});
        
        // Toggle Tools Menu
        toolsBtn.addEventListener('click', (e) => {{
            e.stopPropagation();
            toolsMenu.classList.toggle('show');
            plusMenu.classList.remove('show');
        }});
        
        // Tool selection
        document.querySelectorAll('.menu-item[data-tool]').forEach(item => {{
            item.addEventListener('click', () => {{
                const tool = item.getAttribute('data-tool');
                activeTool = tool;
                toolLabel.textContent = tool;
                
                // Update active state
                document.querySelectorAll('.menu-item[data-tool]').forEach(i => {{
                    i.classList.remove('active');
                    i.querySelector('.checkmark')?.remove();
                }});
                item.classList.add('active');
                const check = document.createElement('span');
                check.className = 'checkmark';
                check.textContent = '✓';
                item.appendChild(check);
                
                toolsMenu.classList.remove('show');
                toolsBtn.classList.add('active');
                
                // Send to Streamlit
                window.parent.postMessage({{
                    type: 'streamlit:setComponentValue',
                    value: {{action: 'set_tool', tool: tool}}
                }}, '*');
            }});
        }});
        
        // Close menus on outside click
        document.addEventListener('click', () => {{
            plusMenu.classList.remove('show');
            toolsMenu.classList.remove('show');
        }});
        
        // Auto-resize textarea
        userInput.addEventListener('input', () => {{
            userInput.style.height = 'auto';
            userInput.style.height = userInput.scrollHeight + 'px';
            submitBtn.disabled = userInput.value.trim() === '';
        }});
        
        // Submit on Enter
        userInput.addEventListener('keydown', (e) => {{
            if (e.key === 'Enter' && !e.shiftKey) {{
                e.preventDefault();
                if (userInput.value.trim()) {{
                    submitQuery();
                }}
            }}
        }});
        
        submitBtn.addEventListener('click', submitQuery);
        
        function submitQuery() {{
            const query = userInput.value.trim();
            if (query) {{
                window.parent.postMessage({{
                    type: 'streamlit:setComponentValue',
                    value: {{action: 'submit', query: query, tool: activeTool}}
                }}, '*');
                userInput.value = '';
                userInput.style.height = 'auto';
                submitBtn.disabled = true;
            }}
        }}
        
        function triggerUpload() {{
            plusMenu.classList.remove('show');
            window.parent.postMessage({{
                type: 'streamlit:setComponentValue',
                value: {{action: 'upload'}}
            }}, '*');
        }}
    </script>
</body>
</html>
    """
    
    # Render the Gemini UI
    result = components.html(gemini_html, height=800, scrolling=False)
    
    # Handle UI actions
    if result:
        action = result.get('action')
        
        if action == 'set_tool':
            st.session_state.workflow = result['tool']
            st.rerun()
        
        elif action == 'submit':
            query = result['query']
            tool = result['tool']
            st.session_state.workflow = tool
            
            st.session_state.chat_history.append({
                'role': 'user',
                'content': query,
                'timestamp': datetime.now().isoformat()
            })
            
            with st.spinner("✨ Thinking..."):
                response = generate_response(query)
            
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
        
        elif action == 'upload':
            st.session_state.show_upload = True
            st.rerun()
    
    # Upload modal
    if st.session_state.get('show_upload', False):
        st.markdown("---")
        st.markdown("### 📎 Upload Study Materials")
        uploaded_files = st.file_uploader(
            "PDF, DOCX, or TXT files",
            type=['pdf', 'txt', 'docx'],
            accept_multiple_files=True,
            key="file_uploader"
        )
        
        if uploaded_files:
            if st.button("Process Files"):
                process_uploaded_files(uploaded_files)
                st.session_state.show_upload = False
                st.rerun()


if __name__ == "__main__":
    main()
