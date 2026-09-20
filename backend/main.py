"""
FastAPI Backend for NotebookPRO
Handles RAG, LLM, file processing, and chat management
"""
from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
from datetime import datetime
import uuid
import sys
import warnings
import logging
import os
import shutil
import tempfile
import zipfile
import re
import html
import time
from io import BytesIO
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask
import weasyprint
import markdown
import latex2mathml.converter
import ziamath

# Suppress warnings
warnings.filterwarnings('ignore')
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '2')
logging.getLogger().setLevel(logging.ERROR)

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

import config
from utils.document_processor import DocumentProcessor
from utils.vector_db import VectorDatabase
from utils.hybrid_retriever import HybridRetriever
from utils.llm_generator import LLMGenerator
from utils.config_manager import ConfigManager
from utils.spaces_manager import SpacesManager
from utils.studio_manager import StudioManager
from utils.studio_generator import StudioGenerator
from models.studio_models import (
    NotebookEntry,
    NotebookEntryCreate,
    NotebookEntryUpdate,
    Flashcard,
    FlashcardCreate,
    FlashcardUpdate,
    FlashcardReview,
    FlashcardGenerateRequest,
    MasteryLevel,
    Quiz,
    QuizCreate,
    QuizGenerateRequest,
    QuizSubmission,
    QuizResult,
    QuizHistory,
)

# Initialize FastAPI
app = FastAPI(title="NotebookPRO API", version="2.0.0")

# CORS - Allow Flutter web to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Flutter web URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
config_manager = ConfigManager()
spaces_manager = SpacesManager()
studio_manager = StudioManager()
studio_generator = None  # Will be initialized after LLM
vector_db = None
hybrid_retriever = None
llm_generator = None
current_space = None

# ==================== Pydantic Models ====================

class ChatMessage(BaseModel):
    role: str
    content: str
    timestamp: str
    sources: Optional[List[Dict[str, Any]]] = None
    suggested_follow_ups: Optional[List[str]] = None

class ChatRequest(BaseModel):
    query: str
    space_id: str
    chat_id: Optional[str] = None
    workflow: str = "chat"

class ChatResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    chat_id: str
    timestamp: str
    ui_notice: Optional[str] = None
    suggested_follow_ups: Optional[List[str]] = None

class DeleteMessageRequest(BaseModel):
    timestamp: str
    role: str = "assistant"
    content: Optional[str] = None
    delete_related_user: bool = False

class SpaceCreate(BaseModel):
    name: str

class SpaceResponse(BaseModel):
    id: str
    name: str
    created_at: str
    file_count: int

class ChatInfo(BaseModel):
    id: str
    title: str
    preview: str
    created_at: str
    updated_at: str
    message_count: int

class ConfigResponse(BaseModel):
    gemini_api_key: Optional[str]
    nvidia_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None

class ConfigUpdate(BaseModel):
    gemini_api_key: Optional[str] = None
    nvidia_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None

class ChatToNotebookRequest(BaseModel):
    space_id: str
    question: str
    answer: str
    chat_id: Optional[str] = None
    assistant_timestamp: Optional[str] = None
    tags: List[str] = []
    space_name: Optional[str] = None

class ExportAnswerPdfRequest(BaseModel):
    space_id: str
    question: str
    answer: str
    title: Optional[str] = None

class QAItem(BaseModel):
    question: str
    answer: str

class ExportCombinedAnswersPdfRequest(BaseModel):
    space_id: str
    chat_title: Optional[str] = None
    items: List[QAItem]


class ExportSelectedNotebookPdfRequest(BaseModel):
    space_id: str
    entry_ids: List[str]
    title: Optional[str] = None

# ==================== Helper Functions ====================

def get_data_dir():
    """Get data directory path"""
    return Path(config.DATA_DIR)

def get_space_dir(space_id: str):
    """Get space-specific directory"""
    return get_data_dir() / "spaces" / space_id

def load_chats_for_space(space_id: str) -> List[Dict]:
    """Load all chats for a space"""
    chats_file = get_space_dir(space_id) / "chats.json"
    if chats_file.exists():
        with open(chats_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_chats_for_space(space_id: str, chats: List[Dict]):
    """Save chats for a space"""
    chats_file = get_space_dir(space_id) / "chats.json"
    chats_file.parent.mkdir(parents=True, exist_ok=True)
    with open(chats_file, 'w', encoding='utf-8') as f:
        json.dump(chats, f, indent=2, ensure_ascii=False)

def get_chat_title(messages: List[Dict]) -> str:
    """Generate chat title from first user message"""
    for msg in messages:
        if msg['role'] == 'user':
            content = msg['content'][:50]
            return content + "..." if len(msg['content']) > 50 else content
    return "New Chat"


def _is_nlp_metrics_query(query: str) -> bool:
    """Detect metric-heavy NLP evaluation questions that need broader retrieval."""
    q = (query or "").lower()
    triggers = (
        "bleu",
        "rouge",
        "meteor",
        "bertscore",
        "sequence generation",
        "machine translation",
        "summarization",
        "n-gram",
        "brevity penalty",
        "evaluation metric",
        "nlp metric",
        "transformer",
    )
    return any(t in q for t in triggers)


def _extract_year_from_filename(filename: str) -> Optional[int]:
    """Infer publication year from filename (best-effort)."""
    if not filename:
        return None

    matches = re.findall(r'(?:19|20)\d{2}', filename)
    if not matches:
        return None

    try:
        years = [int(y) for y in matches]
        valid_years = [y for y in years if 1900 <= y <= 2100]
        if not valid_years:
            return None
        return max(valid_years)
    except Exception:
        return None


def _resolve_publication_year(meta: Optional[Dict[str, Any]]) -> Optional[int]:
    """Resolve publication year from metadata; fallback to filename inference."""
    if not meta:
        return None

    raw_year = meta.get('publication_year')
    if raw_year is not None:
        try:
            year = int(raw_year)
            if 1900 <= year <= 2100:
                return year
        except Exception:
            pass

    filename = str(meta.get('filename', ''))
    return _extract_year_from_filename(filename)


def _build_retrieval_queries(query: str) -> List[str]:
    """Build retrieval query set; add targeted expansions for NLP metric topics."""
    queries = [query]
    if not _is_nlp_metrics_query(query):
        return queries

    queries.extend([
        f"{query} BLEU brevity penalty BP n-gram precision",
        f"{query} ROUGE-N ROUGE-L longest common subsequence LCS F1 recall precision",
        f"{query} METEOR stemming synonym WordNet exact-match limitation",
        f"{query} BERTScore contextual embeddings semantic similarity deep learning transformer",
    ])
    return queries


def _retrieve_documents_for_query(
    retriever: HybridRetriever,
    query: str,
    n_results: int,
    score_threshold: float,
):
    """Retrieve with optional query expansions and merge by max score."""
    retrieval_queries = _build_retrieval_queries(query)
    per_query_n = max(12, n_results // 2) if len(retrieval_queries) > 1 else n_results
    per_query_threshold = max(0.0, score_threshold - 0.01) if len(retrieval_queries) > 1 else score_threshold

    merged = {}
    for q in retrieval_queries:
        docs, metas, scores = retriever.retrieve(
            query=q,
            n_results=per_query_n,
            score_threshold=per_query_threshold,
        )
        for doc, meta, score in zip(docs, metas, scores):
            current = merged.get(doc)
            score_value = float(score)
            if current is None or score_value > current["score"]:
                merged[doc] = {
                    "doc": doc,
                    "meta": meta,
                    "score": score_value,
                }

    ranked = sorted(merged.values(), key=lambda item: item["score"], reverse=True)
    selected = ranked[:n_results]
    return (
        [item["doc"] for item in selected],
        [item["meta"] for item in selected],
        [item["score"] for item in selected],
    )


def _sanitize_retrieved_chunk(text: str) -> str:
    """Clean retrieval chunks to remove OCR/table-of-contents numeric noise."""
    if not text:
        return ""

    cleaned_lines: List[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            cleaned_lines.append("")
            continue

        # Drop pathological section-number runs like: 1.1.1.1.1.1...
        if re.search(r'(?:\d+\.){8,}\d*', line):
            continue

        compact = re.sub(r'\s+', '', line)
        # Drop long punctuation/digit-only lines that come from OCR artifacts.
        if len(compact) >= 40 and re.fullmatch(r'[\d\W_]+', compact):
            continue

        line = re.sub(r'(?:\d+\.){8,}\d*', ' ', line)
        line = re.sub(r'\s{2,}', ' ', line).strip()
        if line:
            cleaned_lines.append(line)

    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned).strip()
    return cleaned


def _is_usable_context_chunk(text: str) -> bool:
    """Reject chunks that are mostly numeric garbage after cleanup."""
    if not text or len(text) < 80:
        return False

    letters = len(re.findall(r'[A-Za-z]', text))
    digits = len(re.findall(r'\d', text))

    if letters < 35:
        return False

    # If a chunk is heavily numeric and has little natural language, skip it.
    if digits > 0 and letters < max(80, int(digits * 0.35)):
        return False

    return True


def _format_sse_event(event: str, payload: Dict[str, Any]) -> str:
    """Format one Server-Sent Event frame."""
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

def ensure_notebooks_for_existing_spaces() -> int:
    """Ensure every existing space has an associated notebook metadata record."""
    created_count = 0
    spaces = spaces_manager.get_all_spaces()

    for space in spaces:
        space_id = space.get('id')
        if not space_id:
            continue

        existing_notebook = studio_manager.get_space_notebook(space_id)
        if existing_notebook:
            continue

        studio_manager.ensure_space_notebook(space_id, space.get('name', space_id))
        created_count += 1

    return created_count

def rebuild_space_index_if_missing(space_id: str, force: bool = False) -> int:
    """Rebuild a space index from uploaded files if the current index is empty."""
    if not vector_db:
        return 0

    if not force:
        try:
            existing_count = vector_db.get_collection_count()
            if existing_count > 0:
                print(f"[INDEX_REBUILD {space_id}] skip existing_vectors={existing_count}")
                return 0
        except Exception:
            # If count check fails, continue with a best-effort rebuild.
            pass

    uploads_dir = get_space_dir(space_id) / "uploads"
    if not uploads_dir.exists():
        return 0

    files = [
        p for p in uploads_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".pdf", ".docx", ".txt"}
    ]
    if not files:
        print(f"[INDEX_REBUILD {space_id}] skip no_supported_uploads")
        return 0

    print(f"[INDEX_REBUILD {space_id}] start files={len(files)}")

    processor = DocumentProcessor()
    texts: List[str] = []
    metadatas: List[Dict[str, Any]] = []
    ids: List[str] = []

    for file_path in files:
        try:
            print(f"[INDEX_REBUILD {space_id}] processing_file={file_path.name}")
            file_data = processor.process_file(file_path)
            chunk_filename = file_data.get('virtual_filename', file_path.name)
            chunks = processor.chunk_text(
                file_data['content'],
                chunk_size=config.CHUNK_SIZE,
                overlap=config.CHUNK_OVERLAP,
                semantic=True,
                source_filename=chunk_filename,
            )
            total_chunks = len(chunks)
            print(f"[INDEX_REBUILD {space_id}] file_chunks={total_chunks} file={chunk_filename}")
            publication_year = file_data.get('publication_year', '0000')
            for idx, chunk in enumerate(chunks):
                texts.append(chunk)
                metadatas.append({
                    'filename': chunk_filename,
                    'chunk_index': idx,
                    'total_chunks': total_chunks,
                    'source_type': file_data['format'],
                    'publication_year': publication_year,
                })
                ids.append(f"{space_id}_rebuild_{len(ids)}_{uuid.uuid4().hex[:8]}")
        except Exception as e:
            print(f"Index rebuild skipped {file_path.name}: {e}")

    if not texts:
        print(f"[INDEX_REBUILD {space_id}] skip no_chunks_generated")
        return 0

    batch_size = 5000
    for i in range(0, len(texts), batch_size):
        end = min(i + batch_size, len(texts))
        print(f"[INDEX_REBUILD {space_id}] batch_add start={i} end={end}")
        vector_db.add_documents(
            texts[i:i + batch_size],
            metadatas[i:i + batch_size],
            ids[i:i + batch_size],
        )

    print(f"[INDEX_REBUILD {space_id}] done chunks={len(texts)}")
    return len(texts)

def initialize_space(space_id: str):
    """Initialize vector DB and components for a space"""
    global vector_db, hybrid_retriever, llm_generator, studio_generator, current_space

    # Fast path: reuse already initialized components for the active space.
    if current_space == space_id and vector_db is not None and llm_generator is not None:
        return

    # On space switch, release references and clear caches to reduce stale VRAM pressure.
    switching_space = current_space is not None and current_space != space_id
    if switching_space:
        vector_db = None
        hybrid_retriever = None
        llm_generator = None
        studio_generator = None
        if config.SPACE_SWITCH_CLEAR_CUDA_CACHE:
            VectorDatabase.clear_runtime_caches(
                unload_embedding_model=config.SPACE_SWITCH_UNLOAD_EMBEDDING_MODEL,
            )
    
    # Get API keys for Gemini answers and NVIDIA follow-up questions.
    gemini_key = config_manager.get_api_key('gemini')
    nvidia_key = config_manager.get_api_key('nvidia')
    groq_key = config_manager.get_api_key('groq')

    if not gemini_key and not nvidia_key and not groq_key:
        raise HTTPException(
            status_code=400,
            detail="No API key configured. Please add a Gemini or NVIDIA API key.",
        )
    
    # Initialize vector database for this space (space-local persistence path).
    space_dir = get_space_dir(space_id)
    vector_db = VectorDatabase(
        collection_name=f"space_{space_id}",
        persist_directory=str(space_dir / "vector_db"),
    )

    collection_reset = vector_db.ensure_collection_embedding_compatibility()
    if collection_reset:
        print(f"[INDEX_REBUILD {space_id}] reset_due_to_embedding_dimension_change=1")

    hybrid_retriever = HybridRetriever(vector_db, alpha=0.6)

    # Backward-compatibility: rebuild embeddings from uploaded files if index is empty.
    rebuild_space_index_if_missing(space_id)
    
    # Initialize LLM generator (Gemini for answers, NVIDIA for follow-ups).
    llm_generator = LLMGenerator(
        provider="gemini",
        gemini_api_key=gemini_key,
        nvidia_api_key=nvidia_key,
        groq_api_key=groq_key,
    )
    
    # Initialize studio generator with LLM
    studio_generator = StudioGenerator(llm_generator, studio_manager)
    current_space = space_id

@app.on_event("startup")
async def startup_sync_notebooks():
    """Auto-create missing notebooks for pre-existing spaces when backend starts and preload GPU models."""
    try:
        from utils.vector_db import VectorDatabase
        print("Preloading embedding and reranker models to GPU...")
        VectorDatabase._get_or_create_embedding_model()
        VectorDatabase._get_or_create_reranker()
        print("Models successfully preloaded to GPU.")
    except Exception as e:
        print(f"Failed to preload models: {e}")

    try:
        created = ensure_notebooks_for_existing_spaces()
        if created > 0:
            print(f"Created {created} missing notebook(s) for existing spaces")
    except Exception as e:
        # Keep server startup resilient even if sync fails.
        print(f"Notebook startup sync failed: {e}")

# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """Health check"""
    return {"status": "NotebookPRO API is running", "version": "2.0.0"}

@app.get("/api/health")
async def health_check():
    """Fast health endpoint used by the Flutter app connection checks."""
    return {"status": "ok"}

@app.get("/api/config", response_model=ConfigResponse)
async def get_config():
    """Get current API keys (masked)"""
    gemini_key = config_manager.get_api_key('gemini')
    nvidia_key = config_manager.get_api_key('nvidia')
    groq_key = config_manager.get_api_key('groq')
    masked_gemini_key = "***" + gemini_key[-4:] if gemini_key else None
    masked_nvidia_key = "***" + nvidia_key[-4:] if nvidia_key else None
    masked_groq_key = "***" + groq_key[-4:] if groq_key else None
    
    return ConfigResponse(
        gemini_api_key=masked_gemini_key,
        nvidia_api_key=masked_nvidia_key,
        groq_api_key=masked_groq_key,
    )

@app.get("/api/system/export")
async def export_data():
    """Export the entire data directory as a ZIP file."""
    try:
        tmp_dir = Path(tempfile.gettempdir())
        zip_base_path = tmp_dir / f"notebookpro_backup_{int(time.time())}"
        
        # This creates notebookpro_backup_xxx.zip
        shutil.make_archive(
            base_name=str(zip_base_path),
            format="zip",
            root_dir=str(get_data_dir())
        )
        
        zip_path = str(zip_base_path) + ".zip"
        
        return FileResponse(
            path=zip_path,
            filename="notebookpro_backup.zip",
            media_type="application/zip",
            background=BackgroundTask(lambda: os.remove(zip_path))
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {e}")

@app.post("/api/system/import")
async def import_data(file: UploadFile = File(...)):
    """Import a ZIP file and overwrite the data directory."""
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Must be a ZIP file")
        
    try:
        tmp_dir = Path(tempfile.gettempdir())
        zip_path = tmp_dir / f"notebookpro_import_{int(time.time())}.zip"
        
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Verify zip structure minimally
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # For a basic check, we don't strict-enforce specific files in case of empty states
            pass
            
        extract_dir = tmp_dir / f"notebookpro_import_extracted_{int(time.time())}"
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            
        # Re-initialize managers to close DB connections if any? 
        # Best approach: replace data and ask user to restart.
        
        # Replace data directory
        data_dir = get_data_dir()
        for item in extract_dir.iterdir():
            target = data_dir / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(item), str(target))
            
        # Clean up
        os.remove(zip_path)
        shutil.rmtree(extract_dir)
        
        return {"status": "success", "message": "Data imported successfully. Please restart the backend."}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Import failed: {e}")


@app.post("/api/config")
async def update_config(config_update: ConfigUpdate):
    """Update API keys"""
    if config_update.gemini_api_key:
        config_manager.set_api_key('gemini', config_update.gemini_api_key)

    if config_update.nvidia_api_key:
        config_manager.set_api_key('nvidia', config_update.nvidia_api_key)

    if config_update.groq_api_key:
        config_manager.set_api_key('groq', config_update.groq_api_key)
    
    return {"status": "success", "message": "Configuration updated"}

@app.get("/api/spaces", response_model=List[SpaceResponse])
async def get_spaces():
    """Get all spaces"""
    # Self-healing check in case spaces were created externally while server is running.
    ensure_notebooks_for_existing_spaces()
    spaces = spaces_manager.get_all_spaces()
    
    result = []
    for space in spaces:
        space_id = space['id']
        space_dir = get_space_dir(space_id)
        processed_file = space_dir / "processed_files.json"
        
        file_count = 0
        if processed_file.exists():
            with open(processed_file, 'r', encoding='utf-8-sig') as f:
                file_count = len(json.load(f))
        
        result.append(SpaceResponse(
            id=space_id,
            name=space['name'],
            created_at=space['created_at'],
            file_count=file_count
        ))
    
    return result

@app.post("/api/spaces", response_model=SpaceResponse)
async def create_space(space_data: SpaceCreate):
    """Create a new space"""
    try:
        space = spaces_manager.create_space(space_data.name)

        # Create associated notebook metadata with the same name as the space.
        studio_manager.ensure_space_notebook(space['id'], space['name'])
        
        return SpaceResponse(
            id=space['id'],
            name=space['name'],
            created_at=space['created_at'],
            file_count=0
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/api/spaces/{space_id}")
async def delete_space(space_id: str):
    """Delete a space"""
    try:
        spaces_manager.delete_space(space_id)
        
        # Delete space directory
        space_dir = get_space_dir(space_id)
        if space_dir.exists():
            shutil.rmtree(space_dir)
        
        return {"status": "success", "message": f"Space {space_id} deleted"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting space: {str(e)}")

@app.get("/api/spaces/{space_id}/chats", response_model=List[ChatInfo])
async def get_chats(space_id: str):
    """Get all chats for a space"""
    chats = load_chats_for_space(space_id)
    
    result = []
    for chat in chats:
        messages = chat.get('messages', [])
        result.append(ChatInfo(
            id=chat['id'],
            title=get_chat_title(messages),
            preview=messages[0]['content'][:100] if messages else "",
            created_at=chat.get('created_at', ''),
            updated_at=chat.get('updated_at', ''),
            message_count=len(messages)
        ))
    
    return result

@app.get("/api/spaces/{space_id}/chats/{chat_id}")
async def get_chat(space_id: str, chat_id: str):
    """Get specific chat by ID"""
    chats = load_chats_for_space(space_id)
    
    for chat in chats:
        if chat['id'] == chat_id:
            return chat
    
    raise HTTPException(status_code=404, detail="Chat not found")

@app.delete("/api/spaces/{space_id}/chats/{chat_id}")
async def delete_chat(space_id: str, chat_id: str):
    """Delete a chat"""
    chats = load_chats_for_space(space_id)
    chats = [c for c in chats if c['id'] != chat_id]
    save_chats_for_space(space_id, chats)
    
    return {"status": "success", "message": f"Chat {chat_id} deleted"}

@app.post("/api/spaces/{space_id}/chats/{chat_id}/messages/delete")
async def delete_chat_message(space_id: str, chat_id: str, request: DeleteMessageRequest):
    """Delete a specific message from a chat.

    Optionally deletes the nearest preceding user message as a Q/A pair.
    """
    chats = load_chats_for_space(space_id)

    chat = next((c for c in chats if c.get('id') == chat_id), None)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")

    messages = chat.get('messages', [])
    removed = False
    target_index = -1

    # Primary match: role + timestamp (+ optional exact content)
    for idx, msg in enumerate(messages):
        if msg.get('role') != request.role:
            continue
        if msg.get('timestamp') != request.timestamp:
            continue
        if request.content is not None and msg.get('content') != request.content:
            continue

        target_index = idx
        removed = True
        break

    # Fallback: role + exact content (handles rare timestamp drift)
    if not removed and request.content is not None:
        for idx in range(len(messages) - 1, -1, -1):
            msg = messages[idx]
            if msg.get('role') == request.role and msg.get('content') == request.content:
                target_index = idx
                removed = True
                break

    if not removed:
        raise HTTPException(status_code=404, detail="Message not found")

    indexes_to_remove = {target_index}

    if request.delete_related_user and target_index > 0:
        for idx in range(target_index - 1, -1, -1):
            if messages[idx].get('role') == 'user':
                indexes_to_remove.add(idx)
                break

    for idx in sorted(indexes_to_remove, reverse=True):
        if 0 <= idx < len(messages):
            del messages[idx]

    chat['messages'] = messages
    chat['updated_at'] = datetime.now().isoformat()
    save_chats_for_space(space_id, chats)

    return {
        "status": "success",
        "message": "Message deleted",
        "remaining_messages": len(messages),
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process a chat message with RAG"""
    try:
        trace_id = uuid.uuid4().hex[:8]
        t_start = time.perf_counter()
        print(
            f"[CHAT_TIMING {trace_id}] start space={request.space_id} workflow={request.workflow} "
            f"query_len={len((request.query or '').strip())}"
        )

        # Initialize space if needed
        t_init = time.perf_counter()
        initialize_space(request.space_id)
        print(f"[CHAT_TIMING {trace_id}] init_space_ms={(time.perf_counter() - t_init) * 1000:.1f}")
        
        # Reuse per-space retriever to avoid rebuilding BM25 index every request.
        global hybrid_retriever
        if hybrid_retriever is None:
            hybrid_retriever = HybridRetriever(vector_db, alpha=0.6)
            print(f"[CHAT_TIMING {trace_id}] retriever_cache=MISS")
        else:
            print(f"[CHAT_TIMING {trace_id}] retriever_cache=HIT")

        retrieval_n = int(os.getenv("CHAT_RETRIEVAL_RESULTS", "16"))
        retrieval_n = max(8, min(retrieval_n, 30))
        retrieval_threshold = float(os.getenv("CHAT_RETRIEVAL_THRESHOLD", "0.01"))
        chat_max_tokens = int(os.getenv("CHAT_MAX_TOKENS", "1800"))
        chat_max_tokens = max(700, min(chat_max_tokens, 2600))
        
        # Retrieve relevant documents (with targeted query expansion for metric-heavy NLP prompts).
        t_retrieve = time.perf_counter()
        documents, metadatas, scores = _retrieve_documents_for_query(
            retriever=hybrid_retriever,
            query=request.query,
            n_results=retrieval_n,
            score_threshold=retrieval_threshold,
        )
        print(
            f"[CHAT_TIMING {trace_id}] retrieval_ms={(time.perf_counter() - t_retrieve) * 1000:.1f} "
            f"docs={len(documents)} query_expansions={len(_build_retrieval_queries(request.query))}"
        )
        
        # Build context from retrieved documents
        t_context = time.perf_counter()
        context_parts = []
        sources = []
        filtered_metadatas = []
        dropped_chunks = 0
        seen_context_fingerprints = set()
        
        for doc, meta, score in zip(documents, metadatas, scores):
            cleaned_doc = _sanitize_retrieved_chunk(doc)
            if not _is_usable_context_chunk(cleaned_doc):
                dropped_chunks += 1
                continue

            fingerprint = re.sub(r'\s+', ' ', cleaned_doc[:320]).lower()
            if fingerprint in seen_context_fingerprints:
                dropped_chunks += 1
                continue
            seen_context_fingerprints.add(fingerprint)

            # Extract clean filename for source citation
            filename = meta.get('filename', 'Unknown')
            clean_name = filename.replace('.pdf', '').replace('.docx', '').replace('.txt', '')
            publication_year = _resolve_publication_year(meta)
            year_label = str(publication_year) if publication_year else "UNKNOWN"
            source_idx = len(context_parts) + 1
            context_parts.append(f"Source [{source_idx}] [YEAR: {year_label}] ({clean_name}):\n{cleaned_doc}\n")
            filtered_metadatas.append(meta)
            sources.append({
                "content": cleaned_doc[:200] + "..." if len(cleaned_doc) > 200 else cleaned_doc,
                "metadata": meta,
                "score": float(score)
            })

        print(
            f"[CHAT_TIMING {trace_id}] context_filter kept={len(context_parts)} dropped={dropped_chunks}"
        )

        if not context_parts:
            return ChatResponse(
                response=(
                    "I found related retrieval hits, but they were too noisy to trust for a grounded answer. "
                    "Please re-upload a cleaner source file (or a text-based PDF) and try again."
                ),
                sources=[],
                chat_id=request.chat_id or str(uuid.uuid4()),
                timestamp=datetime.now().isoformat(),
                ui_notice="Context chunks were filtered due to low text quality.",
                suggested_follow_ups=[],
            )
        
        context = "\n".join(context_parts)
        print(
            f"[CHAT_TIMING {trace_id}] context_ms={(time.perf_counter() - t_context) * 1000:.1f} "
            f"context_chars={len(context)}"
        )
        
        # Use the advanced generate_response method which has the new NotebookLM-style prompt
        t_llm = time.perf_counter()
        response = llm_generator.generate_response(
            prompt=request.query,
            context=context,
            use_case=request.workflow if request.workflow in ["summary", "explanation", "qa", "notes"] else "qa",
            metadatas=filtered_metadatas,
            temperature=0.35,
            max_tokens=chat_max_tokens,
        )
        print(
            f"[CHAT_TIMING {trace_id}] llm_ms={(time.perf_counter() - t_llm) * 1000:.1f} "
            f"response_chars={len(response or '')}"
        )

        t_follow = time.perf_counter()
        suggested_follow_ups = llm_generator.generate_follow_up_questions(
            prompt=request.query,
            context=context,
            answer=response,
            max_questions=3,
        )
        print(
            f"[CHAT_TIMING {trace_id}] followups_ms={(time.perf_counter() - t_follow) * 1000:.1f} "
            f"count={len(suggested_follow_ups)}"
        )
        ui_notice = getattr(llm_generator, 'last_notice', None)

        # Create or update chat
        chat_id = request.chat_id or str(uuid.uuid4())
        chats = load_chats_for_space(request.space_id)
        
        # Find existing chat or create new
        chat = None
        for c in chats:
            if c['id'] == chat_id:
                chat = c
                break
        
        if not chat:
            chat = {
                'id': chat_id,
                'messages': [],
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            chats.append(chat)
        
        # Add messages
        timestamp = datetime.now().isoformat()
        chat['messages'].extend([
            {'role': 'user', 'content': request.query, 'timestamp': timestamp},
            {
                'role': 'assistant',
                'content': response,
                'timestamp': timestamp,
                'sources': sources,
                'suggested_follow_ups': suggested_follow_ups,
            }
        ])
        chat['updated_at'] = timestamp
        
        # Save chats
        t_save = time.perf_counter()
        save_chats_for_space(request.space_id, chats)
        print(f"[CHAT_TIMING {trace_id}] save_ms={(time.perf_counter() - t_save) * 1000:.1f}")
        print(f"[CHAT_TIMING {trace_id}] total_ms={(time.perf_counter() - t_start) * 1000:.1f}")
        
        return ChatResponse(
            response=response,
            sources=sources,
            chat_id=chat_id,
            timestamp=timestamp,
            ui_notice=ui_notice,
            suggested_follow_ups=suggested_follow_ups,
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """SSE streaming compatibility endpoint for the Flutter client."""
    trace_id = "unknown"
    t_stream = time.perf_counter()
    try:
        trace_id = uuid.uuid4().hex[:8]
        t_stream = time.perf_counter()
        print(f"[CHAT_STREAM {trace_id}] start space={request.space_id}")

        def event_stream():
            try:
                # 1. Initialization & Retrieval
                initialize_space(request.space_id)
                global hybrid_retriever
                if hybrid_retriever is None:
                    hybrid_retriever = HybridRetriever(vector_db, alpha=0.6)
                
                retrieval_n = int(os.getenv("CHAT_RETRIEVAL_RESULTS", "16"))
                retrieval_n = max(8, min(retrieval_n, 30))
                retrieval_threshold = float(os.getenv("CHAT_RETRIEVAL_THRESHOLD", "0.01"))
                chat_max_tokens = int(os.getenv("CHAT_MAX_TOKENS", "1800"))
                chat_max_tokens = max(700, min(chat_max_tokens, 2600))

                documents, metadatas, scores = _retrieve_documents_for_query(
                    retriever=hybrid_retriever,
                    query=request.query,
                    n_results=retrieval_n,
                    score_threshold=retrieval_threshold,
                )

                # 2. Context Building
                context_parts = []
                sources = []
                filtered_metadatas = []
                seen_context_fingerprints = set()
                
                for doc, meta, score in zip(documents, metadatas, scores):
                    cleaned_doc = _sanitize_retrieved_chunk(doc)
                    if not _is_usable_context_chunk(cleaned_doc):
                        continue
                    
                    fingerprint = re.sub(r'\s+', ' ', cleaned_doc[:320]).lower()
                    if fingerprint in seen_context_fingerprints:
                        continue
                    seen_context_fingerprints.add(fingerprint)

                    filename = meta.get('filename', 'Unknown')
                    clean_name = filename.replace('.pdf', '').replace('.docx', '').replace('.txt', '')
                    publication_year = _resolve_publication_year(meta)
                    year_label = str(publication_year) if publication_year else "UNKNOWN"
                    source_idx = len(context_parts) + 1
                    
                    context_parts.append(f"Source [{source_idx}] [YEAR: {year_label}] ({clean_name}):\n{cleaned_doc}\n")
                    filtered_metadatas.append(meta)
                    sources.append({
                        "content": cleaned_doc[:200] + "..." if len(cleaned_doc) > 200 else cleaned_doc,
                        "metadata": meta,
                        "score": float(score)
                    })

                chat_id = request.chat_id or str(uuid.uuid4())
                timestamp = datetime.now().isoformat()

                if not context_parts:
                    yield _format_sse_event("meta", {
                        "chat_id": chat_id,
                        "timestamp": timestamp,
                        "sources": [],
                        "ui_notice": "Context chunks were filtered due to low text quality.",
                    })
                    msg = "I found related retrieval hits, but they were too noisy to trust for a grounded answer. Please re-upload a cleaner source file (or a text-based PDF) and try again."
                    yield _format_sse_event("token", {"delta": msg})
                    yield _format_sse_event("done", {
                        "chat_id": chat_id,
                        "timestamp": timestamp,
                        "response": msg,
                        "sources": [],
                        "ui_notice": "Context chunks were filtered due to low text quality.",
                        "suggested_follow_ups": [],
                    })
                    return

                context = "\n".join(context_parts)

                # 3. Yield Meta (unblocks UI, shows citations)
                yield _format_sse_event("meta", {
                    "chat_id": chat_id,
                    "timestamp": timestamp,
                    "sources": sources,
                    "ui_notice": getattr(llm_generator, 'last_notice', None),
                })

                # 4. Stream LLM tokens live
                full_response = ""
                generator = llm_generator.generate_stream(
                    prompt=request.query,
                    context=context,
                    use_case=request.workflow if request.workflow in ["summary", "explanation", "qa", "notes"] else "qa",
                    metadatas=filtered_metadatas,
                    temperature=0.35,
                    max_tokens=chat_max_tokens,
                )

                for chunk in generator:
                    if chunk:
                        full_response += chunk
                        yield _format_sse_event("token", {"delta": chunk})

                # 5. Generate follow-ups and save history
                suggested_follow_ups = llm_generator.generate_follow_up_questions(
                    prompt=request.query,
                    context=context,
                    answer=full_response,
                    max_questions=3,
                )

                if suggested_follow_ups:
                    yield _format_sse_event("followups", {
                        "suggested_follow_ups": suggested_follow_ups,
                    })

                chats = load_chats_for_space(request.space_id)
                chat = next((c for c in chats if c['id'] == chat_id), None)
                if not chat:
                    chat = {
                        'id': chat_id,
                        'messages': [],
                        'created_at': timestamp,
                        'updated_at': timestamp
                    }
                    chats.append(chat)
                
                chat['messages'].extend([
                    {'role': 'user', 'content': request.query, 'timestamp': timestamp},
                    {
                        'role': 'assistant',
                        'content': full_response,
                        'timestamp': timestamp,
                        'sources': sources,
                        'suggested_follow_ups': suggested_follow_ups,
                    }
                ])
                chat['updated_at'] = timestamp
                save_chats_for_space(request.space_id, chats)

                # 6. Yield Done
                yield _format_sse_event("done", {
                    "chat_id": chat_id,
                    "timestamp": timestamp,
                    "response": full_response,
                    "sources": sources,
                    "ui_notice": getattr(llm_generator, 'last_notice', None),
                    "suggested_follow_ups": suggested_follow_ups,
                })

            except Exception as e:
                print(f"[CHAT_STREAM ERROR] {trace_id}: {str(e)}")
                yield _format_sse_event("error", {"message": str(e)})

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    except Exception as exc:
        error_message = str(exc)
        elapsed_ms = (time.perf_counter() - t_stream) * 1000
        print(f"[CHAT_STREAM {trace_id}] error ms={elapsed_ms:.1f} detail={error_message}")

        def error_stream(message: str = error_message):
            yield _format_sse_event("error", {"message": message})

        return StreamingResponse(
            error_stream(),
            media_type="text/event-stream",
            status_code=500,
        )

@app.post("/api/spaces/{space_id}/upload")
async def upload_files(space_id: str, files: List[UploadFile] = File(...)):
    """Upload and process files for a space"""
    try:
        global hybrid_retriever
        # Initialize space
        initialize_space(space_id)
        
        # Save uploaded files temporarily
        space_dir = get_space_dir(space_id)
        uploads_dir = space_dir / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        
        processor = DocumentProcessor()
        all_chunks = []
        processed_files = []
        skipped_files = []

        processed_file_path = space_dir / "processed_files.json"
        existing_filenames = set()
        if processed_file_path.exists():
            with open(processed_file_path, 'r', encoding='utf-8-sig') as f:
                try:
                    for item in json.load(f):
                        existing_filenames.add(item.get('original_filename', item.get('filename')))
                except Exception:
                    pass
        
        for file in files:
            filename = (file.filename or "unnamed_upload").strip()

            if filename in existing_filenames:
                print(f"[UPLOAD] skip file={filename} reason=already_exists")
                skipped_files.append({"filename": filename, "reason": "File already exists in this space"})
                continue

            # Ensure we read from the start of the upload stream.
            try:
                await file.seek(0)
            except Exception:
                pass

            content = await file.read()
            content_size = len(content) if content else 0
            if content_size == 0:
                reason = "Upload payload was empty (0 bytes)"
                skipped_files.append({"filename": filename, "reason": reason})
                print(f"[UPLOAD] skip file={filename} reason={reason}")
                continue

            # Save file
            file_path = uploads_dir / filename
            with open(file_path, "wb") as f:
                f.write(content)
                f.flush()
                os.fsync(f.fileno())

            try:
                if file_path.stat().st_size == 0:
                    reason = "File became 0 bytes after write"
                    skipped_files.append({"filename": filename, "reason": reason})
                    print(f"[UPLOAD] skip file={filename} reason={reason}")
                    file_path.unlink(missing_ok=True)
                    continue
            except Exception:
                pass
            
            # Process file and extract content
            try:
                print(f"[UPLOAD] Extracting text from newly uploaded file: {filename}")
                file_data = processor.process_file(file_path)
                content = file_data['content']
                chunk_filename = file_data.get('virtual_filename', filename)
                
                # Chunk the content
                chunks = processor.chunk_text(
                    content,
                    chunk_size=config.CHUNK_SIZE,
                    overlap=config.CHUNK_OVERLAP,
                    semantic=True,
                    source_filename=chunk_filename,
                )
                
                # Format chunks for vector database
                formatted_chunks = []
                publication_year = file_data.get('publication_year', '0000')
                for idx, chunk in enumerate(chunks):
                    formatted_chunks.append({
                        'content': chunk,
                        'metadata': {
                            'filename': chunk_filename,
                            'chunk_index': idx,
                            'total_chunks': len(chunks),
                            'source_type': file_data['format'],
                            'publication_year': publication_year,
                        }
                    })
                
                all_chunks.extend(formatted_chunks)
                processed_files.append({
                    'filename': chunk_filename,
                    'original_filename': filename,
                    'publication_year': publication_year,
                    'chunks': len(chunks),
                    'processed_at': datetime.now().isoformat()
                })
                
                # Progressively save to processed_files.json so UI can see them immediately
                try:
                    p_file = space_dir / "processed_files.json"
                    ex = []
                    if p_file.exists():
                        with open(p_file, 'r', encoding='utf-8-sig') as f:
                            ex = json.load(f)
                    # Filter out ones we just added to avoid duplicates if we run this multiple times
                    ex = [x for x in ex if x.get('original_filename') != filename and x.get('filename') != chunk_filename]
                    ex.append(processed_files[-1])
                    with open(p_file, 'w', encoding='utf-8') as f:
                        json.dump(ex, f, indent=2)
                except Exception as e:
                    print(f"Error saving progressive processed_files.json: {e}")
                    
            except Exception as e:
                # Log error but continue with other files
                err = str(e)
                print(f"Error processing {filename}: {err}")
                skipped_files.append({"filename": filename, "reason": err})
                continue
        
        # Add to vector database in batches to avoid size limits
        if all_chunks:
            print(f"[UPLOAD] Finished extracting text. Now embedding {len(all_chunks)} chunks for the new files. This may take a while on CPU...")
            # Extract texts, metadatas, and generate IDs
            texts = [chunk['content'] for chunk in all_chunks]
            metadatas = [chunk['metadata'] for chunk in all_chunks]
            ids = [f"{space_id}_{idx}_{uuid.uuid4().hex[:8]}" for idx in range(len(all_chunks))]
            
            # Process in batches of 5000 to avoid ChromaDB batch size limit
            batch_size = 5000
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]
                batch_ids = ids[i:i + batch_size]
                
                vector_db.add_documents(batch_texts, batch_metadatas, batch_ids)
                print(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

            # Refresh retriever cache so BM25 index includes newly uploaded chunks.
            hybrid_retriever = HybridRetriever(vector_db, alpha=0.6)
            print(f"[UPLOAD] retriever_cache_refreshed space={space_id} chunks={len(all_chunks)}")
        
        # Save processed files info — write atomically to avoid corruption
        processed_file = space_dir / "processed_files.json"
        existing = []
        if processed_file.exists():
            try:
                with open(processed_file, 'r', encoding='utf-8-sig') as f:
                    existing = json.load(f)
            except Exception:
                existing = []
        
        # Deduplicate to prevent double-adding from the progressive save
        new_originals = {p.get('original_filename') for p in processed_files}
        new_virtuals = {p.get('filename') for p in processed_files}
        existing = [x for x in existing if x.get('original_filename') not in new_originals and x.get('filename') not in new_virtuals]
        
        existing.extend(processed_files)
        tmp_file = processed_file.with_suffix('.tmp')
        with open(tmp_file, 'w', encoding='utf-8') as f:
            json.dump(existing, f, indent=2, ensure_ascii=False)
        tmp_file.replace(processed_file)  # atomic rename
        
        print(f"[UPLOAD] Finished successfully! Returning response to UI.")
        
        return {
            "status": "success",
            "files_processed": len(processed_files),
            "total_chunks": len(all_chunks),
            "files_skipped": len(skipped_files),
            "skipped_files": skipped_files,
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/spaces/{space_id}/files")
async def get_files(space_id: str):
    """Get processed files for a space"""
    processed_file = get_space_dir(space_id) / "processed_files.json"
    
    if processed_file.exists():
        with open(processed_file, 'r', encoding='utf-8-sig') as f:
            return json.load(f)
    
    return []

@app.delete("/api/spaces/{space_id}/files/{filename}")
async def delete_file(space_id: str, filename: str):
    """Delete a specific file from a space"""
    try:
        original_filename = filename
        processed_file = get_space_dir(space_id) / "processed_files.json"
        if processed_file.exists():
            with open(processed_file, 'r', encoding='utf-8-sig') as f:
                files_data = json.load(f)

            # Support both virtual ([YEAR] name.pdf) and original filename lookups.
            for item in files_data:
                if item.get('filename') == filename:
                    original_filename = item.get('original_filename', filename)
                    break
                if item.get('original_filename') == filename:
                    original_filename = item.get('original_filename', filename)
                    break

            files_data = [
                f for f in files_data
                if f.get('filename') != filename and f.get('original_filename') != filename
            ]
            with open(processed_file, 'w', encoding='utf-8') as f:
                json.dump(files_data, f, indent=2, ensure_ascii=False)

        # Remove physical file from uploads directory.
        file_path = get_space_dir(space_id) / "uploads" / original_filename
        if file_path.exists():
            file_path.unlink()

        # Remove associated chunks from vector DB metadata.
        try:
            ids_to_delete: List[str] = []
            candidate_names = {filename, original_filename}
            try:
                for candidate in candidate_names:
                    filtered = vector_db.collection.get(where={"filename": candidate})
                    if filtered and filtered.get('ids'):
                        ids_to_delete.extend(filtered.get('ids', []))
            except Exception:
                # Fallback path for older Chroma versions where `where` can be inconsistent.
                all_results = vector_db.collection.get()
                all_ids = all_results.get('ids', []) if all_results else []
                all_metas = all_results.get('metadatas', []) if all_results else []
                for doc_id, metadata in zip(all_ids, all_metas):
                    if metadata and metadata.get('filename') in candidate_names:
                        ids_to_delete.append(doc_id)

            if ids_to_delete:
                vector_db.collection.delete(ids=list(set(ids_to_delete)))
                vector_db.client.persist()
                print(f"Deleted {len(ids_to_delete)} chunks for {filename}")
        except Exception as e:
            print(f"Vector cleanup warning for {filename}: {e}")

        return {
            "status": "success",
            "message": f"File {filename} deleted"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


import weasyprint
import markdown
import latex2mathml.converter
import re
from datetime import datetime
import html

def _render_latex_to_mathml(text: str) -> str:
    """Find math expressions in text and convert using ziamath."""
    # Remove citations like [13], **[1, 2]**, etc.
    text = re.sub(r'\s*(?:\*\*)?\[\d+(?:,\s*\d+)*\](?:\*\*)?', '', text)

    def render_math(m, display_mode):
        try:
            math_str = m.group(1).strip()
            # ziamath handles rendering SVG from latex
            svg = ziamath.Math.fromlatex(math_str).svg()
            if display_mode == 'block':
                return f'<div style="text-align: center; margin: 1em 0;">{svg}</div>'
            else:
                return f'<span style="display:inline-block; vertical-align:middle;">{svg}</span>'
        except Exception as e:
            print("MATH RENDER ERROR: ", e)
            return m.group(0)

    # Block math using \[ ... \]
    text = re.sub(r'\\\[([\s\S]+?)\\\]', lambda m: render_math(m, 'block'), text)

    # Inline math using \( ... \)
    text = re.sub(r'\\\(([\s\S]+?)\\\)', lambda m: render_math(m, 'inline'), text)

    # Block math using $$
    text = re.sub(r'\$\$([\s\S]+?)\$\$', lambda m: render_math(m, 'block'), text)

    # Inline math using $
    text = re.sub(r'\$(?!\$)([\s\S]+?)\$(?!\$)', lambda m: render_math(m, 'inline'), text)

    return text

def _render_html_page(title: str, body_html: str, space_id: str) -> bytes:
    """Generates WeasyPrint PDF from HTML."""
    exported_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{html.escape(title)}</title>
        <style>
            body {{ font-family: sans-serif; font-size: 11pt; line-height: 1.5; color: #333; }}
            h1 {{ font-size: 18pt; margin-bottom: 5px; }}
            .meta {{ font-size: 9pt; color: #666; margin-bottom: 20px; }}
            h2 {{ font-size: 14pt; margin-top: 15px; margin-bottom: 5px; }}
            h3 {{ font-size: 12.5pt; margin-top: 10px; margin-bottom: 5px; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 10px; }}
            th, td {{ border: 1px solid #ccc; padding: 6px; text-align: left; font-size: 10pt; }}
            pre {{ background-color: #f5f5f5; padding: 10px; font-size: 9.5pt; overflow-x: auto; }}
            code {{ font-family: monospace; background-color: #f5f5f5; padding: 2px 4px; }}
            blockquote {{ border-left: 3px solid #ccc; margin: 0; padding-left: 10px; color: #555; }}
            math {{ font-family: "Cambria Math", "Latin Modern Math", "STIX Two Math", math, serif; font-size: 1.1em; }}
            .answer-block {{ margin-bottom: 20px; }}
        </style>
    </head>
    <body>
        <h1>{html.escape(title)}</h1>
        <div class="meta">Space: {html.escape(space_id)} <br> Exported at: {exported_at}</div>
        {body_html}
    </body>
    </html>
    """
    pdf_bytes = weasyprint.HTML(string=html_content).write_pdf()
    return pdf_bytes

def _build_answers_pdf_bytes(title: str, items: list[dict], space_id: str) -> bytes:
    body_parts = []
    for item in items:
        q = item.get('question', '')
        a = item.get('answer', '')
        
        q_math = _render_latex_to_mathml(q)
        a_math = _render_latex_to_mathml(a)
        
        q_html = markdown.markdown(q_math, extensions=['tables', 'fenced_code'])
        a_html = markdown.markdown(a_math, extensions=['tables', 'fenced_code'])
        
        body_parts.append(f"<div class='answer-block'><h2>Question: {q_html}</h2><div class='answer'>{a_html}</div></div><hr>")
        
    return _render_html_page(title, "".join(body_parts), space_id)

def _safe_pdf_filename(name: str) -> str:
    base = (name or 'notebookpro_export').strip().replace(' ', '_')
    base = re.sub(r'[^A-Za-z0-9_\-]', '', base)
    if not base:
        base = 'notebookpro_export'
    return f"{base}.pdf"


def _apply_inline_format(text: str, attrs: dict) -> str:
    if not text: return text
    
    # Preserve leading/trailing whitespace
    l_space = len(text) - len(text.lstrip(' \t\n'))
    r_space = len(text) - len(text.rstrip(' \t\n'))
    
    inner = text.strip(' \t\n')
    if not inner: return text # If it's all spaces, don't wrap it
    
    if attrs.get('bold'): inner = f"**{inner}**"
    if attrs.get('italic'): inner = f"*{inner}*"
    if attrs.get('strike'): inner = f"~~{inner}~~"
    if attrs.get('code'): inner = f"`{inner}`"
    if attrs.get('link'): inner = f"[{inner}]({attrs.get('link')})"
    
    return text[:l_space] + inner + text[len(text)-r_space:] if r_space > 0 else text[:l_space] + inner

def _quill_or_plain_to_text(content: str) -> str:
    """Convert plain text or Quill delta JSON string into Markdown text."""
    if not content:
        return ""

    stripped = content.strip()
    if not stripped:
        return ""

    try:
        parsed = json.loads(stripped)
        ops = []
        if isinstance(parsed, dict) and isinstance(parsed.get('ops'), list):
            ops = parsed.get('ops', [])
        elif isinstance(parsed, list):
            ops = parsed
        else:
            return stripped

        lines = []
        current_line_ops = []
        
        for op in ops:
            if not isinstance(op, dict):
                continue
                
            insert_value = op.get('insert')
            if not isinstance(insert_value, str):
                continue
                
            attrs = op.get('attributes', {})
            parts = insert_value.split('\n')
            
            for i, part in enumerate(parts):
                if i > 0:
                    line_md = ""
                    for l_op, l_attrs in current_line_ops:
                        line_md += _apply_inline_format(l_op, l_attrs)
                    
                    if attrs.get('header') == 1: line_md = f"# {line_md}"
                    elif attrs.get('header') == 2: line_md = f"## {line_md}"
                    elif attrs.get('header') == 3: line_md = f"### {line_md}"
                    elif attrs.get('header') == 4: line_md = f"#### {line_md}"
                    elif attrs.get('blockquote'): line_md = f"> {line_md}"
                    elif attrs.get('code-block'): line_md = f"```\n{line_md}\n```"
                    elif attrs.get('list') == 'bullet': line_md = f"* {line_md}"
                    elif attrs.get('list') == 'ordered': line_md = f"1. {line_md}"
                    
                    lines.append(line_md)
                    current_line_ops = []
                
                if part:
                    current_line_ops.append((part, attrs))
                    
        if current_line_ops:
            line_md = ""
            for l_op, l_attrs in current_line_ops:
                line_md += _apply_inline_format(l_op, l_attrs)
            lines.append(line_md)
            
        text = '\n'.join(lines)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()
    except Exception:
        return stripped


def _extract_chat_qa(text: str) -> Optional[Dict[str, str]]:
    """Extract chat-style Q/A blocks from plain text."""
    if not text:
        return None

    match = re.match(r'^\s*Q:\s*(.*?)\s*\n\s*\n\s*A:\s*(.*)$', text, flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return None

    question = (match.group(1) or '').strip()
    answer = (match.group(2) or '').strip()
    if not question or not answer:
        return None
    return {'question': question, 'answer': answer}


def _normalize_compare_text(value: str) -> str:
    v = (value or '').lower().strip()
    v = v.replace('...', ' ')
    v = re.sub(r'[^a-z0-9\s]', ' ', v)
    v = re.sub(r'\s+', ' ', v).strip()
    return v


def _is_same_question_title(title: str, question: str) -> bool:
    """Fuzzy equality to match truncated titles with full questions."""
    t = _normalize_compare_text(title)
    q = _normalize_compare_text(question)
    if not t or not q:
        return False

    if t == q:
        return True

    # Prefix checks for truncated titles.
    if len(t) >= 18 and q.startswith(t):
        return True
    if len(q) >= 18 and t.startswith(q):
        return True

    # Token-prefix overlap check (first several tokens usually define the question).
    t_tokens = t.split()
    q_tokens = q.split()
    common = min(len(t_tokens), len(q_tokens), 8)
    if common >= 4 and t_tokens[:common] == q_tokens[:common]:
        return True

    return False


def _normalize_chat_entry_body(entry: NotebookEntry, body_text: str) -> str:
    """For chat entries, avoid repeating question in body when title already carries it."""
    qa = _extract_chat_qa(body_text)
    if not qa:
        return body_text

    title = (entry.title or '').strip()
    question = qa['question'].strip()

    if not title:
        return f"Question: {qa['question']}\n\nAnswer: {qa['answer']}"

    # If title and question are effectively the same, keep only the answer body.
    if _is_same_question_title(title, question):
        return qa['answer']

    return f"Question: {qa['question']}\n\nAnswer: {qa['answer']}"


def _build_notebook_entries_pdf_bytes(title: str, entries: list, space_id: str) -> bytes:
    body_parts = []
    for idx, entry in enumerate(entries, 1):
        body_text = _quill_or_plain_to_text(entry.content)
        entry_title = entry.title or f"Note {idx}"

        if entry.source_type == 'chat':
            qa = _extract_chat_qa(body_text)
            if qa and qa.get('question'):
                entry_title = qa['question']
            body_text = _normalize_chat_entry_body(entry, body_text)

        updated = entry.updated_at.strftime('%Y-%m-%d %H:%M') if entry.updated_at else ''
        
        body_text = _render_latex_to_mathml(body_text)
        body_html = markdown.markdown(body_text, extensions=['tables', 'fenced_code'])
        
        updated_str = f"<div class='meta'>Updated: {updated}</div>" if updated else ""
        body_parts.append(f"<div class='answer-block'><h2>{idx}. {html.escape(entry_title)}</h2>{updated_str}<div>{body_html}</div></div><hr>")
        
    return _render_html_page(title, "".join(body_parts), space_id)


@app.post("/api/studio/notebook", response_model=NotebookEntry)
async def create_notebook_entry(entry_data: NotebookEntryCreate):
    """Create a new notebook entry"""
    try:
        entry = studio_manager.create_notebook_entry(entry_data)
        return entry
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/studio/notebook/space/{space_id}")
async def get_space_notebook(space_id: str):
    """Get or create notebook metadata for a space."""
    try:
        space = spaces_manager.get_space(space_id)
        space_name = space['name'] if space else space_id
        notebook = studio_manager.ensure_space_notebook(space_id, space_name)
        return notebook
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/studio/notebook/from-chat", response_model=NotebookEntry)
async def add_chat_to_notebook(request: ChatToNotebookRequest):
    """Add a chat question/answer pair into a space notebook."""
    try:
        space = spaces_manager.get_space(request.space_id)
        resolved_space_name = request.space_name or (space['name'] if space else request.space_id)

        entry = studio_manager.create_notebook_entry_from_chat(
            space_id=request.space_id,
            question=request.question,
            answer=request.answer,
            chat_id=request.chat_id,
            assistant_timestamp=request.assistant_timestamp,
            tags=request.tags,
            space_name=resolved_space_name
        )
        return entry
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/studio/notebook", response_model=List[NotebookEntry])
async def list_notebook_entries(space_id: Optional[str] = None):
    """List all notebook entries, optionally filtered by space"""
    try:
        entries = studio_manager.list_notebook_entries(space_id)
        return entries
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/studio/notebook/{entry_id}", response_model=NotebookEntry)
async def get_notebook_entry(entry_id: str):
    """Get a single notebook entry"""
    entry = studio_manager.get_notebook_entry(entry_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Notebook entry not found")
    return entry

@app.put("/api/studio/notebook/{entry_id}", response_model=NotebookEntry)
async def update_notebook_entry(entry_id: str, update_data: NotebookEntryUpdate):
    """Update a notebook entry"""
    entry = studio_manager.update_notebook_entry(entry_id, update_data)
    if not entry:
        raise HTTPException(status_code=404, detail="Notebook entry not found")
    return entry

@app.delete("/api/studio/notebook/{entry_id}")
async def delete_notebook_entry(entry_id: str):
    """Delete a notebook entry"""
    success = studio_manager.delete_notebook_entry(entry_id)
    if not success:
        raise HTTPException(status_code=404, detail="Notebook entry not found")
    return {"status": "success", "message": "Notebook entry deleted"}


@app.post("/api/studio/notebook/export/answer-pdf")
async def export_single_answer_pdf(request: ExportAnswerPdfRequest):
    """Export one Q/A pair as a downloadable PDF."""
    try:
        title = request.title or "NotebookPRO Answer Export"
        pdf_bytes = _build_answers_pdf_bytes(
            title=title,
            space_id=request.space_id,
            items=[{"question": request.question, "answer": request.answer}],
        )
        filename = _safe_pdf_filename(title)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export answer PDF: {e}")


@app.post("/api/studio/notebook/export/combined-pdf")
async def export_combined_answers_pdf(request: ExportCombinedAnswersPdfRequest):
    """Export all provided Q/A pairs as one combined downloadable PDF."""
    try:
        if not request.items:
            raise HTTPException(status_code=400, detail="No answers provided for export")

        title = request.chat_title or "NotebookPRO Combined Answers"
        pdf_bytes = _build_answers_pdf_bytes(
            title=title,
            space_id=request.space_id,
            items=[{"question": i.question, "answer": i.answer} for i in request.items],
        )
        filename = _safe_pdf_filename(title)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export combined PDF: {e}")


@app.get("/api/studio/notebook/export/entry/{entry_id}/pdf")
async def export_notebook_entry_pdf(entry_id: str):
    """Export one notebook entry as a downloadable PDF."""
    try:
        entry = studio_manager.get_notebook_entry(entry_id)
        if not entry:
            raise HTTPException(status_code=404, detail="Notebook entry not found")

        title = f"Notebook Entry - {entry.title}"
        pdf_bytes = _build_notebook_entries_pdf_bytes(title, [entry], entry.space_id)
        filename = _safe_pdf_filename(entry.title or f"entry_{entry.id}")
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export notebook entry PDF: {e}")


@app.get("/api/studio/notebook/export/space/{space_id}/combined-pdf")
async def export_space_notebook_pdf(space_id: str):
    """Export all notebook entries for a space as a single PDF."""
    try:
        entries = studio_manager.list_notebook_entries(space_id=space_id)
        if not entries:
            raise HTTPException(status_code=404, detail="No notebook entries found for this space")

        space = spaces_manager.get_space(space_id)
        space_name = space['name'] if space else space_id
        title = f"Notebook - {space_name}"
        pdf_bytes = _build_notebook_entries_pdf_bytes(title, entries, space_id)
        filename = _safe_pdf_filename(f"{space_name}_notebook")
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export space notebook PDF: {e}")


@app.post("/api/studio/notebook/export/selected-pdf")
async def export_selected_notebook_pdf(request: ExportSelectedNotebookPdfRequest):
    """Export selected notebook entries for a space as a single combined PDF."""
    try:
        if not request.entry_ids:
            raise HTTPException(status_code=400, detail="No notebook entries selected for export")

        entries = []
        for entry_id in request.entry_ids:
            entry = studio_manager.get_notebook_entry(entry_id)
            if not entry:
                raise HTTPException(status_code=404, detail=f"Notebook entry not found: {entry_id}")
            if entry.space_id != request.space_id:
                raise HTTPException(
                    status_code=400,
                    detail=f"Notebook entry does not belong to this space: {entry_id}",
                )
            entries.append(entry)

        if not entries:
            raise HTTPException(status_code=404, detail="No notebook entries found for export")

        space = spaces_manager.get_space(request.space_id)
        space_name = space['name'] if space else request.space_id
        title = request.title or f"Notebook Selection - {space_name}"
        pdf_bytes = _build_notebook_entries_pdf_bytes(title, entries, request.space_id)
        filename = _safe_pdf_filename(f"{space_name}_notebook_selected")

        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export selected notebook PDF: {e}")


# ===== FLASHCARD ROUTES =====

@app.post("/api/studio/flashcards", response_model=Flashcard)
async def create_flashcard(card_data: FlashcardCreate):
    """Create a new flashcard"""
    try:
        card = studio_manager.create_flashcard(card_data)
        return card
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/studio/flashcards", response_model=List[Flashcard])
async def list_flashcards(
    space_id: Optional[str] = None,
    mastery: Optional[MasteryLevel] = None
):
    """List all flashcards, optionally filtered"""
    try:
        cards = studio_manager.list_flashcards(space_id, mastery)
        return cards
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/studio/flashcards/{card_id}", response_model=Flashcard)
async def get_flashcard(card_id: str):
    """Get a single flashcard"""
    card = studio_manager.get_flashcard(card_id)
    if not card:
        raise HTTPException(status_code=404, detail="Flashcard not found")
    return card

@app.put("/api/studio/flashcards/{card_id}", response_model=Flashcard)
async def update_flashcard(card_id: str, update_data: FlashcardUpdate):
    """Update a flashcard"""
    card = studio_manager.update_flashcard(card_id, update_data)
    if not card:
        raise HTTPException(status_code=404, detail="Flashcard not found")
    return card

@app.post("/api/studio/flashcards/{card_id}/review", response_model=Flashcard)
async def review_flashcard(card_id: str, review: FlashcardReview):
    """Record a flashcard review"""
    card = studio_manager.review_flashcard(card_id, review)
    if not card:
        raise HTTPException(status_code=404, detail="Flashcard not found")
    return card

@app.delete("/api/studio/flashcards/{card_id}")
async def delete_flashcard(card_id: str):
    """Delete a flashcard"""
    success = studio_manager.delete_flashcard(card_id)
    if not success:
        raise HTTPException(status_code=404, detail="Flashcard not found")
    return {"status": "success", "message": "Flashcard deleted"}

@app.post("/api/studio/flashcards/generate", response_model=List[Flashcard])
async def generate_flashcards(request: FlashcardGenerateRequest):
    """Generate flashcards from content using LLM"""
    global studio_generator
    
    if not studio_generator:
        raise HTTPException(status_code=503, detail="LLM not initialized")
    
    try:
        cards = await studio_generator.generate_flashcards(request)
        return cards
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ===== QUIZ ROUTES =====

@app.post("/api/studio/quizzes", response_model=Quiz)
async def create_quiz(quiz_data: QuizCreate):
    """Create a new quiz"""
    try:
        quiz = studio_manager.create_quiz(quiz_data)
        return quiz
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/studio/quizzes", response_model=List[Quiz])
async def list_quizzes(space_id: Optional[str] = None):
    """List all quizzes, optionally filtered by space"""
    try:
        quizzes = studio_manager.list_quizzes(space_id)
        return quizzes
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/studio/quizzes/{quiz_id}", response_model=Quiz)
async def get_quiz(quiz_id: str):
    """Get a quiz by ID"""
    quiz = studio_manager.get_quiz(quiz_id)
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")
    return quiz

@app.delete("/api/studio/quizzes/{quiz_id}")
async def delete_quiz(quiz_id: str):
    """Delete a quiz"""
    success = studio_manager.delete_quiz(quiz_id)
    if not success:
        raise HTTPException(status_code=404, detail="Quiz not found")
    return {"status": "success", "message": "Quiz deleted"}

@app.post("/api/studio/quizzes/generate", response_model=Quiz)
async def generate_quiz(request: QuizGenerateRequest):
    """Generate a quiz from content using LLM"""
    global studio_generator
    
    if not studio_generator:
        raise HTTPException(status_code=503, detail="LLM not initialized")
    
    try:
        quiz = await studio_generator.generate_quiz(request)
        if not quiz:
            raise HTTPException(status_code=500, detail="Failed to generate quiz")
        return quiz
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/studio/quizzes/{quiz_id}/submit", response_model=QuizResult)
async def submit_quiz(quiz_id: str, submission: QuizSubmission):
    """Submit quiz answers and get results"""
    try:
        result = studio_manager.submit_quiz(quiz_id, submission.answers)
        if not result:
            raise HTTPException(status_code=404, detail="Quiz not found")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/studio/quizzes/{quiz_id}/history", response_model=QuizHistory)
async def get_quiz_history(quiz_id: str):
    """Get quiz attempt history"""
    try:
        history = studio_manager.get_quiz_history(quiz_id)
        if not history:
            raise HTTPException(status_code=404, detail="Quiz not found")
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")

# ==================== Run Server ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="error")

