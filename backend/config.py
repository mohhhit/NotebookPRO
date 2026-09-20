import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys

from pathlib import Path

from dotenv import load_dotenv



# Project paths

CODE_ROOT = Path(__file__).parent
if getattr(sys, "frozen", False) or "Program Files" in str(CODE_ROOT) or "ProgramFiles" in str(CODE_ROOT):
    PROJECT_ROOT = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "NotebookPRO"
else:
    PROJECT_ROOT = CODE_ROOT



# Load environment variables from backend/.env and project-root/.env.

for env_file in [PROJECT_ROOT / ".env", PROJECT_ROOT.parent / ".env"]:

    if env_file.exists():

        load_dotenv(env_file, override=False)



DATA_DIR = PROJECT_ROOT / "data"

MODELS_DIR = PROJECT_ROOT / "models"

UPLOADS_DIR = DATA_DIR / "uploads"

VECTOR_DB_DIR = DATA_DIR / "vector_db"

CHATS_DIR = DATA_DIR / "chats"



# Create directories if they don't exist

for dir_path in [DATA_DIR, MODELS_DIR, UPLOADS_DIR, VECTOR_DB_DIR, CHATS_DIR]:

    dir_path.mkdir(parents=True, exist_ok=True)



# Model configuration

# RAG uses pre-trained models directly - no training required!

MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/phi-2")  # Pre-trained model
FOLLOWUP_MODEL = os.getenv("FOLLOWUP_MODEL", "meta/llama-3.3-70b-instruct")  # Fast model for follow-ups

USE_PRETRAINED = os.getenv("USE_PRETRAINED", "true").lower() == "true"  # Use pre-trained by default

MODEL_PATH = os.getenv("MODEL_PATH", str(MODELS_DIR / "trained_model"))  # Only if fine-tuned

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")  # For document embeddings
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))
RERANKER_BATCH_SIZE = int(os.getenv("RERANKER_BATCH_SIZE", "8"))



# API Keys

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")



# Application settings

MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", "200"))  # MB

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))

MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))

# Memory behavior when switching spaces
SPACE_SWITCH_CLEAR_CUDA_CACHE = os.getenv("SPACE_SWITCH_CLEAR_CUDA_CACHE", "true").lower() == "true"
SPACE_SWITCH_UNLOAD_EMBEDDING_MODEL = os.getenv("SPACE_SWITCH_UNLOAD_EMBEDDING_MODEL", "false").lower() == "true"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "700"))

CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))



# Use cases

USE_CASES = {

    "explanation": "Provide detailed explanation of concepts",

    "summary": "Generate concise summary of content",

    "qa": "Answer questions based on content",

    "notes": "Create structured study notes"

}

