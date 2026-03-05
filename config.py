import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent
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
USE_PRETRAINED = os.getenv("USE_PRETRAINED", "true").lower() == "true"  # Use pre-trained by default
MODEL_PATH = os.getenv("MODEL_PATH", str(MODELS_DIR / "trained_model"))  # Only if fine-tuned
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # For document embeddings

# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")

# Application settings
MAX_UPLOAD_SIZE = int(os.getenv("MAX_UPLOAD_SIZE", "200"))  # MB
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2048"))
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Use cases
USE_CASES = {
    "explanation": "Provide detailed explanation of concepts",
    "summary": "Generate concise summary of content",
    "qa": "Answer questions based on content",
    "notes": "Create structured study notes"
}
