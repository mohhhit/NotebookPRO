import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional
from pathlib import Path
import config
import logging
import warnings

# Suppress unnecessary warnings
warnings.filterwarnings('ignore', category=FutureWarning)
logging.getLogger('chromadb').setLevel(logging.ERROR)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)


class VectorDatabase:
    """Manage vector database for document embeddings using ChromaDB."""
    
    def __init__(self, collection_name: str = "documents", persist_directory: str = None):
        """
        Initialize the vector database.
        
        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist the database
        """
        if persist_directory is None:
            persist_directory = str(config.VECTOR_DB_DIR)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize embedding model (downloads on first run)
        print("Loading embedding model (sentence-transformers)...")
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL, device='cpu')
        print("✓ Embedding model loaded!")
    
    def add_documents(self, texts: List[str], metadatas: List[Dict], ids: List[str]):
        """
        Add documents to the vector database.
        
        Args:
            texts: List of text chunks
            metadatas: List of metadata dictionaries
            ids: List of unique IDs for each chunk
        """
        if not texts:
            return
        
        # Generate embeddings (suppress progress bar in production)
        embeddings = self.embedding_model.encode(
            texts, 
            show_progress_bar=False,
            batch_size=32
        ).tolist()
        
        # Add to collection
        self.collection.add(
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
            ids=ids
        )
    
    def query(self, query_text: str, n_results: int = 5, filter_dict: Optional[Dict] = None) -> Dict:
        """
        Query the vector database for similar documents.
        
        Args:
            query_text: The query string
            n_results: Number of results to return
            filter_dict: Optional filter dictionary for metadata
            
        Returns:
            Dictionary containing query results
        """
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query_text]).tolist()
        
        # Query the collection
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=filter_dict
        )
        
        return results
    
    def delete_collection(self):
        """Delete the entire collection."""
        self.client.delete_collection(name=self.collection.name)
    
    def get_collection_count(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()
    
    def get_all_documents(self) -> tuple[List[str], List[Dict]]:
        """
        Get all documents and metadata from the collection.
        
        Returns:
            Tuple of (documents, metadatas)
        """
        count = self.collection.count()
        if count == 0:
            return [], []
        
        # Get all documents
        results = self.collection.get()
        
        return results.get('documents', []), results.get('metadatas', [])
    
    def create_space_collection(self, space_name: str):
        """
        Create a new collection for a specific subject space.
        
        Args:
            space_name: Name of the subject space
        """
        return self.client.get_or_create_collection(
            name=f"space_{space_name}",
            metadata={"hnsw:space": "cosine"}
        )
