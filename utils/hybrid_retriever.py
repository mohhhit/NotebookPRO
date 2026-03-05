"""
Hybrid Retriever: Combines Vector (ChromaDB) + Keyword (BM25) search.
This is the "secret sauce" that makes NotebookLM so accurate.
"""

from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
import numpy as np


class HybridRetriever:
    """
    Combines Dense Retrieval (Embeddings) with Sparse Retrieval (BM25).
    
    This is crucial for accuracy:
    - Vector search finds conceptually similar content
    - BM25 finds exact keyword matches (formulas, terms, names)
    """
    
    def __init__(self, vector_db, alpha: float = 0.5):
        """
        Initialize hybrid retriever.
        
        Args:
            vector_db: VectorDatabase instance
            alpha: Weight balance (0=only BM25, 1=only vector, 0.5=balanced)
        """
        self.vector_db = vector_db
        self.alpha = alpha  # Weight for vector search
        self.bm25 = None
        self.bm25_corpus = []
        self.bm25_metadata = []
        
    def index_documents(self, documents: List[str], metadatas: List[Dict]):
        """
        Index documents for BM25 keyword search.
        
        Args:
            documents: List of document chunks
            metadatas: List of metadata dicts for each chunk
        """
        # Tokenize documents for BM25
        tokenized_corpus = [doc.lower().split() for doc in documents]
        
        # Create BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.bm25_corpus = documents
        self.bm25_metadata = metadatas
        
    def retrieve(
        self, 
        query: str, 
        n_results: int = 5,
        score_threshold: float = 0.0
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """
        Hybrid retrieval: combines vector + keyword search.
        
        Args:
            query: User's question
            n_results: Number of chunks to retrieve
            score_threshold: Minimum score threshold
            
        Returns:
            Tuple of (documents, metadatas, scores)
        """
        if not self.bm25:
            # Fallback to pure vector search if BM25 not initialized
            results = self.vector_db.query(query, n_results=n_results * 2)
            return (
                results['documents'][0] if results['documents'] else [],
                results['metadatas'][0] if results['metadatas'] else [],
                results.get('distances', [[]])[0]
            )
        
        # Get more results than needed for reranking
        fetch_size = n_results * 3
        
        # 1. Vector search (semantic similarity)
        vector_results = self.vector_db.query(query, n_results=fetch_size)
        vector_docs = vector_results['documents'][0] if vector_results['documents'] else []
        vector_meta = vector_results['metadatas'][0] if vector_results['metadatas'] else []
        vector_distances = vector_results.get('distances', [[]])[0]
        
        # Convert distances to similarity scores (chromadb uses cosine distance)
        vector_scores = [1 / (1 + d) for d in vector_distances]
        
        # 2. BM25 search (keyword matching)
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top BM25 results
        top_bm25_indices = np.argsort(bm25_scores)[::-1][:fetch_size]
        
        # 3. Combine results with weighted scoring
        combined_docs = {}  # Use dict to deduplicate by content
        
        # Add vector results
        for doc, meta, score in zip(vector_docs, vector_meta, vector_scores):
            combined_docs[doc] = {
                'doc': doc,
                'meta': meta,
                'score': self.alpha * score
            }
        
        # Add BM25 results (normalize scores to 0-1 range)
        max_bm25_score = max(bm25_scores) if max(bm25_scores) > 0 else 1
        for idx in top_bm25_indices:
            doc = self.bm25_corpus[idx]
            meta = self.bm25_metadata[idx]
            bm25_score = bm25_scores[idx] / max_bm25_score
            
            if doc in combined_docs:
                # Average if document found by both methods
                combined_docs[doc]['score'] += (1 - self.alpha) * bm25_score
            else:
                combined_docs[doc] = {
                    'doc': doc,
                    'meta': meta,
                    'score': (1 - self.alpha) * bm25_score
                }
        
        # 4. Rank by combined score
        ranked_results = sorted(
            combined_docs.values(), 
            key=lambda x: x['score'], 
            reverse=True
        )
        
        # 5. Filter by threshold and limit results
        filtered_results = [
            r for r in ranked_results 
            if r['score'] >= score_threshold
        ][:n_results]
        
        # 6. Return in expected format
        documents = [r['doc'] for r in filtered_results]
        metadatas = [r['meta'] for r in filtered_results]
        scores = [r['score'] for r in filtered_results]
        
        return documents, metadatas, scores
    
    def get_stats(self) -> Dict:
        """Get retriever statistics."""
        return {
            'bm25_indexed': len(self.bm25_corpus) if self.bm25 else 0,
            'vector_count': self.vector_db.get_collection_count(),
            'alpha': self.alpha
        }
