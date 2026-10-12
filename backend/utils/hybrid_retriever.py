"""
Hybrid Retriever: Combines Vector (ChromaDB) + Keyword (BM25) search.
This is the "secret sauce" that makes NotebookLM so accurate.
"""

from typing import List, Dict, Tuple
from rank_bm25 import BM25Okapi
import numpy as np
import re
import time


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

    def _tokenize(self, text: str) -> List[str]:
        """Lightweight tokenizer for BM25."""
        return re.findall(r"\w+", text.lower())

    def ensure_bm25_index(self):
        """Build BM25 index lazily from vector DB when needed."""
        if self.bm25 is not None:
            return

        t0 = time.perf_counter()
        print("[RETRIEVER] bm25_build_start")

        documents, metadatas = self.vector_db.get_all_documents()
        if not documents:
            print("[RETRIEVER] bm25_build_skip empty_corpus")
            return

        self.index_documents(documents, metadatas)
        print(
            f"[RETRIEVER] bm25_build_done ms={(time.perf_counter() - t0) * 1000:.1f} "
            f"docs={len(documents)}"
        )
        
    def index_documents(self, documents: List[str], metadatas: List[Dict]):
        """
        Index documents for BM25 keyword search.
        
        Args:
            documents: List of document chunks
            metadatas: List of metadata dicts for each chunk
        """
        # Tokenize documents for BM25
        tokenized_corpus = [self._tokenize(doc) for doc in documents]
        
        # Create BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.bm25_corpus = documents
        self.bm25_metadata = metadatas
        
    def retrieve(
        self, 
        query: str, 
        n_results: int = 20,
        score_threshold: float = 0.0
    ) -> Tuple[List[str], List[Dict], List[float]]:
        """
        True Hybrid retrieval: combines Dense Vector search + Sparse BM25 keyword search,
        followed by Cross-Encoder reranking.
        
        Args:
            query: User's question
            n_results: Number of chunks to retrieve
            score_threshold: Minimum score threshold
            
        Returns:
            Tuple of (documents, metadatas, scores)
        """
        fetch_size = max(n_results * 3, 40)
        
        # 1. Dense Vector Search (semantic similarity)
        vector_docs: List[str] = []
        vector_meta: List[Dict] = []
        vector_scores: List[float] = []
        try:
            vector_results = self.vector_db.query(query, n_results=fetch_size)
            vector_docs = vector_results.get('documents', [[]])[0]
            vector_meta = vector_results.get('metadatas', [[]])[0]
            vector_distances = vector_results.get('distances', [[]])[0]
            vector_scores = [1.0 / (1.0 + d) for d in vector_distances]
        except Exception as e:
            print(f"[RETRIEVER] vector_query_error: {e}")

        # 2. Sparse BM25 Keyword Search (exact keyword matches like 'Prometheus')
        self.ensure_bm25_index()
        bm25_docs: List[str] = []
        bm25_meta: List[Dict] = []
        bm25_scores: List[float] = []
        if self.bm25:
            try:
                tokenized_query = self._tokenize(query)
                if tokenized_query:
                    raw_bm25_scores = self.bm25.get_scores(tokenized_query)
                    top_bm25_indices = np.argsort(raw_bm25_scores)[::-1][:fetch_size]
                    max_score = float(np.max(raw_bm25_scores)) if len(raw_bm25_scores) > 0 and np.max(raw_bm25_scores) > 0 else 1.0
                    for idx in top_bm25_indices:
                        if raw_bm25_scores[idx] > 0:
                            bm25_docs.append(self.bm25_corpus[idx])
                            bm25_meta.append(self.bm25_metadata[idx])
                            bm25_scores.append(float(raw_bm25_scores[idx]) / max_score)
            except Exception as e:
                print(f"[RETRIEVER] bm25_query_error: {e}")

        # 3. Deduplicate and merge candidates into a unified pool
        candidates_map: Dict[str, Dict] = {}
        
        # Add vector search results
        for doc, meta, score in zip(vector_docs, vector_meta, vector_scores):
            fp = doc.strip()[:240]
            candidates_map[fp] = {
                'doc': doc,
                'meta': meta,
                'vector_score': score,
                'bm25_score': 0.0,
                'score': self.alpha * score,
            }

        # Add BM25 keyword results
        for doc, meta, score in zip(bm25_docs, bm25_meta, bm25_scores):
            fp = doc.strip()[:240]
            if fp in candidates_map:
                candidates_map[fp]['bm25_score'] = score
                candidates_map[fp]['score'] = (
                    self.alpha * candidates_map[fp]['vector_score'] + (1 - self.alpha) * score
                )
            else:
                candidates_map[fp] = {
                    'doc': doc,
                    'meta': meta,
                    'vector_score': 0.0,
                    'bm25_score': score,
                    'score': (1 - self.alpha) * score,
                }

        candidate_list = list(candidates_map.values())
        if not candidate_list:
            return [], [], []

        # 4. Two-stage Cross-Encoder Reranking on merged candidate pool
        if hasattr(self.vector_db, 'rerank_documents'):
            try:
                candidate_docs = [c['doc'] for c in candidate_list]
                rerank_scores = self.vector_db.rerank_documents(query, candidate_docs)
                if rerank_scores and len(rerank_scores) == len(candidate_list):
                    for c, r_score in zip(candidate_list, rerank_scores):
                        c['score'] = r_score
            except Exception as e:
                print(f"[RETRIEVER] cross_encoder_rerank_failed fallback_to_hybrid_score: {e}")

        # 5. Rank by final score
        ranked_results = sorted(
            candidate_list,
            key=lambda x: x['score'],
            reverse=True,
        )

        # 6. Filter by threshold (keep at least top n_results if all fall below strict threshold)
        filtered_results = [
            r for r in ranked_results
            if r['score'] >= score_threshold
        ]
        if not filtered_results and ranked_results:
            filtered_results = ranked_results[:n_results]

        # 7. Select with soft per-source caps for multi-source diversity
        selected = []
        source_counts = {}
        max_per_source = max(3, n_results // 2)
        for r in filtered_results:
            filename = r['meta'].get('filename', '')
            current = source_counts.get(filename, 0)
            if current < max_per_source:
                selected.append(r)
                source_counts[filename] = current + 1
            if len(selected) >= n_results:
                break

        if len(selected) < n_results:
            for r in filtered_results:
                if r in selected:
                    continue
                selected.append(r)
                if len(selected) >= n_results:
                    break

        return (
            [r['doc'] for r in selected],
            [r['meta'] for r in selected],
            [r['score'] for r in selected],
        )
    
    def get_stats(self) -> Dict:
        """Get retriever statistics."""
        return {
            'bm25_indexed': len(self.bm25_corpus) if self.bm25 else 0,
            'vector_count': self.vector_db.get_collection_count(),
            'alpha': self.alpha
        }
