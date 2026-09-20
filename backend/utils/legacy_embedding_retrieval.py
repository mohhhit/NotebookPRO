"""
LEGACY REFERENCE ONLY (DO NOT IMPORT IN RUNTIME)

This file keeps the old MiniLM embedding and hybrid chunk retrieval approach for
historical reference, rollback comparison, or migration audits.

Current production path uses:
- Embedding: BAAI/bge-m3
- Retrieval: Chroma dense retrieval + BGE reranker

This module is intentionally non-usable by runtime and is never imported.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# Legacy embedding model used before migration
LEGACY_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Legacy chunking defaults previously used in uploads
LEGACY_CHUNK_SIZE = 900
LEGACY_CHUNK_OVERLAP = 120


@dataclass
class LegacyChunkRetrievalSettings:
    """Reference settings from the old retrieval stack."""

    chunk_size: int = LEGACY_CHUNK_SIZE
    chunk_overlap: int = LEGACY_CHUNK_OVERLAP
    semantic_chunking: bool = True
    retrieval_n: int = 16
    retrieval_threshold: float = 0.01
    hybrid_alpha: float = 0.6


class LegacyVectorDatabaseReference:
    """
    Reference-only placeholder for old vector DB behavior.

    Legacy behavior summary:
    - sentence-transformers/all-MiniLM-L6-v2 embeddings (384 dimensions)
    - Chroma collection persisted per space
    - query() performed direct dense retrieval without cross-encoder reranking
    """

    def add_documents(self, texts: List[str], metadatas: List[Dict], ids: List[str]) -> None:
        raise RuntimeError("Legacy reference only. Do not use in production runtime.")

    def query(self, query_text: str, n_results: int = 20, filter_dict: Optional[Dict] = None) -> Dict:
        raise RuntimeError("Legacy reference only. Do not use in production runtime.")


class LegacyHybridRetrieverReference:
    """
    Reference-only placeholder for old hybrid retrieval path.

    Legacy behavior summary:
    - Combined Chroma dense retrieval and BM25 sparse retrieval
    - No FlagEmbedding reranker stage
    - Combined scores using alpha weighting and per-source caps
    """

    def retrieve(
        self,
        query: str,
        n_results: int = 20,
        score_threshold: float = 0.0,
    ) -> Tuple[List[str], List[Dict], List[float]]:
        raise RuntimeError("Legacy reference only. Do not use in production runtime.")
