import logging
import os
import threading
import uuid
import warnings
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import CrossEncoder, SentenceTransformer
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

import config

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("chromadb").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


class VectorDatabase:
    """Manage vector database for document embeddings using ChromaDB."""

    _embedding_model = None
    _embedding_model_name = None
    _embedding_model_lock = threading.Lock()
    _tokenizer = None

    _reranker_model = None
    _reranker_model_name = None
    _reranker_lock = threading.Lock()

    @staticmethod
    def _empty_query_result() -> Dict:
        return {
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "ids": [[]],
        }

    @staticmethod
    def _is_dimension_mismatch_error(err: Exception) -> bool:
        msg = str(err).lower()
        return (
            ("expecting embedding with dimension" in msg and "got" in msg)
            or ("does not match index dimensionality" in msg)
            or ("dimensionality of" in msg and "index dimensionality" in msg)
        )

    @staticmethod
    def _is_index_not_found_error(err: Exception) -> bool:
        msg = str(err).lower()
        return "index not found" in msg or "create an instance before querying" in msg

    def _recreate_collection(self) -> None:
        """Recreate collection to recover from stale/missing index internals."""
        collection_name = self.collection.name
        try:
            self.client.delete_collection(name=collection_name)
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _resolve_torch_device(env_var_name: str) -> str:
        """Resolve target device with optional env override."""
        import torch

        preference = os.getenv(env_var_name, "auto").strip().lower()
        if preference == "cpu":
            return "cpu"
        if preference == "cuda":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, collection_name: str = "documents", persist_directory: str = None):
        if persist_directory is None:
            persist_directory = str(config.VECTOR_DB_DIR)

        self.client = self._create_chroma_client(persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self.embedding_model = self._get_or_create_embedding_model()

    def _create_chroma_client(self, persist_directory: str):
        """Create a Chroma client compatible with both legacy and modern APIs."""
        if hasattr(chromadb, "PersistentClient"):
            try:
                return chromadb.PersistentClient(
                    path=persist_directory,
                    settings=Settings(anonymized_telemetry=False),
                )
            except TypeError:
                return chromadb.PersistentClient(path=persist_directory)
            except Exception:
                pass

        return chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=persist_directory,
                anonymized_telemetry=False,
            )
        )

    def ensure_collection_embedding_compatibility(self) -> bool:
        """
        Ensure persisted collection dimensionality matches current embedding model.

        Returns:
            True if collection was reset due to mismatch, else False.
        """
        try:
            count = self.collection.count()
        except Exception:
            return False

        try:
            if count == 0:
                # Query probing on empty indexes may not trigger dimension validation.
                probe_id = f"_dim_probe_{uuid.uuid4().hex[:12]}"
                probe_text = "embedding-dimension-probe"
                probe_embedding = self._encode([probe_text])
                self.collection.add(
                    embeddings=probe_embedding,
                    documents=[probe_text],
                    metadatas=[{"_probe": True}],
                    ids=[probe_id],
                )
                self.collection.delete(ids=[probe_id])
                return False

            probe = self._encode(["embedding-dimension-probe"])
            self.collection.query(query_embeddings=probe, n_results=1)
            return False
        except Exception as e:
            if not (self._is_dimension_mismatch_error(e) or self._is_index_not_found_error(e)):
                raise

            collection_name = self.collection.name
            reason = "missing_index" if self._is_index_not_found_error(e) else "dimension_mismatch"
            print(
                "[VECTOR_DB] collection_incompatible "
                f"reason={reason} collection={collection_name} model={config.EMBEDDING_MODEL}; resetting index"
            )
            self._recreate_collection()
            return True

    @classmethod
    def _get_or_create_embedding_model(cls):
        """Create embedding model once and reuse it for all vector DB instances."""
        with cls._embedding_model_lock:
            if cls._embedding_model is None or cls._embedding_model_name != config.EMBEDDING_MODEL:
                device = cls._resolve_torch_device("EMBEDDING_DEVICE")
                model_name = config.EMBEDDING_MODEL

                print(f"Loading embedding model ({model_name}) on {device}...")
                cls._tokenizer = None
                cls._embedding_model = SentenceTransformer(
                    model_name,
                    device=device,
                    model_kwargs={"torch_dtype": torch.float16},
                    trust_remote_code=True
                )
                cls._embedding_model_name = model_name
                print(f"Embedding model {model_name} loaded on {device}.")

            return cls._embedding_model

    @classmethod
    def _get_or_create_reranker(cls):
        """Create reranker model once and reuse it for all vector DB instances."""
        target_model = getattr(config, "RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-12-v2")

        with cls._reranker_lock:
            if cls._reranker_model is None or cls._reranker_model_name != target_model:
                device = cls._resolve_torch_device("RERANKER_DEVICE")
                print(f"Loading reranker model ({target_model}) on {device}...")
                cls._reranker_model = CrossEncoder(target_model, device=device, trust_remote_code=True, model_kwargs={"torch_dtype": torch.float16})
                
                # Qwen3-Reranker lacks a default pad token, which breaks batch sizes > 1
                if cls._reranker_model.tokenizer.pad_token is None:
                    cls._reranker_model.tokenizer.pad_token = cls._reranker_model.tokenizer.eos_token
                    
                cls._reranker_model_name = target_model
                print(f"Reranker model {target_model} loaded on {device}.")

            return cls._reranker_model

    @classmethod
    def clear_runtime_caches(cls, unload_embedding_model: bool = False):
        """Best-effort cleanup for GPU/CPU caches between large space workloads."""
        try:
            import gc

            gc.collect()
        except Exception:
            pass

        try:
            import torch

            if unload_embedding_model and cls._embedding_model is not None:
                try:
                    cls._embedding_model.to("cpu")
                except Exception:
                    pass

                try:
                    if cls._reranker_model is not None and hasattr(cls._reranker_model, "model"):
                        cls._reranker_model.model.to("cpu")
                except Exception:
                    pass

                cls._embedding_model = None
                cls._embedding_model_name = None
                cls._reranker_model = None
                cls._reranker_model_name = None

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                if hasattr(torch.cuda, "ipc_collect"):
                    torch.cuda.ipc_collect()
        except Exception:
            pass

    def _embedding_batch_size(self) -> int:
        """Embedding batch size tuned for low-VRAM GPUs with env override."""
        if getattr(config, "EMBEDDING_BATCH_SIZE", 0) > 0:
            return int(config.EMBEDDING_BATCH_SIZE)

        model_name = (config.EMBEDDING_MODEL or "").lower()
        if "nomic-embed-text" in model_name:
            return 32
        if "bge-m3" in model_name:
            return 16
        if "bge-large" in model_name:
            return 24
        return 64

    def _reranker_batch_size(self) -> int:
        if getattr(config, "RERANKER_BATCH_SIZE", 0) > 0:
            return int(config.RERANKER_BATCH_SIZE)
        return 16

    def _encode(self, texts: list, instruction: str = None) -> list:
        """Encode texts using SentenceTransformer."""
        if instruction:
            texts = [f"Instruct: {instruction}\nQuery: {t}" for t in texts]

        batch_size = self._embedding_batch_size()
        
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_tensor=False,
            normalize_embeddings=True
        )
        return embeddings.tolist()

    def add_documents(self, texts: List[str], metadatas: List[Dict], ids: List[str]):
        """Embed documents — no instruction prefix for docs."""
        if not texts:
            return
        
        # Documents are encoded WITHOUT instruction
        embeddings = self._encode(texts)

        try:
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids,
            )
        except Exception as e:
            if not (self._is_dimension_mismatch_error(e) or self._is_index_not_found_error(e)):
                raise

            # Auto-recover from stale dimensionality or missing collection index internals.
            collection_name = self.collection.name
            reason = "missing_index" if self._is_index_not_found_error(e) else "dimension_mismatch"
            print(
                "[VECTOR_DB] add_recover "
                f"reason={reason} collection={collection_name} model={config.EMBEDDING_MODEL}; resetting index and retrying"
            )
            self._recreate_collection()
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids,
            )

        if hasattr(self.client, "persist"):
            self.client.persist()

    def query(self, query_text: str, n_results: int = 20, filter_dict: Optional[Dict] = None) -> Dict:
        """Embed query WITH instruction for better retrieval."""
        try:
            if self.collection.count() == 0:
                return self._empty_query_result()
        except Exception:
            pass

        instruction = (
            "Given a student's question, retrieve relevant passages "
            "from academic textbooks that answer the question"
        )
        query_embedding = self._encode([query_text], instruction=instruction)

        try:
            return self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                where=filter_dict,
            )
        except Exception as e:
            if "index not found" in str(e).lower() or "create an instance" in str(e).lower():
                return self._empty_query_result()

            if self._is_dimension_mismatch_error(e):
                print(
                    "[VECTOR_DB] query_dimension_mismatch "
                    f"model={config.EMBEDDING_MODEL}; returning empty results"
                )
                return self._empty_query_result()
            raise

    def query_and_rerank(
        self,
        query_text: str,
        n_retrieve: int = 20,
        n_final: int = 6,
        filter_dict: Optional[Dict] = None,
    ) -> Dict:
        """Two-stage retrieval: dense retrieval in Chroma followed by reranking."""
        raw = self.query(query_text=query_text, n_results=n_retrieve, filter_dict=filter_dict)

        docs = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]
        ids = raw.get("ids", [[]])[0]

        if not docs:
            return raw

        reranker = self._get_or_create_reranker()
        pairs = [[query_text, doc] for doc in docs]
        scores = reranker.predict(
            pairs,
            batch_size=self._reranker_batch_size(),
            show_progress_bar=False,
        )
        
        # Qwen3 outputs raw logits (e.g. -11 to +8). Convert them to 0-1 probabilities for the UI.
        scores = torch.sigmoid(torch.tensor(list(scores))).tolist()
        scores = [float(s) for s in scores]

        ranked = sorted(
            zip(scores, docs, metadatas, distances, ids),
            key=lambda x: x[0],
            reverse=True,
        )[: max(1, n_final)]

        scores_out, docs_out, metas_out, dists_out, ids_out = zip(*ranked)
        return {
            "documents": [list(docs_out)],
            "metadatas": [list(metas_out)],
            "distances": [list(dists_out)],
            "ids": [list(ids_out)],
            "reranker_scores": [list(scores_out)],
        }

    def delete_collection(self):
        """Delete the entire collection."""
        self.client.delete_collection(name=self.collection.name)

    def get_collection_count(self) -> int:
        """Get the number of documents in the collection."""
        return self.collection.count()

    def get_all_documents(self) -> tuple[List[str], List[Dict]]:
        """Get all documents and metadata from the collection."""
        count = self.collection.count()
        if count == 0:
            return [], []

        results = self.collection.get()
        return results.get("documents", []), results.get("metadatas", [])

    def create_space_collection(self, space_name: str):
        """Create a new collection for a specific subject space."""
        return self.client.get_or_create_collection(
            name=f"space_{space_name}",
            metadata={"hnsw:space": "cosine"},
        )
