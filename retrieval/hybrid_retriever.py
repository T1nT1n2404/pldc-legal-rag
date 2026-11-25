from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
from google import genai
from google.genai import types

from config import (
    DATA_DIR,
    DEFAULT_TOP_K,
    EMBEDDING_DIM,
    GEMINI_API_KEY,
    GEMINI_EMBEDDING_MODEL,
    INDEX_DIR,
    LOCAL_EMBEDDING_MODEL,
    RRF_K,
)
from ingestion.indexer import tokenize_vietnamese_legal
from ingestion.schemas import DocumentChunk


class HybridRetriever:
    """
    Hybrid Sparse (BM25) + Dense (Gemini Embedding Cosine) Retriever
    fused via Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, index_dir: Path = INDEX_DIR):
        self.index_dir = index_dir
        self.chunks: List[DocumentChunk] = []
        self.bm25 = None
        self.vectors: Optional[np.ndarray] = None
        self.manifest: dict = {}
        self._local_embedder = None
        self._local_model_name: Optional[str] = None
        self.reload()

    def reload(self) -> None:
        chunks_path = self.index_dir / "chunks.json"
        bm25_path = self.index_dir / "bm25.pkl"
        vectors_path = self.index_dir / "vectors.npz"
        manifest_path = self.index_dir / "manifest.json"

        self.chunks = []
        self.bm25 = None
        self.vectors = None
        self.manifest = {}

        if chunks_path.exists():
            raw = json.loads(chunks_path.read_text(encoding="utf-8"))
            self.chunks = [DocumentChunk(**item) for item in raw]
        if bm25_path.exists():
            with bm25_path.open("rb") as f:
                self.bm25 = pickle.load(f)
        if vectors_path.exists():
            data = np.load(vectors_path)
            self.vectors = data["vectors"].astype(np.float32)
        if manifest_path.exists():
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    @property
    def is_ready(self) -> bool:
        return bool(self.chunks and self.bm25 is not None)

    @property
    def has_dense(self) -> bool:
        return self.vectors is not None and len(self.vectors) == len(self.chunks)

    def _embed_query(self, query: str, api_key: Optional[str] = None) -> Optional[np.ndarray]:
        if not self.has_dense:
            return None

        provider = (self.manifest.get("embedding_provider") or "gemini").lower()
        if provider == "local":
            try:
                from fastembed import TextEmbedding

                model_name = self.manifest.get("embedding_model") or LOCAL_EMBEDDING_MODEL
                if self._local_embedder is None or self._local_model_name != model_name:
                    cache_dir = DATA_DIR / "models_cache"
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    self._local_embedder = TextEmbedding(
                        model_name=model_name, cache_dir=str(cache_dir)
                    )
                    self._local_model_name = model_name
                vec = np.array(next(self._local_embedder.embed([query])), dtype=np.float32)
                norm = np.linalg.norm(vec)
                return vec / norm if norm > 0 else vec
            except Exception as exc:
                print(f"[Retriever] Local query embedding failed (falling back to BM25): {exc}")
                return None

        key = (api_key or GEMINI_API_KEY).strip()
        if not key:
            return None
        try:
            client = genai.Client(api_key=key)
            resp = client.models.embed_content(
                model=self.manifest.get("embedding_model") or GEMINI_EMBEDDING_MODEL,
                contents=query,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_QUERY",
                    output_dimensionality=int(self.manifest.get("embedding_dim") or EMBEDDING_DIM),
                ),
            )
            vec = np.array(resp.embeddings[0].values, dtype=np.float32)
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec
        except Exception as exc:
            print(f"[Retriever] Gemini query embedding failed (falling back to BM25): {exc}")
            return None

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        source_type: Literal["all", "slide", "textbook"] = "all",
        mode: Literal["hybrid", "bm25", "dense"] = "hybrid",
        api_key: Optional[str] = None,
        dense_weight: float = 0.55,
        bm25_weight: float = 0.45,
    ) -> dict:
        t0 = time.perf_counter()
        if not self.is_ready:
            return {
                "query": query,
                "mode_used": "none",
                "latency_ms": 0.0,
                "results": [],
                "warning": "Index chưa được xây dựng. Hãy chạy pipeline nạp dữ liệu trước.",
            }

        # Filter allowed indices by metadata (source_type)
        allowed_mask = np.array(
            [
                (source_type == "all") or (c.source_type == source_type)
                for c in self.chunks
            ],
            dtype=bool,
        )
        allowed_indices = np.where(allowed_mask)[0]
        if len(allowed_indices) == 0:
            return {
                "query": query,
                "mode_used": mode,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                "results": [],
            }

        # 1. BM25 lexical scores
        bm25_scores_all = np.zeros(len(self.chunks), dtype=np.float32)
        if mode in ("hybrid", "bm25"):
            q_tokens = tokenize_vietnamese_legal(query)
            if q_tokens:
                bm25_scores_all = np.array(
                    self.bm25.get_scores(q_tokens), dtype=np.float32
                )

        # 2. Dense vector cosine scores
        dense_scores_all = np.zeros(len(self.chunks), dtype=np.float32)
        used_dense = False
        if mode in ("hybrid", "dense"):
            q_vec = self._embed_query(query, api_key=api_key)
            if q_vec is not None and self.vectors is not None:
                dense_scores_all = self.vectors @ q_vec
                used_dense = True

        # Effective mode
        if mode == "dense" and not used_dense:
            effective_mode = "bm25_fallback"
        elif mode == "hybrid" and not used_dense:
            effective_mode = "bm25_only_no_dense_index"
        else:
            effective_mode = mode

        # Compute rankings within allowed_indices
        bm25_order = allowed_indices[np.argsort(-bm25_scores_all[allowed_indices])]
        bm25_rank_map = {
            int(idx): rank for rank, idx in enumerate(bm25_order, start=1)
        }

        dense_rank_map: dict[int, int] = {}
        if used_dense:
            dense_order = allowed_indices[np.argsort(-dense_scores_all[allowed_indices])]
            dense_rank_map = {
                int(idx): rank for rank, idx in enumerate(dense_order, start=1)
            }

        # Score fusion (RRF)
        candidate_scores: list[tuple[int, float]] = []
        for idx in allowed_indices:
            i = int(idx)
            if effective_mode == "hybrid":
                rrf = (
                    dense_weight / (RRF_K + dense_rank_map[i])
                    + bm25_weight / (RRF_K + bm25_rank_map[i])
                )
                candidate_scores.append((i, float(rrf)))
            elif effective_mode == "dense":
                candidate_scores.append((i, float(dense_scores_all[i])))
            else:
                candidate_scores.append((i, float(bm25_scores_all[i])))

        candidate_scores.sort(key=lambda x: x[1], reverse=True)
        top_items = candidate_scores[:top_k]

        results = []
        for rank, (i, score) in enumerate(top_items, start=1):
            c = self.chunks[i]
            results.append(
                {
                    "rank": rank,
                    "chunk_id": c.chunk_id,
                    "citation": c.citation(),
                    "source_type": c.source_type,
                    "source_file": c.source_file,
                    "chapter": c.chapter,
                    "location_label": c.location_label,
                    "page_or_slide": c.page_or_slide,
                    "title": c.title,
                    "text": c.text,
                    "final_score": round(score, 5),
                    "bm25_score": round(float(bm25_scores_all[i]), 4),
                    "bm25_rank": bm25_rank_map.get(i),
                    "dense_score": round(float(dense_scores_all[i]), 4) if used_dense else None,
                    "dense_rank": dense_rank_map.get(i) if used_dense else None,
                }
            )

        latency_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "query": query,
            "source_type": source_type,
            "mode_requested": mode,
            "mode_used": effective_mode,
            "latency_ms": latency_ms,
            "results": results,
        }
