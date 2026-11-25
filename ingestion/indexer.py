from __future__ import annotations

import hashlib
import json
import pickle
import re
import time
import unicodedata
from pathlib import Path
from typing import List, Optional

import numpy as np
from google import genai
from google.genai import types
from rank_bm25 import BM25Okapi

from config import (
    DATA_DIR,
    EMBEDDING_DIM,
    EMBEDDING_PROVIDER,
    GEMINI_API_KEY,
    GEMINI_EMBEDDING_MODEL,
    INDEX_DIR,
    LOCAL_EMBEDDING_MODEL,
)
from ingestion.chunker import load_all_chunks
from ingestion.schemas import DocumentChunk


def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    cleaned = "".join(ch for ch in nfkd if not unicodedata.combining(ch))
    return cleaned.replace("đ", "d").replace("Đ", "D")


def tokenize_vietnamese_legal(text: str) -> List[str]:
    """
    Domain-aware tokenizer for Vietnamese legal BM25:
    - Preserves exact legal references (e.g., 'điều 644' -> 'dieu_644', 'khoản 1' -> 'khoan_1')
    - Keeps both accented tokens (high precision) and ASCII-folded tokens (typo/no-diacritic recall)
    """
    low = unicodedata.normalize("NFC", text.lower())
    # Capture compound legal tokens like "điều 644", "khoản 2", "điểm a"
    compound_tokens = []
    for m in re.finditer(r"\b(điều|khoản|chương|mục)\s+(\d+)\b", low):
        compound_tokens.append(f"{_strip_accents(m.group(1))}_{m.group(2)}")

    words = re.findall(r"[0-9a-zà-ỹ]+", low)
    # Emit accented tokens + folded tokens for robustness when user types without accents
    tokens = list(words) + compound_tokens
    for w in words:
        folded = _strip_accents(w)
        if folded != w and len(folded) >= 2:
            tokens.append(folded)
    return tokens


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _embed_chunks_gemini(
    chunks: List[DocumentChunk],
    api_key: str,
    model: str = GEMINI_EMBEDDING_MODEL,
    dim: int = EMBEDDING_DIM,
    batch_size: int = 40,
) -> np.ndarray:
    cache_path = INDEX_DIR / "embedding_cache.json"
    cache: dict[str, list[float]] = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            cache = {}

    client = genai.Client(api_key=api_key)
    hashes = [_text_hash(f"{model}:{dim}:{c.text}") for c in chunks]
    missing_indices = [i for i, h in enumerate(hashes) if h not in cache]

    if missing_indices:
        print(
            f"[Indexer] Generating Gemini embeddings ({model}, dim={dim}) "
            f"for {len(missing_indices)}/{len(chunks)} chunks...",
            flush=True,
        )
        total_batches = (len(missing_indices) + batch_size - 1) // batch_size
        for b_num, start in enumerate(range(0, len(missing_indices), batch_size), start=1):
            batch_idx = missing_indices[start : start + batch_size]
            batch_texts = [chunks[i].text for i in batch_idx]
            for attempt in range(1, 7):
                try:
                    resp = client.models.embed_content(
                        model=model,
                        contents=batch_texts,
                        config=types.EmbedContentConfig(
                            task_type="RETRIEVAL_DOCUMENT",
                            output_dimensionality=dim,
                        ),
                    )
                    for idx_in_chunks, emb in zip(batch_idx, resp.embeddings):
                        cache[hashes[idx_in_chunks]] = list(emb.values)
                    break
                except Exception as exc:
                    err_s = str(exc)
                    wait = min(15 * attempt, 65) if ("429" in err_s or "RESOURCE_EXHAUSTED" in err_s) else min(2**attempt * 2, 20)
                    print(
                        f"[Indexer] Batch {b_num}/{total_batches} attempt {attempt} error: {err_s[:160]}. Retrying in {wait}s...",
                        flush=True,
                    )
                    time.sleep(wait)
            else:
                raise RuntimeError(f"Failed embedding batch {b_num}")

            if b_num % 10 == 0 or b_num == total_batches:
                cache_path.write_text(json.dumps(cache), encoding="utf-8")
                print(
                    f"[Indexer] Embedded {min(start + batch_size, len(missing_indices))}/{len(missing_indices)} chunks (batch {b_num}/{total_batches})",
                    flush=True,
                )

        cache_path.write_text(json.dumps(cache), encoding="utf-8")
        print(f"[Indexer] Embeddings cached -> {cache_path}", flush=True)
    else:
        print(f"[Indexer] Reused all {len(chunks)} cached embeddings.")

    mat = np.array([cache[h] for h in hashes], dtype=np.float32)
    # L2 normalize so dot product == cosine similarity
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return mat / norms


def _embed_chunks_local(
    chunks: List[DocumentChunk],
    model_name: str = LOCAL_EMBEDDING_MODEL,
) -> np.ndarray:
    from fastembed import TextEmbedding

    cache_dir = DATA_DIR / "models_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"[Indexer] Generating local ONNX embeddings ({model_name}) for {len(chunks)} chunks...",
        flush=True,
    )
    t0 = time.time()
    embedder = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))
    texts = [c.text for c in chunks]
    vecs = list(embedder.embed(texts, batch_size=64))
    mat = np.array(vecs, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    print(
        f"[Indexer] Local ONNX embeddings built {mat.shape} in {time.time()-t0:.1f}s!",
        flush=True,
    )
    return mat / norms


def build_hybrid_index(
    chunks: Optional[List[DocumentChunk]] = None,
    api_key: Optional[str] = None,
    skip_dense: bool = False,
    provider: Optional[str] = None,
) -> dict:
    chunks = chunks if chunks is not None else load_all_chunks()
    if not chunks:
        raise ValueError("Chưa có chunk nào để index! Hãy chạy PPTX parser / PDF OCR trước.")

    # 1. Persist chunks metadata & text
    chunks_path = INDEX_DIR / "chunks.json"
    chunks_path.write_text(
        json.dumps([c.model_dump() for c in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 2. Build & persist BM25 lexical index
    tokenized_corpus = [tokenize_vietnamese_legal(c.text) for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_path = INDEX_DIR / "bm25.pkl"
    with bm25_path.open("wb") as f:
        pickle.dump(bm25, f)
    print(f"[Indexer] Built BM25 index for {len(chunks)} chunks -> {bm25_path}", flush=True)

    # 3. Build & persist Dense Vector index (local ONNX or Gemini API)
    emb_provider = (provider or EMBEDDING_PROVIDER).strip().lower()
    key = (api_key or GEMINI_API_KEY).strip()
    has_dense = False
    used_model = None
    used_dim = None
    vectors_path = INDEX_DIR / "vectors.npz"

    if not skip_dense:
        if emb_provider == "local":
            vectors = _embed_chunks_local(chunks, model_name=LOCAL_EMBEDDING_MODEL)
            np.savez_compressed(vectors_path, vectors=vectors)
            has_dense = True
            used_model = LOCAL_EMBEDDING_MODEL
            used_dim = int(vectors.shape[1])
            print(f"[Indexer] Saved local dense vectors {vectors.shape} -> {vectors_path}", flush=True)
        elif emb_provider == "gemini" and key:
            vectors = _embed_chunks_gemini(chunks, api_key=key)
            np.savez_compressed(vectors_path, vectors=vectors)
            has_dense = True
            used_model = GEMINI_EMBEDDING_MODEL
            used_dim = int(vectors.shape[1])
            print(f"[Indexer] Saved Gemini dense vectors {vectors.shape} -> {vectors_path}", flush=True)
        else:
            print(
                "[Indexer] Skipping dense vectors (no GEMINI_API_KEY for gemini provider).",
                flush=True,
            )

    slide_count = sum(1 for c in chunks if c.source_type == "slide")
    textbook_count = sum(1 for c in chunks if c.source_type == "textbook")
    manifest = {
        "total_chunks": len(chunks),
        "slide_chunks": slide_count,
        "textbook_chunks": textbook_count,
        "has_bm25": True,
        "has_dense_vectors": has_dense,
        "embedding_provider": emb_provider if has_dense else None,
        "embedding_model": used_model,
        "embedding_dim": used_dim,
        "built_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (INDEX_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


if __name__ == "__main__":
    build_hybrid_index()
