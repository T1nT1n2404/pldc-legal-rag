from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent
DOCS_DIR = APP_DIR.parent  # Contains the 6 .pptx files and 1 textbook .pdf
DATA_DIR = APP_DIR / "data"
PARSED_DIR = DATA_DIR / "parsed"
OCR_CACHE_DIR = DATA_DIR / "ocr_cache"
INDEX_DIR = DATA_DIR / "index"

for _d in (DATA_DIR, PARSED_DIR, OCR_CACHE_DIR, INDEX_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Load .env from legal_rag/ or workspace root
load_dotenv(APP_DIR / ".env")
load_dotenv(DOCS_DIR / ".env")

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_CHAT_MODEL: str = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.6-flash")
GEMINI_OCR_MODEL: str = os.getenv("GEMINI_OCR_MODEL", "gemini-3.5-flash-lite")
GEMINI_FALLBACK_MODELS: list[str] = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.8-flash",
]
GEMINI_EMBEDDING_MODEL: str = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
LOCAL_EMBEDDING_MODEL: str = os.getenv(
    "LOCAL_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
EMBEDDING_PROVIDER: str = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()  # "local" or "gemini"
EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "768"))

# Retrieval defaults
DEFAULT_TOP_K = 6
RRF_K = 60  # Standard Reciprocal Rank Fusion constant
TEXTBOOK_CHUNK_SIZE = 850
TEXTBOOK_CHUNK_OVERLAP = 150
