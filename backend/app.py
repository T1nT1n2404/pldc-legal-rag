from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent.rag_agent import LegalRAGAgent
from config import APP_DIR, GEMINI_API_KEY, GEMINI_CHAT_MODEL, OCR_CACHE_DIR, PARSED_DIR
from ingestion.chunker import chunk_textbook_ocr, load_all_chunks
from ingestion.indexer import build_hybrid_index
from ingestion.pdf_ocr import ocr_textbook_pdf
from ingestion.pptx_parser import parse_all_pptx
from retrieval.hybrid_retriever import HybridRetriever
from tools.criminal_calc import (
    calculate_multi_crime_sentence,
    calculate_multi_judgment_sentence,
    calculate_suspended_sentence,
)
from tools.inheritance_calc import calculate_inheritance
from tools.labor_calc import (
    calculate_overtime_pay,
    calculate_probation_rules,
    calculate_severance_allowance,
)

FRONTEND_DIR = APP_DIR / "frontend"

app = FastAPI(
    title="PLDC Legal Agentic RAG API",
    description="Hybrid RAG (BM25 + Gemini Dense + RRF) + Deterministic Legal Calculators",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

retriever = HybridRetriever()
agent = LegalRAGAgent(retriever=retriever)

# Track background OCR/index jobs safely
_job_state: Dict[str, Any] = {
    "running": False,
    "stage": "idle",
    "message": "",
    "error": None,
}
_job_lock = threading.Lock()


class ChatRequest(BaseModel):
    message: str
    history: List[dict] = Field(default_factory=list)
    source_filter: Literal["all", "slide", "textbook"] = "all"
    retrieval_mode: Literal["hybrid", "bm25", "dense"] = "hybrid"
    model: str = GEMINI_CHAT_MODEL


class SearchRequest(BaseModel):
    query: str
    top_k: int = 6
    source_type: Literal["all", "slide", "textbook"] = "all"
    mode: Literal["hybrid", "bm25", "dense"] = "hybrid"


class RebuildRequest(BaseModel):
    run_pdf_ocr: bool = False
    ocr_batch_size: int = 2
    ocr_workers: int = 4


def _count_ocr_cached_pages() -> int:
    return len(list(OCR_CACHE_DIR.glob("tess_p*.json")))


@app.get("/")
def serve_index():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/api/stats")
def get_stats():
    ocr_json = PARSED_DIR / "textbook_ocr.json"
    ocr_total_pages = 248
    ocr_done_pages = _count_ocr_cached_pages()
    if ocr_json.exists():
        try:
            meta = json.loads(ocr_json.read_text(encoding="utf-8"))
            ocr_total_pages = int(meta.get("total_pages_in_pdf", 248))
            ocr_done_pages = max(ocr_done_pages, int(meta.get("ocr_pages_count", 0)))
        except Exception:
            pass

    return {
        "index_ready": retriever.is_ready,
        "has_dense_vectors": retriever.has_dense,
        "has_env_api_key": bool(GEMINI_API_KEY),
        "default_model": GEMINI_CHAT_MODEL,
        "manifest": retriever.manifest,
        "ocr_progress": {
            "done_pages": ocr_done_pages,
            "total_pages": ocr_total_pages,
            "completed": ocr_done_pages >= ocr_total_pages and ocr_json.exists(),
        },
        "job": _job_state,
    }


@app.post("/api/chat")
def chat_endpoint(
    req: ChatRequest,
    x_gemini_api_key: Optional[str] = Header(default=None),
):
    try:
        return agent.chat(
            message=req.message,
            history=req.history,
            api_key=x_gemini_api_key,
            source_filter=req.source_filter,
            retrieval_mode=req.retrieval_mode,
            model_name=req.model or GEMINI_CHAT_MODEL,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý Gemini Agent: {exc}")


@app.post("/api/search")
def search_endpoint(
    req: SearchRequest,
    x_gemini_api_key: Optional[str] = Header(default=None),
):
    return retriever.search(
        query=req.query,
        top_k=min(max(req.top_k, 1), 15),
        source_type=req.source_type,
        mode=req.mode,
        api_key=x_gemini_api_key,
    )


@app.post("/api/calc/inheritance")
def calc_inheritance_endpoint(payload: dict):
    try:
        return calculate_inheritance(**payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/calc/criminal/multi-crime")
def calc_multi_crime_endpoint(payload: dict):
    try:
        return calculate_multi_crime_sentence(**payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/calc/criminal/multi-judgment")
def calc_multi_judgment_endpoint(payload: dict):
    try:
        return calculate_multi_judgment_sentence(**payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/calc/criminal/suspended")
def calc_suspended_endpoint(payload: dict):
    try:
        return calculate_suspended_sentence(**payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/calc/labor")
def calc_labor_endpoint(payload: dict):
    try:
        calc_type = payload.pop("calc_type", "overtime")
        if calc_type == "overtime":
            return calculate_overtime_pay(**payload)
        if calc_type == "severance":
            return calculate_severance_allowance(**payload)
        return calculate_probation_rules(**payload)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/index/rebuild")
def rebuild_index_endpoint(
    req: RebuildRequest,
    x_gemini_api_key: Optional[str] = Header(default=None),
):
    key = (x_gemini_api_key or GEMINI_API_KEY).strip()
    if req.run_pdf_ocr and not key:
        raise HTTPException(
            status_code=400,
            detail="Cần GEMINI_API_KEY để chạy OCR giáo trình PDF và tạo Dense Embeddings.",
        )

    with _job_lock:
        if _job_state["running"]:
            return {"status": "already_running", "job": _job_state}
        _job_state.update(
            {"running": True, "stage": "starting", "message": "Đang khởi chạy pipeline...", "error": None}
        )

    def _worker():
        try:
            _job_state.update({"stage": "pptx", "message": "Đang phân tích 6 file slide PPTX..."})
            parse_all_pptx()

            if req.run_pdf_ocr:
                _job_state.update(
                    {"stage": "pdf_ocr", "message": "Đang OCR giáo trình PDF 248 trang bằng Gemini (có cache resumable)..."}
                )
                ocr_textbook_pdf(
                    api_key=key,
                    batch_size=req.ocr_batch_size,
                    max_workers=req.ocr_workers,
                )

            _job_state.update({"stage": "chunking", "message": "Đang chia chunk giáo trình & hợp nhất dữ liệu..."})
            chunk_textbook_ocr()

            _job_state.update({"stage": "indexing", "message": "Đang xây dựng chỉ mục Hybrid (BM25 + Dense Vector)..."})
            chunks = load_all_chunks()
            build_hybrid_index(chunks=chunks, api_key=key, skip_dense=not bool(key))
            retriever.reload()

            _job_state.update(
                {"running": False, "stage": "done", "message": "Hoàn tất cập nhật kho tri thức Hybrid RAG!", "error": None}
            )
        except Exception as exc:
            _job_state.update(
                {"running": False, "stage": "error", "message": "Pipeline gặp lỗi.", "error": str(exc)}
            )

    threading.Thread(target=_worker, daemon=True).start()
    return {"status": "started", "job": _job_state}
