# PLĐC Legal Agentic RAG — Chatbot Pháp luật & Máy tính Pháp lý

Hệ thống **Agentic Hybrid RAG** kết hợp **Công cụ tính toán pháp lý xác định (Deterministic Tool Calling)** xây dựng trên kho dữ liệu thực tế môn **Pháp luật Đại cương** (6 bộ slide `.pptx` + 1 giáo trình PDF scan 248 trang).

## Kiến trúc tổng quan

```
Documents (6 PPTX + 248-page Scanned PDF)
   ├── PPTX Parser (python-pptx: slide, table, notes) ──┐
   └── Resumable Gemini Vision OCR (batch PDF → MD) ────┤
                                                        ▼
                                            Source-aware Chunker
                                  (1 slide = 1 chunk | recursive overlap)
                                                        │
                              ┌─────────────────────────┴──────────────────────────┐
                              ▼                                                    ▼
                 BM25 Sparse Index                                   Dense Vector Index
        (Vietnamese legal tokenizer: điều_644)               (gemini-embedding-001, L2 norm)
                              └─────────────────────────┬──────────────────────────┘
                                                        ▼
                                       Hybrid Retriever (RRF Fusion, k=60)
                                                        │
                                                        ▼
                                      Gemini 3.8 Flash Agent (Tool Calling)
                         ┌──────────────────────────────┼──────────────────────────────┐
                         ▼                              ▼                              ▼
             search_legal_knowledge        Deterministic Legal Calculators     Trace Inspector UI
                                      (Thừa kế BLDS | Điều 55/56/65 BLHS | BLLĐ)
```

## Cách chạy nhanh

### 1. Cấu hình `GEMINI_API_KEY`
```bash
cd /Users/t1nt1n/Documents/HK261/PLDC/legal_rag
cp .env.example .env
# Mở .env và điền GEMINI_API_KEY=...
```
*(Hoặc bạn có thể nhập trực tiếp API Key ở thanh bên trái giao diện Web App).*

### 2. Chạy OCR toàn bộ 248 trang Giáo trình PDF + Build Dense Vector Index
Pipeline có **checkpoint từng batch** (`data/ocr_cache/`) nên nếu mạng gián đoạn có thể chạy lại mà không mất tiến độ:
```bash
.venv/bin/python -m ingestion.run_pipeline --ocr-pdf
```
*(Hoặc bấm nút **"Chạy OCR PDF + Build Dense Index"** ngay trên giao diện Web App).*

### 3. Khởi động Web App
```bash
.venv/bin/uvicorn backend.app:app --reload --port 8000
```
Mở trình duyệt tại: `http://localhost:8000`

### 4. Chạy Unit Test & API Test
```bash
.venv/bin/pytest -v
```
