from __future__ import annotations

import argparse
import sys
from pathlib import Path

_APP_DIR = str(Path(__file__).resolve().parent.parent)
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

from ingestion.chunker import chunk_textbook_ocr, load_all_chunks
from ingestion.indexer import build_hybrid_index
from ingestion.pdf_ocr import ocr_textbook_pdf
from ingestion.pptx_parser import parse_all_pptx


def main():
    parser = argparse.ArgumentParser(
        description="Run full PLDC Legal RAG ingestion pipeline (PPTX + PDF OCR + Hybrid Index)."
    )
    parser.add_argument(
        "--ocr-pdf",
        action="store_true",
        help="Run Gemini OCR on the 248-page scanned textbook PDF (resumable via cache).",
    )
    parser.add_argument("--ocr-start", type=int, default=1)
    parser.add_argument("--ocr-end", type=int, default=None)
    parser.add_argument("--ocr-batch-size", type=int, default=2)
    parser.add_argument("--ocr-workers", type=int, default=4)
    parser.add_argument(
        "--skip-dense",
        action="store_true",
        help="Build BM25 index only (offline mode without GEMINI_API_KEY).",
    )
    args = parser.parse_args()

    print("=== Step 1: Parsing PPTX slides ===")
    parse_all_pptx()

    if args.ocr_pdf:
        print("\n=== Step 2: Running Gemini OCR on Textbook PDF ===")
        ocr_textbook_pdf(
            batch_size=args.ocr_batch_size,
            max_workers=args.ocr_workers,
            start_page=args.ocr_start,
            end_page=args.ocr_end,
        )

    print("\n=== Step 3: Chunking Textbook OCR (if available) ===")
    chunk_textbook_ocr()

    print("\n=== Step 4: Building Hybrid Index (BM25 + Dense Vectors) ===")
    chunks = load_all_chunks()
    manifest = build_hybrid_index(chunks=chunks, skip_dense=args.skip_dense)
    print("\n=== Pipeline Summary ===")
    for k, v in manifest.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
