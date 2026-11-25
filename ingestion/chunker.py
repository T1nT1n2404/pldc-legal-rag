from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import List

from config import PARSED_DIR, TEXTBOOK_CHUNK_OVERLAP, TEXTBOOK_CHUNK_SIZE
from ingestion.schemas import DocumentChunk


def _recursive_split(text: str, max_chars: int, overlap: int) -> List[str]:
    """Heading/paragraph/sentence-aware recursive splitter for legal prose."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    # Split primarily on markdown headings or double newlines
    separators = ["\n## ", "\n### ", "\n\n", "\n", ". ", " "]
    parts: List[str] = [text]
    for sep in separators:
        if all(len(p) <= max_chars for p in parts):
            break
        next_parts: List[str] = []
        for p in parts:
            if len(p) <= max_chars:
                next_parts.append(p)
            else:
                pieces = p.split(sep)
                rebuilt = [pieces[0]] + [
                    (sep.lstrip("\n") + piece if sep.startswith("\n#") else piece)
                    for piece in pieces[1:]
                ]
                next_parts.extend([x.strip() for x in rebuilt if x.strip()])
        parts = next_parts

    # Merge small pieces into target window with overlap
    chunks: List[str] = []
    buf = ""
    for piece in parts:
        if not buf:
            buf = piece
        elif len(buf) + len(piece) + 2 <= max_chars:
            buf = f"{buf}\n\n{piece}"
        else:
            chunks.append(buf.strip())
            tail = buf[-overlap:].strip() if overlap > 0 and len(buf) > overlap else ""
            buf = f"{tail}\n{piece}".strip() if tail else piece
    if buf.strip():
        chunks.append(buf.strip())

    return chunks


def chunk_textbook_ocr(
    ocr_json_path: Path | None = None,
    chunk_size: int = TEXTBOOK_CHUNK_SIZE,
    overlap: int = TEXTBOOK_CHUNK_OVERLAP,
) -> List[DocumentChunk]:
    path = ocr_json_path or (PARSED_DIR / "textbook_ocr.json")
    if not path.exists():
        print(f"[Chunker] Textbook OCR file not found yet at {path} (skipping).")
        return []

    payload = json.loads(path.read_text(encoding="utf-8"))
    source_file = unicodedata.normalize(
        "NFC", payload.get("source_file", "Giáo trình Pháp luật Đại cương.pdf")
    )
    pages = payload.get("pages", [])

    chunks: List[DocumentChunk] = []
    for p in pages:
        page_num = int(p.get("page", 0))
        chapter = unicodedata.normalize("NFC", str(p.get("chapter", "")).strip())
        md = unicodedata.normalize("NFC", str(p.get("markdown", "")).strip())
        if len(re.sub(r"[\d\s\W]+", "", md)) < 15:
            continue

        # Infer local title from first markdown heading on page
        heading_match = re.search(r"^#{1,4}\s+(.+)$", md, re.MULTILINE)
        page_title = heading_match.group(1).strip() if heading_match else chapter

        sub_texts = _recursive_split(md, max_chars=chunk_size, overlap=overlap)
        for sub_idx, sub in enumerate(sub_texts, start=1):
            header = f"[Nguồn: Giáo trình - Trang {page_num} | {chapter or 'Pháp luật Đại cương'}]"
            full_text = f"{header}\n{sub}"
            chunk_id = f"textbook::p{page_num:03d}::c{sub_idx:02d}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    source_type="textbook",
                    source_file=source_file,
                    chapter=chapter or "Giáo trình Pháp luật Đại cương",
                    location_label=f"Trang {page_num}",
                    page_or_slide=page_num,
                    title=page_title[:200],
                    text=full_text,
                    char_count=len(full_text),
                    metadata={"sub_chunk": sub_idx, "total_sub_chunks": len(sub_texts)},
                )
            )

    out_file = PARSED_DIR / "textbook_chunks.json"
    out_file.write_text(
        json.dumps([c.model_dump() for c in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[Chunker] Created {len(chunks)} textbook chunks -> {out_file}")
    return chunks


def load_all_chunks() -> List[DocumentChunk]:
    chunks: List[DocumentChunk] = []
    for fname in ("pptx_chunks.json", "textbook_chunks.json"):
        p = PARSED_DIR / fname
        if p.exists():
            raw = json.loads(p.read_text(encoding="utf-8"))
            chunks.extend(DocumentChunk(**item) for item in raw)
    return chunks
