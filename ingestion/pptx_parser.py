from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import List

from pptx import Presentation

from config import DOCS_DIR, PARSED_DIR
from ingestion.schemas import DocumentChunk


def _normalize_str(s: str) -> str:
    return unicodedata.normalize("NFC", s).strip()


def _infer_chapter(filename: str) -> str:
    name = _normalize_str(Path(filename).stem)
    low = name.lower()
    if "dân sự" in low:
        return "Chương 2 - Pháp luật Dân sự"
    if "chương 2-phần 2" in low or "chương 2 - phần 2" in low:
        return "Chương 2 (Phần 2) - Thừa kế & Hôn nhân Gia đình"
    if "lao động" in low:
        return "Chương Pháp luật Lao động"
    if "hình sự" in low:
        return "Chương Pháp luật Hình sự"
    if "chương 1_p2" in low:
        return "Chương 1 (Phần 2) - Lý luận chung về Nhà nước & Pháp luật"
    if "chương 1" in low:
        return "Chương 1 - Đại cương về Nhà nước & Pháp luật"
    return name


def _extract_table_markdown(table) -> str:
    rows = []
    for row in table.rows:
        cells = [re.sub(r"\s+", " ", cell.text.strip()) for cell in row.cells]
        if any(cells):
            rows.append(cells)
    if not rows:
        return ""
    header = "| " + " | ".join(rows[0]) + " |"
    sep = "| " + " | ".join(["---"] * len(rows[0])) + " |"
    body = ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join([header, sep, *body])


def _extract_shapes_text(shapes, title_shape_id=None) -> tuple[str, list[str], list[str]]:
    title_text = ""
    body_lines: list[str] = []
    tables_md: list[str] = []

    for shape in shapes:
        # Recursively handle grouped shapes
        if shape.shape_type == 6 and hasattr(shape, "shapes"):  # MSO_SHAPE_TYPE.GROUP
            sub_t, sub_b, sub_tbl = _extract_shapes_text(shape.shapes)
            if not title_text and sub_t:
                title_text = sub_t
            body_lines.extend(sub_b)
            tables_md.extend(sub_tbl)
            continue

        if shape.has_table:
            md = _extract_table_markdown(shape.table)
            if md:
                tables_md.append(md)
            continue

        if shape.has_text_frame:
            paragraphs = []
            for p in shape.text_frame.paragraphs:
                txt = re.sub(r"\s+", " ", p.text).strip()
                if txt:
                    indent = "  " * (p.level or 0)
                    paragraphs.append(f"{indent}{txt}")
            block = "\n".join(paragraphs).strip()
            if not block:
                continue

            if title_shape_id is not None and shape.shape_id == title_shape_id:
                title_text = block.replace("\n", " ")
            else:
                body_lines.append(block)

    return title_text, body_lines, tables_md


def parse_pptx_file(pptx_path: Path) -> List[DocumentChunk]:
    prs = Presentation(str(pptx_path))
    file_name = _normalize_str(pptx_path.name)
    chapter = _infer_chapter(file_name)
    chunks: List[DocumentChunk] = []

    for idx, slide in enumerate(prs.slides, start=1):
        title_shape_id = None
        if slide.shapes.title is not None:
            title_shape_id = slide.shapes.title.shape_id

        title_text, body_lines, tables_md = _extract_shapes_text(
            slide.shapes, title_shape_id=title_shape_id
        )

        notes_text = ""
        if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
            notes_text = re.sub(
                r"\s+", " ", slide.notes_slide.notes_text_frame.text
            ).strip()

        # Fallback title from first short line if no explicit title placeholder
        if not title_text and body_lines:
            first_line = body_lines[0].split("\n")[0].strip()
            if len(first_line) <= 120:
                title_text = first_line

        sections = [f"[Nguồn: Slide {idx} | {file_name} | {chapter}]"]
        if title_text:
            sections.append(f"Tiêu đề: {title_text}")
        if body_lines:
            sections.append("\n".join(body_lines))
        if tables_md:
            sections.append("Bảng:\n" + "\n\n".join(tables_md))
        if notes_text:
            sections.append(f"Ghi chú giảng viên: {notes_text}")

        full_text = _normalize_str("\n".join(sections))
        # Skip nearly-empty decorative/divider slides with no real text
        content_only = "".join(body_lines) + title_text + "".join(tables_md) + notes_text
        if len(re.sub(r"[\d\s\W]+", "", content_only)) < 8:
            continue

        chunk_id = f"slide::{pptx_path.stem}::s{idx:03d}"
        chunks.append(
            DocumentChunk(
                chunk_id=_normalize_str(chunk_id),
                source_type="slide",
                source_file=file_name,
                chapter=chapter,
                location_label=f"Slide {idx}",
                page_or_slide=idx,
                title=title_text[:200],
                text=full_text,
                char_count=len(full_text),
                metadata={
                    "has_table": bool(tables_md),
                    "has_notes": bool(notes_text),
                },
            )
        )

    return chunks


def parse_all_pptx(docs_dir: Path = DOCS_DIR) -> List[DocumentChunk]:
    pptx_files = sorted(docs_dir.glob("*.pptx"))
    all_chunks: List[DocumentChunk] = []
    for f in pptx_files:
        if f.name.startswith("~$"):
            continue
        file_chunks = parse_pptx_file(f)
        print(f"[PPTX] {f.name}: {len(file_chunks)} slides indexed")
        all_chunks.extend(file_chunks)

    out_file = PARSED_DIR / "pptx_chunks.json"
    out_file.write_text(
        json.dumps([c.model_dump() for c in all_chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"[PPTX] Saved {len(all_chunks)} slide chunks -> {out_file}")
    return all_chunks


if __name__ == "__main__":
    parse_all_pptx()
