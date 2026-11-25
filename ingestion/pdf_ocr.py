from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Optional

import pymupdf as fitz

from config import DATA_DIR, DOCS_DIR, OCR_CACHE_DIR, PARSED_DIR

TESSDATA_DIR = DATA_DIR / "tessdata"


def _find_textbook_pdf(docs_dir: Path = DOCS_DIR) -> Path:
    pdfs = sorted(docs_dir.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"Không tìm thấy file PDF nào trong {docs_dir}")
    return max(pdfs, key=lambda p: p.stat().st_size)


def _detect_chapter_heading(text: str) -> str:
    m = re.search(
        r"(?:^|\n)\s*(CHƯƠNG\s+[IVX0-9]+[^\n]*)", text, re.IGNORECASE
    )
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()[:140]
    return ""


def _clean_ocr_text(raw: str) -> str:
    text = unicodedata.normalize("NFC", raw.strip())
    # Remove isolated gutter/binding noise characters at start of lines
    text = re.sub(r"(?m)^[\|\[\]\\/iIl!]{1,2}\s+(?=[A-ZÀ-Ỹa-zà-ỹ0-9“\"(])", "", text)
    # Join hyphenated line breaks cleanly
    text = re.sub(r"-\n([a-zà-ỹ])", r"\1", text)
    # Merge soft-wrapped lines inside paragraphs while keeping headings/bullets/paragraph breaks
    lines = [ln.strip() for ln in text.splitlines()]
    merged: list[str] = []
    buf = ""
    for ln in lines:
        if not ln:
            if buf:
                merged.append(buf)
                buf = ""
            continue
        is_new_block = bool(
            re.match(
                r"^(CHƯƠNG\s|Mục\s|\d+\.\s|\d+\.\d+|[a-zđ]\)|[-•+]\s|Điều\s+\d+)",
                ln,
                re.IGNORECASE,
            )
        )
        if is_new_block:
            if buf:
                merged.append(buf)
            buf = ln
        elif buf:
            buf = f"{buf} {ln}"
        else:
            buf = ln
    if buf:
        merged.append(buf)
    return "\n\n".join(merged).strip()


def _ocr_single_pdf_page(
    pdf_path: Path, page_num: int, dpi: int = 210
) -> dict:
    """
    OCR 1 PDF page (1-based) locally using Tesseract Vietnamese (tessdata_best).
    Automatically splits landscape double-page book spreads into Left + Right halves
    and trims book-spine gutter margins so reading order and accuracy stay clean.
    """
    cache_file = OCR_CACHE_DIR / f"tess_p{page_num:04d}.json"
    if cache_file.exists():
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        if "markdown" in cached:
            return cached

    doc = fitz.open(str(pdf_path))
    page = doc[page_num - 1]
    w, h = page.rect.width, page.rect.height
    is_spread = w > h * 1.15

    if is_spread:
        # Trim 2.5% outer margin and 1.5% center spine gutter to prevent dark-crease artifacts
        clips = [
            fitz.Rect(w * 0.025, h * 0.02, w * 0.488, h * 0.98),
            fitz.Rect(w * 0.512, h * 0.02, w * 0.975, h * 0.98),
        ]
    else:
        clips = [fitz.Rect(w * 0.02, h * 0.02, w * 0.98, h * 0.98)]

    parts_text: list[str] = []
    with tempfile.TemporaryDirectory(dir=str(DATA_DIR), prefix="tmp_ocr_") as tmpdir:
        for idx, clip in enumerate(clips):
            img_path = os.path.join(tmpdir, f"p{page_num}_{idx}.png")
            pix = page.get_pixmap(dpi=dpi, clip=clip, colorspace=fitz.csGRAY)
            pix.save(img_path)
            res = subprocess.run(
                [
                    "tesseract",
                    img_path,
                    "stdout",
                    "--tessdata-dir",
                    str(TESSDATA_DIR),
                    "-l",
                    "vie",
                    "--psm",
                    "6",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            cleaned = _clean_ocr_text(res.stdout or "")
            if cleaned:
                parts_text.append(cleaned)
    doc.close()

    merged_md = "\n\n".join(parts_text).strip()
    result = {
        "page": page_num,
        "is_spread": is_spread,
        "chapter": _detect_chapter_heading(merged_md),
        "markdown": merged_md,
    }
    cache_file.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def ocr_textbook_pdf(
    pdf_path: Optional[Path] = None,
    api_key: Optional[str] = None,
    batch_size: int = 2,
    max_workers: int = 6,
    start_page: int = 1,
    end_page: Optional[int] = None,
    model_name: str = "tesseract-vie-best",
) -> Path:
    if not (TESSDATA_DIR / "vie.traineddata").exists():
        raise FileNotFoundError(
            f"Thiếu {TESSDATA_DIR / 'vie.traineddata'}. Hãy tải tessdata_best/vie.traineddata trước."
        )

    pdf_path = pdf_path or _find_textbook_pdf()
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    doc.close()

    end_p = min(end_page or total_pages, total_pages)
    page_nums = list(range(start_page, end_p + 1))

    print(
        f"[OCR] Starting Local Tesseract (vie tessdata_best) OCR on {pdf_path.name}: "
        f"pages {start_page}-{end_p}/{total_pages} (workers={max_workers}, auto spread-split=True)",
        flush=True,
    )

    t0 = time.time()
    results_map: dict[int, dict] = {}
    done_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(_ocr_single_pdf_page, pdf_path, p_num): p_num
            for p_num in page_nums
        }
        for future in concurrent.futures.as_completed(future_map):
            p_num = future_map[future]
            results_map[p_num] = future.result()
            done_count += 1
            if done_count % 25 == 0 or done_count == len(page_nums):
                elapsed = time.time() - t0
                rate = done_count / max(elapsed, 0.1)
                eta = (len(page_nums) - done_count) / max(rate, 0.01)
                print(
                    f"[OCR] Progress: {done_count}/{len(page_nums)} PDF spreads "
                    f"({elapsed:.1f}s elapsed, ~{eta:.0f}s remaining)",
                    flush=True,
                )

    all_pages = [results_map[p] for p in sorted(results_map.keys())]

    current_chapter = ""
    for p in all_pages:
        if p.get("chapter"):
            current_chapter = p["chapter"]
        else:
            p["chapter"] = current_chapter

    out_payload = {
        "source_file": unicodedata.normalize("NFC", pdf_path.name),
        "ocr_engine": "Tesseract OCR (vie tessdata_best, auto double-page spread split)",
        "total_pages_in_pdf": total_pages,
        "ocr_pages_count": len(all_pages),
        "pages": all_pages,
    }
    out_file = PARSED_DIR / "textbook_ocr.json"
    out_file.write_text(
        json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total_chars = sum(len(p.get("markdown", "")) for p in all_pages)
    print(
        f"[OCR] Completed! Saved {len(all_pages)} PDF pages ({total_chars:,} chars) in {time.time()-t0:.1f}s -> {out_file}",
        flush=True,
    )
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Local Vietnamese Tesseract OCR")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--start-page", type=int, default=1)
    parser.add_argument("--end-page", type=int, default=None)
    args = parser.parse_args()
    ocr_textbook_pdf(
        max_workers=args.workers,
        start_page=args.start_page,
        end_page=args.end_page,
    )


if __name__ == "__main__":
    main()
