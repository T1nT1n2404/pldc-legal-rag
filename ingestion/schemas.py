from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    chunk_id: str
    source_type: Literal["slide", "textbook"]
    source_file: str
    chapter: str = ""
    location_label: str  # e.g., "Slide 45" or "Trang 112"
    page_or_slide: int
    title: str = ""
    text: str
    char_count: int = 0
    metadata: dict = Field(default_factory=dict)

    def citation(self) -> str:
        if self.source_type == "slide":
            return f"[Slide: {self.source_file} - {self.location_label}]"
        return f"[Giáo trình: {self.source_file} - {self.location_label}]"
