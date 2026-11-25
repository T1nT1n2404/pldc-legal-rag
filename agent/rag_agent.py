from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_CHAT_MODEL, GEMINI_FALLBACK_MODELS
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

SYSTEM_INSTRUCTION = """Bạn là **Trợ lý Pháp luật Đại cương (PLĐC) — Hệ thống Agentic RAG**.

Nguyên tắc hoạt động bắt buộc:
1. **Tra cứu tài liệu trước khi trả lời lý thuyết (Grounded RAG):**
   - Luôn gọi công cụ `search_legal_knowledge` để tìm kiếm trong bộ Slide bài giảng và Giáo trình Pháp luật Đại cương.
   - Bám sát nội dung truy xuất được; luôn ghi trích dẫn nguồn rõ ràng theo đúng nhãn như `[Slide: 261_Chương 2_Dân sự.pptx - Slide 221]` hoặc `[Giáo trình: ... - Trang 112]`.
   - Nếu tài liệu chưa đề cập trực tiếp chi tiết nào, nói rõ phần nào lấy từ tài liệu và phần nào bổ sung theo quy định chung của pháp luật Việt Nam hiện hành (BLDS 2015, BLHS 2015, BLLĐ 2019).

2. **KHÔNG tự tính nhẩm — BẮT BUỘC dùng công cụ tính toán (Deterministic Tool Calling):**
   - Mọi bài tập chia thừa kế → gọi `tool_calculate_inheritance`.
   - Mọi bài tập tổng hợp hình phạt nhiều tội (Điều 55 BLHS) → gọi `tool_calculate_multi_crime_sentence`.
   - Mọi bài tập tổng hợp hình phạt nhiều bản án (Điều 56 BLHS: tội cũ K1 hay tội mới K2) → gọi `tool_calculate_multi_judgment_sentence`.
   - Mọi bài kiểm tra điều kiện án treo & thời gian thử thách (Điều 65 BLHS) → gọi `tool_calculate_suspended_sentence`.
   - Mọi bài tính lương làm thêm giờ / làm đêm (Điều 98), trợ cấp thôi việc / mất việc (Điều 46, 47), thử việc (Điều 25, 26) → gọi `tool_calculate_labor_law`.
   - Sau khi công cụ trả kết quả chính xác, hãy trình bày lời giải rõ ràng theo chuẩn bài thi PLĐC: (1) Cơ sở pháp lý, (2) Phân tích dữ kiện, (3) Bảng/từng bước tính toán, (4) Kết luận cuối cùng.

3. **Lưu ý nghiệp vụ quan trọng khi trích xuất tham số cho công cụ:**
   - Với **chia thừa kế**:
     - Con riêng của vợ/chồng KHÔNG phải hàng thừa kế thứ nhất của người chết (trừ khi có quan hệ chăm sóc nuôi dưỡng như cha con/mẹ con theo Điều 654).
     - Vợ/chồng hợp pháp, cha/mẹ, con chưa thành niên (< 18 tuổi), con thành niên mất khả năng lao động thuộc diện Điều 644 BLDS (nhận tối thiểu 2/3 suất theo pháp luật).
     - Tài sản chung vợ chồng phải chia đôi trước để xác định di sản của người chết (`marital_joint_assets`).
   - Với **Điều 56 BLHS**: Phân biệt rõ **Khoản 1 (tội cũ — phạm trước khi có bản án 1: cộng trước, trừ sau)** vs **Khoản 2 (tội mới — phạm khi đang chấp hành bản án 1: trừ trước, cộng sau)**.

Trả lời bằng tiếng Việt mạch lạc, sử dụng Markdown gọn đẹp (tiêu đề, bảng, in đậm con số kết quả). Viết phân số dạng `2/3` thay vì LaTeX `$\frac{2}{3}$` để hiển thị sạch trên web. Ghi trích dẫn nguồn chính xác từng thẻ riêng như `[Slide: ...]` hoặc `[Giáo trình: ...]`."""


# JSON Schema declarations for Gemini Function Calling
def _build_tool_declarations() -> types.Tool:
    return types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="search_legal_knowledge",
                description=(
                    "Tìm kiếm kiến thức pháp luật trong kho dữ liệu Hybrid RAG gồm 6 bộ Slide bài giảng "
                    "và Giáo trình Pháp luật Đại cương. Luôn gọi hàm này khi cần tra cứu khái niệm, điều luật, ví dụ, bài tập mẫu."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "query": {
                            "type": "STRING",
                            "description": "Câu truy vấn tiếng Việt ngắn gọn, giàu từ khóa pháp lý (ví dụ: 'Điều 644 người thừa kế không phụ thuộc nội dung di chúc').",
                        },
                        "source_type": {
                            "type": "STRING",
                            "enum": ["all", "slide", "textbook"],
                            "description": "Lọc nguồn: 'all' (cả slide và giáo trình), 'slide', hoặc 'textbook'.",
                        },
                        "top_k": {
                            "type": "INTEGER",
                            "description": "Số đoạn tài liệu trả về (mặc định 6, tối đa 10).",
                        },
                    },
                    "required": ["query"],
                },
            ),
            types.FunctionDeclaration(
                name="tool_calculate_inheritance",
                description=(
                    "Tính toán chính xác chia thừa kế theo Bộ luật Dân sự 2015: xác định di sản ròng (chia đôi tài sản chung vợ chồng, trừ nợ/mai táng phí), "
                    "chia theo di chúc, chia theo pháp luật, thừa kế thế vị (Điều 652), và trích bù suất bắt buộc 2/3 cho người thừa kế không phụ thuộc di chúc (Điều 644)."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "marital_joint_assets": {
                            "type": "NUMBER",
                            "description": "Tổng tài sản chung của vợ chồng (hàm sẽ tự động chia đôi để lấy phần của người chết).",
                        },
                        "personal_assets": {
                            "type": "NUMBER",
                            "description": "Tài sản riêng của người chết (hoặc tổng di sản nếu đề bài đã cho sẵn di sản riêng của người chết).",
                        },
                        "other_joint_assets": {
                            "type": "NUMBER",
                            "description": "Tài sản đồng sở hữu khác (nếu có).",
                        },
                        "other_joint_ownership_ratio": {
                            "type": "NUMBER",
                            "description": "Tỷ lệ sở hữu của người chết trong tài sản đồng sở hữu khác (mặc định 0.5, ví dụ tài sản chung với người tình chia 4 -> 0.25).",
                        },
                        "debts_and_obligations": {
                            "type": "NUMBER",
                            "description": "Nghĩa vụ tài sản / tiền nợ của người chết cần thanh toán từ di sản.",
                        },
                        "funeral_cost": {
                            "type": "NUMBER",
                            "description": "Chi phí hợp lý cho việc mai táng (Điều 658).",
                        },
                        "has_valid_will": {
                            "type": "BOOLEAN",
                            "description": "True nếu có di chúc hợp pháp; False nếu không có hoặc di chúc vô hiệu toàn bộ.",
                        },
                        "will_beneficiaries": {
                            "type": "ARRAY",
                            "description": "Danh sách người được chỉ định trong di chúc.",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "name": {"type": "STRING"},
                                    "share_amount": {"type": "NUMBER", "description": "Số tiền cụ thể theo di chúc (nếu có)."},
                                    "share_fraction": {"type": "NUMBER", "description": "Tỷ lệ phần di sản ròng (vd: 1.0 cho toàn bộ, 0.5 cho 1/2)."},
                                    "is_alive_at_opening": {"type": "BOOLEAN"},
                                    "refused_or_disqualified": {"type": "BOOLEAN"},
                                },
                                "required": ["name"],
                            },
                        },
                        "statutory_heirs": {
                            "type": "ARRAY",
                            "description": "Danh sách những người thuộc hàng thừa kế theo pháp luật đang được hưởng (thường là hàng 1).",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "name": {"type": "STRING"},
                                    "relation": {"type": "STRING", "description": "VD: 'vợ', 'chồng', 'cha', 'mẹ', 'con chưa thành niên', 'con thành niên mất khả năng lao động', 'con thành niên'."},
                                    "is_forced_heir_644": {"type": "BOOLEAN", "description": "True nếu thuộc diện Điều 644 (vợ/chồng/cha/mẹ/con <18t/con thành niên mất khả năng LĐ)."},
                                    "disinherited_by_will": {"type": "BOOLEAN", "description": "True nếu bị truất quyền hưởng di sản trong di chúc."},
                                    "refused_or_disqualified_621": {"type": "BOOLEAN", "description": "True nếu từ chối nhận di sản hoặc vi phạm Điều 621."},
                                    "predeceased_substitutes": {
                                        "type": "ARRAY",
                                        "items": {"type": "STRING"},
                                        "description": "Tên các cháu/chắt hưởng thừa kế thế vị (Điều 652) nếu người con này đã chết trước.",
                                    },
                                },
                                "required": ["name", "relation"],
                            },
                        },
                        "currency_unit": {
                            "type": "STRING",
                            "description": "Đơn vị tiền tệ dùng trong đề bài, ví dụ 'triệu đồng', 'tỷ đồng', 'VNĐ'.",
                        },
                    },
                    "required": ["statutory_heirs"],
                },
            ),
            types.FunctionDeclaration(
                name="tool_calculate_multi_crime_sentence",
                description=(
                    "Tổng hợp hình phạt khi phạm nhiều tội (Điều 55 & Điều 38 BLHS 2015): quy đổi 3 ngày CTKGG = 1 ngày tù, "
                    "áp trần 3 năm CTKGG / 30 năm tù có thời hạn, chung thân/tử hình thu hút, trừ tạm giữ/tạm giam, hình phạt bổ sung."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "penalties": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "crime_name": {"type": "STRING"},
                                    "penalty_type": {
                                        "type": "STRING",
                                        "enum": [
                                            "tu_hinh",
                                            "tu_chung_than",
                                            "tu_co_thoi_han",
                                            "cai_tao_khong_giam_giu",
                                            "phat_tien",
                                            "truc_xuat",
                                            "canh_cao",
                                        ],
                                    },
                                    "years": {"type": "NUMBER"},
                                    "months": {"type": "NUMBER"},
                                    "days": {"type": "NUMBER"},
                                    "amount_vnd": {"type": "NUMBER"},
                                },
                                "required": ["penalty_type"],
                            },
                        },
                        "detention_days": {"type": "NUMBER"},
                        "detention_months": {"type": "NUMBER"},
                        "supplementary_same_type_months": {
                            "type": "ARRAY",
                            "items": {"type": "NUMBER"},
                        },
                        "supplementary_fines_vnd": {
                            "type": "ARRAY",
                            "items": {"type": "NUMBER"},
                        },
                    },
                    "required": ["penalties"],
                },
            ),
            types.FunctionDeclaration(
                name="tool_calculate_multi_judgment_sentence",
                description=(
                    "Tổng hợp hình phạt của nhiều bản án theo Điều 56 BLHS 2015: "
                    "Khoản 1 (phạm tội CŨ trước khi có BA1: C: + -) hoặc Khoản 2 (đang chấp hành BA1 lại phạm tội MỚI: M: - +)."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "case_type": {
                            "type": "STRING",
                            "enum": ["khoan_1_toi_cu", "khoan_2_toi_moi"],
                        },
                        "ba1_type": {
                            "type": "STRING",
                            "enum": ["tu_co_thoi_han", "cai_tao_khong_giam_giu", "tu_chung_than"],
                        },
                        "ba1_months": {"type": "NUMBER", "description": "Tổng thời hạn bản án 1 tính theo tháng (VD: 10 năm = 120 tháng)."},
                        "ba1_served_months": {"type": "NUMBER", "description": "Thời gian đã chấp hành bản án 1 tính theo tháng."},
                        "ba2_type": {
                            "type": "STRING",
                            "enum": ["tu_co_thoi_han", "cai_tao_khong_giam_giu", "tu_chung_than", "tu_hinh"],
                        },
                        "ba2_months": {"type": "NUMBER", "description": "Thời hạn bản án 2 tính theo tháng."},
                        "ba2_detention_months": {"type": "NUMBER", "description": "Thời gian tạm giam ở vụ án 2 tính theo tháng (nếu có)."},
                    },
                    "required": ["case_type", "ba1_type", "ba1_months", "ba1_served_months", "ba2_type"],
                },
            ),
            types.FunctionDeclaration(
                name="tool_calculate_suspended_sentence",
                description="Kiểm tra điều kiện hưởng án treo và tính thời gian thử thách theo Điều 65 BLHS 2015 & NQ 02/2018/NQ-HĐTP.",
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "prison_months": {"type": "NUMBER", "description": "Mức phạt tù bị tuyên tính theo tháng."},
                        "has_good_character": {"type": "BOOLEAN"},
                        "has_stable_residence_or_work": {"type": "BOOLEAN"},
                        "mitigating_factors_count": {"type": "INTEGER"},
                        "aggravating_factors_count": {"type": "INTEGER"},
                        "is_recidivist": {"type": "BOOLEAN"},
                        "fled_and_wanted": {"type": "BOOLEAN"},
                        "multiple_crimes_same_trial": {"type": "BOOLEAN"},
                        "is_under_18": {"type": "BOOLEAN"},
                        "new_crime_during_probation": {"type": "BOOLEAN"},
                    },
                    "required": ["prison_months"],
                },
            ),
            types.FunctionDeclaration(
                name="tool_calculate_labor_law",
                description=(
                    "Tính toán theo Bộ luật Lao động 2019: 'overtime' (lương làm thêm giờ/ban đêm Điều 98), "
                    "'severance' (trợ cấp thôi việc Điều 46 / mất việc làm Điều 47), hoặc 'probation' (lương & thời gian thử việc Điều 25, 26)."
                ),
                parameters={
                    "type": "OBJECT",
                    "properties": {
                        "calc_type": {
                            "type": "STRING",
                            "enum": ["overtime", "severance", "probation"],
                        },
                        "hourly_wage": {"type": "NUMBER"},
                        "hours": {"type": "NUMBER"},
                        "day_type": {
                            "type": "STRING",
                            "enum": ["ngay_thuong", "ngay_nghi_hang_tuan", "ngay_le_tet"],
                        },
                        "is_night_shift": {"type": "BOOLEAN"},
                        "worked_daytime_before": {"type": "BOOLEAN"},
                        "allowance_type": {
                            "type": "STRING",
                            "enum": ["thoi_viec", "mat_viec_lam"],
                        },
                        "avg_salary_6_months": {"type": "NUMBER"},
                        "total_working_months": {"type": "NUMBER"},
                        "unemployment_insurance_months": {"type": "NUMBER"},
                        "official_salary": {"type": "NUMBER"},
                        "role_level": {
                            "type": "STRING",
                            "enum": ["quan_ly_dn", "cao_dang_tro_len", "trung_cap_ktv", "khac", "hd_duoi_1_thang"],
                        },
                        "currency_unit": {"type": "STRING"},
                    },
                    "required": ["calc_type"],
                },
            ),
        ]
    )


class LegalRAGAgent:
    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever
        self.tools = _build_tool_declarations()

    def _execute_tool(
        self,
        name: str,
        args: Dict[str, Any],
        api_key: str,
        retrieval_mode: str,
        source_filter: str,
    ) -> tuple[dict, List[dict]]:
        retrieved_chunks: List[dict] = []
        if name == "search_legal_knowledge":
            q = str(args.get("query", ""))
            st = args.get("source_type") or source_filter or "all"
            if source_filter != "all":
                st = source_filter
            top_k = min(int(args.get("top_k", 6)), 10)
            res = self.retriever.search(
                query=q,
                top_k=top_k,
                source_type=st,
                mode=retrieval_mode,
                api_key=api_key,
            )
            retrieved_chunks = res.get("results", [])
            # Compact payload sent back to LLM to save context tokens while keeping full text & citations
            llm_payload = {
                "mode_used": res.get("mode_used"),
                "latency_ms": res.get("latency_ms"),
                "documents": [
                    {
                        "citation": r["citation"],
                        "chapter": r["chapter"],
                        "score": r["final_score"],
                        "content": r["text"],
                    }
                    for r in retrieved_chunks
                ],
            }
            return llm_payload, retrieved_chunks

        if name == "tool_calculate_inheritance":
            return calculate_inheritance(**args), []

        if name == "tool_calculate_multi_crime_sentence":
            return calculate_multi_crime_sentence(**args), []

        if name == "tool_calculate_multi_judgment_sentence":
            return calculate_multi_judgment_sentence(**args), []

        if name == "tool_calculate_suspended_sentence":
            return calculate_suspended_sentence(**args), []

        if name == "tool_calculate_labor_law":
            ct = args.get("calc_type")
            cu = args.get("currency_unit", "VNĐ")
            if ct == "overtime":
                return (
                    calculate_overtime_pay(
                        hourly_wage=float(args.get("hourly_wage", 0)),
                        hours=float(args.get("hours", 0)),
                        day_type=args.get("day_type", "ngay_thuong"),
                        is_night_shift=bool(args.get("is_night_shift", False)),
                        worked_daytime_before=bool(args.get("worked_daytime_before", False)),
                        currency_unit=cu,
                    ),
                    [],
                )
            if ct == "severance":
                return (
                    calculate_severance_allowance(
                        allowance_type=args.get("allowance_type", "thoi_viec"),
                        avg_salary_6_months=float(args.get("avg_salary_6_months", 0)),
                        total_working_months=float(args.get("total_working_months", 0)),
                        unemployment_insurance_months=float(
                            args.get("unemployment_insurance_months", 0)
                        ),
                        currency_unit=cu,
                    ),
                    [],
                )
            return (
                calculate_probation_rules(
                    official_salary=float(args.get("official_salary", 0)),
                    role_level=args.get("role_level", "cao_dang_tro_len"),
                    currency_unit=cu,
                ),
                [],
            )

        return {"error": f"Unknown tool: {name}"}, []

    def chat(
        self,
        message: str,
        history: Optional[List[dict]] = None,
        api_key: Optional[str] = None,
        source_filter: str = "all",
        retrieval_mode: str = "hybrid",
        model_name: str = GEMINI_CHAT_MODEL,
    ) -> dict:
        t0 = time.perf_counter()
        key = (api_key or GEMINI_API_KEY).strip()
        if not key:
            raise ValueError(
                "Chưa có GEMINI_API_KEY. Hãy thêm vào file legal_rag/.env hoặc nhập khóa API ở góc trên bên phải giao diện."
            )

        client = genai.Client(api_key=key)

        contents: List[types.Content] = []
        for turn in (history or [])[-8:]:
            role = "user" if turn.get("role") == "user" else "model"
            text = str(turn.get("content", "")).strip()
            if text:
                contents.append(
                    types.Content(role=role, parts=[types.Part.from_text(text=text)])
                )
        contents.append(
            types.Content(role="user", parts=[types.Part.from_text(text=message)])
        )

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            tools=[self.tools],
            temperature=0.15,
        )

        tool_trace: List[dict] = []
        all_citations: List[dict] = []
        seen_chunk_ids: set[str] = set()
        final_text = ""

        # Agentic tool-calling loop (max 6 iterations) with automatic 503/404 model fallback
        active_model = model_name
        candidate_models = [model_name] + [
            m for m in GEMINI_FALLBACK_MODELS if m != model_name
        ]

        for _step in range(6):
            response = None
            last_exc = None
            for m_try in candidate_models:
                try:
                    response = client.models.generate_content(
                        model=m_try,
                        contents=contents,
                        config=config,
                    )
                    active_model = m_try
                    # Promote working model to front for subsequent turns
                    candidate_models = [m_try] + [m for m in candidate_models if m != m_try]
                    break
                except Exception as exc:
                    last_exc = exc
                    err_s = str(exc)
                    if any(code in err_s for code in ("503", "UNAVAILABLE", "404", "NOT_FOUND")):
                        continue
                    raise
            if response is None and last_exc is not None:
                raise last_exc

            candidate = response.candidates[0] if response.candidates else None
            if not candidate or not candidate.content:
                break

            contents.append(candidate.content)
            fn_calls = [
                part.function_call
                for part in (candidate.content.parts or [])
                if part.function_call is not None
            ]

            if not fn_calls:
                final_text = response.text or ""
                break

            fn_response_parts: List[types.Part] = []
            for fc in fn_calls:
                call_t0 = time.perf_counter()
                fc_args = dict(fc.args) if fc.args else {}
                try:
                    result_obj, chunks = self._execute_tool(
                        name=fc.name,
                        args=fc_args,
                        api_key=key,
                        retrieval_mode=retrieval_mode,
                        source_filter=source_filter,
                    )
                except Exception as exc:
                    result_obj = {"error": str(exc)}
                    chunks = []
                call_ms = round((time.perf_counter() - call_t0) * 1000, 2)

                for ch in chunks:
                    if ch["chunk_id"] not in seen_chunk_ids:
                        seen_chunk_ids.add(ch["chunk_id"])
                        all_citations.append(ch)

                tool_trace.append(
                    {
                        "tool": fc.name,
                        "args": fc_args,
                        "duration_ms": call_ms,
                        "output_preview": result_obj
                        if fc.name != "search_legal_knowledge"
                        else {
                            "mode_used": result_obj.get("mode_used"),
                            "hits": len(chunks),
                            "top_citations": [c["citation"] for c in chunks[:4]],
                        },
                    }
                )

                fn_response_parts.append(
                    types.Part.from_function_response(
                        name=fc.name,
                        response={"result": result_obj},
                    )
                )

            contents.append(types.Content(role="user", parts=fn_response_parts))

        if not final_text:
            final_text = "Xin lỗi, hệ thống chưa tạo được câu trả lời hoàn chỉnh. Vui lòng thử lại."

        return {
            "answer": final_text,
            "citations": all_citations,
            "tool_trace": tool_trace,
            "total_latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "model": active_model,
        }
