from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field

MAX_CTKGG_MONTHS = 36.0  # 3 năm (Điều 55 khoản 1 điểm a)
MAX_PRISON_MONTHS = 360.0  # 30 năm (Điều 55 khoản 1 điểm a, b)
MAX_SUPPLEMENTARY_MONTHS = 60.0  # 5 năm (Điều 55 khoản 2)


def format_duration_months(total_months: float) -> str:
    """Format fractional months cleanly as X năm Y tháng Z ngày (1 tháng = 30 ngày)."""
    total_days = int(round(max(0.0, total_months) * 30))
    years = total_days // 360
    rem = total_days % 360
    months = rem // 30
    days = rem % 30
    parts = []
    if years:
        parts.append(f"{years} năm")
    if months:
        parts.append(f"{months} tháng")
    if days:
        parts.append(f"{days} ngày")
    return " ".join(parts) if parts else "0 ngày"


class PenaltyItem(BaseModel):
    crime_name: str = Field(default="Tội phạm")
    penalty_type: Literal[
        "tu_hinh",
        "tu_chung_than",
        "tu_co_thoi_han",
        "cai_tao_khong_giam_giu",
        "phat_tien",
        "truc_xuat",
        "canh_cao",
    ]
    years: float = 0.0
    months: float = 0.0
    days: float = 0.0
    amount_vnd: float = 0.0

    def total_months(self) -> float:
        return self.years * 12.0 + self.months + (self.days / 30.0)


def calculate_multi_crime_sentence(
    penalties: List[dict],
    detention_days: float = 0.0,
    detention_months: float = 0.0,
    supplementary_same_type_months: Optional[List[float]] = None,
    supplementary_fines_vnd: Optional[List[float]] = None,
) -> dict:
    """
    Tổng hợp hình phạt trong trường hợp phạm nhiều tội (Điều 55 & Điều 38 BLHS 2015).
    """
    items = [PenaltyItem(**p) for p in penalties]
    steps: List[str] = []
    types_present = {p.penalty_type for p in items}
    total_detention_months = detention_months + (detention_days / 30.0)

    # Separate fines & deportation (do not merge with prison)
    total_main_fine = sum(p.amount_vnd for p in items if p.penalty_type == "phat_tien")
    has_deportation = "truc_xuat" in types_present

    final_custodial_type = "none"
    final_months = 0.0
    remaining_months = 0.0
    summary = ""

    if "tu_hinh" in types_present:
        final_custodial_type = "tu_hinh"
        summary = "Tử hình"
        steps.append(
            "**Điều 55 khoản 1 điểm d BLHS 2015:** Hình phạt nặng nhất là **Tử hình** → hình phạt chung là **Tử hình**."
        )
    elif "tu_chung_than" in types_present:
        final_custodial_type = "tu_chung_than"
        summary = "Tù chung thân"
        steps.append(
            "**Điều 55 khoản 1 điểm c BLHS 2015:** Hình phạt nặng nhất là **Tù chung thân** → hình phạt chung là **Tù chung thân**."
        )
    elif "tu_co_thoi_han" in types_present:
        final_custodial_type = "tu_co_thoi_han"
        prison_sum = sum(
            p.total_months() for p in items if p.penalty_type == "tu_co_thoi_han"
        )
        ctkgg_sum = sum(
            p.total_months() for p in items if p.penalty_type == "cai_tao_khong_giam_giu"
        )
        converted_ctkgg = ctkgg_sum / 3.0

        steps.append(
            f"**Bước 1 — Tổng hợp hình phạt chính (Điều 55 khoản 1 điểm a, b BLHS 2015):**\n"
            f"- Tổng tù có thời hạn: `{format_duration_months(prison_sum)}` ({prison_sum:.2f} tháng)."
        )
        if ctkgg_sum > 0:
            steps.append(
                f"- Quy đổi Cải tạo không giam giữ sang Tù (3 ngày CTKGG = 1 ngày tù): "
                f"`{format_duration_months(ctkgg_sum)}` / 3 = `{format_duration_months(converted_ctkgg)}`."
            )

        raw_total = prison_sum + converted_ctkgg
        final_months = min(raw_total, MAX_PRISON_MONTHS)
        if raw_total > MAX_PRISON_MONTHS:
            steps.append(
                f"- Tổng cộng `{format_duration_months(raw_total)}` vượt quá mức trần **30 năm** → áp dụng mức tối đa **30 năm**."
            )
        else:
            steps.append(
                f"- Hình phạt chung (tù có thời hạn): **{format_duration_months(final_months)}**."
            )

        remaining_months = max(0.0, final_months - total_detention_months)
        if total_detention_months > 0:
            steps.append(
                f"**Bước 2 — Trừ thời gian tạm giữ/tạm giam (Điều 38 BLHS, 1 ngày = 1 ngày tù):**\n"
                f"- Trừ `{format_duration_months(total_detention_months)}` → còn phải chấp hành: **{format_duration_months(remaining_months)}**."
            )
        summary = f"Tù có thời hạn: {format_duration_months(remaining_months)}"
    elif "cai_tao_khong_giam_giu" in types_present:
        final_custodial_type = "cai_tao_khong_giam_giu"
        ctkgg_sum = sum(
            p.total_months() for p in items if p.penalty_type == "cai_tao_khong_giam_giu"
        )
        final_months = min(ctkgg_sum, MAX_CTKGG_MONTHS)
        steps.append(
            f"**Điều 55 khoản 1 điểm a BLHS 2015 (Cùng là Cải tạo không giam giữ):**\n"
            f"- Tổng các hình phạt: `{format_duration_months(ctkgg_sum)}` (trần tối đa **03 năm**).\n"
            f"- Hình phạt chung: **{format_duration_months(final_months)}**."
        )
        # 1 day detention = 3 days CTKGG (Art. 38)
        deduct_ctkgg = total_detention_months * 3.0
        remaining_months = max(0.0, final_months - deduct_ctkgg)
        if total_detention_months > 0:
            steps.append(
                f"**Trừ tạm giữ/tạm giam (1 ngày tạm giam = 3 ngày CTKGG):** "
                f"trừ `{format_duration_months(deduct_ctkgg)}` → còn lại **{format_duration_months(remaining_months)}**."
            )
        summary = f"Cải tạo không giam giữ: {format_duration_months(remaining_months)}"
    elif total_main_fine > 0:
        summary = f"Phạt tiền: {total_main_fine:,.0f} VNĐ"
    else:
        summary = "Cảnh cáo / Hình phạt khác"

    if total_main_fine > 0:
        steps.append(
            f"**Phạt tiền là hình phạt chính (Điều 55 khoản 1 điểm đ):** Cộng các khoản tiền (không tổng hợp với tù) = **{total_main_fine:,.0f} VNĐ**."
        )
    if has_deportation:
        steps.append(
            "**Trục xuất:** Không tổng hợp với các loại hình phạt khác, thi hành độc lập."
        )

    # Supplementary penalties (Khoản 2 Điều 55)
    supp_months = 0.0
    if supplementary_same_type_months:
        raw_supp = sum(supplementary_same_type_months)
        supp_months = min(raw_supp, MAX_SUPPLEMENTARY_MONTHS)
        steps.append(
            f"**Hình phạt bổ sung cùng loại (Điều 55 khoản 2):** Tổng `{format_duration_months(raw_supp)}` → áp trần tối đa 5 năm = **{format_duration_months(supp_months)}**."
        )
    supp_fine = sum(supplementary_fines_vnd or [])
    if supp_fine > 0:
        steps.append(
            f"**Phạt tiền bổ sung:** Cộng lại = **{supp_fine:,.0f} VNĐ**."
        )

    return {
        "final_custodial_type": final_custodial_type,
        "combined_months_before_detention": round(final_months, 3),
        "remaining_months": round(remaining_months, 3),
        "remaining_human": format_duration_months(remaining_months),
        "total_main_fine_vnd": total_main_fine,
        "has_deportation": has_deportation,
        "supplementary_months": round(supp_months, 3),
        "supplementary_fine_vnd": supp_fine,
        "summary": summary,
        "steps": steps,
        "legal_basis": ["Điều 55 BLHS 2015 (sửa đổi, bổ sung 2017)", "Điều 38 BLHS 2015"],
    }


def calculate_multi_judgment_sentence(
    case_type: Literal["khoan_1_toi_cu", "khoan_2_toi_moi"],
    ba1_type: Literal["tu_co_thoi_han", "cai_tao_khong_giam_giu", "tu_chung_than"],
    ba1_months: float,
    ba1_served_months: float,
    ba2_type: Literal["tu_co_thoi_han", "cai_tao_khong_giam_giu", "tu_chung_than", "tu_hinh"],
    ba2_months: float = 0.0,
    ba2_detention_months: float = 0.0,
) -> dict:
    """
    Tổng hợp hình phạt của nhiều bản án (Điều 56 BLHS 2015):
    - khoan_1_toi_cu (C: + -): Phạm tội TRƯỚC khi có bản án 1 -> Tổng hợp (BA1 + BA2, áp trần 30 năm) rồi mới trừ đã chấp hành BA1.
    - khoan_2_toi_moi (M: - +): Đang chấp hành bản án 1 lại phạm tội MỚI -> Tính phần còn lại của BA1 (BA1 - đã chấp hành), rồi tổng hợp với BA2 (áp trần 30 năm).
    """
    steps: List[str] = []

    if ba2_type == "tu_hinh":
        return {
            "case_type": case_type,
            "final_type": "tu_hinh",
            "remaining_months": 0.0,
            "remaining_human": "Tử hình",
            "steps": ["Hình phạt nặng nhất là **Tử hình** → hình phạt chung là **Tử hình** (Điều 55, 56 BLHS)."],
            "legal_basis": ["Điều 55, 56 BLHS 2015"],
        }
    if ba1_type == "tu_chung_than" or ba2_type == "tu_chung_than":
        return {
            "case_type": case_type,
            "final_type": "tu_chung_than",
            "remaining_months": 0.0,
            "remaining_human": "Tù chung thân",
            "steps": ["Hình phạt nặng nhất là **Tù chung thân** → hình phạt chung là **Tù chung thân** (Điều 55, 56 BLHS)."],
            "legal_basis": ["Điều 55, 56 BLHS 2015"],
        }

    # Normalize to prison months if either is prison; else both CTKGG
    both_ctkgg = ba1_type == "cai_tao_khong_giam_giu" and ba2_type == "cai_tao_khong_giam_giu"
    cap = MAX_CTKGG_MONTHS if both_ctkgg else MAX_PRISON_MONTHS
    final_type = "cai_tao_khong_giam_giu" if both_ctkgg else "tu_co_thoi_han"

    b1 = ba1_months / 3.0 if (not both_ctkgg and ba1_type == "cai_tao_khong_giam_giu") else ba1_months
    b1_served = ba1_served_months / 3.0 if (not both_ctkgg and ba1_type == "cai_tao_khong_giam_giu") else ba1_served_months
    b2 = ba2_months / 3.0 if (not both_ctkgg and ba2_type == "cai_tao_khong_giam_giu") else ba2_months

    if not both_ctkgg and (ba1_type == "cai_tao_khong_giam_giu" or ba2_type == "cai_tao_khong_giam_giu"):
        steps.append(
            f"**Quy đổi CTKGG sang tù (3 ngày CTKGG = 1 ngày tù):** "
            f"BA1 = `{format_duration_months(b1)}`, đã chấp hành BA1 = `{format_duration_months(b1_served)}`, BA2 = `{format_duration_months(b2)}`."
        )

    if case_type == "khoan_1_toi_cu":
        # C: + -
        raw_sum = b1 + b2
        capped = min(raw_sum, cap)
        remaining = max(0.0, capped - b1_served - ba2_detention_months)
        steps.append(
            f"**Áp dụng Khoản 1 Điều 56 BLHS 2015 (Tội cũ — công thức `C: + -`):**\n"
            f"- Bước 1: Tổng hợp hình phạt 2 bản án: `{format_duration_months(b1)} + {format_duration_months(b2)} = {format_duration_months(raw_sum)}`.\n"
            f"- Áp trần tối đa ({format_duration_months(cap)}): `{format_duration_months(capped)}`.\n"
            f"- Bước 2: Trừ thời gian đã chấp hành BA1 (`{format_duration_months(b1_served)}`)"
            + (
                f" và tạm giam BA2 (`{format_duration_months(ba2_detention_months)}`)"
                if ba2_detention_months > 0
                else ""
            )
            + f" → **Còn phải chấp hành: `{format_duration_months(remaining)}`**."
        )
    else:
        # M: - +
        b1_rem = max(0.0, b1 - b1_served)
        raw_sum = b1_rem + b2
        capped = min(raw_sum, cap)
        remaining = max(0.0, capped - ba2_detention_months)
        steps.append(
            f"**Áp dụng Khoản 2 Điều 56 BLHS 2015 (Phạm tội mới khi đang chấp hành án — công thức `M: - +`):**\n"
            f"- Bước 1: Tính phần chưa chấp hành của BA1: `{format_duration_months(b1)} - {format_duration_months(b1_served)} = {format_duration_months(b1_rem)}`.\n"
            f"- Bước 2: Tổng hợp với hình phạt của tội mới (BA2): `{format_duration_months(b1_rem)} + {format_duration_months(b2)} = {format_duration_months(raw_sum)}`.\n"
            f"- Áp trần tối đa ({format_duration_months(cap)}): `{format_duration_months(capped)}`."
            + (
                f"\n- Trừ thời gian tạm giam BA2 (`{format_duration_months(ba2_detention_months)}`)."
                if ba2_detention_months > 0
                else ""
            )
            + f"\n→ **Hình phạt chung còn phải chấp hành: `{format_duration_months(remaining)}`**."
        )

    return {
        "case_type": case_type,
        "final_type": final_type,
        "remaining_months": round(remaining, 3),
        "remaining_human": format_duration_months(remaining),
        "steps": steps,
        "legal_basis": ["Điều 55 BLHS 2015", "Điều 56 BLHS 2015"],
    }


def calculate_suspended_sentence(
    prison_months: float,
    has_good_character: bool = True,
    has_stable_residence_or_work: bool = True,
    mitigating_factors_count: int = 2,
    aggravating_factors_count: int = 0,
    is_recidivist: bool = False,
    fled_and_wanted: bool = False,
    multiple_crimes_same_trial: bool = False,
    is_under_18: bool = False,
    new_crime_during_probation: bool = False,
) -> dict:
    """
    Kiểm tra điều kiện hưởng án treo và tính thời gian thử thách (Điều 65 BLHS 2015 & NQ 02/2018/NQ-HĐTP).
    """
    disqualifiers: List[str] = []
    if prison_months > 36.0:
        disqualifiers.append(f"Mức phạt tù ({format_duration_months(prison_months)}) vượt quá 03 năm.")
    if is_recidivist:
        disqualifiers.append("Thuộc trường hợp tái phạm hoặc tái phạm nguy hiểm.")
    if fled_and_wanted:
        disqualifiers.append("Bỏ trốn và đã bị cơ quan tiến hành tố tụng đề nghị truy nã.")
    if new_crime_during_probation:
        disqualifiers.append("Đang hưởng án treo mà phạm tội mới trong thời gian thử thách.")
    if multiple_crimes_same_trial and not is_under_18:
        disqualifiers.append("Bị xét xử cùng một lần về nhiều tội (không phải người dưới 18 tuổi).")
    if not has_good_character:
        disqualifiers.append("Không đáp ứng điều kiện về nhân thân tốt.")
    if not has_stable_residence_or_work:
        disqualifiers.append("Không có nơi cư trú rõ ràng hoặc nơi làm việc ổn định.")
    if mitigating_factors_count < 2 and aggravating_factors_count == 0:
        disqualifiers.append("Chưa đủ từ 02 tình tiết giảm nhẹ trở lên theo Điều 2 NQ 02/2018/NQ-HĐTP.")
    elif aggravating_factors_count > 0 and (mitigating_factors_count - aggravating_factors_count) < 2:
        disqualifiers.append("Số tình tiết giảm nhẹ không nhiều hơn số tình tiết tăng nặng từ 02 tình tiết trở lên.")

    eligible = len(disqualifiers) == 0
    raw_probation = prison_months * 2.0
    probation_months = min(max(raw_probation, 12.0), 60.0) if eligible else 0.0

    steps = []
    if eligible:
        steps.append(
            f"**Đủ điều kiện xem xét hưởng án treo (Điều 65 BLHS 2015, NQ 02/2018/NQ-HĐTP).**\n"
            f"- Mức phạt tù: `{format_duration_months(prison_months)}` (≤ 3 năm).\n"
            f"- Thời gian thử thách = 2 × mức phạt tù = `{format_duration_months(raw_probation)}`, "
            f"giới hạn trong khung **[1 năm — 5 năm]** → **Thời gian thử thách: `{format_duration_months(probation_months)}`**."
        )
    else:
        steps.append(
            "**KHÔNG đủ điều kiện hưởng án treo** vì các lý do sau:\n"
            + "\n".join(f"- {d}" for d in disqualifiers)
        )

    return {
        "eligible": eligible,
        "prison_months": prison_months,
        "probation_months": round(probation_months, 2),
        "probation_human": format_duration_months(probation_months) if eligible else "Không áp dụng",
        "disqualifiers": disqualifiers,
        "steps": steps,
        "legal_basis": ["Điều 65 BLHS 2015", "Nghị quyết 02/2018/NQ-HĐTP (sửa đổi bởi NQ 01/2022)"],
    }
