from __future__ import annotations

import math
from typing import Literal


def calculate_overtime_pay(
    hourly_wage: float,
    hours: float,
    day_type: Literal["ngay_thuong", "ngay_nghi_hang_tuan", "ngay_le_tet"] = "ngay_thuong",
    is_night_shift: bool = False,
    worked_daytime_before: bool = False,
    currency_unit: str = "VNĐ",
) -> dict:
    """
    Tính tiền lương làm thêm giờ và làm việc ban đêm (Điều 98 BLLĐ 2019 & Điều 55-57 NĐ 145/2020/NĐ-CP).
    """
    base_rate_map = {
        "ngay_thuong": (1.5, "150% (ngày thường)"),
        "ngay_nghi_hang_tuan": (2.0, "200% (ngày nghỉ hằng tuần)"),
        "ngay_le_tet": (3.0, "300% (ngày nghỉ lễ, tết, ngày nghỉ có hưởng lương)"),
    }
    ot_mult, day_label = base_rate_map[day_type]
    steps = [f"- Đơn giá lương giờ thực trả: `{hourly_wage:,.0f}` {currency_unit}/giờ, số giờ làm thêm: `{hours}` giờ."]

    if not is_night_shift:
        effective_mult = ot_mult
        total = hourly_wage * effective_mult * hours
        steps.append(
            f"- Làm thêm ban ngày ({day_label}): `{hourly_wage:,.0f} × {ot_mult*100:.0f}% × {hours}` = **`{total:,.0f}` {currency_unit}**."
        )
    else:
        # Điều 57 Nghị định 145/2020/NĐ-CP:
        # = Tiền lương giờ làm thêm (150%/200%/300%) + 30% lương giờ ngày thường + 20% * (đơn giá ban ngày của ngày đó)
        if day_type == "ngay_thuong":
            daytime_ref_mult = 1.5 if worked_daytime_before else 1.0
        elif day_type == "ngay_nghi_hang_tuan":
            daytime_ref_mult = 2.0
        else:
            daytime_ref_mult = 3.0

        night_addon = 0.30
        night_ot_addon = 0.20 * daytime_ref_mult
        effective_mult = ot_mult + night_addon + night_ot_addon
        total = hourly_wage * effective_mult * hours
        steps.append(
            f"- Làm thêm giờ vào **ban đêm** ({day_label}, Điều 98 khoản 3 BLLĐ 2019 & Điều 57 NĐ 145/2020):\n"
            f"  - Mức làm thêm giờ: `{ot_mult*100:.0f}%`\n"
            f"  - Cộng làm việc ban đêm: `+30%`\n"
            f"  - Cộng 20% tiền lương ban ngày của ngày tương ứng (`20% × {daytime_ref_mult*100:.0f}%`): `+{night_ot_addon*100:.0f}%`\n"
            f"  - **Tổng hệ số = `{effective_mult*100:.0f}%`** → `{hourly_wage:,.0f} × {effective_mult:.2f} × {hours}` = **`{total:,.0f}` {currency_unit}**."
        )

    return {
        "effective_multiplier_percent": round(effective_mult * 100, 1),
        "total_pay": round(total, 2),
        "currency_unit": currency_unit,
        "steps": steps,
        "legal_basis": ["Điều 98 Bộ luật Lao động 2019", "Điều 55-57 Nghị định 145/2020/NĐ-CP"],
    }


def calculate_severance_allowance(
    allowance_type: Literal["thoi_viec", "mat_viec_lam"],
    avg_salary_6_months: float,
    total_working_months: float,
    unemployment_insurance_months: float = 0.0,
    currency_unit: str = "VNĐ",
) -> dict:
    """
    Tính trợ cấp thôi việc (Điều 46 BLLĐ 2019) và trợ cấp mất việc làm (Điều 47 BLLĐ 2019).
    """
    eligible_months = max(0.0, total_working_months - unemployment_insurance_months)
    # Làm tròn theo Điều 8 NĐ 145/2020: lẻ 1-6 tháng = 0.5 năm; > 6 tháng = 1 năm
    full_years = int(eligible_months // 12)
    rem_months = eligible_months % 12
    if rem_months == 0:
        rounded_years = float(full_years)
    elif rem_months <= 6:
        rounded_years = full_years + 0.5
    else:
        rounded_years = float(full_years + 1)

    if total_working_months < 12:
        return {
            "eligible": False,
            "rounded_years": 0.0,
            "total_allowance": 0.0,
            "currency_unit": currency_unit,
            "steps": [
                "Người lao động làm việc thường xuyên chưa đủ **12 tháng** → không đủ điều kiện hưởng trợ cấp thôi việc / mất việc làm."
            ],
            "legal_basis": ["Điều 46, 47 Bộ luật Lao động 2019"],
        }

    if allowance_type == "thoi_viec":
        total = rounded_years * 0.5 * avg_salary_6_months
        basis = "Điều 46 Bộ luật Lao động 2019 (0,5 tháng lương / năm)"
    else:
        raw_total = rounded_years * 1.0 * avg_salary_6_months
        min_total = 2.0 * avg_salary_6_months
        total = max(raw_total, min_total)
        basis = "Điều 47 Bộ luật Lao động 2019 (1 tháng lương / năm, tối thiểu 2 tháng lương)"

    steps = [
        f"- Tổng thời gian làm việc: `{total_working_months:.1f}` tháng; trừ thời gian đóng BHTN (`{unemployment_insurance_months:.1f}` tháng) = `{eligible_months:.1f}` tháng.",
        f"- Quy đổi số năm hưởng trợ cấp (lẻ 1–6 tháng → 0,5 năm; >6 tháng → 1 năm): **`{rounded_years}` năm**.",
        f"- Áp dụng {basis} với bình quân 6 tháng lương liền kề `{avg_salary_6_months:,.0f}` {currency_unit}:",
        f"→ **Tiền trợ cấp = `{total:,.0f}` {currency_unit}**.",
    ]

    return {
        "eligible": True,
        "allowance_type": allowance_type,
        "eligible_months": round(eligible_months, 1),
        "rounded_years": rounded_years,
        "total_allowance": round(total, 2),
        "currency_unit": currency_unit,
        "steps": steps,
        "legal_basis": ["Điều 46, 47 Bộ luật Lao động 2019", "Điều 8 Nghị định 145/2020/NĐ-CP"],
    }


def calculate_probation_rules(
    official_salary: float,
    role_level: Literal["quan_ly_dn", "cao_dang_tro_len", "trung_cap_ktv", "khac", "hd_duoi_1_thang"] = "cao_dang_tro_len",
    currency_unit: str = "VNĐ",
) -> dict:
    """
    Tính tiền lương thử việc tối thiểu và thời gian thử việc tối đa (Điều 25, 26 BLLĐ 2019).
    """
    max_days_map = {
        "quan_ly_dn": (180, "Quản lý doanh nghiệp (Luật Doanh nghiệp)"),
        "cao_dang_tro_len": (60, "Chức danh cần trình độ chuyên môn từ Cao đẳng trở lên"),
        "trung_cap_ktv": (30, "Chức danh cần trình độ Trung cấp, công nhân kỹ thuật, nhân viên nghiệp vụ"),
        "khac": (6, "Công việc khác"),
        "hd_duoi_1_thang": (0, "Hợp đồng lao động dưới 01 tháng (không áp dụng thử việc)"),
    }
    max_days, label = max_days_map[role_level]
    min_salary = official_salary * 0.85

    return {
        "role_description": label,
        "max_probation_days": max_days,
        "min_probation_salary": round(min_salary, 2),
        "currency_unit": currency_unit,
        "steps": [
            f"- Nhóm công việc: **{label}** → Thời gian thử việc tối đa: **`{max_days}` ngày** (Điều 25 BLLĐ 2019).",
            f"- Tiền lương thử việc tối thiểu (85% lương chính thức `{official_salary:,.0f}`): **`{min_salary:,.0f}` {currency_unit}** (Điều 26 BLLĐ 2019).",
        ],
        "legal_basis": ["Điều 24, 25, 26 Bộ luật Lao động 2019"],
    }
