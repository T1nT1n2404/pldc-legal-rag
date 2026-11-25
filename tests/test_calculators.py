from __future__ import annotations

import pytest

from tools.inheritance_calc import calculate_inheritance
from tools.criminal_calc import (
    calculate_multi_crime_sentence,
    calculate_multi_judgment_sentence,
    calculate_suspended_sentence,
)
from tools.labor_calc import (
    calculate_overtime_pay,
    calculate_probation_rules,
    calculate_severance_allowance,
)


# ============================================================
# 1. INHERITANCE TESTS (Verified directly against PLDC Slides)
# ============================================================

def test_slide_219_ba_thuy_ong_tai():
    """
    Slide 219 (261_Chương 2_Dân sự.pptx):
    Vợ chồng bà Thủy & ông Tài tài sản chung 18 tỷ.
    Ông Tài có con riêng là Nhân (không phải con bà Thủy).
    Bà Thủy di chúc cho chị gái Loan toàn bộ di sản, truất quyền ông Tài.
    Kết quả: Di sản = 9 tỷ; Ông Tài (chồng, Điều 644) = 2/3 * 9 = 6 tỷ; Loan = 3 tỷ.
    """
    res = calculate_inheritance(
        marital_joint_assets=18.0,
        has_valid_will=True,
        will_beneficiaries=[{"name": "Bà Loan", "share_fraction": 1.0}],
        statutory_heirs=[
            {"name": "Ông Tài", "relation": "chồng", "disinherited_by_will": True}
        ],
        currency_unit="tỷ đồng",
    )
    assert res["net_estate"] == pytest.approx(9.0)
    assert res["shares"]["Ông Tài"]["total"] == pytest.approx(6.0)
    assert res["shares"]["Bà Loan"]["total"] == pytest.approx(3.0)


def test_slide_220_221_ong_a_240_trieu():
    """
    Slide 220-221 (261_Chương 2_Dân sự.pptx):
    Di sản ông A = 240 triệu. Di chúc cho bạn M hưởng toàn bộ.
    Vợ B và 3 con C, D, E đều chưa đủ 18 tuổi (4 người diện Điều 644).
    1 suất PL = 60tr -> 2/3 suất = 40tr.
    Kết quả: B=C=D=E = 40tr; M = 240 - 160 = 80tr.
    """
    res = calculate_inheritance(
        personal_assets=240.0,
        has_valid_will=True,
        will_beneficiaries=[{"name": "Bà M", "share_fraction": 1.0}],
        statutory_heirs=[
            {"name": "Bà B", "relation": "vợ"},
            {"name": "C", "relation": "con chưa thành niên"},
            {"name": "D", "relation": "con chưa thành niên"},
            {"name": "E", "relation": "con chưa thành niên"},
        ],
        currency_unit="triệu đồng",
    )
    assert res["one_statutory_share"] == pytest.approx(60.0)
    assert res["mandatory_min_644"] == pytest.approx(40.0)
    assert res["shares"]["Bà M"]["total"] == pytest.approx(80.0)
    for name in ["Bà B", "C", "D", "E"]:
        assert res["shares"][name]["total"] == pytest.approx(40.0)


def test_slide_229_mixed_will_and_intestate():
    """
    Slide 229: Di sản 720tr. Di chúc 1/3 (=240tr) chia đều cho B, C, X (mỗi người 80tr).
    2/3 còn lại (=480tr) chia theo pháp luật cho hàng 1 (B, C, D, E -> mỗi người 120tr).
    """
    res = calculate_inheritance(
        personal_assets=720.0,
        has_valid_will=True,
        will_beneficiaries=[
            {"name": "B", "share_amount": 80.0},
            {"name": "C", "share_amount": 80.0},
            {"name": "X", "share_amount": 80.0},
        ],
        statutory_heirs=[
            {"name": "B", "relation": "vợ"},
            {"name": "C", "relation": "con thành niên"},
            {"name": "D", "relation": "con thành niên"},
            {"name": "E", "relation": "con thành niên"},
        ],
    )
    assert res["shares"]["B"]["total"] == pytest.approx(200.0)
    assert res["shares"]["C"]["total"] == pytest.approx(200.0)
    assert res["shares"]["X"]["total"] == pytest.approx(80.0)
    assert res["shares"]["D"]["total"] == pytest.approx(120.0)
    assert res["shares"]["E"]["total"] == pytest.approx(120.0)


def test_inheritance_substitution_652():
    """Thừa kế thế vị (Điều 652): Con C chết trước, có 2 cháu C1, C2 thế vị."""
    res = calculate_inheritance(
        personal_assets=600.0,
        has_valid_will=False,
        statutory_heirs=[
            {"name": "Vợ B", "relation": "vợ"},
            {"name": "Con C", "relation": "con", "predeceased_substitutes": ["Cháu C1", "Cháu C2"]},
            {"name": "Con D", "relation": "con"},
        ],
    )
    assert res["shares"]["Vợ B"]["total"] == pytest.approx(200.0)
    assert res["shares"]["Con D"]["total"] == pytest.approx(200.0)
    assert res["shares"]["Cháu C1"]["total"] == pytest.approx(100.0)
    assert res["shares"]["Cháu C2"]["total"] == pytest.approx(100.0)


# ============================================================
# 2. CRIMINAL SENTENCING TESTS (BLHS 2015 Art. 55, 56, 65)
# ============================================================

def test_art55_prison_cap_and_ctkgg_conversion():
    # 18 years prison + 15 years prison -> capped at 30 years (360 months)
    res = calculate_multi_crime_sentence(
        penalties=[
            {"crime_name": "Tội A", "penalty_type": "tu_co_thoi_han", "years": 18},
            {"crime_name": "Tội B", "penalty_type": "tu_co_thoi_han", "years": 15},
        ]
    )
    assert res["remaining_months"] == pytest.approx(360.0)

    # 2 years prison + 3 years CTKGG (= 1 year prison) -> 3 years (36 months)
    res2 = calculate_multi_crime_sentence(
        penalties=[
            {"crime_name": "Tội A", "penalty_type": "tu_co_thoi_han", "years": 2},
            {"crime_name": "Tội B", "penalty_type": "cai_tao_khong_giam_giu", "years": 3},
        ],
        detention_months=6,
    )
    assert res2["combined_months_before_detention"] == pytest.approx(36.0)
    assert res2["remaining_months"] == pytest.approx(30.0)


def test_art56_k1_vs_k2_cap_difference():
    """
    Classic exam trap! BA1 = 20 năm, đã chấp hành 5 năm. BA2 = 15 năm.
    - Khoản 1 (Tội cũ, C: + -): min(20 + 15, 30) - 5 = 30 - 5 = 25 năm (300 tháng).
    - Khoản 2 (Tội mới, M: - +): min((20 - 5) + 15, 30) = min(30, 30) = 30 năm (360 tháng).
    """
    r1 = calculate_multi_judgment_sentence(
        case_type="khoan_1_toi_cu",
        ba1_type="tu_co_thoi_han",
        ba1_months=240,
        ba1_served_months=60,
        ba2_type="tu_co_thoi_han",
        ba2_months=180,
    )
    assert r1["remaining_months"] == pytest.approx(300.0)

    r2 = calculate_multi_judgment_sentence(
        case_type="khoan_2_toi_moi",
        ba1_type="tu_co_thoi_han",
        ba1_months=240,
        ba1_served_months=60,
        ba2_type="tu_co_thoi_han",
        ba2_months=180,
    )
    assert r2["remaining_months"] == pytest.approx(360.0)


def test_slide_125_suspended_sentence():
    # 3 năm tù treo -> 3*2 = 6 năm -> trần 5 năm (60 tháng)
    res = calculate_suspended_sentence(prison_months=36)
    assert res["eligible"] is True
    assert res["probation_months"] == pytest.approx(60.0)

    # 4 tháng tù treo -> 4*2 = 8 tháng -> sàn 1 năm (12 tháng)
    res_min = calculate_suspended_sentence(prison_months=4)
    assert res_min["eligible"] is True
    assert res_min["probation_months"] == pytest.approx(12.0)


# ============================================================
# 3. LABOR LAW TESTS (BLLĐ 2019)
# ============================================================

def test_labor_calculators():
    ot = calculate_overtime_pay(hourly_wage=100_000, hours=4, day_type="ngay_thuong", is_night_shift=True, worked_daytime_before=False)
    # 150% + 30% + 20%*100% = 200%
    assert ot["effective_multiplier_percent"] == pytest.approx(200.0)
    assert ot["total_pay"] == pytest.approx(800_000.0)

    sev = calculate_severance_allowance(
        allowance_type="thoi_viec",
        avg_salary_6_months=10_000_000,
        total_working_months=40,
        unemployment_insurance_months=10,
    )
    # 30 months = 2 years 6 months -> rounded to 2.5 years -> 2.5 * 0.5 * 10M = 12.5M
    assert sev["rounded_years"] == pytest.approx(2.5)
    assert sev["total_allowance"] == pytest.approx(12_500_000.0)

    prob = calculate_probation_rules(official_salary=20_000_000, role_level="cao_dang_tro_len")
    assert prob["max_probation_days"] == 60
    assert prob["min_probation_salary"] == pytest.approx(17_000_000.0)
