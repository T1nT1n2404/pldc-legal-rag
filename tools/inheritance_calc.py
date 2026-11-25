from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class WillBeneficiary(BaseModel):
    name: str
    share_amount: Optional[float] = Field(
        default=None, description="Số tiền/giá trị cụ thể được hưởng theo di chúc (cùng đơn vị tiền tệ)"
    )
    share_fraction: Optional[float] = Field(
        default=None, description="Tỷ lệ hưởng trên toàn bộ di sản ròng, ví dụ 1.0 (toàn bộ), 0.5 (1/2), 0.3333..."
    )
    is_alive_at_opening: bool = Field(
        default=True, description="Còn sống tại thời điểm mở thừa kế (False -> phần di chúc này chia theo pháp luật)"
    )
    refused_or_disqualified: bool = Field(
        default=False, description="Từ chối nhận di sản hoặc không có quyền hưởng theo Điều 621 BLDS"
    )


class StatutoryHeir(BaseModel):
    name: str
    relation: str = Field(
        default="con",
        description="Quan hệ: vợ, chồng, cha, mẹ, con chưa thành niên, con thành niên mất khả năng lao động, con thành niên, ...",
    )
    is_forced_heir_644: Optional[bool] = Field(
        default=None,
        description="Thuộc diện người thừa kế không phụ thuộc nội dung di chúc (Điều 644 BLDS). Nếu None sẽ tự suy luận từ relation.",
    )
    disinherited_by_will: bool = Field(
        default=False,
        description="Bị người lập di chúc truất quyền hưởng di sản (không nhận phần chia theo pháp luật nếu di chúc hợp pháp, nhưng vẫn nhận Điều 644 nếu thuộc diện).",
    )
    refused_or_disqualified_621: bool = Field(
        default=False,
        description="Từ chối nhận di sản hoặc vi phạm Điều 621 BLDS (không được hưởng cả Điều 644).",
    )
    predeceased_substitutes: List[str] = Field(
        default_factory=list,
        description="Danh sách cháu/chắt hưởng thừa kế thế vị (Điều 652 BLDS) nếu người con này chết trước/cùng thời điểm.",
    )


def _infer_forced_heir_644(relation: str) -> bool:
    rel = relation.lower().strip()
    keywords = [
        "vợ",
        "chồng",
        "cha",
        "mẹ",
        "bố",
        "chưa thành niên",
        "dưới 18",
        "chưa đủ 18",
        "mất khả năng lao động",
        "không có khả năng lao động",
        "khuyết tật nặng",
    ]
    return any(k in rel for k in keywords)


def calculate_inheritance(
    marital_joint_assets: float = 0.0,
    personal_assets: float = 0.0,
    other_joint_assets: float = 0.0,
    other_joint_ownership_ratio: float = 0.5,
    debts_and_obligations: float = 0.0,
    funeral_cost: float = 0.0,
    has_valid_will: bool = False,
    will_beneficiaries: Optional[List[dict]] = None,
    statutory_heirs: Optional[List[dict]] = None,
    currency_unit: str = "triệu đồng",
) -> dict:
    """
    Tính chia thừa kế theo Bộ luật Dân sự 2015 (Điều 609, 612, 621, 644, 650, 651, 652, 658).
    """
    steps: List[str] = []
    will_list = [WillBeneficiary(**w) for w in (will_beneficiaries or [])]
    heir_list = [StatutoryHeir(**h) for h in (statutory_heirs or [])]

    # Step 1: Determine net estate (Xác định di sản thừa kế)
    marital_share = marital_joint_assets / 2.0
    other_share = other_joint_assets * other_joint_ownership_ratio
    gross_estate = personal_assets + marital_share + other_share
    deductions = debts_and_obligations + funeral_cost
    net_estate = max(0.0, gross_estate - deductions)

    steps.append(
        f"**Bước 1 — Xác định di sản ròng (Điều 612, 658 BLDS; Điều 33 Luật HNGĐ):**\n"
        f"- Tài sản riêng của người chết: `{personal_assets:,.2f}` {currency_unit}\n"
        f"- Phần từ tài sản chung vợ chồng ({marital_joint_assets:,.2f} / 2): `{marital_share:,.2f}` {currency_unit}\n"
        + (
            f"- Phần từ tài sản đồng sở hữu khác ({other_joint_assets:,.2f} × {other_joint_ownership_ratio:.2f}): `{other_share:,.2f}` {currency_unit}\n"
            if other_joint_assets > 0
            else ""
        )
        + f"- Trừ nghĩa vụ tài sản & mai táng phí: `{deductions:,.2f}` {currency_unit}\n"
        f"→ **Tổng di sản ròng đem chia = `{net_estate:,.2f}` {currency_unit}**."
    )

    if net_estate <= 0:
        return {
            "currency_unit": currency_unit,
            "net_estate": 0.0,
            "one_statutory_share": 0.0,
            "mandatory_min_644": 0.0,
            "shares": {},
            "steps": steps + ["Di sản ròng bằng 0 nên không có di sản để phân chia."],
            "legal_basis": ["Điều 612 BLDS 2015", "Điều 658 BLDS 2015"],
        }

    # Track allocations per beneficiary
    from_will: Dict[str, float] = {}
    from_intestate: Dict[str, float] = {}
    from_644_adjustment: Dict[str, float] = {}

    # Step 2: Divide by valid will (Chia theo di chúc)
    valid_will_total = 0.0
    if has_valid_will and will_list:
        raw_will_alloc: Dict[str, float] = {}
        invalid_will_notes: List[str] = []
        for wb in will_list:
            amt = 0.0
            if wb.share_amount is not None:
                amt = max(0.0, wb.share_amount)
            elif wb.share_fraction is not None:
                amt = max(0.0, wb.share_fraction * net_estate)

            if not wb.is_alive_at_opening or wb.refused_or_disqualified:
                reason = (
                    "chết trước/cùng thời điểm mở thừa kế"
                    if not wb.is_alive_at_opening
                    else "từ chối nhận hoặc không có quyền hưởng (Điều 621)"
                )
                invalid_will_notes.append(
                    f"  - {wb.name} ({amt:,.2f} {currency_unit}): {reason} → phần này chuyển chia theo pháp luật."
                )
                continue
            raw_will_alloc[wb.name] = raw_will_alloc.get(wb.name, 0.0) + amt

        total_requested = sum(raw_will_alloc.values())
        scale = 1.0
        if total_requested > net_estate and total_requested > 0:
            scale = net_estate / total_requested

        for name, amt in raw_will_alloc.items():
            actual = amt * scale
            from_will[name] = from_will.get(name, 0.0) + actual
            valid_will_total += actual

        will_lines = [
            f"  - {n}: `{v:,.2f}` {currency_unit}" for n, v in from_will.items()
        ]
        if invalid_will_notes:
            will_lines.extend(invalid_will_notes)
        steps.append(
            f"**Bước 2 — Chia theo di chúc hợp pháp:**\n"
            + ("\n".join(will_lines) if will_lines else "  - Không có phần di chúc nào có hiệu lực.")
            + f"\n→ Tổng di sản chia theo di chúc = `{valid_will_total:,.2f}` {currency_unit}."
        )
    else:
        steps.append(
            "**Bước 2 — Chia theo di chúc:** Không có di chúc hợp pháp → toàn bộ di sản chia theo pháp luật."
        )

    # Step 3: Divide remaining pool by law (Chia theo pháp luật - Điều 650, 651, 652)
    intestate_pool = max(0.0, net_estate - valid_will_total)

    # Eligible in denominator of "1 suất thừa kế theo pháp luật" (for Art. 644):
    # All statutory heirs in the active rank EXCEPT those refused or disqualified under Art. 621.
    # (Disinherited heirs ARE counted when computing the hypothetical statutory share!)
    hypothetical_heirs = [
        h for h in heir_list if not h.refused_or_disqualified_621
    ]
    n_hypothetical = len(hypothetical_heirs)
    one_statutory_share = (
        net_estate / n_hypothetical if n_hypothetical > 0 else 0.0
    )
    mandatory_min_644 = (2.0 / 3.0) * one_statutory_share

    # Actual recipients of intestate_pool exclude disinherited heirs (if valid will exists)
    actual_intestate_heirs = [
        h
        for h in hypothetical_heirs
        if not (has_valid_will and h.disinherited_by_will)
    ]

    if intestate_pool > 1e-9 and actual_intestate_heirs:
        per_intestate = intestate_pool / len(actual_intestate_heirs)
        int_lines = []
        for h in actual_intestate_heirs:
            if h.predeceased_substitutes:
                sub_share = per_intestate / len(h.predeceased_substitutes)
                for sub_name in h.predeceased_substitutes:
                    from_intestate[sub_name] = (
                        from_intestate.get(sub_name, 0.0) + sub_share
                    )
                int_lines.append(
                    f"  - Nhóm thế vị của {h.name} ({', '.join(h.predeceased_substitutes)} — Điều 652): "
                    f"hưởng chung `{per_intestate:,.2f}` → mỗi người `{sub_share:,.2f}` {currency_unit}"
                )
            else:
                from_intestate[h.name] = from_intestate.get(h.name, 0.0) + per_intestate
                int_lines.append(
                    f"  - {h.name} ({h.relation}): `{per_intestate:,.2f}` {currency_unit}"
                )
        steps.append(
            f"**Bước 3 — Chia phần di sản theo pháp luật (Điều 650, 651, 652 BLDS):**\n"
            f"- Di sản chia theo pháp luật: `{intestate_pool:,.2f}` {currency_unit} cho {len(actual_intestate_heirs)} suất.\n"
            + "\n".join(int_lines)
        )
    else:
        steps.append(
            "**Bước 3 — Chia theo pháp luật:** Không còn phần di sản dư chưa định đoạt trong di chúc."
        )

    # Step 4: Forced heirs protection (Điều 644 BLDS - tối thiểu 2/3 một suất theo pháp luật)
    forced_heirs: List[StatutoryHeir] = []
    if has_valid_will and n_hypothetical > 0:
        for h in hypothetical_heirs:
            # Predeceased heirs (substitutes) do not personally get Art. 644
            if h.predeceased_substitutes:
                continue
            is_644 = (
                h.is_forced_heir_644
                if h.is_forced_heir_644 is not None
                else _infer_forced_heir_644(h.relation)
            )
            if is_644:
                forced_heirs.append(h)

    if forced_heirs:
        deficits: Dict[str, float] = {}
        for h in forced_heirs:
            curr = from_will.get(h.name, 0.0) + from_intestate.get(h.name, 0.0)
            if curr + 1e-6 < mandatory_min_644:
                deficits[h.name] = mandatory_min_644 - curr

        if deficits:
            total_deficit = sum(deficits.values())
            # Deduct proportionally from surplus holders (prefer will beneficiaries first, then any surplus > min)
            forced_names = {h.name for h in forced_heirs}
            all_names = sorted(set(from_will.keys()) | set(from_intestate.keys()))

            # Compute available surplus for each recipient
            surpluses: Dict[str, float] = {}
            for name in all_names:
                curr_total = from_will.get(name, 0.0) + from_intestate.get(name, 0.0)
                floor = mandatory_min_644 if name in forced_names else 0.0
                avail = max(0.0, curr_total - floor)
                if avail > 1e-9:
                    surpluses[name] = avail

            total_surplus = sum(surpluses.values())
            deduct_lines = []
            if total_surplus > 1e-9:
                ratio = min(1.0, total_deficit / total_surplus)
                for name, avail in surpluses.items():
                    cut = avail * ratio
                    from_644_adjustment[name] = from_644_adjustment.get(name, 0.0) - cut
                    deduct_lines.append(
                        f"  - Trích giảm từ {name}: `-{cut:,.2f}` {currency_unit}"
                    )

            add_lines = []
            for name, d_amt in deficits.items():
                from_644_adjustment[name] = from_644_adjustment.get(name, 0.0) + d_amt
                add_lines.append(
                    f"  - Bù cho {name} đủ 2/3 suất: `+{d_amt:,.2f}` {currency_unit}"
                )

            steps.append(
                f"**Bước 4 — Bảo vệ người thừa kế không phụ thuộc nội dung di chúc (Điều 644 BLDS):**\n"
                f"- Giả sử chia toàn bộ di sản theo pháp luật cho {n_hypothetical} người → 1 suất PL = `{net_estate:,.2f} / {n_hypothetical}` = `{one_statutory_share:,.2f}` {currency_unit}.\n"
                f"- Mức tối thiểu bắt buộc (2/3 suất PL) = `{mandatory_min_644:,.2f}` {currency_unit}.\n"
                f"- Tổng số tiền cần trích bù = `{total_deficit:,.2f}` {currency_unit}:\n"
                + "\n".join(add_lines + deduct_lines)
            )
        else:
            steps.append(
                f"**Bước 4 — Kiểm tra Điều 644 BLDS:** 1 suất PL = `{one_statutory_share:,.2f}`, "
                f"mức 2/3 suất = `{mandatory_min_644:,.2f}` {currency_unit}. "
                f"Tất cả người thừa kế diện Điều 644 đều đã nhận đủ mức tối thiểu → không cần trích bù."
            )

    # Aggregate final shares
    all_recipients = sorted(
        set(from_will.keys())
        | set(from_intestate.keys())
        | set(from_644_adjustment.keys())
    )
    shares: Dict[str, dict] = {}
    for name in all_recipients:
        w_v = round(from_will.get(name, 0.0), 4)
        i_v = round(from_intestate.get(name, 0.0), 4)
        adj_v = round(from_644_adjustment.get(name, 0.0), 4)
        total_v = round(max(0.0, w_v + i_v + adj_v), 4)
        shares[name] = {
            "from_will": round(w_v, 2),
            "from_intestate": round(i_v, 2),
            "adjustment_644": round(adj_v, 2),
            "total": round(total_v, 2),
        }

    legal_basis = ["Điều 609, 612, 658 BLDS 2015"]
    if marital_joint_assets > 0:
        legal_basis.append("Điều 33, 66 Luật Hôn nhân và Gia đình 2014")
    if has_valid_will:
        legal_basis.append("Điều 624, 644 BLDS 2015 (Thừa kế theo di chúc & Điều 644)")
    if intestate_pool > 0 or not has_valid_will:
        legal_basis.append("Điều 650, 651, 652 BLDS 2015 (Thừa kế theo pháp luật & thế vị)")

    return {
        "currency_unit": currency_unit,
        "net_estate": round(net_estate, 2),
        "one_statutory_share": round(one_statutory_share, 2),
        "mandatory_min_644": round(mandatory_min_644, 2),
        "shares": shares,
        "total_distributed": round(sum(s["total"] for s in shares.values()), 2),
        "steps": steps,
        "legal_basis": legal_basis,
    }
