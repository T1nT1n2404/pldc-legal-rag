from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app import app

client = TestClient(app)


def test_stats_and_frontend():
    r_home = client.get("/")
    assert r_home.status_code == 200
    assert "PLĐC Legal RAG" in r_home.text

    r_stats = client.get("/api/stats")
    assert r_stats.status_code == 200
    data = r_stats.json()
    assert data["index_ready"] is True
    assert data["manifest"]["slide_chunks"] > 500


def test_hybrid_search_api():
    r = client.post(
        "/api/search",
        json={"query": "Điều 644 người thừa kế không phụ thuộc nội dung di chúc", "top_k": 5, "mode": "bm25"},
    )
    assert r.status_code == 200
    data = r.json()
    assert len(data["results"]) == 5
    combined_text = " ".join(item["text"] for item in data["results"]).lower()
    assert "644" in combined_text or "di chúc" in combined_text


def test_calculator_endpoints():
    r1 = client.post(
        "/api/calc/inheritance",
        json={
            "marital_joint_assets": 18.0,
            "has_valid_will": True,
            "will_beneficiaries": [{"name": "Bà Loan", "share_fraction": 1.0}],
            "statutory_heirs": [{"name": "Ông Tài", "relation": "chồng", "disinherited_by_will": True}],
            "currency_unit": "tỷ đồng",
        },
    )
    assert r1.status_code == 200
    assert r1.json()["shares"]["Bà Loan"]["total"] == 3.0

    r2 = client.post(
        "/api/calc/criminal/multi-judgment",
        json={
            "case_type": "khoan_1_toi_cu",
            "ba1_type": "tu_co_thoi_han",
            "ba1_months": 240,
            "ba1_served_months": 60,
            "ba2_type": "tu_co_thoi_han",
            "ba2_months": 180,
        },
    )
    assert r2.status_code == 200
    assert r2.json()["remaining_months"] == 300.0
