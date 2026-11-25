from __future__ import annotations

import argparse
from retrieval.hybrid_retriever import HybridRetriever

# Golden benchmark queries mapped to expected keywords / slide sources in PLDC
GOLDEN_SET = [
    {
        "query": "Điều 644 người thừa kế không phụ thuộc vào nội dung di chúc 2/3 suất",
        "expected_file_substr": "Chương 2",
        "expected_terms": ["644", "không phụ thuộc"],
    },
    {
        "query": "tài sản chung vợ chồng chia đôi xác định di sản người quá cố",
        "expected_file_substr": "Chương 2",
        "expected_terms": ["tài sản chung", "di sản"],
    },
    {
        "query": "tổng hợp hình phạt của nhiều bản án Khoản 1 Khoản 2 Điều 56 BLHS",
        "expected_file_substr": "hình sự",
        "expected_terms": ["56", "bản án"],
    },
    {
        "query": "3 ngày cải tạo không giam giữ bằng 1 ngày tù tối đa 30 năm Điều 55",
        "expected_file_substr": "hình sự",
        "expected_terms": ["55", "30 năm"],
    },
    {
        "query": "ấn định thời gian thử thách án treo gấp hai lần mức phạt tù không quá 5 năm Điều 65",
        "expected_file_substr": "hình sự",
        "expected_terms": ["án treo", "thử thách"],
    },
    {
        "query": "tiền lương thử việc ít nhất bằng 85% mức lương Điều 26",
        "expected_file_substr": "Lao động",
        "expected_terms": ["85%", "thử việc"],
    },
    {
        "query": "đơn phương chấm dứt hợp đồng lao động trái pháp luật Điều 40",
        "expected_file_substr": "Lao động",
        "expected_terms": ["chấm dứt", "hợp đồng"],
    },
    {
        "query": "hàng thừa kế thứ nhất vợ chồng cha đẻ mẹ đẻ con đẻ con nuôi",
        "expected_file_substr": "Dân sự",
        "expected_terms": ["hàng thừa kế thứ nhất"],
    },
]


def evaluate(mode: str = "hybrid", k: int = 5):
    retriever = HybridRetriever()
    hits = 0
    mrr_sum = 0.0
    latencies = []

    print(f"=== Retrieval Evaluation (mode={mode}, top_k={k}, chunks={len(retriever.chunks)}) ===")
    for idx, case in enumerate(GOLDEN_SET, start=1):
        res = retriever.search(query=case["query"], top_k=k, mode=mode)
        latencies.append(res["latency_ms"])
        first_hit_rank = None
        for item in res["results"]:
            txt = item["text"].lower()
            file_ok = case["expected_file_substr"].lower() in item["source_file"].lower()
            terms_ok = all(t.lower() in txt for t in case["expected_terms"])
            if file_ok and terms_ok:
                first_hit_rank = item["rank"]
                break

        if first_hit_rank is not None:
            hits += 1
            mrr_sum += 1.0 / first_hit_rank
            status = f"HIT@{first_hit_rank}"
        else:
            status = "MISS"
        print(f"[{idx:02d}] {status:8s} ({res['latency_ms']:5.2f} ms) | {case['query'][:65]}")

    n = len(GOLDEN_SET)
    recall_at_k = hits / n
    mrr = mrr_sum / n
    avg_lat = sum(latencies) / n
    print(f"\nSummary -> Recall@{k}: {recall_at_k:.1%} ({hits}/{n}) | MRR@{k}: {mrr:.3f} | Avg Latency: {avg_lat:.2f} ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="bm25", choices=["hybrid", "bm25", "dense"])
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    evaluate(mode=args.mode, k=args.k)
