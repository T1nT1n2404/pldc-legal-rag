# Hướng dẫn Trả lời Phỏng vấn Hệ thống RAG (Dựa trên chính Dự án này)

Tài liệu này tóm tắt các câu hỏi phỏng vấn phổ biến nhất về **Production RAG / Agentic RAG** kèm câu trả lời gắn trực tiếp với các quyết định thiết kế trong dự án `legal_rag` của bạn.

---

## 1. Tại sao không dùng Naive RAG (chỉ cắt text + vector search)?
Trong miền **Pháp luật Việt Nam**, Naive RAG gặp 4 lỗ hổng lớn — và dự án này giải quyết từng cái:
1. **Dữ liệu đa định dạng & PDF scan khổ đôi (Real-world War Story):** Giáo trình 248 trang là PDF scan 100% ảnh (0 font chữ), và mỗi trang PDF thực chất là **2 trang sách ghép ngang**. Khi thử OCR bằng LLM Vision, gặp 2 vấn đề thực tế: (a) bộ lọc bản quyền `FinishReason.RECITATION` chặn chép nguyên văn giáo trình xuất bản, và (b) đọc nguyên khổ ngang dễ lẫn dòng giữa 2 trang.
   - Giải pháp thực tế (`ingestion/pdf_ocr.py`): Tự động nhận diện khổ ngang (`width > 1.15 * height`), **cắt đôi trang trái/phải + cắt lề gáy sách 1,5%** bằng PyMuPDF, chạy **Tesseract `tessdata_best` tiếng Việt nội bộ song song 6 luồng** (chi phí \$0, không giới hạn rate limit, có checkpoint từng trang), sau đó dùng **Gemini Embedding (`gemini-embedding-001`)** cho tầng truy xuất ngữ nghĩa.
2. **Chiến lược Chunking khác nhau theo cấu trúc nguồn (`ingestion/chunker.py`):**
   - Slide PPTX có cấu trúc tự nhiên theo từng khung ý → **1 Slide = 1 Semantic Chunk** (giữ nguyên tiêu đề + gạch đầu dòng + bảng + ghi chú giảng viên), không cắt đứt giữa chừng.
   - Giáo trình văn xuôi dài → **Recursive Heading/Paragraph Splitter** (850 ký tự, overlap 150 ký tự) + gắn **Contextual Header** `[Nguồn: Giáo trình - Trang X | Chương ...]` vào đầu mỗi chunk (kỹ thuật *Contextual Retrieval* giúp embedding không bị mất ngữ cảnh chương).
3. **Truy vấn điều luật cần khớp chính xác từ khóa:** Dense embedding giỏi hiểu ngữ nghĩa nhưng hay trượt số điều chính xác (ví dụ `Điều 644` vs `Điều 651`, `Điều 55` vs `Điều 56`).
   - Giải pháp: **Hybrid Search = Dense Vector (Cosine) + Sparse BM25** (với tokenizer chuyên biệt tạo token ghép `dieu_644`, `khoan_1` và song song token có dấu + không dấu) rồi hợp nhất bằng **Reciprocal Rank Fusion (RRF, $k=60$)** (`retrieval/hybrid_retriever.py`).
4. **LLM hay sai toán & ảo giác công thức pháp lý:** Chia thừa kế Điều 644 (2/3 suất) hay bẫy trần 30 năm giữa Khoản 1 (`C: + -`) vs Khoản 2 (`M: - +`) Điều 56 BLHS rất dễ bị LLM tính nhẩm sai.
   - Giải pháp: **Agentic RAG + Deterministic Function Calling (`agent/rag_agent.py` + `tools/`)** — tách biệt *suy luận ngữ nghĩa / trích xuất tham số* (LLM) khỏi *tính toán xác định* (Python code có unit test 100%).

---

## 2. Kết quả Benchmark thực đo trên 2.163 chunks & Công thức RRF
$$\text{RRF}(d) = \frac{w_{\text{dense}}}{k + \text{rank}_{\text{dense}}(d)} + \frac{w_{\text{bm25}}}{k + \text{rank}_{\text{bm25}}(d)}$$
- **Số liệu thực đo (`eval/eval_retrieval.py`, 829 slide + 1.334 chunk giáo trình = 2.163 chunk):**
  - **Chỉ Dense Vector** (`paraphrase-multilingual-MiniLM-L12-v2`, 384-dim): `Recall@5 = 100%`, **`MRR@5 = 0.698`**, độ trễ warm ~6–8 ms (nhưng câu hỏi chứa số điều luật như `Điều 644` chỉ đạt `HIT@4`, `Điều 56` đạt `HIT@2`).
  - **Hybrid (Dense + BM25 + RRF, $k=60$)**: `Recall@5 = 100%`, **`MRR@5 = 0.844` (+20,9% MRR so với Dense thuần!)**, độ trễ warm **~16–26 ms** — đưa cả `Điều 644` và `Điều 56` lên **`HIT@1`**!
- **Vì sao dùng RRF thay vì cộng thẳng điểm BM25 + Cosine?**
  - BM25 có miền giá trị không bị chặn (ví dụ 0 đến 25+), trong khi Cosine Similarity nằm trong `[-1, 1]`. Cộng điểm trực tiếp rất nhạy với phân phối điểm (score scaling).
  - RRF chỉ dựa trên **thứ hạng (rank)** nên miễn nhiễm với khác biệt thang đo, ổn định và không cần train model reranker nặng. Hằng số $k=60$ làm mượt tác động của các outlier ở top 1–2.

---

## 3. Làm sao giảm Hallucination (Ảo giác) và tăng Observability?
- **Grounded Citations:** Mọi chunk đều mang nhãn nguồn chuẩn (`Slide X` / `Trang Y`), prompt bắt buộc trích dẫn rõ nguồn.
- **Deterministic Tools + Unit Tests:** Các phép tính pháp lý có `pytest` đối chiếu trực tiếp với đáp án bài tập trong slide (`tests/test_calculators.py`).
- **RAG & Tool Trace Inspector (Observability):** Giao diện Web App hiển thị trực tiếp ngay dưới câu trả lời: latency từng bước, JSON input/output của mỗi tool call, điểm BM25 / Dense / RRF và đoạn text gốc được truy xuất — giúp debug lỗi thuộc về **Retrieval** hay **Generation** ngay lập tức.

---

## 4. Nếu mở rộng từ ~2.000 chunk lên 10 triệu văn bản luật (Scale-up)?
Câu hỏi phỏng vấn kinh điển — hướng trả lời chuẩn hệ thống:
1. **Vector Store:** Thay mảng `numpy` in-memory (exact search, <1ms ở 2k chunks) bằng **Qdrant / pgvector / Milvus** dùng chỉ mục ANN (**HNSW** hoặc **IVF-PQ**) hỗ trợ metadata pre-filtering theo `loại_văn_bản`, `hiệu_lực`, `ngày_ban_hành`.
2. **Sparse Search:** Thay `rank-bm25` in-memory bằng **Elasticsearch / OpenSearch** hoặc **Qdrant Sparse Vectors (SPLADE / BM25)**.
3. **Two-stage Retrieval + Reranking:** Lấy top-50 từ Hybrid RRF → đưa qua **Cross-Encoder Reranker** (`bge-reranker-v2-m3` hoặc Cohere Rerank) chọn top-5..8 chất lượng cao nhất vào LLM.
4. **Evaluation Pipeline (Đánh giá định lượng):**
   - *Retrieval metrics:* **Recall@K**, **MRR (Mean Reciprocal Rank)**, **NDCG@K** trên bộ Golden Dataset (câu hỏi + danh sách slide/trang đúng).
   - *Generation metrics (RAGAS / LLM-as-a-Judge):* **Faithfulness** (câu trả lời có trung thực với context không), **Answer Relevancy**, và **Tool Call Accuracy** (trích xuất đúng tham số cho máy tính pháp lý không).
5. **Caching & Versioning:** Semantic cache cho câu hỏi phổ biến + quản lý phiên bản hiệu lực văn bản luật (luật cũ hết hiệu lực không được lẫn vào kết quả truy vấn luật hiện hành).

---

## 5. Kịch bản Demo 3 phút gây ấn tượng khi phỏng vấn
1. Mở tab **Kiểm tra Hybrid Retrieval**: Gõ `Điều 644 người thừa kế không phụ thuộc di chúc` → cho xem sự khác biệt giữa **BM25** vs **Dense** vs **Hybrid RRF** cùng độ trễ tính bằng mili-giây.
2. Mở tab **Chatbot Pháp luật**: Bấm câu mẫu **"Tổng hợp bản án Điều 56 (K1 vs K2)"** (bẫy 20 năm đã ở 5 năm + 15 năm → K1 ra **25 năm**, K2 ra **30 năm**).
3. Mở **RAG & Tool Trace Inspector** ngay dưới câu trả lời: Chỉ cho người phỏng vấn thấy Agent đã tự gọi cả `search_legal_knowledge` (lấy Slide 115–117) lẫn `tool_calculate_multi_judgment_sentence` 2 lần để so sánh chính xác 100% không bị LLM tính sai!
