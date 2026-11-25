// PLDC Legal Agentic RAG — Frontend Controller

const state = {
  history: [],
  sending: false,
};

// --- Safe lightweight Markdown renderer (works 100% offline) ---
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function renderInline(text) {
  let s = escapeHtml(text);
  // Highlight source citations like [Slide: ...] or [Giáo trình: ...]
  s = s.replace(/\[(Slide:[^\]]+|Giáo trình:[^\]]+)\]/g, '<span class="cite-badge">[$1]</span>');
  s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  return s;
}

function renderMarkdown(md) {
  const lines = String(md || "").split("\n");
  let html = "";
  let inUl = false;
  let inTable = false;
  let tableHeaderDone = false;

  const closeLists = () => {
    if (inUl) { html += "</ul>"; inUl = false; }
  };
  const closeTable = () => {
    if (inTable) { html += "</tbody></table>"; inTable = false; tableHeaderDone = false; }
  };

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i].trim();

    if (!line) {
      closeLists();
      closeTable();
      continue;
    }

    // Markdown table
    if (line.startsWith("|") && line.endsWith("|")) {
      closeLists();
      const cells = line.slice(1, -1).split("|").map(c => c.trim());
      if (cells.every(c => /^[-:\s]+$/.test(c))) continue; // separator row
      if (!inTable) {
        html += "<table><thead><tr>" + cells.map(c => `<th>${renderInline(c)}</th>`).join("") + "</tr></thead><tbody>";
        inTable = true;
        tableHeaderDone = true;
      } else {
        html += "<tr>" + cells.map(c => `<td>${renderInline(c)}</td>`).join("") + "</tr>";
      }
      continue;
    } else {
      closeTable();
    }

    // Headings
    const hMatch = line.match(/^(#{1,4})\s+(.+)$/);
    if (hMatch) {
      closeLists();
      const lvl = Math.min(hMatch[1].length + 1, 4);
      html += `<h${lvl}>${renderInline(hMatch[2])}</h${lvl}>`;
      continue;
    }

    // Bullet list
    const liMatch = line.match(/^[-*•]\s+(.+)$/);
    if (liMatch) {
      if (!inUl) { html += "<ul>"; inUl = true; }
      html += `<li>${renderInline(liMatch[1])}</li>`;
      continue;
    }

    closeLists();
    html += `<p>${renderInline(line)}</p>`;
  }
  closeLists();
  closeTable();
  return html;
}

function getApiHeaders() {
  const headers = { "Content-Type": "application/json" };
  const key = document.getElementById("input-api-key").value.trim();
  if (key) headers["X-Gemini-Api-Key"] = key;
  return headers;
}

// --- Init API key from localStorage ---
const apiKeyInput = document.getElementById("input-api-key");
apiKeyInput.value = localStorage.getItem("pldc_gemini_key") || "";
apiKeyInput.addEventListener("input", () => {
  localStorage.setItem("pldc_gemini_key", apiKeyInput.value.trim());
});

// --- Tabs switching ---
document.querySelectorAll(".tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("tab-" + btn.dataset.tab).classList.add("active");
  });
});

document.querySelectorAll(".subtab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".subtab").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".calc-pane").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById("calc-" + btn.dataset.calc).classList.add("active");
  });
});

// --- Fetch & refresh system stats ---
async function refreshStats() {
  try {
    const res = await fetch("/api/stats");
    const data = await res.json();
    const m = data.manifest || {};
    document.getElementById("stat-slides").textContent = m.slide_chunks ?? 0;
    document.getElementById("stat-textbook").textContent = m.textbook_chunks ?? 0;

    const badge = document.getElementById("status-badge");
    if (data.index_ready) {
      badge.textContent = "Sẵn sàng";
      badge.className = "badge badge-ok";
    } else {
      badge.textContent = "Chưa Index";
      badge.className = "badge badge-warn";
    }

    const pBm25 = document.getElementById("pill-bm25");
    pBm25.textContent = `BM25: ${m.has_bm25 ? "ON" : "OFF"}`;
    pBm25.className = `pill ${m.has_bm25 ? "ok" : "warn"}`;

    const pDense = document.getElementById("pill-dense");
    pDense.textContent = `Dense Vec: ${data.has_dense_vectors ? "ON" : "OFF"}`;
    pDense.className = `pill ${data.has_dense_vectors ? "ok" : "warn"}`;

    const ocr = data.ocr_progress || { done_pages: 0, total_pages: 248 };
    const pct = Math.min(100, Math.round((ocr.done_pages / Math.max(1, ocr.total_pages)) * 100));
    document.getElementById("ocr-label").textContent = `${ocr.done_pages} / ${ocr.total_pages} trang (${pct}%)`;
    document.getElementById("ocr-progress-fill").style.width = `${pct}%`;

    const jobMsg = document.getElementById("job-status-msg");
    if (data.job && data.job.running) {
      jobMsg.textContent = `⏳ ${data.job.message}`;
      document.getElementById("btn-run-ocr").disabled = true;
      setTimeout(refreshStats, 3000);
    } else {
      document.getElementById("btn-run-ocr").disabled = false;
      if (data.job && data.job.error) {
        jobMsg.textContent = `❌ ${data.job.error}`;
      } else if (data.job && data.job.stage === "done") {
        jobMsg.textContent = `✅ ${data.job.message}`;
      }
    }
  } catch (err) {
    console.error("Failed to fetch stats:", err);
  }
}

document.getElementById("btn-run-ocr").addEventListener("click", async () => {
  const msgEl = document.getElementById("job-status-msg");
  msgEl.textContent = "Đang gửi yêu cầu OCR & Build Index...";
  try {
    const res = await fetch("/api/index/rebuild", {
      method: "POST",
      headers: getApiHeaders(),
      body: JSON.stringify({ run_pdf_ocr: true, ocr_batch_size: 8, ocr_workers: 3 }),
    });
    const data = await res.json();
    if (!res.ok) {
      msgEl.textContent = `❌ ${data.detail || "Không thể chạy OCR"}`;
      return;
    }
    refreshStats();
  } catch (err) {
    msgEl.textContent = `❌ Lỗi mạng: ${err.message}`;
  }
});

// --- Chat functionality ---
const chatMessages = document.getElementById("chat-messages");
const chatInput = document.getElementById("chat-input");
const btnSend = document.getElementById("btn-send");

function appendMessage(role, contentHtml, extraNode = null) {
  const wrap = document.createElement("div");
  wrap.className = `msg msg-${role}`;
  wrap.innerHTML = `
    <div class="msg-avatar">${role === "user" ? "👤" : "⚖️"}</div>
    <div class="msg-body">
      <div class="msg-content">${contentHtml}</div>
    </div>
  `;
  if (extraNode) wrap.querySelector(".msg-body").appendChild(extraNode);
  chatMessages.appendChild(wrap);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return wrap;
}

function buildTraceNode(data) {
  const details = document.createElement("details");
  details.className = "trace-details";
  const toolCount = (data.tool_trace || []).length;
  const citeCount = (data.citations || []).length;
  details.innerHTML = `<summary>🔎 RAG &amp; Tool Trace Inspector — ${toolCount} tool call(s), ${citeCount} nguồn truy xuất, ${data.total_latency_ms} ms (${escapeHtml(data.model)})</summary>`;

  (data.tool_trace || []).forEach((t, idx) => {
    const div = document.createElement("div");
    div.className = "trace-block";
    div.innerHTML = `
      <div><strong>Tool #${idx + 1}:</strong> <code>${escapeHtml(t.tool)}</code> (${t.duration_ms} ms)</div>
      <pre>Args: ${escapeHtml(JSON.stringify(t.args, null, 2))}\nOutput: ${escapeHtml(JSON.stringify(t.output_preview, null, 2))}</pre>
    `;
    details.appendChild(div);
  });

  if (citeCount > 0) {
    const cWrap = document.createElement("div");
    cWrap.className = "trace-block";
    cWrap.innerHTML = `<div><strong>Tài liệu đã truy xuất (${citeCount}):</strong></div>`;
    data.citations.forEach(c => {
      const item = document.createElement("div");
      item.style.marginTop = "6px";
      item.innerHTML = `
        <span class="cite-badge">${escapeHtml(c.citation)}</span>
        <span class="pill">Final: ${c.final_score}</span>
        <span class="pill">BM25 #${c.bm25_rank} (${c.bm25_score})</span>
        ${c.dense_score !== null ? `<span class="pill ok">Dense #${c.dense_rank} (${c.dense_score})</span>` : ""}
        <pre>${escapeHtml(c.text.slice(0, 420))}${c.text.length > 420 ? "..." : ""}</pre>
      `;
      cWrap.appendChild(item);
    });
    details.appendChild(cWrap);
  }
  return details;
}

async function sendChatMessage(text) {
  if (!text.trim() || state.sending) return;
  state.sending = true;
  btnSend.disabled = true;

  appendMessage("user", renderMarkdown(text));
  const loadingMsg = appendMessage("assistant", "<p>⏳ Đang tra cứu Hybrid RAG và chạy công cụ tính toán...</p>");

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: getApiHeaders(),
      body: JSON.stringify({
        message: text,
        history: state.history,
        source_filter: document.getElementById("select-source-filter").value,
        retrieval_mode: document.getElementById("select-retrieval-mode").value,
      }),
    });
    const data = await res.json();
    if (!res.ok) {
      loadingMsg.querySelector(".msg-content").innerHTML = `<p style="color:#fca5a5;">❌ ${escapeHtml(data.detail || "Lỗi gọi API")}</p>`;
    } else {
      loadingMsg.querySelector(".msg-content").innerHTML = renderMarkdown(data.answer);
      loadingMsg.querySelector(".msg-body").appendChild(buildTraceNode(data));
      state.history.push({ role: "user", content: text });
      state.history.push({ role: "assistant", content: data.answer });
    }
  } catch (err) {
    loadingMsg.querySelector(".msg-content").innerHTML = `<p style="color:#fca5a5;">❌ Lỗi kết nối server: ${escapeHtml(err.message)}</p>`;
  } finally {
    state.sending = false;
    btnSend.disabled = false;
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }
}

document.getElementById("chat-form").addEventListener("submit", e => {
  e.preventDefault();
  const txt = chatInput.value.trim();
  if (!txt) return;
  chatInput.value = "";
  sendChatMessage(txt);
});

chatInput.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    document.getElementById("chat-form").requestSubmit();
  }
});

document.querySelectorAll(".sample-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelector('.tab[data-tab="chat"]').click();
    sendChatMessage(btn.dataset.prompt);
  });
});

document.getElementById("btn-clear-chat").addEventListener("click", () => {
  state.history = [];
  chatMessages.innerHTML = "";
  appendMessage("assistant", "<p>Đã xóa lịch sử hội thoại. Bạn hãy đặt câu hỏi mới nhé!</p>");
});

// --- Direct Calculators ---
document.getElementById("preset-slide-219").addEventListener("click", () => {
  document.getElementById("inh-marital").value = 18;
  document.getElementById("inh-personal").value = 0;
  document.getElementById("inh-debts").value = 0;
  document.getElementById("inh-funeral").value = 0;
  document.getElementById("inh-unit").value = "tỷ đồng";
  document.getElementById("inh-has-will").checked = true;
  document.getElementById("inh-will-json").value = JSON.stringify([{ name: "Bà Loan", share_fraction: 1.0 }], null, 2);
  document.getElementById("inh-heirs-json").value = JSON.stringify([{ name: "Ông Tài", relation: "chồng", disinherited_by_will: true }], null, 2);
});

document.getElementById("preset-slide-220").addEventListener("click", () => {
  document.getElementById("inh-marital").value = 0;
  document.getElementById("inh-personal").value = 240;
  document.getElementById("inh-debts").value = 0;
  document.getElementById("inh-funeral").value = 0;
  document.getElementById("inh-unit").value = "triệu đồng";
  document.getElementById("inh-has-will").checked = true;
  document.getElementById("inh-will-json").value = JSON.stringify([{ name: "Bà M", share_fraction: 1.0 }], null, 2);
  document.getElementById("inh-heirs-json").value = JSON.stringify([
    { name: "Bà B", relation: "vợ" },
    { name: "C", relation: "con chưa thành niên" },
    { name: "D", relation: "con chưa thành niên" },
    { name: "E", relation: "con chưa thành niên" }
  ], null, 2);
});

document.getElementById("preset-slide-229").addEventListener("click", () => {
  document.getElementById("inh-marital").value = 0;
  document.getElementById("inh-personal").value = 720;
  document.getElementById("inh-debts").value = 0;
  document.getElementById("inh-funeral").value = 0;
  document.getElementById("inh-unit").value = "triệu đồng";
  document.getElementById("inh-has-will").checked = true;
  document.getElementById("inh-will-json").value = JSON.stringify([
    { name: "B", share_amount: 80 },
    { name: "C", share_amount: 80 },
    { name: "X", share_amount: 80 }
  ], null, 2);
  document.getElementById("inh-heirs-json").value = JSON.stringify([
    { name: "B", relation: "vợ" },
    { name: "C", relation: "con thành niên" },
    { name: "D", relation: "con thành niên" },
    { name: "E", relation: "con thành niên" }
  ], null, 2);
});

document.getElementById("btn-calc-inheritance").addEventListener("click", async () => {
  const out = document.getElementById("inh-result");
  try {
    const payload = {
      marital_joint_assets: parseFloat(document.getElementById("inh-marital").value || "0"),
      personal_assets: parseFloat(document.getElementById("inh-personal").value || "0"),
      debts_and_obligations: parseFloat(document.getElementById("inh-debts").value || "0"),
      funeral_cost: parseFloat(document.getElementById("inh-funeral").value || "0"),
      currency_unit: document.getElementById("inh-unit").value || "triệu đồng",
      has_valid_will: document.getElementById("inh-has-will").checked,
      will_beneficiaries: JSON.parse(document.getElementById("inh-will-json").value || "[]"),
      statutory_heirs: JSON.parse(document.getElementById("inh-heirs-json").value || "[]"),
    };
    const res = await fetch("/api/calc/inheritance", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const d = await res.json();
    if (!res.ok) throw new Error(d.detail);

    let md = d.steps.join("\n\n") + "\n\n### Bảng kết quả cuối cùng\n";
    md += "| Người nhận | Theo di chúc | Theo pháp luật | Điều chỉnh Đ644 | **Tổng nhận** |\n|---|---|---|---|---|\n";
    for (const [name, s] of Object.entries(d.shares)) {
      md += `| ${name} | ${s.from_will} | ${s.from_intestate} | ${s.adjustment_644} | **${s.total} ${d.currency_unit}** |\n`;
    }
    md += `\n**Cơ sở pháp lý:** ${d.legal_basis.join("; ")}`;
    out.innerHTML = `<div class="msg-content">${renderMarkdown(md)}</div>`;
  } catch (err) {
    out.innerHTML = `<p style="color:#fca5a5;">❌ ${escapeHtml(err.message)}</p>`;
  }
});

document.getElementById("btn-calc-d56").addEventListener("click", async () => {
  const out = document.getElementById("crim-result");
  try {
    const payload = {
      case_type: document.getElementById("crim-case-type").value,
      ba1_type: document.getElementById("crim-ba1-type").value,
      ba1_months: parseFloat(document.getElementById("crim-ba1-months").value || "0"),
      ba1_served_months: parseFloat(document.getElementById("crim-ba1-served").value || "0"),
      ba2_type: document.getElementById("crim-ba2-type").value,
      ba2_months: parseFloat(document.getElementById("crim-ba2-months").value || "0"),
    };
    const res = await fetch("/api/calc/criminal/multi-judgment", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const d = await res.json();
    if (!res.ok) throw new Error(d.detail);
    const md = `${d.steps.join("\n\n")}\n\n**Kết luận:** **${d.remaining_human}**\n\n**Cơ sở pháp lý:** ${d.legal_basis.join("; ")}`;
    out.innerHTML = `<div class="msg-content">${renderMarkdown(md)}</div>`;
  } catch (err) {
    out.innerHTML = `<p style="color:#fca5a5;">❌ ${escapeHtml(err.message)}</p>`;
  }
});

document.getElementById("btn-calc-d55").addEventListener("click", async () => {
  const out = document.getElementById("crim-result");
  try {
    const payload = {
      penalties: JSON.parse(document.getElementById("crim-d55-json").value || "[]"),
      detention_months: parseFloat(document.getElementById("crim-detention-months").value || "0"),
      detention_days: parseFloat(document.getElementById("crim-detention-days").value || "0"),
    };
    const res = await fetch("/api/calc/criminal/multi-crime", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const d = await res.json();
    if (!res.ok) throw new Error(d.detail);
    const md = `${d.steps.join("\n\n")}\n\n**Kết luận:** **${d.summary}**\n\n**Cơ sở pháp lý:** ${d.legal_basis.join("; ")}`;
    out.innerHTML = `<div class="msg-content">${renderMarkdown(md)}</div>`;
  } catch (err) {
    out.innerHTML = `<p style="color:#fca5a5;">❌ ${escapeHtml(err.message)}</p>`;
  }
});

document.getElementById("btn-calc-sev").addEventListener("click", async () => {
  const out = document.getElementById("lab-result");
  try {
    const payload = {
      calc_type: "severance",
      allowance_type: document.getElementById("lab-allow-type").value,
      avg_salary_6_months: parseFloat(document.getElementById("lab-avg-sal").value || "0"),
      total_working_months: parseFloat(document.getElementById("lab-total-m").value || "0"),
      unemployment_insurance_months: parseFloat(document.getElementById("lab-bhtn-m").value || "0"),
    };
    const res = await fetch("/api/calc/labor", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const d = await res.json();
    if (!res.ok) throw new Error(d.detail);
    const md = `${d.steps.join("\n")}\n\n**Cơ sở pháp lý:** ${d.legal_basis.join("; ")}`;
    out.innerHTML = `<div class="msg-content">${renderMarkdown(md)}</div>`;
  } catch (err) {
    out.innerHTML = `<p style="color:#fca5a5;">❌ ${escapeHtml(err.message)}</p>`;
  }
});

document.getElementById("btn-calc-ot").addEventListener("click", async () => {
  const out = document.getElementById("lab-result");
  try {
    const payload = {
      calc_type: "overtime",
      hourly_wage: parseFloat(document.getElementById("lab-wage").value || "0"),
      hours: parseFloat(document.getElementById("lab-hours").value || "0"),
      day_type: document.getElementById("lab-day-type").value,
      is_night_shift: document.getElementById("lab-night").checked,
    };
    const res = await fetch("/api/calc/labor", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const d = await res.json();
    if (!res.ok) throw new Error(d.detail);
    const md = `${d.steps.join("\n")}\n\n**Cơ sở pháp lý:** ${d.legal_basis.join("; ")}`;
    out.innerHTML = `<div class="msg-content">${renderMarkdown(md)}</div>`;
  } catch (err) {
    out.innerHTML = `<p style="color:#fca5a5;">❌ ${escapeHtml(err.message)}</p>`;
  }
});

// --- Hybrid Retrieval Inspector ---
async function runInspectorSearch() {
  const q = document.getElementById("inspect-query").value.trim();
  if (!q) return;
  const metaEl = document.getElementById("inspect-meta");
  const resEl = document.getElementById("inspect-results");
  metaEl.textContent = "Đang truy xuất...";
  resEl.innerHTML = "";

  try {
    const res = await fetch("/api/search", {
      method: "POST",
      headers: getApiHeaders(),
      body: JSON.stringify({
        query: q,
        mode: document.getElementById("inspect-mode").value,
        source_type: document.getElementById("inspect-source").value,
        top_k: 8,
      }),
    });
    const d = await res.json();
    metaEl.textContent = `Chế độ thực thi: ${d.mode_used} | Độ trễ: ${d.latency_ms} ms | Kết quả: ${(d.results || []).length}`;
    (d.results || []).forEach(r => {
      const card = document.createElement("div");
      card.className = "chunk-card";
      card.innerHTML = `
        <div class="chunk-head">
          <div><strong>#${r.rank}</strong> <span class="cite-badge">${escapeHtml(r.citation)}</span> <span style="color:var(--muted);font-size:12px;">${escapeHtml(r.chapter)}</span></div>
          <div class="chunk-scores">
            <span class="pill">Score: ${r.final_score}</span>
            <span class="pill">BM25 #${r.bm25_rank} (${r.bm25_score})</span>
            ${r.dense_score !== null ? `<span class="pill ok">Dense #${r.dense_rank} (${r.dense_score})</span>` : `<span class="pill warn">Dense: N/A</span>`}
          </div>
        </div>
        <div class="chunk-text">${escapeHtml(r.text)}</div>
      `;
      resEl.appendChild(card);
    });
  } catch (err) {
    metaEl.textContent = `❌ Lỗi: ${err.message}`;
  }
}

document.getElementById("btn-inspect-search").addEventListener("click", runInspectorSearch);
document.getElementById("inspect-query").addEventListener("keydown", e => {
  if (e.key === "Enter") runInspectorSearch();
});

// Initial load
refreshStats();
