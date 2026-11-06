import { API_BASE, API_AUTH_TOKEN } from "../config.js";

class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

function readableDetail(payload, status) {
  const detail = payload?.detail;
  if (typeof detail === "string" && detail) return detail;
  if (Array.isArray(detail) && detail.length > 0) return detail[0].msg || "Request failed.";
  if (status === 401) return "Authentication failed.";
  if (status === 413) return "Uploaded file too large.";
  if (status === 422) return "Invalid request.";
  if (status === 429) return "Rate limit/quota reached.";
  if (status === 502) return "LLM could not be reached.";
  if (status === 504) return "LLM request timed out.";
  if (status >= 500) return "Server error. Check backend logs.";
  return "Request rejected.";
}

async function postJson(path, body, timeoutMs = 45000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  const headers = {};
  if (API_AUTH_TOKEN) headers["X-API-Key"] = API_AUTH_TOKEN;

  try {
    const res = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    const payload = await res.json().catch(() => null);
    if (!res.ok) throw new ApiError(res.status, readableDetail(payload, res.status));
    return payload;
  } finally {
    clearTimeout(timeoutId);
  }
}

async function postMultipartExtract(files, timeoutMs = 900000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  const headers = {};
  if (API_AUTH_TOKEN) headers["X-API-Key"] = API_AUTH_TOKEN;

  try {
    const form = new FormData();
    for (const f of files) form.append("files", f, f.name);
    const res = await fetch(`${API_BASE}/pdf/extract`, {
      method: "POST",
      headers,
      body: form,
      signal: controller.signal,
    });
    const payload = await res.json().catch(() => null);
    if (!res.ok) throw new ApiError(res.status, readableDetail(payload, res.status));
    return payload;
  } finally {
    clearTimeout(timeoutId);
  }
}

function el(id) {
  return document.getElementById(id);
}

const pdfInput = el("pdf-input");
const pdfList = el("pdf-list");
const summaryQuestion = el("summary-question");
const extractButton = el("extract-button");
const summarizeButton = el("summarize-button");
const statusEl = el("status");
const errorEl = el("error");

let selectedFiles = []; // {file, filename}
let lastSessionId = null;
let lastExtracted = null;

function setStatus(msg) {
  statusEl.textContent = msg || "";
}

function setError(msg) {
  errorEl.hidden = !msg;
  errorEl.textContent = msg || "";
}

function renderFiles() {
  pdfList.innerHTML = "";

  if (!selectedFiles.length) {
    pdfList.innerHTML = "<div class='status'>No PDFs selected.</div>";
    return;
  }

  for (let i = 0; i < selectedFiles.length; i++) {
    const f = selectedFiles[i];

    const item = document.createElement("div");
    item.className = "pdf-item";

    const meta = document.createElement("div");
    meta.className = "pdf-meta";
    const name = document.createElement("div");
    name.className = "pdf-name";
    name.textContent = f.filename;

    const badge = document.createElement("div");
    badge.className = "badge";
    badge.textContent = f.pages ? `${f.pageExtracted}/${f.totalPages} extracted pages` : "Ready";
    meta.appendChild(name);
    meta.appendChild(badge);

    const actions = document.createElement("div");
    actions.className = "pdf-actions";

    const removeBtn = document.createElement("button");
    removeBtn.className = "small-btn";
    removeBtn.type = "button";
    removeBtn.textContent = "Remove";
    removeBtn.disabled = !!lastSessionId;
    removeBtn.addEventListener("click", () => {
      selectedFiles.splice(i, 1);
      lastSessionId = null;
      lastExtracted = null;
      summarizeButton.disabled = true;
      setError("");
      setStatus("");
      renderFiles();
    });

    actions.appendChild(removeBtn);
    item.appendChild(meta);
    item.appendChild(actions);

    pdfList.appendChild(item);
  }
}

pdfInput.addEventListener("change", () => {
  const files = Array.from(pdfInput.files || []);
  for (const f of files) {
    if (f.type !== "application/pdf" && !f.name.toLowerCase().endsWith(".pdf")) continue;
    if (selectedFiles.some(x => x.filename === f.name)) continue;
    selectedFiles.push({ file: f, filename: f.name });
  }
  lastSessionId = null;
  lastExtracted = null;
  summarizeButton.disabled = true;
  setError("");
  setStatus("");
  renderFiles();
});

extractButton.addEventListener("click", async () => {
  if (!selectedFiles.length) {
    setError("Select at least one PDF.");
    return;
  }

  if (!summaryQuestion.value.trim()) {
    setError("Enter a topic / question.");
    return;
  }

  setError("");
  setStatus("Extracting text from PDFs...");
  extractButton.disabled = true;
  summarizeButton.disabled = true;

  try {
    const files = selectedFiles.map(x => x.file);
    const res = await postMultipartExtract(files);

    lastSessionId = res.session_id;
    lastExtracted = res;

    const pdfs = res.pdfs || [];
    const byName = new Map(pdfs.map(p => [p.filename, p]));

    selectedFiles = selectedFiles.map(sf => {
      const r = byName.get(sf.filename);
      if (!r) return sf;
      return {
        ...sf,
        pages: r.extracted_pages,
        totalPages: r.total_pages,
        ok: r.success,
        error: r.error,
      };
    });

    renderFiles();

    const allSuccess = pdfs.every(p => p.success);
    setStatus(allSuccess ? "Extraction complete. Ready to synthesize." : "Extraction completed with some failures. You can still synthesize if at least one PDF extracted text.");
    summarizeButton.disabled = false;
  } catch (e) {
    setError(e?.message || String(e));
    setStatus("");
  } finally {
    extractButton.disabled = false;
  }
});

summarizeButton.addEventListener("click", async () => {
  if (!lastSessionId) {
    setError("No extracted session found. Extract PDFs first.");
    return;
  }

  setError("");
  setStatus("Synthesizing across documents and generating PDF...");
  summarizeButton.disabled = true;

  try {
    await postJson(`/pdf/synthesize/${encodeURIComponent(lastSessionId)}`, { question: summaryQuestion.value.trim() }, 900000);
    const gen = await postJson(`/pdf/generate/${encodeURIComponent(lastSessionId)}`, {}, 900000);

    const headers = {};
    if (API_AUTH_TOKEN) headers["X-API-Key"] = API_AUTH_TOKEN;
    const dlRes = await fetch(`${API_BASE}${gen.download_url}`, { headers });
    if (!dlRes.ok) throw new ApiError(dlRes.status, "Failed to download generated PDF report.");
    const blob = await dlRes.blob();
    const blobUrl = URL.createObjectURL(blob);

    const a = document.createElement("a");
    a.href = blobUrl;
    a.download = gen.filename || "synthesis-report.pdf";
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(blobUrl), 60000);

    setStatus("Done. Output PDF opened in a new tab.");
  } catch (e) {
    setError(e?.message || String(e));
    setStatus("");
    summarizeButton.disabled = false;
  }
});

renderFiles();
