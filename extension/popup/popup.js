import {
  askQuestion,
  askQuestionStream,
  rememberMemory,
  rememberWebpage,
  getDocumentTags,
  getTagsForUrl,
  checkServerHealth,
  getProviderStatus,
  testProviderConnection,
  configureProvider,
  removeProviderCredentials,
  storageService,
  OfflineError,
} from "../api.js";
import { API_BASE, API_AUTH_TOKEN } from "../config.js";
import { pdfService } from "../pdf-retrieval.js";

console.log("[MemoryMind] popup.js loaded successfully. Initializing...");

// Storage keys
const STORAGE_KEY_ACTIVATED = "memorymind_is_activated";
const STORAGE_KEY_CONFIG = "memorymind_provider_config";
const STORAGE_KEY_LAST_VERIFIED = "memorymind_last_verified";

// View panels
const loadingView = document.getElementById("loading-view");
const activationView = document.getElementById("activation-view");
const mainView = document.getElementById("main-view");
const settingsView = document.getElementById("settings-view");
const summarizeView = document.getElementById("summarize-view");

const ALL_VIEWS = ["loading-view", "activation-view", "main-view", "settings-view", "summarize-view"];

function showView(viewId) {
  for (const id of ALL_VIEWS) {
    const el = document.getElementById(id);
    if (el) {
      if (id === viewId) {
        el.hidden = false;
        el.removeAttribute("hidden");
      } else {
        el.hidden = true;
        el.setAttribute("hidden", "");
      }
    }
  }
}

// Main Dashboard elements
const questionInput = document.getElementById("question");
const askButton = document.getElementById("ask-button");
const rememberButton = document.getElementById("remember-button");
const rememberStatus = document.getElementById("remember-status");
const rememberPageButton = document.getElementById("remember-page-button");
const messageBox = document.getElementById("message");
const resultPanel = document.getElementById("result");
const answerText = document.getElementById("answer");
const sourcesBox = document.getElementById("sources");
const sourceList = document.getElementById("source-list");
const relatedConceptsBox = document.getElementById("related-concepts");
const conceptList = document.getElementById("concept-list");
const settingsOpenBtn = document.getElementById("settings-open-btn");

// First-Run Activation UI elements
const activationProviderSelect = document.getElementById("activation-provider-select");
const activationKeyGroup = document.getElementById("activation-key-group");
const activationApiKey = document.getElementById("activation-api-key");
const activationOllamaGroup = document.getElementById("activation-ollama-group");
const activationOllamaUrl = document.getElementById("activation-ollama-url");
const activationOllamaModel = document.getElementById("activation-ollama-model");
const activationError = document.getElementById("activation-error");
const activationStatus = document.getElementById("activation-status");
const activateSubmitBtn = document.getElementById("activate-submit-btn");

// Settings UI elements
const settingsBackBtn = document.getElementById("settings-back-btn");
const settingsStatusBadge = document.getElementById("settings-status-badge");
const currentProviderName = document.getElementById("current-provider-name");
const currentModelRow = document.getElementById("current-model-row");
const currentProviderModel = document.getElementById("current-provider-model");
const currentKeyRow = document.getElementById("current-key-row");
const currentProviderKey = document.getElementById("current-provider-key");
const currentProviderStateText = document.getElementById("current-provider-state-text");

const settingsProviderSelect = document.getElementById("settings-provider-select");
const settingsKeyGroup = document.getElementById("settings-key-group");
const settingsApiKey = document.getElementById("settings-api-key");
const settingsKeyHelper = document.getElementById("settings-key-helper");
const settingsOllamaGroup = document.getElementById("settings-ollama-group");
const settingsOllamaUrl = document.getElementById("settings-ollama-url");
const settingsOllamaModel = document.getElementById("settings-ollama-model");
const settingsError = document.getElementById("settings-error");
const settingsStatus = document.getElementById("settings-status");
const settingsTestBtn = document.getElementById("settings-test-btn");
const settingsSaveBtn = document.getElementById("settings-save-btn");
const settingsRemoveBtn = document.getElementById("settings-remove-btn");



// --- Small UI helpers -------------------------------------------------------

function showError(message) {
  messageBox.textContent = message;
  messageBox.hidden = false;
}

function clearError() {
  messageBox.hidden = true;
  messageBox.textContent = "";
}

function setBusy(button, busyLabel, idleLabel) {
  button.dataset.idleLabel = idleLabel;
  button.textContent = busyLabel;
  button.disabled = true;
}

function setIdle(button) {
  button.textContent = button.dataset.idleLabel || button.textContent;
  button.disabled = false;
}

function showRememberStatus(message, isOk) {
  rememberStatus.textContent = message;
  rememberStatus.classList.toggle("ok", isOk);
  rememberStatus.hidden = false;
}

function isSafeHttpUrl(url) {
  if (!url || typeof url !== "string") return false;
  try {
    const parsed = new URL(url);
    return parsed.protocol === "http:" || parsed.protocol === "https:";
  } catch {
    return false;
  }
}

let isOperationInProgress = false;

function setAllBusy(isBusy) {
  askButton.disabled = isBusy;
  rememberButton.disabled = isBusy;
  rememberPageButton.disabled = isBusy;
}

// Any network or API problem is shown the same way to the user. Raw error
// text stays in the console.
function handleError(error) {
  if (error instanceof OfflineError) {
    showError(error.message);
  } else {
    showError(error.message || "Something went wrong.");
  }
  console.warn("MemoryMind:", error);
}

function escapeHtml(text) {
  return (text || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function parseInlineMarkdown(text) {
  // 1. Escape HTML for strict XSS prevention
  let safe = escapeHtml(text);

  // 2. Normalize backslash-escaped markdown formatting characters (e.g. \*\* -> **)
  safe = safe.replace(/\\([*_`#])/g, "$1");

  // 3. Inline code: `code`
  safe = safe.replace(/`([^`]+)`/g, "<code>$1</code>");

  // 4. Bold: **text** or __text__
  safe = safe.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  safe = safe.replace(/__([^_]+)__/g, "<strong>$1</strong>");

  // 5. Italic: *text* or _text_
  safe = safe.replace(/(^|[^*])\*([^*]+)\*([^*]|$)/g, "$1<em>$2</em>$3");
  safe = safe.replace(/(^|[^_])_([^_]+)_([^_]|$)/g, "$1<em>$2</em>$3");

  // 6. Interactive citation badges: [1], [2], etc.
  safe = safe.replace(
    /\[(\d+)\]/g,
    '<button type="button" class="citation-badge" data-source-id="$1" title="Jump to source [$1]">[$1]</button>'
  );

  return safe;
}

function renderSafeMarkdownWithCitations(container, markdownText) {
  container.innerHTML = "";
  if (!markdownText) return;

  const lines = markdownText.split("\n");
  let inList = false;
  let listEl = null;

  for (let i = 0; i < lines.length; i++) {
    const rawLine = lines[i];
    const trimmed = rawLine.trim();

    // Check for bullet list item: "- " or "* "
    const isBullet = /^[-*]\s+/.test(trimmed);

    if (isBullet) {
      if (!inList) {
        inList = true;
        listEl = document.createElement("ul");
        container.appendChild(listEl);
      }
      const itemContent = trimmed.replace(/^[-*]\s+/, "");
      const li = document.createElement("li");
      li.innerHTML = parseInlineMarkdown(itemContent);
      listEl.appendChild(li);
    } else {
      if (inList) {
        inList = false;
        listEl = null;
      }

      if (trimmed) {
        const p = document.createElement("p");
        p.innerHTML = parseInlineMarkdown(trimmed);
        container.appendChild(p);
      }
    }
  }

  // Attach interactive click handlers to all generated citation badges
  container.querySelectorAll(".citation-badge").forEach((badge) => {
    badge.addEventListener("click", () => {
      const sourceIdx = badge.getAttribute("data-source-id");
      const card = document.getElementById(`source-card-${sourceIdx}`);
      if (card) {
        card.scrollIntoView({ behavior: "smooth", block: "center" });
        card.classList.add("highlight");
        setTimeout(() => card.classList.remove("highlight"), 2000);
      }
    });
  });
}

// --- Asking ----------------------------------------------------------------

async function handleAsk() {
  if (isOperationInProgress || askButton.disabled) return;

  const question = questionInput.value.trim();
  if (!question) {
    showError("Type a question first.");
    return;
  }

  isOperationInProgress = true;
  clearError();
  resultPanel.hidden = true;
  answerText.innerHTML = "";
  setAllBusy(true);
  setBusy(askButton, "Searching memories...", "Ask");

  let accumulated = "";
  let streamStarted = false;

  const stageTimer = setTimeout(() => {
    if (isOperationInProgress && !streamStarted) {
      askButton.textContent = "Synthesizing answer...";
    }
  }, 750);

  try {
    await askQuestionStream(
      question,
      (token) => {
        streamStarted = true;
        accumulated += token;
        resultPanel.hidden = false;
        renderSafeMarkdownWithCitations(answerText, accumulated);
      },
      (doneEvent) => {
        const finalAnswer = doneEvent.cleaned_answer || accumulated;
        renderAnswer({
          answer: finalAnswer,
          sources: doneEvent.sources || [],
          related_concepts: doneEvent.related_concepts || [],
        });
      }
    );
  } catch (streamError) {
    // Fall back to regular endpoint if streaming fails
    try {
      const data = await askQuestion(question);
      renderAnswer(data);
    } catch (error) {
      handleError(error);
    }
  } finally {
    clearTimeout(stageTimer);
    setIdle(askButton);
    setAllBusy(false);
    isOperationInProgress = false;
  }
}

function formatReadTime(value) {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function renderAnswer(data) {
  const answer = data.answer || "";
  renderSafeMarkdownWithCitations(answerText, answer);
  renderSources(data.sources || []);
  renderRelatedConcepts(data.related_concepts || []);
  resultPanel.hidden = false;
}

function renderSources(sources) {
  sourceList.innerHTML = "";

  if (!sources.length) {
    sourcesBox.hidden = true;
    return;
  }

  for (let i = 0; i < sources.length; i++) {
    const source = sources[i];
    const idx = source.source_index || (i + 1);
    const item = document.createElement("li");
    item.className = "source-card";
    item.id = `source-card-${idx}`;
    if (source.is_cited) {
      item.classList.add("cited");
    }

    const title = document.createElement("div");
    title.className = "source-title";

    const indexBadge = document.createElement("span");
    indexBadge.className = "source-num-badge";
    indexBadge.textContent = `[${idx}] `;
    title.appendChild(indexBadge);

    if (source.url && isSafeHttpUrl(source.url)) {
      const link = document.createElement("a");
      link.href = source.url;
      link.target = "_blank";
      link.rel = "noopener noreferrer";
      link.textContent = source.title || source.url;
      title.appendChild(link);
    } else {
      const span = document.createElement("span");
      span.textContent = source.title || "Saved memory";
      title.appendChild(span);
    }

    const meta = document.createElement("div");
    meta.className = "source-meta";
    const domain = source.domain || "";
    const readTime = formatReadTime(source.visited_at);
    const relevance = Number.isFinite(Number(source.score))
      ? `Score: ${Number(source.score).toFixed(2)}`
      : "";
    meta.textContent = [domain, readTime ? `Read ${readTime}` : "", relevance]
      .filter(Boolean)
      .join(" · ");

    const tags = [...(source.topics || []), ...(source.tags || [])]
      .map((tag) => tag.label)
      .filter(Boolean);
    const uniqueTags = [...new Set(tags)].slice(0, 8);
    if (uniqueTags.length) {
      const tagRow = document.createElement("div");
      tagRow.className = "source-tags";
      tagRow.textContent = uniqueTags.join(" · ");
      item.appendChild(tagRow);
    }

    if (source.snippet) {
      const snippet = document.createElement("p");
      snippet.className = "source-snippet";
      snippet.textContent = `Relevant section: ${source.snippet}`;
      item.appendChild(snippet);
    }

    if (source.url && isSafeHttpUrl(source.url)) {
      const open = document.createElement("a");
      open.href = source.url;
      open.target = "_blank";
      open.rel = "noopener noreferrer";
      open.className = "open-article";
      open.textContent = "Open Article ↗";
      item.appendChild(open);
    }

    item.prepend(meta);
    item.prepend(title);
    sourceList.appendChild(item);
  }

  sourcesBox.hidden = false;
}

function renderRelatedConcepts(concepts) {
  conceptList.innerHTML = "";

  if (!concepts.length) {
    relatedConceptsBox.hidden = true;
    return;
  }

  for (const concept of concepts.slice(0, 12)) {
    const item = document.createElement("li");
    item.className = "concept-item";

    const label = document.createElement("strong");
    label.textContent = concept.label || "Concept";

    const count = document.createElement("span");
    count.className = "concept-count";
    count.textContent = `${concept.source_count || 0} source${concept.source_count === 1 ? "" : "s"}`;

    item.appendChild(label);
    item.appendChild(count);
    conceptList.appendChild(item);
  }

  relatedConceptsBox.hidden = false;
}


// --- Remembering the current selection -------------------------------------

async function handleRemember() {
  if (isOperationInProgress || rememberButton.disabled) return;
  
  isOperationInProgress = true;
  clearError();
  setAllBusy(true);
  setBusy(rememberButton, "Checking page...", "Remember selected text");
  showRememberStatus("Checking page...", false);
  
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    // Check if it's a PDF
    if (tab?.url && pdfService.isPdfUrl(tab.url)) {
      setBusy(rememberButton, "Retrieving PDF...", "Remember selected text");
      showRememberStatus("Retrieving PDF...", false);
      
      try {
        const pdfData = await pdfService.retrievePdfFromActiveTab();
        setBusy(rememberButton, "Saving PDF...", "Remember selected text");
        showRememberStatus("Extracting and saving PDF...", false);
        
        // Convert to FormData and upload
        const blob = new Blob([pdfData.bytes], { type: 'application/pdf' });
        const formData = new FormData();
        formData.append('file', blob, pdfData.filename);
        formData.append('source_url', pdfData.url);
        formData.append('source_title', pdfData.filename);
        
        const pdfHeaders = {};
        if (API_AUTH_TOKEN) {
          pdfHeaders["X-API-Key"] = API_AUTH_TOKEN;
        }

        const response = await fetch(`${API_BASE}/memory/pdf`, {
          method: 'POST',
          headers: pdfHeaders,
          body: formData
        });
        
        if (!response.ok) {
          throw new Error(`Failed to save PDF: ${response.status}`);
        }
        
        const data = await response.json();
        showRememberStatus("✓ PDF saved", true);
      } catch (pdfErr) {
        if (pdfErr.message === 'FILE_ACCESS_REQUIRED' || pdfErr.message.includes('file access') || pdfErr.message.includes('Allow access to file URLs')) {
          showRememberFileAccessPrompt();
        } else {
          showRememberStatus(pdfErr.message, false);
        }
      }
    } else {
      // Existing webpage/selection logic
      setBusy(rememberButton, "Reading selection...", "Remember selected text");
      showRememberStatus("Reading highlighted text...", false);
      
      const selection = await getSelectionFromActiveTab();
      
      if (!selection) {
        showRememberStatus("Select some text on the webpage first.", false);
        return;
      }
      
      setBusy(rememberButton, "Saving memory...", "Remember selected text");
      showRememberStatus("Embedding & saving memory...", false);
      
      const data = await rememberMemory(selection.text, selection.url, selection.title);
      const labelText = formatTagsOrTopics(data.tags, data.topics);
      showRememberStatus(
        labelText ? `✓ Remembered · ${labelText}` : "✓ Remembered",
        true,
      );
      if ((!data.tags || data.tags.length === 0) && (data.document_id || selection.url)) {
        pollForTags(data.document_id, "✓ Remembered", 20, selection.url);
      }
    }
  } catch (error) {
    handleError(error);
  } finally {
    setIdle(rememberButton);
    setAllBusy(false);
    isOperationInProgress = false;
  }
}

function formatTagsOrTopics(tags, topics) {
  if (tags && tags.length > 0) {
    const labels = tags
      .map((t) => (typeof t === "string" ? t : t?.label))
      .map((l) => (l || "").trim())
      .filter(Boolean);
    const unique = [...new Set(labels)];
    if (unique.length > 0) {
      return unique.join(", ");
    }
  }
  if (topics && topics.length > 0) {
    const meaningful = topics
      .map((t) => (typeof t === "string" ? t : t?.label))
      .map((l) => (l || "").trim())
      .filter((l) => l && l.toLowerCase() !== "other");
    const unique = [...new Set(meaningful)];
    if (unique.length > 0) {
      return unique.join(", ");
    }
  }
  return "";
}

let currentPollingId = 0;

async function pollForTags(documentId, prefix, maxAttempts = 20, url = null) {
  const pollSession = ++currentPollingId;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    await new Promise((resolve) => setTimeout(resolve, 1500));
    if (pollSession !== currentPollingId) return;

    try {
      let res = documentId ? await getDocumentTags(documentId) : null;
      if ((!res || !res.tags || res.tags.length === 0) && url) {
        res = await getTagsForUrl(url);
      }

      if (res && res.tags && res.tags.length > 0) {
        const tagText = formatTagsOrTopics(res.tags, res.topics || []);
        if (tagText) {
          showRememberStatus(`${prefix} · ${tagText}`, true);
        }
        break;
      }
    } catch {
      // Non-blocking best-effort update
    }
  }
}

// Reads the text the user has highlighted in the active tab. The activeTab
// permission is granted when the popup opens, so no broad access is needed.
async function getSelectionFromActiveTab() {
  const [tab] = await chrome.tabs.query({
    active: true,
    currentWindow: true,
  });
  if (!tab?.id) return null;

  if (tab.url && !isSafeHttpUrl(tab.url)) {
    throw new Error("Can't save selection on this page. Try a normal http or https webpage.");
  }

  let results;
  try {
    results = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => window.getSelection().toString(),
    });
  } catch {
    throw new Error("Can't read the selection on this page. Try a normal webpage.");
  }

  const text = (results?.[0]?.result || "").trim();
  if (!text) return null;

  return { text, url: tab.url || "", title: tab.title || "" };
}



async function handleRememberPage() {
  if (isOperationInProgress || rememberPageButton.disabled) return;

  isOperationInProgress = true;
  clearError();
  setAllBusy(true);
  setBusy(rememberPageButton, "Reading page...", "Remember current page");
  showRememberStatus("Extracting page content...", false);

  try {
    const page = await getCurrentPageHTML();
    if (!page?.html) {
      showRememberStatus("No webpage content was available.", false);
      return;
    }
    setBusy(rememberPageButton, "Saving to memory...", "Remember current page");
    showRememberStatus("Embedding & saving to memory...", false);

    const data = await rememberWebpage(page.url, page.title, page.html);
    const labelText = formatTagsOrTopics(data.tags, data.topics);
    showRememberStatus(
      labelText ? `✓ Page saved · ${labelText}` : "✓ Page saved",
      true,
    );
    if ((!data.tags || data.tags.length === 0) && (data.document_id || page.url)) {
      pollForTags(data.document_id, "✓ Page saved", 20, page.url);
    }
  } catch (error) {
    handleError(error);
  } finally {
    setIdle(rememberPageButton);
    setAllBusy(false);
    isOperationInProgress = false;
  }
}

async function getCurrentPageHTML() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) return null;

  if (tab.url && !isSafeHttpUrl(tab.url)) {
    throw new Error("Can't save this page. Try a normal http or https webpage.");
  }

  let results;
  try {
    results = await chrome.scripting.executeScript({
      target: { tabId: tab.id },
      func: () => {
        const url = location.href;
        const title = document.title || "";
        let html = document.documentElement?.outerHTML || "";

        // If the HTML is large (> 1,500,000 characters), prune non-content bloat
        // (scripts, styles, SVGs, canvas, large data URIs) before sending.
        if (html.length > 1500000) {
          try {
            const clone = document.documentElement.cloneNode(true);
            const elementsToRemove = clone.querySelectorAll(
              "script, style, svg, canvas, iframe, noscript, audio, video"
            );
            elementsToRemove.forEach((el) => el.remove());

            const images = clone.querySelectorAll("img[src^='data:']");
            images.forEach((img) => {
              if (img.getAttribute("src")?.length > 500) {
                img.removeAttribute("src");
              }
            });

            const cleanedHtml = clone.outerHTML;
            if (cleanedHtml && cleanedHtml.length > 100) {
              html = cleanedHtml;
            }
          } catch {
            // Keep original html if DOM clone fails
          }
        }

        return { html, title, url };
      },
    });
  } catch {
    throw new Error("Can't read this page. Try a normal webpage.");
  }
  return results?.[0]?.result || null;
}

// --- Check and restore tags when popup opens -------------------------------

async function checkActivePageStatus() {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab?.url || !isSafeHttpUrl(tab.url)) return;

    const res = await getTagsForUrl(tab.url);
    if (res && res.saved) {
      if (res.tags && res.tags.length > 0) {
        const tagText = formatTagsOrTopics(res.tags, res.topics || []);
        if (tagText) {
          showRememberStatus(`✓ Page saved · ${tagText}`, true);
        }
      } else {
        const topicText = formatTagsOrTopics([], res.topics || []);
        showRememberStatus(
          topicText ? `✓ Page saved · ${topicText}` : "✓ Page saved",
          true,
        );
        pollForTags(res.document_id, "✓ Page saved", 20, tab.url);
      }
    }
  } catch {
    // Non-blocking best-effort check
  }
}

// --- Wiring ----------------------------------------------------------------

askButton.addEventListener("click", handleAsk);
rememberButton.addEventListener("click", handleRemember);
rememberPageButton.addEventListener("click", handleRememberPage);

questionInput.addEventListener("keydown", (event) => {
  // Enter asks, Shift+Enter adds a newline.
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    handleAsk();
  }
});

// Seed the idle labels so setBusy can restore them after a request.
askButton.dataset.idleLabel = askButton.textContent;
rememberButton.dataset.idleLabel = rememberButton.textContent;
rememberPageButton.dataset.idleLabel = rememberPageButton.textContent;

// --- Summarize unified view --------------------------------------------------

const summarizeMainButton = document.getElementById("summarize-main-button");
const backToMainBtn = document.getElementById("back-to-main");
const addCurrentPageSum = document.getElementById("add-current-page-sum");
const addUrlsSum = document.getElementById("add-urls-sum");
const addPdfsSum = document.getElementById("add-pdfs-sum");
const urlsInputSection = document.getElementById("urls-input-section");
const urlsInputSum = document.getElementById("urls-input-sum");
const addUrlsConfirm = document.getElementById("add-urls-confirm");
const pdfInputSum = document.getElementById("pdf-input-sum");
const summarySourcesList = document.getElementById("summary-sources-list");
const summaryStyle = document.getElementById("summary-style");
const summaryTopic = document.getElementById("summary-topic");
const summaryLength = document.getElementById("summary-length");
const generateSummaryBtn = document.getElementById("generate-summary-btn");
const summaryProgress = document.getElementById("summary-progress");
const summaryProgressFill = document.getElementById("summary-progress-fill");
const summaryProgressMessage = document.getElementById("summary-progress-message");
const summaryResult = document.getElementById("summary-result");
const summaryResultMessage = document.getElementById("summary-result-message");
const viewPdfBtn = document.getElementById("view-pdf-btn");
const summaryError = document.getElementById("summary-error");

let summarySources = [];
let summaryJobId = null;

function showSummarizeView() {
  showView("summarize-view");
}

function showMainView() {
  showView("main-view");
}

function updateSummarySourcesList() {
  summarySourcesList.innerHTML = "";
  summarySources.forEach((source, index) => {
    const item = document.createElement("div");
    item.className = "summary-source-item";
    item.innerHTML = `
      <div class="summary-source-info">
        <div class="summary-source-name">${escapeHtml(source.name)}</div>
        <div class="summary-source-type">${source.type}</div>
      </div>
      <button class="summary-source-remove" type="button" data-index="${index}">×</button>
    `;
    item.querySelector(".summary-source-remove").addEventListener("click", () => {
      summarySources.splice(index, 1);
      updateSummarySourcesList();
      generateSummaryBtn.disabled = summarySources.length === 0;
    });
    summarySourcesList.appendChild(item);
  });
  generateSummaryBtn.disabled = summarySources.length === 0;
}

summarizeMainButton?.addEventListener("click", showSummarizeView);
backToMainBtn?.addEventListener("click", showMainView);
addCurrentPageSum?.addEventListener("click", async () => {
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    
    if (tab?.url && pdfService.isPdfUrl(tab.url)) {
      try {
        const pdfData = await pdfService.retrievePdfFromActiveTab();
        summarySources.push({
          type: "pdf",
          name: pdfData.filename,
          url: pdfData.url,
          fileData: pdfData.bytes,
          pdfRetrieved: true
        });
        updateSummarySourcesList();
        return;
      } catch (pdfErr) {
        if (pdfErr.message === 'FILE_ACCESS_REQUIRED' || pdfErr.message.includes('file access') || pdfErr.message.includes('Allow access to file URLs')) {
          showSummaryFileAccessPrompt();
        } else {
          showSummaryError(pdfErr.message);
        }
        return;
      }
    }
    
    if (!tab?.url || !tab.url.startsWith('http')) {
      showSummaryError("Cannot add current page.");
      return;
    }
    
    summarySources.push({
      type: "webpage",
      name: tab.title || "Current Page",
      url: tab.url
    });
    updateSummarySourcesList();
  } catch (err) {
    showSummaryError("Cannot add current page.");
  }
});
addUrlsSum?.addEventListener("click", () => {
  urlsInputSection.hidden = !urlsInputSection.hidden;
});

addUrlsConfirm?.addEventListener("click", () => {
  const text = urlsInputSum.value.trim();
  if (!text) return;
  const urls = text.split('\n').map(u => u.trim()).filter(u => u && (u.startsWith('http://') || u.startsWith('https://')));
  urls.forEach(url => {
    summarySources.push({
      type: "webpage",
      name: url,
      url: url
    });
  });
  urlsInputSum.value = "";
  urlsInputSection.hidden = true;
  updateSummarySourcesList();
});

addPdfsSum?.addEventListener("click", () => {
  pdfInputSum.click();
});

pdfInputSum?.addEventListener("change", () => {
  const files = Array.from(pdfInputSum.files || []);
  files.forEach(file => {
    if (file.type === 'application/pdf' || file.name.toLowerCase().endsWith('.pdf')) {
      summarySources.push({
        type: "pdf",
        name: file.name,
        file: file
      });
    }
  });
  pdfInputSum.value = "";
  updateSummarySourcesList();
});

function showRememberFileAccessPrompt() {
  rememberStatus.innerHTML = "";
  rememberStatus.classList.remove("ok");
  
  const text = document.createElement("span");
  text.textContent = "Enable 'Allow access to file URLs' to save local PDFs. ";
  rememberStatus.appendChild(text);

  const link = document.createElement("a");
  link.href = "#";
  link.textContent = "Open Settings";
  link.style.color = "var(--accent)";
  link.style.textDecoration = "underline";
  link.style.cursor = "pointer";
  link.addEventListener("click", (e) => {
    e.preventDefault();
    chrome.tabs.create({ url: `chrome://extensions/?id=${chrome.runtime.id}` });
  });
  rememberStatus.appendChild(link);
  rememberStatus.hidden = false;
}

function showSummaryFileAccessPrompt() {
  summaryError.innerHTML = "";
  
  const text = document.createElement("div");
  text.style.fontWeight = "600";
  text.textContent = "Cannot access local PDF";
  summaryError.appendChild(text);

  const subtext = document.createElement("div");
  subtext.style.marginTop = "4px";
  subtext.style.fontSize = "12px";
  subtext.textContent = "Chrome restricts extensions from reading local files by default. To use this PDF:";
  summaryError.appendChild(subtext);

  const list = document.createElement("ul");
  list.style.margin = "4px 0 8px 16px";
  list.style.padding = "0";
  list.style.fontSize = "12px";
  
  const step1 = document.createElement("li");
  step1.textContent = "Enable 'Allow access to file URLs' in extension settings, or";
  list.appendChild(step1);

  const step2 = document.createElement("li");
  step2.textContent = "Select the PDF file manually using the file picker.";
  list.appendChild(step2);

  summaryError.appendChild(list);

  const btnRow = document.createElement("div");
  btnRow.style.display = "flex";
  btnRow.style.gap = "6px";
  btnRow.style.flexWrap = "wrap";
  btnRow.style.marginTop = "6px";

  const settingsBtn = document.createElement("button");
  settingsBtn.type = "button";
  settingsBtn.className = "secondary small-btn";
  settingsBtn.textContent = "Open Extension Settings";
  settingsBtn.addEventListener("click", () => {
    chrome.tabs.create({ url: `chrome://extensions/?id=${chrome.runtime.id}` });
  });
  btnRow.appendChild(settingsBtn);

  const uploadBtn = document.createElement("button");
  uploadBtn.type = "button";
  uploadBtn.className = "primary small-btn";
  uploadBtn.textContent = "Select PDF Manually";
  uploadBtn.addEventListener("click", () => {
    summaryError.hidden = true;
    pdfInputSum.click();
  });
  btnRow.appendChild(uploadBtn);

  summaryError.appendChild(btnRow);
  summaryError.hidden = false;
}

function showSummaryError(msg) {
  if (!msg) {
    summaryError.textContent = "";
    summaryError.hidden = true;
    return;
  }
  summaryError.textContent = msg;
  summaryError.hidden = false;
}

generateSummaryBtn?.addEventListener("click", async () => {
  showSummaryError("");
  summaryProgress.hidden = false;
  summaryResult.hidden = true;
  generateSummaryBtn.disabled = true;
  
  try {
    const urls = summarySources.filter(s => s.type === "webpage").map(s => s.url);
    const pdfFiles = summarySources.filter(s => s.type === "pdf");
    
    const config = {
      summary_style: summaryStyle.value,
      topic: summaryTopic.value,
      max_length: parseInt(summaryLength.value) || 2000,
      urls: urls
    };
    
    const formData = new FormData();
    formData.append('request', JSON.stringify(config));
    
    // Add uploaded files
    pdfFiles.forEach(source => {
      if (source.file) {
        formData.append('files', source.file);
      } else if (source.fileData && source.pdfRetrieved) {
        // Handle retrieved PDFs
        const blob = new Blob([source.fileData], { type: 'application/pdf' });
        formData.append('files', blob, source.name);
      }
    });
    
    summaryProgressFill.style.width = "20%";
    summaryProgressMessage.textContent = "Creating job...";
    
    const authHeaders = {};
    if (API_AUTH_TOKEN) {
      authHeaders["X-API-Key"] = API_AUTH_TOKEN;
    }
    const res = await fetch(`${API_BASE}/jobs/create`, {
      method: "POST",
      headers: authHeaders,
      body: formData
    });
      
      if (!res.ok) {
        const errorText = await res.text();
        let errorMsg = `Failed to create job (${res.status})`;
        try {
          const errorJson = JSON.parse(errorText);
          if (typeof errorJson.detail === 'string') {
            errorMsg = errorJson.detail;
          } else if (Array.isArray(errorJson.detail)) {
            errorMsg = errorJson.detail.map(d => d.msg || d).join(', ');
          } else if (errorJson.detail && typeof errorJson.detail === 'object') {
            errorMsg = JSON.stringify(errorJson.detail);
          }
        } catch (e) {
          if (errorText && errorText.length < 200) {
            errorMsg = errorText;
          }
        }
        throw new Error(errorMsg);
      }
    const jobData = await res.json();
    summaryJobId = jobData.job_id;
    
    summaryProgressFill.style.width = "50%";
    summaryProgressMessage.textContent = "Processing sources...";
    
    // Poll for completion (bounded: max 15 minutes)
    const pollStartTime = Date.now();
    const maxPollMs = 15 * 60 * 1000;
    const pollHeaders = {};
    if (API_AUTH_TOKEN) {
      pollHeaders["X-API-Key"] = API_AUTH_TOKEN;
    }
    const poll = setInterval(async () => {
      if (Date.now() - pollStartTime > maxPollMs) {
        clearInterval(poll);
        showSummaryError("Job polling timed out after 15 minutes.");
        summaryProgress.hidden = true;
        generateSummaryBtn.disabled = false;
        return;
      }
      try {
        const statusRes = await fetch(`${API_BASE}/jobs/${summaryJobId}/status`, {
          headers: pollHeaders,
        });
        const status = await statusRes.json();
        
        summaryProgressFill.style.width = `${status.progress * 100}%`;
        summaryProgressMessage.textContent = status.message || "Processing...";
        
        if (status.status === "completed") {
          clearInterval(poll);
          summaryProgress.hidden = true;
          summaryResult.hidden = false;
          summaryResultMessage.textContent = `Completed with ${status.sources_summary?.successful || 0} sources`;
          generateSummaryBtn.disabled = false;
        } else if (status.status === "failed" || status.status === "expired") {
          clearInterval(poll);
          showSummaryError(status.message || "Job failed");
          summaryProgress.hidden = true;
          generateSummaryBtn.disabled = false;
        }
      } catch (err) {
        // ignore transient network errors during polling
      }
    }, 1000);
    
  } catch (err) {
    showSummaryError(err.message);
    summaryProgress.hidden = true;
    generateSummaryBtn.disabled = false;
  }
});

viewPdfBtn?.addEventListener("click", async () => {
  if (!summaryJobId) return;
  try {
    const authHeaders = {};
    if (API_AUTH_TOKEN) {
      authHeaders["X-API-Key"] = API_AUTH_TOKEN;
    }
    const res = await fetch(`${API_BASE}/jobs/${summaryJobId}/pdf`, {
      headers: authHeaders,
    });
    if (!res.ok) {
      throw new Error(`Failed to load PDF (${res.status})`);
    }
    const blob = await res.blob();
    const blobUrl = URL.createObjectURL(blob);
    window.open(blobUrl, "_blank", "noopener,noreferrer");
    setTimeout(() => URL.revokeObjectURL(blobUrl), 60000);
  } catch (err) {
    showSummaryError(`Failed to open PDF: ${err.message}`);
  }
});

// Store job in localStorage for persistence
setInterval(() => {
  if (summaryJobId) {
    localStorage.setItem('summarize_job_id', summaryJobId);
  }
}, 2000);

// Restore job on load
const savedJobId = localStorage.getItem('summarize_job_id');
if (savedJobId) {
  // Could restore, but keep simple
}

// --- First-Run Activation & Settings Management -----------------------------

function getProviderDisplayName(providerId) {
  const map = {
    grok: "xAI Grok",
    openai: "OpenAI",
    gemini: "Google Gemini",
    anthropic: "Anthropic Claude",
    ollama: "Ollama (Local)",
    none: "Offline / Echo Mode",
  };
  return map[providerId] || (providerId ? providerId.toUpperCase() : "None");
}

function updateActivationFormFields() {
  const provider = activationProviderSelect.value;
  if (provider === "ollama") {
    activationKeyGroup.hidden = true;
    activationOllamaGroup.hidden = false;
  } else {
    activationKeyGroup.hidden = false;
    activationOllamaGroup.hidden = true;
  }
}

activationProviderSelect?.addEventListener("change", updateActivationFormFields);

activateSubmitBtn?.addEventListener("click", async () => {
  const provider = activationProviderSelect.value;
  const apiKey = activationApiKey.value.trim();
  const ollamaUrl = activationOllamaUrl.value.trim();
  const ollamaModel = activationOllamaModel.value.trim();

  activationError.hidden = true;
  activationStatus.hidden = true;

  if (provider !== "ollama" && provider !== "none" && !apiKey) {
    activationError.textContent = "Please enter an API key to activate.";
    activationError.hidden = false;
    activationApiKey.focus();
    return;
  }

  activateSubmitBtn.disabled = true;
  activateSubmitBtn.textContent = "Verifying & Activating...";

  try {
    const configPayload = {
      provider,
      apiKey: apiKey || null,
      model: provider === "ollama" ? ollamaModel : null,
      baseUrl: provider === "ollama" ? ollamaUrl : null,
      testFirst: true,
    };

    const res = await configureProvider(configPayload);

    // Save verified persistent state
    await storageService.set(STORAGE_KEY_ACTIVATED, true);
    await storageService.set(STORAGE_KEY_CONFIG, res);
    await storageService.set(STORAGE_KEY_LAST_VERIFIED, Date.now());

    activationStatus.textContent = "✓ Activated successfully! Loading dashboard...";
    activationStatus.hidden = false;

    setTimeout(() => {
      showView("main-view");
      checkActivePageStatus();
      activationApiKey.value = "";
      activationStatus.hidden = true;
      activateSubmitBtn.disabled = false;
      activateSubmitBtn.textContent = "Activate MemoryMind";
    }, 700);
  } catch (err) {
    activationError.textContent = err.message || "Failed to activate provider.";
    activationError.hidden = false;
    activateSubmitBtn.disabled = false;
    activateSubmitBtn.textContent = "Activate MemoryMind";
  }
});

// --- Settings View Logic ----------------------------------------------------

function updateSettingsFormFields() {
  const selectedProvider = settingsProviderSelect.value;
  if (selectedProvider === "ollama") {
    settingsKeyGroup.hidden = true;
    settingsOllamaGroup.hidden = false;
  } else {
    settingsKeyGroup.hidden = false;
    settingsOllamaGroup.hidden = true;
  }

  // Update helper text dynamically
  storageService.get(STORAGE_KEY_CONFIG).then((cached) => {
    const activeProvider = cached?.active_provider;
    if (selectedProvider === activeProvider) {
      settingsApiKey.placeholder = "Leave empty to keep existing verified key";
      settingsKeyHelper.textContent = "Enter a new key only if replacing existing credentials.";
    } else {
      settingsApiKey.placeholder = `Enter API key for ${getProviderDisplayName(selectedProvider)}`;
      settingsKeyHelper.textContent = "New credentials will be tested before becoming active.";
    }
  });
}

settingsProviderSelect?.addEventListener("change", updateSettingsFormFields);

async function renderSettingsView() {
  settingsError.hidden = true;
  settingsStatus.hidden = true;
  settingsApiKey.value = "";

  let providerData = null;
  let isOnline = true;

  try {
    providerData = await getProviderStatus();
    await storageService.set(STORAGE_KEY_CONFIG, providerData);
    await storageService.set(STORAGE_KEY_ACTIVATED, Boolean(providerData.is_activated));
  } catch (err) {
    isOnline = false;
    providerData = await storageService.get(STORAGE_KEY_CONFIG);
  }

  const isActivated = Boolean(providerData?.is_activated);
  const activeProvider = providerData?.active_provider || "none";
  const activeProviderInfo = providerData?.providers?.[activeProvider] || {};

  // Status Badge
  settingsStatusBadge.className = "badge";
  if (!isOnline) {
    settingsStatusBadge.textContent = "NEEDS ATTENTION";
    settingsStatusBadge.classList.add("badge-attention");
    currentProviderStateText.textContent = "Backend server unreachable";
  } else if (isActivated) {
    settingsStatusBadge.textContent = "ACTIVE";
    settingsStatusBadge.classList.add("badge-active");
    currentProviderStateText.textContent = "Active & Verified";
  } else {
    settingsStatusBadge.textContent = "NOT CONFIGURED";
    settingsStatusBadge.classList.add("badge-inactive");
    currentProviderStateText.textContent = "Not Configured";
  }

  // Current Provider Card Details
  currentProviderName.textContent = getProviderDisplayName(activeProvider);

  if (activeProviderInfo.model && activeProvider !== "none") {
    currentModelRow.hidden = false;
    currentProviderModel.textContent = activeProviderInfo.model;
  } else {
    currentModelRow.hidden = true;
  }

  if (activeProvider === "ollama") {
    currentKeyRow.hidden = false;
    currentProviderKey.textContent = activeProviderInfo.base_url || "http://localhost:11434";
  } else if (activeProviderInfo.masked_key) {
    currentKeyRow.hidden = false;
    currentProviderKey.textContent = activeProviderInfo.masked_key;
  } else {
    currentKeyRow.hidden = true;
  }

  // Default selection in edit dropdown
  if (providerData?.providers && providerData.providers[activeProvider]) {
    settingsProviderSelect.value = activeProvider;
  }
  updateSettingsFormFields();
}

settingsOpenBtn?.addEventListener("click", async () => {
  showView("settings-view");
  await renderSettingsView();
});

settingsBackBtn?.addEventListener("click", () => {
  showView("main-view");
});

settingsTestBtn?.addEventListener("click", async () => {
  const provider = settingsProviderSelect.value;
  const apiKey = settingsApiKey.value.trim();
  const ollamaUrl = settingsOllamaUrl.value.trim();
  const ollamaModel = settingsOllamaModel.value.trim();

  settingsError.hidden = true;
  settingsStatus.hidden = true;

  settingsTestBtn.disabled = true;
  settingsTestBtn.textContent = "Testing...";

  try {
    const res = await testProviderConnection({
      provider,
      apiKey: apiKey || null,
      model: provider === "ollama" ? ollamaModel : null,
      baseUrl: provider === "ollama" ? ollamaUrl : null,
    });
    settingsStatus.textContent = `✓ ${res.message || "Connection verified successfully!"}`;
    settingsStatus.hidden = false;
  } catch (err) {
    settingsError.textContent = err.message || "Connection test failed.";
    settingsError.hidden = false;
  } finally {
    settingsTestBtn.disabled = false;
    settingsTestBtn.textContent = "Test Connection";
  }
});

settingsSaveBtn?.addEventListener("click", async () => {
  const provider = settingsProviderSelect.value;
  const apiKey = settingsApiKey.value.trim();
  const ollamaUrl = settingsOllamaUrl.value.trim();
  const ollamaModel = settingsOllamaModel.value.trim();

  settingsError.hidden = true;
  settingsStatus.hidden = true;

  settingsSaveBtn.disabled = true;
  settingsSaveBtn.textContent = "Saving...";

  try {
    const res = await configureProvider({
      provider,
      apiKey: apiKey || null,
      model: provider === "ollama" ? ollamaModel : null,
      baseUrl: provider === "ollama" ? ollamaUrl : null,
      testFirst: true,
    });

    // Update persistent state
    await storageService.set(STORAGE_KEY_ACTIVATED, true);
    await storageService.set(STORAGE_KEY_CONFIG, res);
    await storageService.set(STORAGE_KEY_LAST_VERIFIED, Date.now());

    settingsStatus.textContent = `✓ Successfully activated ${getProviderDisplayName(provider)}!`;
    settingsStatus.hidden = false;
    settingsApiKey.value = "";

    await renderSettingsView();
  } catch (err) {
    settingsError.textContent = err.message || "Failed to switch provider.";
    settingsError.hidden = false;
  } finally {
    settingsSaveBtn.disabled = false;
    settingsSaveBtn.textContent = "Save & Activate";
  }
});

settingsRemoveBtn?.addEventListener("click", async () => {
  const selectedProvider = settingsProviderSelect.value;
  const confirmed = window.confirm(
    `Are you sure you want to remove credentials for ${getProviderDisplayName(selectedProvider)}? ` +
      `You will need to activate MemoryMind again to use AI features.`
  );
  if (!confirmed) return;

  settingsRemoveBtn.disabled = true;
  settingsRemoveBtn.textContent = "Removing...";
  settingsError.hidden = true;
  settingsStatus.hidden = true;

  try {
    const res = await removeProviderCredentials(selectedProvider);
    await storageService.set(STORAGE_KEY_CONFIG, res);

    if (!res.is_activated) {
      await storageService.set(STORAGE_KEY_ACTIVATED, false);
      showView("activation-view");
      updateActivationFormFields();
      activationApiKey.value = "";
    } else {
      await renderSettingsView();
      settingsStatus.textContent = `Credentials for ${getProviderDisplayName(selectedProvider)} removed.`;
      settingsStatus.hidden = false;
    }
  } catch (err) {
    settingsError.textContent = err.message || "Failed to remove credentials.";
    settingsError.hidden = false;
  } finally {
    settingsRemoveBtn.disabled = false;
    settingsRemoveBtn.textContent = "Remove Provider Credentials";
  }
});

// --- Hydration & Initialization ---------------------------------------------

async function initializeApp() {
  showView("loading-view");

  // Read local persisted state
  const wasActivated = await storageService.get(STORAGE_KEY_ACTIVATED, false);

  try {
    const statusData = await getProviderStatus();
    const isAct = Boolean(statusData?.is_activated);

    await storageService.set(STORAGE_KEY_ACTIVATED, isAct);
    await storageService.set(STORAGE_KEY_CONFIG, statusData);

    if (isAct) {
      showView("main-view");
      checkActivePageStatus();
    } else {
      showView("activation-view");
      updateActivationFormFields();
    }
  } catch (err) {
    console.warn("[MemoryMind] Server check failed on popup open:", err);
    if (wasActivated) {
      // User was previously activated; preserve dashboard experience
      showView("main-view");
      showError("MemoryMind backend is offline. Start the backend server to ask questions.");
      checkActivePageStatus();
    } else {
      // First-run without backend
      showView("activation-view");
      updateActivationFormFields();
      activationError.textContent = "MemoryMind backend is offline. Start the backend server and try again.";
      activationError.hidden = false;
    }
  }
}

// Start hydration immediately
initializeApp();

console.log("[MemoryMind] popup initialization complete. Button handlers attached:", {
  ask: !!askButton,
  rememberText: !!rememberButton,
  rememberPage: !!rememberPageButton,
  summarize: !!summarizeMainButton,
  settings: !!settingsOpenBtn,
});



