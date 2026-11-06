// All communication with the FastAPI backend lives here, so the popup and
// the background service worker behave the same way.

import { API_BASE, API_AUTH_TOKEN, NO_MEMORY_ANSWER } from "./config.js";

const OFFLINE_MESSAGE =
  "MemoryMind backend is offline. Start the FastAPI server and try again.";

// Thrown when we cannot reach the server at all.
export class OfflineError extends Error {}

// Thrown when the server answered, but with an error status.
export class ApiError extends Error {
  constructor(status, message) {
    super(message);
    this.status = status;
  }
}

// Turn a FastAPI error response into something worth showing a person.
function readableDetail(payload, status) {
  const detail = payload?.detail;

  if (typeof detail === "string" && detail) {
    return detail;
  }
  if (Array.isArray(detail) && detail.length > 0) {
    return detail[0].msg || "That request was not valid.";
  }
  if (status === 401) {
    return "Authentication failed. Check your API_AUTH_TOKEN configuration.";
  }
  if (status === 403) {
    return detail || "API key activation required. Please activate your provider API key.";
  }
  if (status === 408) {
    return "The request timed out while waiting for a response.";
  }
  if (status === 422) {
    return "That request was not valid.";
  }
  if (status === 429) {
    return "The configured language-model quota or rate limit was reached. Check the provider account.";
  }
  if (status === 502) {
    return "The language model could not be reached. Check the server logs.";
  }
  if (status === 504) {
    return "The language model request timed out. Try asking again.";
  }
  if (status >= 500) {
    return "The MemoryMind server had a problem. Check the server logs.";
  }
  return "The MemoryMind server rejected that request.";
}

async function post(path, body, timeoutMs = 45000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  const headers = { "Content-Type": "application/json" };
  if (API_AUTH_TOKEN) {
    headers["X-API-Key"] = API_AUTH_TOKEN;
  }

  let response;
  try {
    response = await fetch(`${API_BASE}${path}`, {
      method: "POST",
      headers,
      body: JSON.stringify(body),
      signal: controller.signal,
    });
  } catch (err) {
    if (err.name === "AbortError") {
      throw new ApiError(408, "The request timed out while waiting for a response.");
    }
    // fetch only rejects when the server is unreachable, which is the
    // common case during local development.
    throw new OfflineError(OFFLINE_MESSAGE);
  } finally {
    clearTimeout(timeoutId);
  }

  const payload = await response.json().catch(() => null);

  if (!response.ok) {
    throw new ApiError(response.status, readableDetail(payload, response.status));
  }
  return payload;
}

export function rememberMemory(text, sourceUrl, sourceTitle, visitedAt = new Date().toISOString()) {
  return post("/memory", {
    text,
    source_url: sourceUrl || null,
    source_title: sourceTitle || null,
    visited_at: visitedAt,
  });
}

export function askQuestion(question) {
  return post("/ask", { question });
}

export async function askQuestionStream(question, onToken, onDone, timeoutMs = 45000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  const headers = { "Content-Type": "application/json" };
  if (API_AUTH_TOKEN) {
    headers["X-API-Key"] = API_AUTH_TOKEN;
  }

  let response;
  try {
    response = await fetch(`${API_BASE}/ask/stream`, {
      method: "POST",
      headers,
      body: JSON.stringify({ question }),
      signal: controller.signal,
    });
  } catch (err) {
    if (err.name === "AbortError") {
      throw new ApiError(408, "The request timed out while waiting for a response.");
    }
    throw new OfflineError(OFFLINE_MESSAGE);
  } finally {
    clearTimeout(timeoutId);
  }

  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    throw new ApiError(response.status, readableDetail(payload, response.status));
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop();

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      try {
        const event = JSON.parse(trimmed);
        if (event.type === "token" && onToken) {
          onToken(event.token);
        } else if (event.type === "done" && onDone) {
          onDone(event);
        }
      } catch {
        // Ignore unparseable chunks
      }
    }
  }

  if (buffer.trim()) {
    try {
      const event = JSON.parse(buffer.trim());
      if (event.type === "done" && onDone) {
        onDone(event);
      }
    } catch {
      // Ignore
    }
  }
}

export function rememberWebpage(url, title, html, visitedAt = new Date().toISOString()) {
  return post("/webpage", {
    url,
    title: title || null,
    html,
    visited_at: visitedAt,
  });
}

export async function getDocumentTags(documentId) {
  if (!documentId) return { tags: [], topics: [], ready: false, saved: false };
  const headers = {};
  if (API_AUTH_TOKEN) {
    headers["X-API-Key"] = API_AUTH_TOKEN;
  }
  try {
    const res = await fetch(`${API_BASE}/memory/${encodeURIComponent(documentId)}/tags`, {
      method: "GET",
      headers,
    });
    if (!res.ok) return { tags: [], topics: [], ready: false, saved: false };
    return await res.json();
  } catch {
    return { tags: [], topics: [], ready: false, saved: false };
  }
}

export async function getTagsForUrl(url) {
  if (!url) return { tags: [], topics: [], ready: false, saved: false };
  const headers = {};
  if (API_AUTH_TOKEN) {
    headers["X-API-Key"] = API_AUTH_TOKEN;
  }
  try {
    const res = await fetch(`${API_BASE}/tags?url=${encodeURIComponent(url)}`, {
      method: "GET",
      headers,
    });
    if (res.ok) {
      return await res.json();
    }
  } catch {
    // Fall back to /memory/{url}/tags below
  }

  // Fallback to /memory/{url}/tags
  return await getDocumentTags(url);
}

export const storageService = {
  async get(key, defaultValue = null) {
    if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
      try {
        const result = await chrome.storage.local.get(key);
        return result[key] !== undefined ? result[key] : defaultValue;
      } catch (err) {
        console.warn("[MemoryMind] chrome.storage.local.get error:", err);
      }
    }
    try {
      const item = localStorage.getItem(key);
      return item !== null ? JSON.parse(item) : defaultValue;
    } catch {
      return defaultValue;
    }
  },

  async set(key, value) {
    if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
      try {
        await chrome.storage.local.set({ [key]: value });
      } catch (err) {
        console.warn("[MemoryMind] chrome.storage.local.set error:", err);
      }
    }
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch {
      // Ignore fallback storage write error
    }
  },

  async remove(key) {
    if (typeof chrome !== "undefined" && chrome.storage && chrome.storage.local) {
      try {
        await chrome.storage.local.remove(key);
      } catch (err) {
        console.warn("[MemoryMind] chrome.storage.local.remove error:", err);
      }
    }
    try {
      localStorage.removeItem(key);
    } catch {
      // Ignore fallback storage remove error
    }
  },
};

export async function checkServerHealth(timeoutMs = 3000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: controller.signal });
    if (!res.ok) return { ok: false };
    return { ok: true, ...(await res.json()) };
  } catch {
    return { ok: false };
  } finally {
    clearTimeout(timeoutId);
  }
}

export async function getProviderStatus(timeoutMs = 4000) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  const headers = {};
  if (API_AUTH_TOKEN) {
    headers["X-API-Key"] = API_AUTH_TOKEN;
  }
  try {
    const res = await fetch(`${API_BASE}/provider/status`, {
      method: "GET",
      headers,
      signal: controller.signal,
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new ApiError(res.status, data.detail || "Failed to retrieve provider status.");
    }
    return await res.json();
  } catch (error) {
    if (error.name === "AbortError") {
      throw new OfflineError("MemoryMind backend request timed out.");
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
}

export async function testProviderConnection({ provider, apiKey = null, model = null, baseUrl = null }) {
  const headers = { "Content-Type": "application/json" };
  if (API_AUTH_TOKEN) {
    headers["X-API-Key"] = API_AUTH_TOKEN;
  }
  const body = { provider };
  if (apiKey !== null && apiKey !== undefined) body.api_key = apiKey;
  if (model !== null && model !== undefined) body.model = model;
  if (baseUrl !== null && baseUrl !== undefined) body.base_url = baseUrl;

  const res = await fetch(`${API_BASE}/provider/test`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || "Connection test failed.");
  }
  return await res.json();
}

export async function configureProvider({ provider, apiKey = null, model = null, baseUrl = null, testFirst = true }) {
  const headers = { "Content-Type": "application/json" };
  if (API_AUTH_TOKEN) {
    headers["X-API-Key"] = API_AUTH_TOKEN;
  }
  const body = {
    provider,
    test_first: testFirst,
  };
  if (apiKey !== null && apiKey !== undefined) body.api_key = apiKey;
  if (model !== null && model !== undefined) body.model = model;
  if (baseUrl !== null && baseUrl !== undefined) body.base_url = baseUrl;

  const res = await fetch(`${API_BASE}/provider/configure`, {
    method: "POST",
    headers,
    body: JSON.stringify(body),
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || "Failed to configure provider.");
  }
  return await res.json();
}

export async function removeProviderCredentials(provider) {
  const headers = { "Content-Type": "application/json" };
  if (API_AUTH_TOKEN) {
    headers["X-API-Key"] = API_AUTH_TOKEN;
  }
  const res = await fetch(`${API_BASE}/provider/remove`, {
    method: "POST",
    headers,
    body: JSON.stringify({ provider }),
  });

  if (!res.ok) {
    const data = await res.json().catch(() => ({}));
    throw new Error(data.detail || "Failed to remove provider credentials.");
  }
  return await res.json();
}

export async function activateApiKey(apiKey, provider = null) {
  return await configureProvider({
    provider: provider || "grok",
    apiKey,
    testFirst: true,
  });
}



export { NO_MEMORY_ANSWER };
