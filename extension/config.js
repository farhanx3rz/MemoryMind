// Shared settings. There is no LLM API key here on purpose.
// The extension only ever talks to the local FastAPI server, and the
// server is the only thing that knows about the LLM.

export const API_BASE = "http://127.0.0.1:8000";

// Optional local authentication token. If the backend is started with
// MEMORYMIND_API_KEY or API_AUTH_TOKEN, set the matching token string here.
export const API_AUTH_TOKEN = "";

export const NO_MEMORY_ANSWER = "I don't have a memory about that.";
