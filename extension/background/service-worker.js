// Background service worker: owns the right-click MemoryMind actions.

import { rememberMemory, rememberWebpage } from "../api.js";

const SELECTION_MENU_ID = "memorymind-remember-selection";
const PAGE_MENU_ID = "memorymind-remember-page";
const BADGE_COLORS = {
  working: "#f5a623",
  success: "#2e9e4f",
  failure: "#d33f3f",
};

function setBadge(text, colorKey) {
  chrome.action.setBadgeBackgroundColor({ color: BADGE_COLORS[colorKey] });
  chrome.action.setBadgeText({ text });
}

function clearBadge() {
  chrome.action.setBadgeText({ text: "" });
}

chrome.runtime.onInstalled.addListener(() => {
  // Rebuild the menus on install/update so the fixed IDs never accumulate
  // duplicate entries across extension reloads.
  chrome.contextMenus.removeAll(() => {
    chrome.contextMenus.create({
      id: SELECTION_MENU_ID,
      title: "Remember selected text with MemoryMind",
      contexts: ["selection"],
    });
    chrome.contextMenus.create({
      id: PAGE_MENU_ID,
      title: "Remember this page with MemoryMind",
      contexts: ["page"],
    });
  });
});

async function getCurrentPageHTML(tabId) {
  let results;
  try {
    results = await chrome.scripting.executeScript({
      target: { tabId },
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

chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId === SELECTION_MENU_ID) {
    await rememberSelection(info, tab);
    return;
  }
  if (info.menuItemId === PAGE_MENU_ID) {
    await rememberPage(tab);
  }
});

function isSafeHttpUrl(url) {
  if (!url || typeof url !== "string") return false;
  try {
    const parsed = new URL(url);
    return parsed.protocol === "http:" || parsed.protocol === "https:";
  } catch {
    return false;
  }
}

async function rememberSelection(info, tab) {
  const text = (info.selectionText || "").trim();
  if (!text) {
    setBadge("!", "failure");
    setTimeout(clearBadge, 3000);
    return;
  }

  const url = tab?.url || info.pageUrl || "";
  if (url && !isSafeHttpUrl(url)) {
    console.warn("MemoryMind: Cannot save selection from non-http/https URL:", url);
    setBadge("!", "failure");
    setTimeout(clearBadge, 3000);
    return;
  }

  setBadge("...", "working");
  try {
    await rememberMemory(
      text,
      url,
      tab?.title || "",
    );
    setBadge("OK", "success");
  } catch (error) {
    console.warn("MemoryMind could not save the selection:", error.message);
    setBadge("ERR", "failure");
  }
  setTimeout(clearBadge, 4000);
}

async function rememberPage(tab) {
  if (!tab?.id) return;

  const tabUrl = tab?.url || "";
  if (tabUrl && !isSafeHttpUrl(tabUrl)) {
    console.warn("MemoryMind: Cannot save page from non-http/https URL:", tabUrl);
    setBadge("!", "failure");
    setTimeout(clearBadge, 3000);
    return;
  }

  setBadge("...", "working");
  try {
    const page = await getCurrentPageHTML(tab.id);
    if (!page?.html) throw new Error("No webpage HTML was available.");
    const pageUrl = page.url || tabUrl;
    if (pageUrl && !isSafeHttpUrl(pageUrl)) {
      throw new Error("Cannot save page from unsupported URL scheme.");
    }
    await rememberWebpage(
      pageUrl,
      page.title || tab.title || "",
      page.html,
    );
    setBadge("OK", "success");
  } catch (error) {
    console.warn("MemoryMind could not save the page:", error.message);
    setBadge("ERR", "failure");
  }
  setTimeout(clearBadge, 4000);
}
