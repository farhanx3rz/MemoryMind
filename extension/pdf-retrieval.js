// PDF retrieval service for MemoryMind extension
// Handles both hosted PDFs and local file PDFs

export class PDFRetrievalService {
  constructor() {
    this.maxSize = 50 * 1024 * 1024; // 50MB
  }
  
  /**
   * Check if URL is a PDF
   */
  isPdfUrl(url) {
    if (!url) return false;
    try {
      const parsed = new URL(url);
      // Check extension
      if (parsed.pathname.toLowerCase().endsWith('.pdf')) return true;
      // Check content-type indicators in URL
      if (parsed.search.toLowerCase().includes('.pdf')) return true;
      if (parsed.pathname.includes('/pdf/')) return true;
      // Chrome's PDF viewer
      if (url.includes('chrome-extension://') && url.includes('.pdf')) return true;
      return false;
    } catch {
      return false;
    }
  }
  
  /**
   * Check if URL is a file:// URL
   */
  isFileUrl(url) {
    if (!url) return false;
    try {
      const parsed = new URL(url);
      return parsed.protocol === 'file:';
    } catch {
      return false;
    }
  }
  
  /**
   * Get filename from URL
   */
  getFilenameFromUrl(url) {
    if (!url) return 'document.pdf';
    try {
      const parsed = new URL(url);
      const pathname = parsed.pathname;
      const filename = pathname.split('/').pop();
      if (filename && filename.toLowerCase().endsWith('.pdf')) {
        return decodeURIComponent(filename);
      }
      if (this.isFileUrl(url)) {
        // Try to extract from file URL
        const match = url.match(/\/([^\/]+\.pdf)(?:\?|#|$)/i);
        if (match) return match[1];
      }
      return 'document.pdf';
    } catch {
      return 'document.pdf';
    }
  }
  
  /**
   * Check if extension has file access permission
   */
  async hasFileAccess() {
    if (typeof chrome === 'undefined') return false;
    if (chrome.extension && typeof chrome.extension.isAllowedFileSchemeAccess === 'function') {
      return new Promise((resolve) => {
        try {
          chrome.extension.isAllowedFileSchemeAccess((isAllowed) => {
            resolve(!!isAllowed);
          });
        } catch {
          resolve(false);
        }
      });
    }
    return false;
  }
  
  /**
   * Retrieve PDF bytes from active tab
   */
  async retrievePdfFromActiveTab() {
    if (typeof chrome === 'undefined' || !chrome.tabs) {
      throw new Error('Chrome APIs not available');
    }
    
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab) {
      throw new Error('No active tab');
    }
    
    const url = tab.url;
    if (!url) {
      throw new Error('No URL for active tab');
    }
    
    // Check if it's a PDF
    if (!this.isPdfUrl(url)) {
      throw new Error('Active tab is not a PDF');
    }
    
    // Handle file URLs
    if (this.isFileUrl(url)) {
      const allowed = await this.hasFileAccess();
      if (!allowed) {
        throw new Error('FILE_ACCESS_REQUIRED');
      }
      
      try {
        let arrayBuffer = null;

        // 1. Try standard fetch
        try {
          const response = await fetch(url);
          if (response.ok || response.status === 0) {
            arrayBuffer = await response.arrayBuffer();
          }
        } catch (fetchErr) {
          // fetch can fail on file:// in some Chrome environments; fall back to XHR below
        }

        // 2. Fall back to XMLHttpRequest if fetch didn't return a buffer
        if (!arrayBuffer || arrayBuffer.byteLength === 0) {
          arrayBuffer = await new Promise((resolve, reject) => {
            try {
              const xhr = new XMLHttpRequest();
              xhr.open('GET', url, true);
              xhr.responseType = 'arraybuffer';
              xhr.onload = () => {
                if (xhr.response && xhr.response.byteLength > 0) {
                  resolve(xhr.response);
                } else {
                  reject(new Error('Empty file content'));
                }
              };
              xhr.onerror = () => reject(new Error('Failed to fetch file'));
              xhr.send();
            } catch (xhrErr) {
              reject(xhrErr);
            }
          });
        }

        if (!arrayBuffer || arrayBuffer.byteLength === 0) {
          throw new Error('Could not read PDF file content');
        }

        if (arrayBuffer.byteLength > this.maxSize) {
          throw new Error(`File too large (max ${Math.round(this.maxSize/1024/1024)}MB)`);
        }
        return {
          bytes: arrayBuffer,
          filename: this.getFilenameFromUrl(url),
          url: url,
          type: 'local'
        };
      } catch (err) {
        throw new Error(`Cannot access local PDF. Enable "Allow access to file URLs" in extension settings or upload the file manually. Error: ${err.message}`);
      }
    }
    
    // Handle HTTP/HTTPS PDFs (including Chrome PDF viewer)
    if (url.startsWith('http://') || url.startsWith('https://')) {
      // Check if it's Chrome's PDF viewer
      if (url.includes('chrome-extension://') || url.includes('extension/pdf')) {
        // Try to get the actual PDF URL
        // For Chrome's PDF viewer, we need to extract the original URL
        try {
          const results = await chrome.scripting.executeScript({
            target: { tabId: tab.id },
            func: () => {
              // Try to find the PDF URL in the viewer
              const embed = document.querySelector('embed[type="application/pdf"]');
              if (embed && embed.src) return embed.src;
              const iframe = document.querySelector('iframe[src*=".pdf"]');
              if (iframe && iframe.src) return iframe.src;
              // Check if URL contains the PDF
              const params = new URLSearchParams(location.search);
              if (params.get('file')) return params.get('file');
              return location.href;
            }
          });
          
          const actualUrl = results?.[0]?.result || url;
          return await this.fetchPdf(actualUrl);
        } catch (err) {
          return await this.fetchPdf(url);
        }
      }
      
      return await this.fetchPdf(url);
    }
    
    throw new Error('Unsupported URL scheme');
  }
  
  /**
   * Fetch PDF from URL
   */
  async fetchPdf(url) {
    try {
      const response = await fetch(url, {
        method: 'GET',
        headers: {
          'Accept': 'application/pdf,*/*'
        }
      });
      
      if (!response.ok) {
        throw new Error(`Failed to fetch PDF: ${response.status}`);
      }
      
      const contentType = response.headers.get('content-type') || '';
      if (!contentType.includes('pdf') && !url.toLowerCase().endsWith('.pdf')) {
        // Check if it's actually a PDF
        const arrayBuffer = await response.arrayBuffer();
        const bytes = new Uint8Array(arrayBuffer);
        // PDF signature
        if (bytes[0] === 0x25 && bytes[1] === 0x50 && bytes[2] === 0x44 && bytes[3] === 0x46) {
          if (arrayBuffer.byteLength > this.maxSize) {
            throw new Error(`File too large (max ${Math.round(this.maxSize/1024/1024)}MB)`);
          }
          return {
            bytes: arrayBuffer,
            filename: this.getFilenameFromUrl(url),
            url: url,
            type: 'remote'
          };
        } else {
          throw new Error('URL does not contain a PDF');
        }
      }
      
      const arrayBuffer = await response.arrayBuffer();
      if (arrayBuffer.byteLength > this.maxSize) {
        throw new Error(`File too large (max ${Math.round(this.maxSize/1024/1024)}MB)`);
      }
      
      return {
        bytes: arrayBuffer,
        filename: this.getFilenameFromUrl(url),
        url: url,
        type: 'remote'
      };
    } catch (err) {
      throw new Error(`Failed to retrieve PDF: ${err.message}`);
    }
  }
}

export const pdfService = new PDFRetrievalService();
