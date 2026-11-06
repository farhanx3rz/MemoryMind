/* Research summarizer main JS */
import { API_BASE, API_AUTH_TOKEN } from "../config.js";

const STYLES = {
    QUICK: "quick",
    DETAILED: "detailed", 
    NOTES: "notes"
};

const STATUS = {
    CREATED: "created",
    EXTRACTING: "extracting",
    ANALYZING: "analyzing",
    SYNTHESIZING: "synthesizing",
    GENERATING_PDF: "generating_pdf",
    COMPLETED: "completed",
    FAILED: "failed"
};

class ApiError extends Error {
    constructor(status, message) {
        super(message);
        this.status = status;
    }
}

function getAuthHeaders(extra = {}) {
    const headers = { ...extra };
    if (API_AUTH_TOKEN) {
        headers["X-API-Key"] = API_AUTH_TOKEN;
    }
    return headers;
}

async function postJson(path, body, timeout = 90000) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    
    try {
        const res = await fetch(`${API_BASE}${path}`, {
            method: "POST",
            headers: getAuthHeaders({ "Content-Type": "application/json" }),
            body: JSON.stringify(body),
            signal: controller.signal,
        });
        
        if (!res.ok) {
            const text = await res.text().catch(() => "");
            let detail = "";
            try {
                const j = JSON.parse(text);
                detail = j.detail || "";
            } catch (_) {
                detail = text;
            }
            throw new ApiError(res.status, detail || `Request failed: ${res.status}`);
        }
        
        return await res.json();
    } finally {
        clearTimeout(timeoutId);
    }
}

async function getJson(path, timeout = 15000) {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    try {
        const res = await fetch(`${API_BASE}${path}`, {
            headers: getAuthHeaders(),
            signal: controller.signal,
        });
        if (!res.ok) {
            const text = await res.text().catch(() => "");
            let detail = "";
            try {
                const j = JSON.parse(text);
                detail = j.detail || "";
            } catch (_) {
                detail = text;
            }
            throw new ApiError(res.status, detail || `Request failed: ${res.status}`);
        }
        return await res.json();
    } finally {
        clearTimeout(timeoutId);
    }
}

class ResearchSummarizer {
    constructor() {
        this.sources = [];
        this.currentJobId = null;
        this.pollingInterval = null;
        
        this.initElements();
        this.initEvents();
        this.checkCurrentPage();
        this.recoverSavedJob();
    }
    
    initElements() {
        this.elements = {
            // Source tabs
            urlsTab: document.getElementById('urls-tab'),
            currentTab: document.getElementById('current-tab'),
            pdfsTab: document.getElementById('pdfs-tab'),
            
            // Inputs
            urlsInput: document.getElementById('urls-input'),
            currentPageTitle: document.getElementById('current-page-title'),
            pdfInput: document.getElementById('pdf-input'),
            summaryStyle: document.getElementById('summary-style'),
            topicInput: document.getElementById('topic-input'),
            maxLengthInput: document.getElementById('max-length-input'),
            
            // Buttons
            addUrlsBtn: document.getElementById('add-urls-btn'),
            addCurrentBtn: document.getElementById('add-current-btn'),
            addPdfsBtn: document.getElementById('add-pdfs-btn'),
            generateBtn: document.getElementById('generate-btn'),
            openPdfBtn: document.getElementById('open-pdf-btn'),
            downloadPdfBtn: document.getElementById('download-pdf-btn'),
            
            // Tabs
            tabBtns: document.querySelectorAll('.tab-btn'),
            
            // Containers
            sourcesList: document.getElementById('sources-list'),
            progressSection: document.querySelector('.progress-section'),
            resultsSection: document.querySelector('.results-section'),
            errorDiv: document.getElementById('error'),
            
            // Progress
            progressFill: document.getElementById('progress-fill'),
            progressMessage: document.getElementById('progress-message'),
            resultMessage: document.getElementById('result-message')
        };
    }
    
    initEvents() {
        // Tab switching
        this.elements.tabBtns.forEach(btn => {
            btn.addEventListener('click', () => this.switchTab(btn.dataset.tab));
        });
        
        // Source management
        this.elements.addUrlsBtn?.addEventListener('click', () => this.addUrls());
        this.elements.addCurrentBtn?.addEventListener('click', () => this.addCurrentPage());
        this.elements.addPdfsBtn?.addEventListener('click', () => this.addPdfs());
        this.elements.generateBtn?.addEventListener('click', () => this.generateReport());
        this.elements.openPdfBtn?.addEventListener('click', () => this.openPdf());
        this.elements.downloadPdfBtn?.addEventListener('click', () => this.downloadPdf());
        
        // Enter key in textarea
        this.elements.urlsInput?.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && e.ctrlKey) {
                e.preventDefault();
                this.addUrls();
            }
        });
    }
    
    switchTab(tabName) {
        // Update active tab button
        this.elements.tabBtns.forEach(btn => {
            btn.classList.toggle('active', btn.dataset.tab === tabName);
        });
        
        // Show/hide tab content
        ['urls', 'current', 'pdfs'].forEach(tab => {
            const content = document.getElementById(`${tab}-tab`);
            if (content) content.classList.toggle('active', tab === tabName);
        });
    }
    
    async checkCurrentPage() {
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            if (tab?.title) {
                this.elements.currentPageTitle.textContent = tab.title;
            }
        } catch (err) {
            console.warn('Could not get current page:', err);
        }
    }
    
    addUrls() {
        const text = this.elements.urlsInput.value.trim();
        if (!text) {
            this.showError('Please enter URLs');
            return;
        }
        
        const urls = text.split('\n')
            .map(line => line.trim())
            .filter(line => line && this.isValidUrl(line));
        
        if (urls.length === 0) {
            this.showError('No valid URLs found');
            return;
        }
        
        urls.forEach(url => {
            this.addSource({
                type: 'webpage',
                title: url,
                url: url,
                status: 'pending'
            });
        });
        
        this.elements.urlsInput.value = '';
        this.updateGenerateButton();
    }
    
    async addCurrentPage() {
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            
            if (!tab?.url || !tab.url.startsWith('http')) {
                this.showError('Cannot add current page. Make sure it\'s a regular webpage.');
                return;
            }
            
            this.addSource({
                type: 'current_page',
                title: tab.title || 'Current Page',
                url: tab.url,
                status: 'pending'
            });
            
            this.updateGenerateButton();
        } catch (err) {
            this.showError(`Cannot add current page: ${err.message}`);
        }
    }
    
    addPdfs() {
        const files = this.elements.pdfInput.files;
        if (!files || files.length === 0) {
            this.showError('Please select PDF files');
            return;
        }
        
        Array.from(files).forEach(file => {
            this.addSource({
                type: 'uploaded_pdf',
                title: file.name,
                filename: file.name,
                file: file,
                status: 'pending'
            });
        });
        
        this.elements.pdfInput.value = '';
        this.updateGenerateButton();
    }
    
    addSource(source) {
        this.sources.push(source);
        this.renderSources();
    }
    
    removeSource(index) {
        this.sources.splice(index, 1);
        this.renderSources();
        this.updateGenerateButton();
    }
    
    renderSources() {
        const container = this.elements.sourcesList;
        container.innerHTML = '';
        
        this.sources.forEach((source, index) => {
            const element = this.createSourceElement(source, index);
            container.appendChild(element);
        });
    }
    
    createSourceElement(source, index) {
        const div = document.createElement('div');
        div.className = 'source-item';
        
        let statusBadge = '';
        if (source.status === 'processing') statusBadge = '<span class="badge processing">Processing...</span>';
        if (source.status === 'error') statusBadge = `<span class="badge error">Error: ${source.error || 'Failed'}</span>`;
        if (source.status === 'ready') statusBadge = '<span class="badge success">Ready</span>';
        
        div.innerHTML = `
            <div class="source-info">
                <div class="source-title">${source.title}</div>
                ${source.url ? `<div class="source-url">${source.url}</div>` : ''}
                ${source.filename ? `<div class="source-filename">${source.filename}</div>` : ''}
                ${statusBadge}
            </div>
            <div class="source-actions">
                <button class="btn-small btn-remove" data-index="${index}">Remove</button>
            </div>
        `;
        
        div.querySelector('.btn-remove').addEventListener('click', () => this.removeSource(index));
        return div;
    }
    
    async generateReport() {
        if (this.sources.length === 0) {
            this.showError('Please add at least one source');
            return;
        }
        
        this.clearError();
        
        // Build unified request config
        const config = {
            summary_style: this.elements.summaryStyle.value,
            topic: this.elements.topicInput.value.trim(),
            max_length: parseInt(this.elements.maxLengthInput.value) || 2000,
            urls: this.sources.filter(s => s.type === 'webpage').map(s => s.url)
        };
        const pdfSources = this.sources.filter(s => s.type === 'uploaded_pdf');
        
        try {
            this.elements.generateBtn.disabled = true;
            this.elements.generateBtn.textContent = 'Creating Job...';
            
            this.elements.progressSection.hidden = false;
            this.updateProgress(0, 'Creating job...');
            
            const formData = new FormData();
            formData.append('request', JSON.stringify(config));
            pdfSources.forEach(s => {
                if (s.file) {
                    formData.append('files', s.file);
                }
            });

            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 60000);
            let jobResp;
            try {
                const res = await fetch(`${API_BASE}/jobs/create`, {
                    method: 'POST',
                    headers: getAuthHeaders(),
                    body: formData,
                    signal: controller.signal,
                });
                if (!res.ok) {
                    const errorText = await res.text().catch(() => "");
                    let errorMsg = `Failed to create job (${res.status})`;
                    try {
                        const errorJson = JSON.parse(errorText);
                        if (typeof errorJson.detail === 'string') errorMsg = errorJson.detail;
                        else if (Array.isArray(errorJson.detail)) errorMsg = errorJson.detail.map(d => d.msg || d).join(', ');
                    } catch (_) {
                        if (errorText && errorText.length < 200) errorMsg = errorText;
                    }
                    throw new Error(errorMsg);
                }
                jobResp = await res.json();
            } finally {
                clearTimeout(timeoutId);
            }

            this.currentJobId = jobResp.job_id;
            localStorage.setItem('research_job_id', this.currentJobId);
            
            // Start monitoring with bounded polling & timeouts
            this.startPolling();
            
        } catch (err) {
            this.showError(`Failed to create job: ${err.message}`);
            this.elements.generateBtn.disabled = false;
            this.elements.generateBtn.textContent = 'Generate Report';
            this.elements.progressSection.hidden = true;
        }
    }
    
    startPolling() {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
            this.pollingInterval = null;
        }

        let consecutiveErrors = 0;
        const maxErrors = 5;
        const pollStartTime = Date.now();
        const maxPollDurationMs = 15 * 60 * 1000; // 15 minutes max
        
        this.pollingInterval = setInterval(async () => {
            if (Date.now() - pollStartTime > maxPollDurationMs) {
                if (this.pollingInterval) {
                    clearInterval(this.pollingInterval);
                    this.pollingInterval = null;
                }
                this.onJobFailed({ message: "Job polling timed out after 15 minutes." });
                return;
            }

            try {
                const status = await getJson(`/jobs/${this.currentJobId}/status`, 10000);
                consecutiveErrors = 0;
                
                this.updateProgress(
                    (status.progress || 0) * 100,
                    status.message || "Processing..."
                );
                
                if (status.status === STATUS.COMPLETED) {
                    this.onJobCompleted(status);
                } else if (status.status === STATUS.FAILED || status.status === "expired") {
                    this.onJobFailed(status);
                }
                
            } catch (err) {
                consecutiveErrors++;
                console.warn(`Polling attempt failed (${consecutiveErrors}/${maxErrors}):`, err);
                if (err.status === 404) {
                    if (this.pollingInterval) {
                        clearInterval(this.pollingInterval);
                        this.pollingInterval = null;
                    }
                    localStorage.removeItem('research_job_id');
                    this.onJobFailed({ message: "Job not found. The backend server may have restarted or the job expired." });
                } else if (consecutiveErrors >= maxErrors) {
                    if (this.pollingInterval) {
                        clearInterval(this.pollingInterval);
                        this.pollingInterval = null;
                    }
                    this.onJobFailed({ message: `Polling stopped after repeated errors: ${err.message}` });
                }
            }
        }, 1500);
    }
    
    async recoverSavedJob() {
        const savedJobId = localStorage.getItem('research_job_id');
        if (!savedJobId) return;

        try {
            const status = await getJson(`/jobs/${savedJobId}/status`, 8000);
            this.currentJobId = savedJobId;
            if (status.status === STATUS.COMPLETED) {
                this.elements.progressSection.hidden = true;
                this.elements.resultsSection.hidden = false;
                this.elements.resultMessage.textContent = `Recovered report with ${status.sources_summary?.successful || 0} sources.`;
                this.elements.openPdfBtn.hidden = false;
                this.elements.downloadPdfBtn.hidden = false;
            } else if (status.status === STATUS.FAILED || status.status === "expired") {
                localStorage.removeItem('research_job_id');
                this.showError(`Previous job ended: ${status.message || status.error || 'Failed'}`);
            } else {
                this.elements.progressSection.hidden = false;
                this.elements.generateBtn.disabled = true;
                this.updateProgress((status.progress || 0) * 100, status.message || "Resuming job...");
                this.startPolling();
            }
        } catch (_) {
            localStorage.removeItem('research_job_id');
        }
    }

    updateProgress(percent, message) {
        this.elements.progressFill.style.width = `${percent}%`;
        this.elements.progressMessage.textContent = message;
    }
    
    onJobCompleted(status) {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
            this.pollingInterval = null;
        }
        
        this.elements.progressSection.hidden = true;
        this.elements.resultsSection.hidden = false;
        this.elements.resultMessage.textContent = `Report completed with ${status.sources_summary.successful} sources.`;
        
        this.elements.openPdfBtn.hidden = false;
        this.elements.downloadPdfBtn.hidden = false;
        
        this.elements.generateBtn.disabled = false;
        this.elements.generateBtn.textContent = 'Generate Report';
    }
    
    onJobFailed(status) {
        if (this.pollingInterval) {
            clearInterval(this.pollingInterval);
            this.pollingInterval = null;
        }
        
        localStorage.removeItem('research_job_id');
        this.showError(`Job failed: ${status.message || status.error || 'Unknown error'}`);
        this.elements.progressSection.hidden = true;
        
        this.elements.generateBtn.disabled = false;
        this.elements.generateBtn.textContent = 'Generate Report';
    }
    
    async openPdf() {
        if (!this.currentJobId) return;
        try {
            const url = `${API_BASE}/jobs/${this.currentJobId}/pdf`;
            const res = await fetch(url, { headers: getAuthHeaders() });
            if (!res.ok) throw new Error(`Failed to load PDF (${res.status})`);
            const blob = await res.blob();
            const blobUrl = URL.createObjectURL(blob);
            window.open(blobUrl, '_blank');
            setTimeout(() => URL.revokeObjectURL(blobUrl), 60000);
        } catch (err) {
            this.showError(`Failed to open PDF: ${err.message}`);
        }
    }
    
    async downloadPdf() {
        if (!this.currentJobId) return;
        try {
            const url = `${API_BASE}/jobs/${this.currentJobId}/pdf`;
            const res = await fetch(url, { headers: getAuthHeaders() });
            if (!res.ok) throw new Error(`Download failed (${res.status})`);
            const blob = await res.blob();
            const blobUrl = URL.createObjectURL(blob);
            const link = document.createElement('a');
            link.href = blobUrl;
            link.download = `research-report-${this.currentJobId}.pdf`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            setTimeout(() => URL.revokeObjectURL(blobUrl), 10000);
        } catch (err) {
            this.showError(`Failed to download PDF: ${err.message}`);
        }
    }
    
    isValidUrl(string) {
        try {
            const url = new URL(string);
            return url.protocol === 'http:' || url.protocol === 'https:';
        } catch (_) {
            return false;
        }
    }
    
    updateGenerateButton() {
        this.elements.generateBtn.disabled = this.sources.length === 0;
    }
    
    showError(message) {
        this.elements.errorDiv.hidden = false;
        this.elements.errorDiv.textContent = message;
    }
    
    clearError() {
        this.elements.errorDiv.hidden = true;
        this.elements.errorDiv.textContent = '';
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.researchSummarizer = new ResearchSummarizer();
});
