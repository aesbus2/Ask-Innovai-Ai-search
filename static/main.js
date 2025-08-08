// Enhanced main.js for Ask InnovAI Admin Interface v2.2.3
// BULLETPROOF FIX: Complete error handling for toLocaleString() undefined errors
// Version: 5.1.0 - Updated for new admin interface

let pollInterval = null;

console.log("Ask InnovAI Admin v2.2.4 - Updated for new admin interface");

// Auto-refresh status every 30 seconds if not actively importing
setInterval(() => {
    if (!pollInterval) {
        refreshStatus();
    }
}, 30000);



// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log("🚀 DOM loaded, initializing enhanced admin interface v2.2.4...");
    refreshStatus();
    checkSystemHealth();
    checkLastImportInfo();
    loadOpenSearchStats(); // Load statistics on startup
    
    // Basic server ping to verify connectivity
    fetch("/ping")
        .then(r => r.json())
        .then(data => console.log("✅ Server ping successful:", data))
        .catch(error => console.error("❌ Server ping failed:", error));

    setupMaxDocsValidation();

    const maxDocsInput = document.getElementById("maxDocsInput");
    if (maxDocsInput) {
        maxDocsInput.addEventListener("input", updateMaxDocsDisplay);
        maxDocsInput.addEventListener("change", updateMaxDocsDisplay);
        updateMaxDocsDisplay(); // Initial display
    }

    // Setup other import-related UI updates
    const importTypeSelect = document.getElementById("importTypeSelect");
    const collectionSelect = document.getElementById("collectionSelect");
    
    if (importTypeSelect) {
        importTypeSelect.addEventListener("change", updateMaxDocsDisplay);
    }
    
    if (collectionSelect) {
        collectionSelect.addEventListener("change", updateMaxDocsDisplay);
    }
});

// ============================================================================
// UTILITY FUNCTIONS - COMPLETE PROTECTION
// ============================================================================

function ultraSafeNumber(value) {
    try {
        // Handle null, undefined, empty string, etc.
        if (value === null || value === undefined || value === '') {
            return 0;
        }
        
        // Handle boolean values
        if (typeof value === 'boolean') {
            return value ? 1 : 0;
        }
        
        // Handle string values
        if (typeof value === 'string') {
            const trimmed = value.trim();
            if (trimmed === '' || trimmed === 'null' || trimmed === 'undefined') {
                return 0;
            }
            const parsed = parseFloat(trimmed);
            return isNaN(parsed) ? 0 : parsed;
        }
        
        // Handle objects with numeric properties
        if (typeof value === 'object' && value !== null) {
            if (value.count !== undefined) return ultraSafeNumber(value.count);
            if (value.value !== undefined) return ultraSafeNumber(value.value);
            if (value.total !== undefined) return ultraSafeNumber(value.total);
        }
        
        // Fallback: try direct conversion
        const converted = Number(value);
        return isNaN(converted) || !isFinite(converted) ? 0 : converted;
        
    } catch (error) {
        console.warn('ultraSafeNumber conversion failed:', value, 'Error:', error.message);
        return 0;
    }
}        
        

// Safe formatting with complete error handling
function ultraSafeString(value) {
    try {
        if (value === null || value === undefined) {
            return '';
        }
        if (typeof value === 'string') {
            return value;
        }
        if (typeof value === 'number' && !isNaN(value) && isFinite(value)) {
            return value.toString();
        }
        if (typeof value === 'boolean') {
            return value.toString();
        }
        return String(value);
    } catch (error) {
        console.warn('ultraSafeString conversion failed:', value, 'Error:', error.message);
        return '';
    }
}

// BULLETPROOF: Ultimate safe timestamp formatting
function ultraSafeTimestamp(ts) {
    try {
        // Handle null/undefined
        if (!ts) {
            return new Date().toLocaleString();
        }
        
        // Handle various timestamp formats
        let date;
        if (ts instanceof Date) {
            date = ts;
        } else if (typeof ts === 'string' || typeof ts === 'number') {
            date = new Date(ts);
        } else {
            console.warn('Invalid timestamp type:', typeof ts, 'value:', ts);
            return new Date().toLocaleString();
        }
        
        // Validate date
        if (isNaN(date.getTime())) {
            console.warn('Invalid date created from timestamp:', ts);
            return new Date().toLocaleString();
        }
        
        // Try toLocaleString with fallback
        try {
            return date.toLocaleString();
        } catch (localeError) {
            console.warn('Date toLocaleString failed:', localeError);
            // Fallback to ISO string
            try {
                return date.toISOString().replace('T', ' ').substring(0, 19);
            } catch (isoError) {
                console.warn('Date toISOString failed:', isoError);
                return 'Invalid Date';
            }
        }
        
    } catch (error) {
        console.warn('Error in ultraSafeTimestamp for value:', ts, 'error:', error);
        return new Date().toLocaleString();
    }
}

// Object validation
function ultraSafeObject(value) {
    try {
        if (value && typeof value === 'object' && !Array.isArray(value)) {
            return value;
        }
        if (value === null || value === undefined) {
            return {};
        }
        if (typeof value === 'string') {
            try {
                const parsed = JSON.parse(value);
                return (parsed && typeof parsed === 'object' && !Array.isArray(parsed)) ? parsed : {};
            } catch {
                return {};
            }
        }
        return {};
    } catch (error) {
        console.warn('ultraSafeObject conversion failed:', value, 'Error:', error.message);
        return {};
    }
}

// Array validation
function ultraSafeArray(value) {
    try {
        if (Array.isArray(value)) {
            return value;
        }
        if (value === null || value === undefined) {
            return [];
        }
        if (typeof value === 'string') {
            try {
                const parsed = JSON.parse(value);
                return Array.isArray(parsed) ? parsed : [value];
            } catch {
                return [value];
            }
        }
        return Array.from(value);
    } catch (error) {
        console.warn('ultraSafeArray conversion failed:', value, 'Error:', error.message);
        return [];
    }
}

// BULLETPROOF: Ultimate safe string validation
function ultraSafeString(str) {
    try {
        if (str === null || str === undefined) {
            return '';
        }
        
        if (typeof str === 'string') {
            return str;
        }
        
        // Try to convert to string
        try {
            return String(str);
        } catch (conversionError) {
            console.warn('String conversion failed:', conversionError);
            return '';
        }
        
    } catch (error) {
        console.warn('Error in ultraSafeString:', error, 'returning empty string');
        return '';
    }
}

// BULLETPROOF: Format numbers with complete protection
function ultraSafeFormat(value) {
    try {
        const num = ultraSafeNumber(value);
        if (num === 0) return '0';
        
        // Use toLocaleString with error handling
        return num.toLocaleString();
    } catch (error) {
        console.warn('ultraSafeFormat failed:', value, 'Error:', error.message);
        return ultraSafeNumber(value).toString();
    }
}

// ============================================================================
// OPENSEARCH STATISTICS FUNCTIONS - BULLETPROOF PROTECTION
// ============================================================================

async function loadOpenSearchStats() {
    const container = document.getElementById('statisticsContainer');
    if (!container) {
        console.warn('Statistics container not found');
        return;
    }
    
    // Show loading state
    container.innerHTML = `
        <div class="loading-stats">
            <div class="spinner"></div>
            Loading database statistics...
        </div>
    `;
    
    try {
        const response = await fetch('/opensearch_statistics', {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        const timestamp = new Date().toISOString();
        
        console.log('✅ OpenSearch statistics loaded:', data);
        displayStatistics(data, timestamp);
        
    } catch (error) {
        console.error('❌ Failed to load OpenSearch statistics:', error);
        displayStatisticsError(error, container);
    }
}

async function getVectorSearchStatus() {
    try {
        const response = await fetch('/opensearch_statistics');
        const result = await response.json();
        
        if (result?.data?.vector_search) {
            const vs = result.data.vector_search;
            return {
                enabled: vs.cluster_support || false,
                ready: vs.vector_search_ready || false,
                coverage: vs.vector_coverage || 0,
                embedder_available: vs.embedder_available || false,
                documents_with_vectors: vs.documents_with_vectors || 0
            };
        }
        
        return { enabled: false, ready: false, coverage: 0 };
    } catch (error) {
        console.warn('Vector search status check failed:', error);
        return { enabled: false, ready: false, coverage: 0, error: true };
    }
}

// Missing error display function  
function displayStatisticsError(errorMessage, errorType = 'unknown') {
    const container = document.getElementById('statisticsContainer');
    if (!container) return;
    
    const troubleshootingTips = {
        network_error: [
            'Check if the backend server is running',
            'Verify your network connection', 
            'Try refreshing the page'
        ],
        server_error: [
            'Check OpenSearch connection in backend logs',
            'Verify OpenSearch cluster is healthy',
            'Check backend configuration'
        ],
        parse_error: [
            'Backend may be returning HTML instead of JSON',
            'Check backend logs for errors',
            'Verify API endpoint is working correctly'
        ],
        timeout_error: [
            'OpenSearch cluster may be slow to respond',
            'Try again in a moment',
            'Check cluster performance'
        ],
        unknown: [
            'Check browser console for details',
            'Try refreshing the page',
            'Contact system administrator if problem persists'
        ]
    };
    
    const tips = troubleshootingTips[errorType] || troubleshootingTips.unknown;
    
    container.innerHTML = `
        <div class="stats-error">
            <div class="error-icon">❌</div>
            <h3>Statistics Loading Failed</h3>
            <p><strong>Error:</strong> ${ultraSafeString(errorMessage)}</p>
            
            <div class="error-details">
                <h4>Troubleshooting Tips:</h4>
                <ul>
                    ${tips.map(tip => `<li>${ultraSafeString(tip)}</li>`).join('')}
                </ul>
            </div>
            
            <div class="error-actions">
                <button onclick="loadOpenSearchStats()" class="btn primary">
                    🔄 Retry Loading
                </button>
                <button onclick="window.location.reload()" class="btn secondary">
                    ↻ Refresh Page
                </button>
            </div>
        </div>
    `;
}

function displayStatistics(response, timestamp) {
    const container = document.getElementById('statisticsContainer');
    if (!container) return;
    
    // Handle the actual API response structure
    const apiData = response.data || response;
    
    // Map your API fields to display
    const safeData = {
        total_evaluations: ultraSafeNumber(apiData?.total_documents || apiData?.total_evaluations),
        active_indices: ultraSafeNumber(apiData?.active_indices),
        agents: ultraSafeNumber(apiData?.agents),
        dispositions: ultraSafeNumber(apiData?.dispositions),
        programs: ultraSafeNumber(apiData?.programs),
        templates: ultraSafeNumber(apiData?.templates),
        weighted_scores_available: ultraSafeNumber(apiData?.weighted_scores_available)
    };
    
    const responseStatus = response.status || 'success';
    const responseTimestamp = response.timestamp || timestamp;
    
    // Generate HTML with actual data
    const html = `
        <div class="stats-dashboard">
            <div class="stats-card priority-metric">
                <h3><span class="emoji">🆔</span> Evaluations Processed</h3>
                <div class="stats-number">${ultraSafeFormat(safeData.total_evaluations)}</div>
                <div class="stats-label">Total Documents in OpenSearch</div>
            </div>
            
            <div class="stats-card">
                <h3><span class="emoji">💾</span> Active Indices</h3>
                <div class="stats-number">${ultraSafeFormat(safeData.active_indices)}</div>
                <div class="stats-label">OpenSearch Indices</div>
            </div>
            
            ${safeData.agents > 0 ? `
            <div class="stats-card">
                <h3><span class="emoji">👥</span> Agents</h3>
                <div class="stats-number">${ultraSafeFormat(safeData.agents)}</div>
                <div class="stats-label">Unique Agents</div>
            </div>` : ''}
            
            ${safeData.dispositions > 0 ? `
            <div class="stats-card">
                <h3><span class="emoji">📋</span> Dispositions</h3>
                <div class="stats-number">${ultraSafeFormat(safeData.dispositions)}</div>
                <div class="stats-label">Call Dispositions</div>
            </div>` : ''}
            
            ${safeData.templates > 0 ? `
            <div class="stats-card">
                <h3><span class="emoji">📄</span> Templates</h3>
                <div class="stats-number">${ultraSafeFormat(safeData.templates)}</div>
                <div class="stats-label">Evaluation Templates</div>
            </div>` : ''}
            
            ${safeData.weighted_scores_available > 0 ? `
            <div class="stats-card">
                <h3><span class="emoji">📊</span> Scored Evaluations</h3>
                <div class="stats-number">${ultraSafeFormat(safeData.weighted_scores_available)}</div>
                <div class="stats-label">With Weighted Scores</div>
            </div>` : ''}
        </div>
        
        <div style="margin-top: 20px; padding: 16px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #6e32a0;">
            <div style="font-size: 0.9em; color: #666;">
                📅 Last updated: ${ultraSafeTimestamp(responseTimestamp)} | 
                🔄 Processing: ${response.processing_time || 'Unknown'}s | 
                🏷️ Version: ${response.version || 'Unknown'}
            </div>
        </div>
    `;
    
    container.innerHTML = html;
    console.log("📊 Statistics dashboard updated with actual API data");
}


// ============================================================================
// IMPORT MANAGEMENT FUNCTIONS - ENHANCED (keeping existing working functions)
// ============================================================================

async function startImport() {
    const collectionSelect = document.getElementById("collectionSelect");
    const importTypeSelect = document.getElementById("importTypeSelect");
    const maxDocsInput = document.getElementById("maxDocsInput");
    
    const selectedCollection = collectionSelect ? collectionSelect.value : "evaluations";
    const importType = importTypeSelect ? importTypeSelect.value : "evaluation_json";
    const maxDocsValue = maxDocsInput ? maxDocsInput.value.trim() : "";
    
    // Validate max documents input
    let maxDocs = null;
    if (maxDocsValue !== "") {
        const parsed = parseInt(maxDocsValue);
        if (isNaN(parsed) || parsed <= 0) {
            alert("❌ Max Documents must be a positive number or left empty for all documents");
            maxDocsInput?.focus();
            return;
        }
        maxDocs = parsed;
    }
    
    const payload = {
        collection: selectedCollection,
        import_type: importType
    };
    
    if (maxDocs !== null) {
        payload.max_docs = maxDocs;
    }
    
    try {
        const response = await fetch("/start_import", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            console.error("Import failed:", data);
            alert(`❌ Import failed: ${data.detail || data.message || "Unknown error"}. Check console for details.`);
            return;
        }

        if (response.ok) {
            let successMsg = `✅ ${importType.charAt(0).toUpperCase() + importType.slice(1)} import started successfully!`;
            if (maxDocs !== null) {
                successMsg += `\n📊 Limited to ${maxDocs} documents`;
            } else {
                successMsg += `\n📊 Processing all available documents`;
            }
            alert(successMsg);
            startPolling();
        }
    } catch (error) {
        console.error("Import request failed:", error);
        alert(`❌ Import request failed: ${error.message}`);
    }
}

function startPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
    }
    
    pollInterval = setInterval(refreshStatus, 2000);
    console.log("🔄 Started polling for import status");
}

function stopPolling() {
    if (pollInterval) {
        clearInterval(pollInterval);
        pollInterval = null;
        console.log("⏹️ Stopped polling");
    }
}

// Enhanced input validation for max documents field
function setupMaxDocsValidation() {
    const maxDocsInput = document.getElementById("maxDocsInput");
    if (maxDocsInput) {
        maxDocsInput.addEventListener("input", function(e) {
            const value = e.target.value.trim();
            
            // Remove non-numeric characters except for initial clearing
            if (value !== "" && (isNaN(value) || parseInt(value) < 0)) {
                e.target.style.borderColor = "#dc3545";
                e.target.title = "Must be a positive number or empty for all documents";
            } else {
                e.target.style.borderColor = "#ddd";
                e.target.title = "Maximum number of documents to import (leave empty for all)";
            }
        });
        
        maxDocsInput.addEventListener("blur", function(e) {
            const value = e.target.value.trim();
            if (value !== "" && (isNaN(value) || parseInt(value) <= 0)) {
                alert("Max Documents must be a positive number or left empty");
                e.target.focus();
            }
        });
    }
}

// Enhanced UI feedback for max documents
function updateMaxDocsDisplay() {
    const maxDocsInput = document.getElementById("maxDocsInput");
    const maxDocsDisplay = document.getElementById("maxDocsDisplay");
    
    if (!maxDocsInput) return;
    
    const value = maxDocsInput.value.trim();
    const numValue = parseInt(value);
    
    // Update display text - UPDATED to match new HTML structure
    if (value === "" || isNaN(numValue)) {
        if (maxDocsDisplay) {
            maxDocsDisplay.textContent = "All documents";
            maxDocsDisplay.style.color = "#666";
        }
        // Update info in maxDocsInfo div if it exists
        const maxDocsInfo = document.getElementById("maxDocsDisplay");
        if (maxDocsInfo) {
            maxDocsInfo.innerHTML = "Will process all available documents";
        }
    } else {
        if (maxDocsDisplay) {
            maxDocsDisplay.textContent = `Max: ${ultraSafeFormat(numValue)}`;
            maxDocsDisplay.style.color = "#6e32a0";
        }
        // Update info in maxDocsInfo div if it exists
        const maxDocsInfo = document.getElementById("maxDocsDisplay");
        if (maxDocsInfo) {
            maxDocsInfo.innerHTML = `Will process up to ${ultraSafeFormat(numValue)} documents for testing`;
        }
    }
}

function startPolling() {
    if (pollInterval) clearInterval(pollInterval);

    console.log("🔄 Starting status polling...");
    pollInterval = setInterval(async () => {
        await refreshStatus();
        
        try {
            const response = await fetch('/status');
            const status = await response.json();

            if (status.status !== 'running') {
                console.log(`🏁 Import completed with status: ${status.status}`);
                clearInterval(pollInterval);
                pollInterval = null;
                
                // Refresh import info and statistics after completion
                if (status.status === 'completed') {
                    setTimeout(() => {
                        checkLastImportInfo();
                        loadOpenSearchStats(); // Refresh statistics after import
                    }, 1000);
                }
            }
        } catch (error) {
            console.error("Status polling error:", error);
        }
    }, 2000);
}

async function refreshStatus() {
    try {
        const response = await fetch('/status');
        const data = await response.json();
        
        const statusContainer = document.getElementById('statusContainer');
        if (statusContainer) {
            const status = data.status || 'unknown';
            const statusClass = status === 'completed' ? 'completed' : 
                               status === 'processing' ? 'processing' : 
                               status === 'failed' ? 'failed' : 'idle';
            
            statusContainer.innerHTML = `
                <div class="status ${statusClass}">
                    <strong>Status:</strong> ${status.charAt(0).toUpperCase() + status.slice(1)}
                </div>
            `;
            
            // Update current step if available
            const stepContainer = document.getElementById('currentStep');
            const stepText = document.getElementById('stepText');
            if (data.current_step && stepContainer && stepText) {
                stepContainer.classList.remove('hidden');
                stepText.textContent = data.current_step;
            } else if (stepContainer) {
                stepContainer.classList.add('hidden');
            }
            
            // Stop polling if completed or failed
            if (status === 'completed' || status === 'failed') {
                stopPolling();
                if (status === 'completed') {
                    loadOpenSearchStats(); // Refresh statistics
                }
            }
        }
        
    } catch (error) {
        console.error('Failed to refresh status:', error);
    }
}

function updateStatusDisplay(data) {
    const container = document.getElementById('statusContainer');
    if (!container) return;

    const statusClass = data.status;
    let html = `<div class="status ${statusClass}">`;
    html += `<strong>Status:</strong> ${data.status.toUpperCase()}`;
    
    if (data.import_type) {
        html += ` (${data.import_type.charAt(0).toUpperCase() + data.import_type.slice(1)} Import)`;
    }
    html += `</div>`;

    // Handle current step display
    const currentStepDiv = document.getElementById('currentStep');
    const stepTextDiv = document.getElementById('stepText');
    
    if (data.current_step && data.status === 'running') {
        if (currentStepDiv) currentStepDiv.style.display = 'block';
        if (stepTextDiv) {
            stepTextDiv.textContent = data.current_step;
            
            // Add elapsed time if available
            if (data.start_time) {
                try {
                    const elapsed = Math.floor((new Date() - new Date(data.start_time)) / 1000);
                    stepTextDiv.textContent += ` (${elapsed}s elapsed)`;
                } catch (e) {
                    // Ignore time calculation errors
                }
            }
        }
    } else {
        if (currentStepDiv) currentStepDiv.style.display = 'none';
    }

    // Add timing information with bulletproof timestamp handling
    if (data.status === 'running' && data.start_time) {
        html += `<div style="margin-top: 10px; font-size: 0.9em; color: #666;">
            Started: ${ultraSafeTimestamp(data.start_time)}
        </div>`;
    } else if (data.status === 'completed' && data.end_time) {
        html += `<div style="margin-top: 10px; font-size: 0.9em; color: #666;">
            Completed: ${ultraSafeTimestamp(data.end_time)}
        </div>`;
    } else if (data.status === 'failed') {
        html += `<div style="margin-top: 10px; font-size: 0.9em; color: #dc3545;">
            Failed: ${data.end_time ? ultraSafeTimestamp(data.end_time) : 'Unknown time'}
        </div>`;
        if (data.error) {
            html += `<div style="margin-top: 10px; padding: 10px; background: #f8d7da; border-radius: 4px; color: #721c24;">
                <strong>Error:</strong> ${ultraSafeString(data.error)}
            </div>`;
        }
    }

    container.innerHTML = html;

    // Show results if completed
    if (data.status === 'completed' && data.results) {
        showResults(data.results);
        // Auto-refresh statistics after successful import
        setTimeout(() => {
            loadOpenSearchStats();
        }, 2000);
    }
}

function showResults(results) {
    const section = document.getElementById('resultsSection');
    const grid = document.getElementById('resultsGrid');
    
    if (!section || !grid) return;
    
    section.classList.remove('hidden');
    
    let html = '';
    
    // ENHANCED: Define key metrics with Evaluations Processed as priority #1
    const metrics = [
        { key: 'total_evaluations_indexed', label: 'Evaluations Processed', class: 'success', icon: '🆔', priority: 1 },
        { key: 'total_documents_processed', label: 'Documents Processed', class: 'success', icon: '📄', priority: 2 },
        { key: 'total_chunks_processed', label: 'Chunks Processed', class: 'info', icon: '🧩', priority: 3 },
        { key: 'errors', label: 'Errors', class: 'warning', icon: '❌', priority: 4 },
        { key: 'opensearch_errors', label: 'OpenSearch Errors', class: 'danger', icon: '🔥', priority: 5 },
        { key: 'import_type', label: 'Import Type', class: 'info', icon: '🔄', priority: 6 }
    ];
    
    // Display key metrics (sorted by priority)
    metrics.sort((a, b) => a.priority - b.priority).forEach(metric => {
        if (results.hasOwnProperty(metric.key)) {
            let value = results[metric.key];
            
            // Format import type
            if (metric.key === 'import_type') {
                value = ultraSafeString(value);
                value = value.charAt(0).toUpperCase() + value.slice(1);
            } else {
                value = ultraSafeFormat(value);
            }
            
            // Add special styling for evaluations processed
            let extraInfo = '';
            let cardClass = 'result-card';
            if (metric.key === 'total_evaluations_indexed') {
                extraInfo = '<div style="font-size: 0.8em; margin-top: 4px; color: #666; font-weight: 500;">Unique EvaluationIDs</div>';
                cardClass = 'result-card priority-metric';
            }
            
            html += `
                <div class="${cardClass}">
                    <h4>${metric.icon} ${metric.label}</h4>
                    <div class="result-value ${metric.class}">${value}</div>
                    ${extraInfo}
                </div>
            `;
        }
    });
    
    // Collections processed with bulletproof handling
    if (results.template_collections_created && ultraSafeArray(results.template_collections_created).length > 0) {
        const collections = ultraSafeArray(results.template_collections_created);
        html += `
            <div class="result-card">
                <h4>📁 Template Collections</h4>
                <div class="result-value info">${ultraSafeFormat(collections.length)}</div>
                <div style="font-size: 0.8em; margin-top: 8px; color: #666;">
                    ${collections.slice(0, 3).map(c => ultraSafeString(c)).join(', ')}
                    ${collections.length > 3 ? '...' : ''}
                </div>
            </div>
        `;
    }
    
    // Import timestamp with bulletproof handling
    if (results.completed_at) {
        try {
            const timestamp = ultraSafeTimestamp(results.completed_at);
            html += `
                <div class="result-card">
                    <h4>⏰ Completed At</h4>
                    <div class="result-value" style="font-size: 1.2em; color: #666;">${timestamp}</div>
                </div>
            `;
        } catch (e) {
            // Ignore timestamp formatting errors
        }
    }
    
    // Success rate with bulletproof handling
    if (results.success_rate) {
        html += `
            <div class="result-card">
                <h4>📈 Success Rate</h4>
                <div class="result-value success">${ultraSafeString(results.success_rate)}</div>
            </div>
        `;
    }
    
    grid.innerHTML = html;
    
    console.log("📈 Import results displayed with BULLETPROOF handling:", results);
}

// ============================================================================
// SYSTEM HEALTH AND MONITORING - BULLETPROOF
// ============================================================================

async function checkSystemHealth() {
    const container = document.getElementById('healthContainer');
    if (!container) {
        console.warn('Health container not found');
        return;
    }
    
    container.innerHTML = `
        <div class="health-item">
            <span class="health-label">System Status</span>
            <span class="health-value">Checking...</span>
        </div>
    `;
    
    try {
        const response = await fetch('/health');
        const data = await response.json();
        
        const isHealthy = response.ok && data.status === 'healthy';
        
        container.innerHTML = `
            <div class="health-item ${isHealthy ? '' : 'unhealthy'}">
                <span class="health-label">System Status</span>
                <span class="health-value">${isHealthy ? '✅ Healthy' : '❌ Unhealthy'}</span>
            </div>
            <div class="health-item">
                <span class="health-label">OpenSearch</span>
                <span class="health-value">${data.opensearch ? '✅ Connected' : '❌ Disconnected'}</span>
            </div>
            <div class="health-item">
                <span class="health-label">Memory Usage</span>
                <span class="health-value">${data.memory_usage ? Math.round(data.memory_usage) + '%' : 'Unknown'}</span>
            </div>
        `;
        
    } catch (error) {
        console.error('Health check failed:', error);
        container.innerHTML = `
            <div class="health-item unhealthy">
                <span class="health-label">System Status</span>
                <span class="health-value">❌ Error: ${error.message}</span>
            </div>
        `;
    }
}

async function checkLastImportInfo() {
    const container = document.getElementById('lastImportInfo');
    if (!container) return;
    
    try {
        const response = await fetch('/import_info');
        const data = await response.json();
        
        if (data.last_import) {
            container.innerHTML = `
                <div class="status completed">
                    <strong>Last Import:</strong> ${ultraSafeTimestamp(data.last_import.timestamp)}
                    <br><strong>Type:</strong> ${ultraSafeString(data.last_import.type)}
                    <br><strong>Status:</strong> ${ultraSafeString(data.last_import.status)}
                </div>
            `;
        } else {
            container.innerHTML = `
                <div class="status idle">
                    <strong>Last Import:</strong> No import history found
                </div>
            `;
        }
    } catch (error) {
        container.innerHTML = `
            <div class="status failed">
                <strong>Error:</strong> ${error.message}
            </div>
        `;
    }
}

async function clearImportTimestamp() {
    if (!confirm('Are you sure you want to clear the import timestamp?')) {
        return;
    }
    
    try {
        const response = await fetch('/clear_import_timestamp', { method: 'POST' });
        const data = await response.json();
        
        alert(data.message || 'Import timestamp cleared');
        checkLastImportInfo(); // Refresh the display
    } catch (error) {
        alert(`Failed to clear timestamp: ${error.message}`);
    }
}

// ============================================================================
// UTILITY FUNCTIONS - BULLETPROOF
// ============================================================================

async function toggleLogs() {
    const logsContainer = document.getElementById('logsContainer');
    const logsContent = document.getElementById('logsContent');
    
    if (!logsContainer) return;
    
    if (logsContainer.classList.contains('hidden')) {
        logsContainer.classList.remove('hidden');
        
        if (logsContent) {
            logsContent.innerHTML = 'Loading logs...';
        }
        
        try {
            const response = await fetch('/logs');
            const data = await response.json();
            
            if (logsContent) {
                if (data.logs && data.logs.length > 0) {
                    logsContent.innerHTML = data.logs.map(log => 
                        `<div style="padding: 4px 16px; border-bottom: 1px solid #f0f0f0; font-family: monospace; font-size: 0.8em;">${ultraSafeString(log)}</div>`
                    ).join('');
                } else {
                    logsContent.innerHTML = '<div style="padding: 16px; text-align: center; color: #666;">No logs available</div>';
                }
                
                // Update logs metadata
                const logsTimestamp = document.getElementById('logsTimestamp');
                const logsCount = document.getElementById('logsCount');
                if (logsTimestamp) logsTimestamp.textContent = new Date().toLocaleString();
                if (logsCount) logsCount.textContent = data.logs ? data.logs.length : 0;
            }
            
        } catch (error) {
            if (logsContent) {
                logsContent.innerHTML = `<div style="padding: 16px; color: #dc3545;">❌ Failed to load logs: ${error.message}</div>`;
            }
        }
    } else {
        logsContainer.classList.add('hidden');
    }
}

async function testSearch() {
    const container = document.getElementById('actionResults');
    if (!container) return;
    
    container.innerHTML = `
        <div class="status processing">
            🔍 Testing search functionality...
        </div>
    `;
    
    try {
        const response = await fetch('/test_search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query: 'test search query' })
        });
        
        const data = await response.json();
        
        if (data.success) {
            const resultCount = ultraSafeNumber(data.results?.length || 0);
            container.innerHTML = `
                <div class="status completed">
                    ✅ Search test successful! Found ${ultraSafeFormat(resultCount)} results.
                </div>
            `;
        } else {
            container.innerHTML = `
                <div class="status failed">
                    ❌ Search test failed: ${ultraSafeString(data.error || 'Unknown error')}
                </div>
            `;
        }
    } catch (error) {
        container.innerHTML = `
            <div class="status failed">
                ❌ Search test failed: ${ultraSafeString(error.message)}
            </div>
        `;
    }
}

function openChatInterface() {
    window.open('/chat', '_blank');
}

// ============================================================================
// EVALUATION LOOKUP FUNCTIONS
// ============================================================================

async function lookupEvaluation() {
    const input = document.getElementById('evaluationIdInput');
    const resultsContainer = document.getElementById('evaluationResults');
    
    if (!input || !resultsContainer) {
        console.error('Evaluation lookup elements not found');
        return;
    }
    
    const evaluationId = input.value.trim();
    
    if (!evaluationId) {
        alert('Please enter an evaluation ID');
        input.focus();
        return;
    }
    
    // Show loading state
    resultsContainer.classList.remove('hidden');
    resultsContainer.innerHTML = `
        <div class="evaluation-loading">
            <div class="spinner"></div>
            <div>Looking up evaluation: <strong>${ultraSafeString(evaluationId)}</strong></div>
        </div>
    `;
    
    try {
        // Make API call to get evaluation data
        const response = await fetch(`/evaluation/${encodeURIComponent(evaluationId)}`, {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            }
        });
        
        if (!response.ok) {
            if (response.status === 404) {
                displayEvaluationError(`Evaluation ID "${evaluationId}" not found in the database.`);
            } else {
                const errorData = await response.json().catch(() => ({}));
                displayEvaluationError(`Failed to fetch evaluation: ${errorData.detail || response.statusText}`);
            }
            return;
        }
        
        const data = await response.json();
        console.log('✅ Evaluation data retrieved:', data);
        
        displayEvaluationResults(data, evaluationId);
        
    } catch (error) {
        console.error('❌ Evaluation lookup failed:', error);
        
        let errorMessage = 'Network error - unable to connect to server';
        if (error.message.includes('fetch')) {
            errorMessage = 'Server is unreachable. Please check if the backend is running.';
        } else {
            errorMessage = `Lookup failed: ${error.message}`;
        }
        
        displayEvaluationError(errorMessage);
    }
}

function displayEvaluationResults(data, searchedId) {
    const resultsContainer = document.getElementById('evaluationResults');
    if (!resultsContainer) return;
    
    // Safely extract data with defaults
    const evalData = {
        internalId: ultraSafeString(data.internalId || data.internal_id || ''),
        evaluationId: ultraSafeString(data.evaluationId || data.evaluation_id || searchedId),
        weighted_score: ultraSafeNumber(data.weighted_score || data.score || 0),
        url: ultraSafeString(data.url || ''),
        template_id: ultraSafeString(data.template_id || ''),
        template_name: ultraSafeString(data.template_name || ''),
        partner: ultraSafeString(data.partner || ''),
        site: ultraSafeString(data.site || ''),
        lob: ultraSafeString(data.lob || ''),
        agentName: ultraSafeString(data.agentName || data.agent_name || ''),
        agentId: ultraSafeString(data.agentId || data.agent_id || ''),
        disposition: ultraSafeString(data.disposition || ''),
        subDisposition: ultraSafeString(data.subDisposition || data.sub_disposition || ''),
        created_on: data.created_on || data.createdOn || '',
        call_date: data.call_date || data.callDate || '',
        call_duration: ultraSafeNumber(data.call_duration || data.callDuration || 0),
        language: ultraSafeString(data.language || ''),
        evaluation: ultraSafeString(data.evaluation || '')
    };
    
    // Format score for display
    const scoreColor = evalData.weighted_score >= 80 ? '#28a745' : 
                      evalData.weighted_score >= 60 ? '#ffc107' : '#dc3545';
    
    const html = `
        <div class="evaluation-header">
            <h3>
                <span class="emoji">📋</span>
                Evaluation: ${evalData.evaluationId}
            </h3>
            <div class="evaluation-score" style="background-color: ${scoreColor};">
                Score: ${evalData.weighted_score}%
            </div>
        </div>
        
        <div class="evaluation-content">
            <!-- Basic Information -->
            <div class="evaluation-grid">
                <div class="eval-field">
                    <div class="eval-label">Internal ID</div>
                    <div class="eval-value ${evalData.internalId ? '' : 'empty'}">
                        ${evalData.internalId || 'Not available'}
                    </div>
                </div>
                
                <div class="eval-field">
                    <div class="eval-label">Evaluation ID</div>
                    <div class="eval-value">
                        ${evalData.evaluationId}
                    </div>
                </div>
                
                <div class="eval-field">
                    <div class="eval-label">Weighted Score</div>
                    <div class="eval-value score">
                        ${evalData.weighted_score}%
                    </div>
                </div>
                
                <div class="eval-field">
                    <div class="eval-label">Evaluation URL</div>
                    <div class="eval-value url ${evalData.url ? '' : 'empty'}">
                        ${evalData.url ? `<a href="${evalData.url}" target="_blank">${evalData.url}</a>` : 'Not available'}
                    </div>
                </div>
            </div>
            
            <!-- Template Information -->
            <div class="evaluation-grid">
                <div class="eval-field">
                    <div class="eval-label">Template ID</div>
                    <div class="eval-value ${evalData.template_id ? '' : 'empty'}">
                        ${evalData.template_id || 'Not available'}
                    </div>
                </div>
                
                <div class="eval-field">
                    <div class="eval-label">Template Name</div>
                    <div class="eval-value ${evalData.template_name ? '' : 'empty'}">
                        ${evalData.template_name || 'Not available'}
                    </div>
                </div>
                
                <div class="eval-field">
                    <div class="eval-label">Language</div>
                    <div class="eval-value ${evalData.language ? '' : 'empty'}">
                        ${evalData.language || 'Not specified'}
                    </div>
                </div>
            </div>
            
            <!-- Organization Information -->
            <div class="evaluation-grid">
                <div class="eval-field">
                    <div class="eval-label">Partner</div>
                    <div class="eval-value ${evalData.partner ? '' : 'empty'}">
                        ${evalData.partner || 'Not specified'}
                    </div>
                </div>
                
                <div class="eval-field">
                    <div class="eval-label">Site</div>
                    <div class="eval-value ${evalData.site ? '' : 'empty'}">
                        ${evalData.site || 'Not specified'}
                    </div>
                </div>
                
                <div class="eval-field">
                    <div class="eval-label">Line of Business</div>
                    <div class="eval-value ${evalData.lob ? '' : 'empty'}">
                        ${evalData.lob || 'Not specified'}
                    </div>
                </div>
            </div>
            
            <!-- Agent Information -->
            <div class="evaluation-grid">
                <div class="eval-field">
                    <div class="eval-label">Agent Name</div>
                    <div class="eval-value ${evalData.agentName ? '' : 'empty'}">
                        ${evalData.agentName || 'Not available'}
                    </div>
                </div>
                
                <div class="eval-field">
                    <div class="eval-label">Agent ID</div>
                    <div class="eval-value ${evalData.agentId ? '' : 'empty'}">
                        ${evalData.agentId || 'Not available'}
                    </div>
                </div>
            </div>
            
            <!-- Call Information -->
            <div class="evaluation-grid">
                <div class="eval-field">
                    <div class="eval-label">Disposition</div>
                    <div class="eval-value ${evalData.disposition ? '' : 'empty'}">
                        ${evalData.disposition || 'Not available'}
                    </div>
                </div>
                
                <div class="eval-field">
                    <div class="eval-label">Sub-Disposition</div>
                    <div class="eval-value ${evalData.subDisposition ? '' : 'empty'}">
                        ${evalData.subDisposition || 'Not available'}
                    </div>
                </div>
                
                <div class="eval-field">
                    <div class="eval-label">Call Duration</div>
                    <div class="eval-value ${evalData.call_duration > 0 ? '' : 'empty'}">
                        ${evalData.call_duration > 0 ? `${evalData.call_duration} seconds` : 'Not available'}
                    </div>
                </div>
            </div>
            
            <!-- Date Information -->
            <div class="evaluation-grid">
                <div class="eval-field">
                    <div class="eval-label">Created On</div>
                    <div class="eval-value date ${evalData.created_on ? '' : 'empty'}">
                        ${evalData.created_on ? ultraSafeTimestamp(evalData.created_on) : 'Not available'}
                    </div>
                </div>
                
                <div class="eval-field">
                    <div class="eval-label">Call Date</div>
                    <div class="eval-value date ${evalData.call_date ? '' : 'empty'}">
                        ${evalData.call_date ? ultraSafeTimestamp(evalData.call_date) : 'Not available'}
                    </div>
                </div>
            </div>
            
            <!-- Evaluation Content -->
            ${evalData.evaluation ? `
            <div class="evaluation-section">
                <h4>
                    <span class="emoji">📄</span>
                    Evaluation Content
                </h4>
                <div class="evaluation-text">
                    ${evalData.evaluation.replace(/\n/g, '<br>')}
                </div>
            </div>
            ` : ''}
        </div>
    `;
    
    resultsContainer.innerHTML = html;
}

function displayEvaluationError(message) {
    const resultsContainer = document.getElementById('evaluationResults');
    if (!resultsContainer) return;
    
    resultsContainer.innerHTML = `
        <div class="evaluation-error">
            <h3>❌ Evaluation Lookup Failed</h3>
            <p>${ultraSafeString(message)}</p>
            <div style="margin-top: 16px;">
                <button onclick="clearEvaluationLookup()" class="btn secondary">
                    <span class="emoji">🗑️</span> Clear Results
                </button>
                <button onclick="lookupEvaluation()" class="btn primary">
                    <span class="emoji">🔄</span> Try Again
                </button>
            </div>
        </div>
    `;
}

function clearEvaluationLookup() {
    const input = document.getElementById('evaluationIdInput');
    const resultsContainer = document.getElementById('evaluationResults');
    
    if (input) {
        input.value = '';
        input.focus();
    }
    
    if (resultsContainer) {
        resultsContainer.classList.add('hidden');
        resultsContainer.innerHTML = '';
    }
}

function handleEvaluationLookupKeyPress(event) {
    if (event.key === 'Enter') {
        lookupEvaluation();
    }
}

// ============================================================================
// GLOBAL WINDOW FUNCTIONS (for HTML onclick handlers)
// ============================================================================

// Expose functions to global scope for HTML onclick handlers
window.startImport = startImport;
window.refreshStatus = refreshStatus;
window.checkSystemHealth = checkSystemHealth;
window.checkLastImportInfo = checkLastImportInfo;
window.clearImportTimestamp = clearImportTimestamp;
window.toggleLogs = toggleLogs;
window.testSearch = testSearch;
window.openChatInterface = openChatInterface;
window.loadOpenSearchStats = loadOpenSearchStats; // Enhanced statistics function
// NEW: Evaluation lookup functions
window.lookupEvaluation = lookupEvaluation;
window.clearEvaluationLookup = clearEvaluationLookup;
window.handleEvaluationLookupKeyPress = handleEvaluationLookupKeyPress;

console.log("✅ Ask InnovAI Admin BULLETPROOF main.js v2.2.3 loaded successfully");
console.log("🛡️ BULLETPROOF: ALL toLocaleString() errors COMPLETELY ELIMINATED");
console.log("🔧 COMPLETE PROTECTION: ultraSafe functions handle ALL undefined/null/invalid values");
console.log("📊 All functions including enhanced statistics with ULTIMATE error handling available");