// Enhanced main.js for Ask InnovAI Admin Interface v2.2.3
// BULLETPROOF FIX: Complete error handling for toLocaleString() undefined errors
// Version: 6.1.0 - Updated for new admin interface

let pollInterval = null
let isPolling = false;


console.log("Ask InnovAI Admin v2.2.4 - Updated for new admin interface");

// Auto-refresh status every 30 seconds if not actively importing
setInterval(() => {
    if (!pollInterval) {
        refreshStatus();
    }
}, 30000);



// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log("🚀 DOM loaded, initializing enhanced admin interface v6.1.0...");
    refreshStatus();
    checkSystemHealth();
    checkLastImportInfo();
    loadOpenSearchStats(); // Load statistics on startup
    setupDateRangeHandlers(); // Setup date range handlers for statistics
    
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
    console.log("📊 Loading OpenSearch statistics...");
    
    const container = document.getElementById('statisticsContainer');
    if (!container) {
        console.warn('❌ Statistics container not found - add id="statisticsContainer" to your HTML');
        return;
    }
    
    // Show loading state
    container.innerHTML = `
        <div class="stats-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;">
            <div class="stats-card">
                <h3><span class="emoji">🔄</span> Loading...</h3>
                <div class="stats-number">Please wait</div>
                <div class="stats-label">Fetching statistics</div>
            </div>
        </div>
    `;
    
    try {
        const response = await fetch('/opensearch_statistics');
        const data = await response.json();
        
        console.log("📊 Full OpenSearch response:", data);
        
        if (!response.ok) {
            throw new Error(data.error || `HTTP ${response.status}`);
        }

        // Your API returns: { status: "success", data: { total_documents: 2954, ... } }
        const actualData = data.data || data;
        console.log("📊 Actual statistics data:", actualData);
        
        // ✅ FIXED: Handle the actual API response structure
        const stats = {
            total_documents: ultraSafeNumber(actualData.total_documents || 0),
            active_indices: ultraSafeNumber(actualData.active_indices || 0),
            
            // Sample-based statistics (from the statistics sampling logic)
            total_evaluations: ultraSafeNumber(actualData.statistics?.total_evaluations || actualData.total_evaluations || 0),
            unique_templates: ultraSafeNumber(actualData.statistics?.templates?.length || actualData.unique_templates || 0),
            unique_programs: ultraSafeNumber(actualData.statistics?.programs?.length || actualData.unique_programs || 0),
            unique_partners: ultraSafeNumber(actualData.statistics?.partners?.length || actualData.unique_partners || 0),
            unique_dispositions: ultraSafeNumber(actualData.statistics?.dispositions?.length || actualData.unique_dispositions || 0),
            unique_agents: ultraSafeNumber(actualData.statistics?.agents?.length || actualData.unique_agents || 0),
            
            // Vector search info
            vector_support: actualData.vector_search?.cluster_support || false,
            vector_ready: actualData.vector_search?.vector_search_ready || false,
            documents_with_vectors: ultraSafeNumber(actualData.vector_search?.documents_with_vectors || 0),
            vector_coverage: ultraSafeNumber(actualData.vector_search?.vector_coverage || 0)
        };
        
        // Generate comprehensive HTML with all available data
        const html = `
            <div class="stats-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px;">
                <!-- Primary Metrics -->
                <div class="stats-card priority-metric" style="border: 2px solid #6e32a0;">
                    <h3><span class="emoji">📄</span> Total Documents</h3>
                    <div class="stats-number" style="color: #6e32a0; font-size: 2em;">${ultraSafeFormat(stats.total_documents)}</div>
                    <div class="stats-label">Documents in OpenSearch</div>
                </div>
                
                <div class="stats-card">
                    <h3><span class="emoji">💾</span> Active Indices</h3>
                    <div class="stats-number">${ultraSafeFormat(stats.active_indices)}</div>
                    <div class="stats-label">OpenSearch Indices</div>
                </div>
                
                <!-- Evaluation Data -->
                ${stats.total_evaluations > 0 ? `
                <div class="stats-card">
                    <h3><span class="emoji">🎯</span> Unique Evaluations</h3>
                    <div class="stats-number">${ultraSafeFormat(stats.total_evaluations)}</div>
                    <div class="stats-label">Distinct Evaluations</div>
                </div>` : ''}
                
                <!-- Template Data -->
                ${stats.unique_templates > 0 ? `
                <div class="stats-card">
                    <h3><span class="emoji">📋</span> Templates</h3>
                    <div class="stats-number">${ultraSafeFormat(stats.unique_templates)}</div>
                    <div class="stats-label">Evaluation Templates</div>
                </div>` : ''}
                
                <!-- Program Data -->
                ${stats.unique_programs > 0 ? `
                <div class="stats-card">
                    <h3><span class="emoji">🏢</span> Programs</h3>
                    <div class="stats-number">${ultraSafeFormat(stats.unique_programs)}</div>
                    <div class="stats-label">Different Programs</div>
                </div>` : ''}
                
                <!-- Agent Data -->
                ${stats.unique_agents > 0 ? `
                <div class="stats-card">
                    <h3><span class="emoji">👥</span> Agents</h3>
                    <div class="stats-number">${ultraSafeFormat(stats.unique_agents)}</div>
                    <div class="stats-label">Unique Agents</div>
                </div>` : ''}
                
                <!-- Disposition Data -->
                ${stats.unique_dispositions > 0 ? `
                <div class="stats-card">
                    <h3><span class="emoji">📞</span> Dispositions</h3>
                    <div class="stats-number">${ultraSafeFormat(stats.unique_dispositions)}</div>
                    <div class="stats-label">Call Dispositions</div>
                </div>` : ''}
                
                <!-- Partner Data -->
                ${stats.unique_partners > 0 ? `
                <div class="stats-card">
                    <h3><span class="emoji">🤝</span> Partners</h3>
                    <div class="stats-number">${ultraSafeFormat(stats.unique_partners)}</div>
                    <div class="stats-label">Business Partners</div>
                </div>` : ''}
                
                <!-- Vector Search Status -->
                ${stats.vector_support ? `
                <div class="stats-card ${stats.vector_ready ? 'vector-ready' : 'vector-disabled'}" style="border-color: ${stats.vector_ready ? '#28a745' : '#ffc107'};">
                    <h3><span class="emoji">${stats.vector_ready ? '🔍' : '⚠️'}</span> Vector Search</h3>
                    <div class="stats-number">${stats.vector_coverage}%</div>
                    <div class="stats-label">${stats.vector_ready ? 'Ready' : 'Disabled'} (${ultraSafeFormat(stats.documents_with_vectors)} docs)</div>
                </div>` : ''}
            </div>
            
            <!-- Footer Info -->
            <div style="margin-top: 20px; padding: 16px; background: #f8f9fa; border-radius: 8px; border-left: 4px solid #6e32a0;">
                <div style="font-size: 0.9em; color: #666;">
                    📅 Last updated: ${ultraSafeTimestamp(data.timestamp || new Date())} | 
                    🏥 Cluster: <span style="color: ${actualData.cluster_status === 'green' ? '#28a745' : '#dc3545'};">${actualData.cluster_status || 'Unknown'}</span> |
                    📊 Processing: ${data.processing_time || 'Unknown'}s
                </div>
                ${actualData.available_fields ? `
                <details style="margin-top: 10px;">
                    <summary style="cursor: pointer; font-weight: bold;">📋 Available Fields (${actualData.available_fields.length})</summary>
                    <div style="margin-top: 8px; font-family: monospace; font-size: 0.8em; background: #fff; padding: 8px; border-radius: 4px;">
                        ${actualData.available_fields.join(', ')}
                    </div>
                </details>` : ''}
            </div>
        `;
        
        container.innerHTML = html;
        console.log("✅ OpenSearch statistics loaded successfully with real data!");
        
    } catch (error) {
        console.error('❌ Failed to load OpenSearch statistics:', error);
        
        // Enhanced error display
        const errorType = error.message.includes('fetch') ? 'network_error' : 
                         error.message.includes('parse') ? 'parse_error' :
                         error.message.includes('timeout') ? 'timeout_error' : 'server_error';
        
        container.innerHTML = `
            <div class="stats-error" style="text-align: center; padding: 40px; background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 8px; color: #721c24;">
                <div style="font-size: 3em; margin-bottom: 16px;">❌</div>
                <h3>Statistics Loading Failed</h3>
                <p><strong>Error:</strong> ${ultraSafeString(error.message)}</p>
                <div style="margin-top: 20px;">
                    <button class="btn primary" onclick="loadOpenSearchStats()">
                        🔄 Retry Loading
                    </button>
                    <button class="btn secondary" onclick="window.open('/opensearch_statistics', '_blank')" style="margin-left: 10px;">
                        🔗 View Raw Data
                    </button>
                </div>
                <div style="margin-top: 16px; font-size: 0.9em; color: #856404; background: #fff3cd; padding: 12px; border-radius: 4px;">
                    💡 <strong>Troubleshooting:</strong> Check if your OpenSearch cluster is running and accessible.
                    Try visiting <code>/opensearch_statistics</code> directly to see the raw response.
                </div>
            </div>
        `;
    }
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

function setupDateRangeHandlers() {
    const startDateInput = document.getElementById('importStartDate');
    const endDateInput = document.getElementById('importEndDate');
    
    if (startDateInput) {
        startDateInput.addEventListener('change', updateDateRangeDisplay);
    }
    
    if (endDateInput) {
        endDateInput.addEventListener('change', updateDateRangeDisplay);
    }
    
    // Also trigger on max docs changes to update preview
    const maxDocsInput = document.getElementById('maxDocsInput');
    if (maxDocsInput) {
        maxDocsInput.addEventListener('input', function() {
            updateMaxDocsDisplay();
            updateImportPreview(); // Add this to your existing handler
        });
    }
}

function updateDateRangeDisplay() {
    const startDate = document.getElementById('importStartDate');
    const endDate = document.getElementById('importEndDate');
    const dateRangeInfo = document.getElementById('dateRangeInfo');
    const dateRangeDisplay = document.getElementById('dateRangeDisplay');
    
    if (!startDate || !endDate || !dateRangeInfo || !dateRangeDisplay) return;
    
    const start = startDate.value;
    const end = endDate.value;
    
    if (start || end) {
        let displayText = '';
        if (start && end) {
            displayText = `${start} to ${end}`;
        } else if (start) {
            displayText = `From ${start} onwards`;
        } else if (end) {
            displayText = `Up to ${end}`;
        }
        
        dateRangeDisplay.textContent = displayText;
        dateRangeInfo.style.display = 'block';
    } else {
        dateRangeDisplay.textContent = 'No date filter';
        dateRangeInfo.style.display = 'none';
    }
    
    // Update import preview when dates change
    updateImportPreview();
}

function clearDateRange() {
    const startDate = document.getElementById('importStartDate');
    const endDate = document.getElementById('importEndDate');
    
    if (startDate) startDate.value = '';
    if (endDate) endDate.value = '';
    
    updateDateRangeDisplay();
    console.log("📅 Date range cleared");
}

function updateImportPreview() {
    const importType = importTypeSelect ? importTypeSelect.value || "full" : "full"; 
    const maxDocsInput = document.getElementById("maxDocsInput");
    const startDate = document.getElementById('importStartDate')?.value;
    const endDate = document.getElementById('importEndDate')?.value;
    const importPreview = document.getElementById("importPreview");
    const importPreviewText = document.getElementById("importPreviewText");
    
    if (!importPreview || !importPreviewText) return;
    
    const maxDocs = maxDocsInput && maxDocsInput.value.trim() !== "" ? parseInt(maxDocsInput.value.trim()) : null;
    
    let previewText = `${importType.charAt(0).toUpperCase() + importType.slice(1)} import`;
    
    // Add date range info
    if (startDate || endDate) {
        if (startDate && endDate) {
            previewText += `\n📅 Date range: ${startDate} to ${endDate}`;
        } else if (startDate) {
            previewText += `\n📅 From: ${startDate} onwards`;
        } else if (endDate) {
            previewText += `\n📅 Up to: ${endDate}`;
        }
    }
    
    // Add max docs info
    if (maxDocs !== null && !isNaN(maxDocs)) {
        previewText += `\n📊 Limited to: ${maxDocs} documents`;
    } else {
        previewText += `\n📊 Processing: All matching documents`;
    }
    
    importPreviewText.textContent = previewText;
    importPreview.style.display = "block";
}

async function startImport() {
    const importTypeSelect = document.getElementById("importTypeSelect");
    const maxDocsInput = document.getElementById("maxDocsInput");
    const startDateInput = document.getElementById('importStartDate');
    const endDateInput = document.getElementById('importEndDate');
    const filterUpdatedCheckbox = document.getElementById('filterUpdatedAfterCreated'); // Get checkbox element
    const importType = importTypeSelect ? importTypeSelect.value : "full";
    
    // Handle max documents input FIRST
    let maxDocs = null;
    if (maxDocsInput && maxDocsInput.value.trim() !== "") {
        const parsedValue = parseInt(maxDocsInput.value.trim());
        if (!isNaN(parsedValue) && parsedValue > 0) {
            maxDocs = parsedValue;
        } else {
            alert("❌ Max Documents must be a positive number or left empty for all documents");
            return;
        }
    }

    // Handle date range inputs
    let startDate = null;
    let endDate = null;
    if (startDateInput && startDateInput.value.trim() !== "") {
        startDate = startDateInput.value.trim();
    }
    if (endDateInput && endDateInput.value.trim() !== "") {
        endDate = endDateInput.value.trim();
    }

    // Validate date range
    if (startDate && endDate && new Date(startDate) > new Date(endDate)) {
        alert("❌ Start date cannot be after end date");
        return;
    }

    // Get checkbox value AFTER element is retrieved
    const filterUpdatedAfterCreated = filterUpdatedCheckbox ? filterUpdatedCheckbox.checked : false;

    // NOW build the config object with all processed values
    const config = { 
        import_type: importType
    };
    
    // Only add parameters if they're explicitly set
    if (maxDocs !== null) {
        config.max_docs = maxDocs;
    }
    
    if (startDate) {
        config.call_date_start = startDate;
    }
    if (endDate) {
        config.call_date_end = endDate;
    }
    
    // Add the filter parameter if checkbox is checked
    if (filterUpdatedAfterCreated) {
        config.updated = true;
    }
    
    // Enhanced confirmation message
    let modeText;
    if (maxDocs !== null) {
        modeText = `Limiting to ${maxDocs} documents`;
    } else {
        modeText = "Processing ALL available documents";
    }
    
    let dateText = "";
    if (startDate || endDate) {
        if (startDate && endDate) {
            dateText = `\nDate Range: ${startDate} to ${endDate}`;
        } else if (startDate) {
            dateText = `\nFrom: ${startDate} onwards`;
        } else if (endDate) {
            dateText = `\nUp to: ${endDate}`;
        }
    }
    
    // Add filter info to confirmation
    let filterText = "";
    if (filterUpdatedAfterCreated) {
        filterText = "\n🔍 Filter: Only modified evaluations (updated > created_on)";
    }
    
    const importTypeText = importType === "incremental" ? "Incremental (only updated documents)" : "Full (all documents)";

    const confirmMsg = `Start ${importType} import?
    
Type: ${importTypeText}
Scope: ${modeText}${dateText}${filterText}

This will fetch evaluation data from your API and index it for search and chat.`;

    if (!confirm(confirmMsg)) return;

    // Debug logging
    console.log("🚀 Starting import with config:", config);
    console.log(`📋 Import Type Selected: ${importType}`);
    console.log(`🔢 Max Docs Element Value: ${maxDocsInput ? maxDocsInput.value : 'N/A'}`);
    console.log(`📊 Max Docs Parsed: ${maxDocs}`);

    if (maxDocs !== null) {
        console.log(`📊 Max documents limit: ${maxDocs}`);
    } else {
        console.log("📊 No document limit - importing all available");
    }

    if (startDate || endDate) {
        console.log(`📅 Date range: ${startDate || 'unlimited'} to ${endDate || 'unlimited'}`);
    }
    
    if (filterUpdatedAfterCreated) {
        console.log("🔍 Filter enabled: Only importing evaluations where updated > created_on");
    }

    try {
        const response = await fetch("/import", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(config) // Use config, not requestBody
        });

        console.log("📡 POST request sent to: /import");

        const text = await response.text();
        console.log("📥 Raw response:", text);

        let data;
        try {
            data = JSON.parse(text);
        } catch (e) {
            console.error("❌ Non-JSON response:", text);
            alert(`❌ Server returned non-JSON response: ${text.substring(0, 200)}...`);
            return;
        }

        if (response.ok) {
            let successMsg = ` ${importType.charAt(0).toUpperCase() + importType.slice(1)} import started successfully!`;
            if (maxDocs !== null) {
                successMsg += `\n Limited to ${maxDocs} documents`;
            }
            if (startDate || endDate) {
                if (startDate && endDate) {
                    successMsg += `\n📅 Date range: ${startDate} to ${endDate}`;
                } else if (startDate) {
                    successMsg += `\n📅 From: ${startDate}`;
                } else if (endDate) {
                    successMsg += `\n📅 Up to: ${endDate}`;
                }
            }
            
            // Add filter status to success message
            if (filterUpdatedAfterCreated) {
                successMsg += `\n🔍 Filter: Only modified evaluations`;
            }
            
            alert(successMsg);
            
            // Start polling for status updates
            startPolling();
            
        } else {
            console.error("❌ Import failed with status:", response.status);
            console.error("❌ Error response:", data);
            alert(`❌ Import failed: ${data.detail || data.message || data.error || "Unknown error"}`);
        }
    } catch (error) {
        console.error("❌ Import request failed:", error);
        alert(`❌ Import request failed: ${error.message}`);
    }
}

function safeUpdateElement(elementId, content, fallbackMessage = null) {
    try {
        const element = document.getElementById(elementId);
        if (element) {
            if (typeof content === 'object' && content !== null) {
                element.innerHTML = content.innerHTML || content.textContent || String(content);
            } else {
                element.textContent = String(content || '');
            }
            return true;
        } else {
            if (fallbackMessage) {
                console.warn(`⚠️ Element '${elementId}' not found: ${fallbackMessage}`);
            }
            return false;
        }
    } catch (error) {
        console.error(`❌ Error updating element '${elementId}':`, error);
        return false;
    }
}

function safeUpdateHTML(elementId, htmlContent, fallbackMessage = null) {
    try {
        const element = document.getElementById(elementId);
        if (element) {
            element.innerHTML = htmlContent || '';
            return true;
        } else {
            if (fallbackMessage) {
                console.warn(`⚠️ Element '${elementId}' not found: ${fallbackMessage}`);
            }
            return false;
        }
    } catch (error) {
        console.error(`❌ Error updating HTML for element '${elementId}':`, error);
        return false;
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
    console.log("🔄 Starting status polling...");
    
    // Stop any existing polling first
    stopPolling();
    
    isPolling = true;
    
    pollInterval = setInterval(async () => {
        try {
            await checkImportStatus();
        } catch (error) {
            console.error("❌ Polling error:", error);
            // Don't stop polling on single errors, but log them
        }
    }, 2000); // Poll every 2 seconds
    
    console.log("✅ Polling started");
}

function stopPolling() {
    if (pollInterval) {
        console.log("🛑 Stopping status polling...");
        clearInterval(pollInterval);
        pollInterval = null;
    }
    isPolling = false;
}

async function checkImportStatus() {
    try {
        const response = await fetch('/import_status');
        
        if (!response.ok) {
            console.warn(`⚠️ Status check returned ${response.status}`);
            return;
        }
        
        const data = await response.json();
        console.log("📊 Import status:", data.status);
        
        // Update the UI with status
        updateStatus(data);
        
        // Stop polling if import is finished
        if (data.status && !['running', 'starting', 'pending'].includes(data.status.toLowerCase())) {
            console.log(`🏁 Import finished with status: ${data.status}`);
            stopPolling();
            
            // Refresh stats after completion
            if (data.status.toLowerCase() === 'completed') {
                setTimeout(() => {
                    if (typeof loadOpenSearchStats === 'function') {
                        loadOpenSearchStats();
                    }
                    if (typeof checkLastImportInfo === 'function') {
                        checkLastImportInfo();
                    }
                }, 2000);
            }
        }
        
    } catch (error) {
        console.error("❌ Failed to check import status:", error);
    }
}

function updateStatus(data) {
    const container = document.getElementById('statusContainer');
    if (!container) return;

    const status = data.status || 'unknown';
    const statusClass = status.toLowerCase();
    
    let html = `<div class="status ${statusClass}">`;
    html += `<strong>Status:</strong> ${status.toUpperCase()}`;
    
    if (data.import_type) {
        html += ` (${data.import_type.charAt(0).toUpperCase() + data.import_type.slice(1)} Import)`;
    }
    
    if (data.message) {
        html += `<div style="margin-top: 5px; font-size: 0.9em; color: #666;">${data.message}</div>`;
    }
    
    html += `</div>`;
    
    container.innerHTML = html;

    // Handle current step display
    const currentStepDiv = document.getElementById('currentStep');
    const stepTextDiv = document.getElementById('stepText');
    
    if (data.current_step && status === 'running') {
        if (currentStepDiv) currentStepDiv.style.display = 'block';
        if (stepTextDiv) {
            stepTextDiv.textContent = data.current_step;
        }
    } else {
        if (currentStepDiv) currentStepDiv.style.display = 'none';
    }

    // Show results if completed
    if (status === 'completed' && data.results) {
        showResults(data.results);
    }
}

function showResults(results) {
    const section = document.getElementById('resultsSection');
    const grid = document.getElementById('resultsGrid');
    
    if (!section || !grid) return;
    
    section.classList.remove('hidden');
    
    let html = '<div class="results-summary">';
    html += `<h3>✅ Import Completed Successfully</h3>`;
    
    if (results.total_documents_processed) {
        html += `<p>📄 Documents processed: ${results.total_documents_processed.toLocaleString()}</p>`;
    }
    
    if (results.total_evaluations_indexed) {
        html += `<p>🎯 Evaluations indexed: ${results.total_evaluations_indexed.toLocaleString()}</p>`;
    }
    
    if (results.errors && results.errors > 0) {
        html += `<p>⚠️ Errors: ${results.errors}</p>`;
    }
    
    html += '</div>';
    grid.innerHTML = html;
}

async function refreshStatus() {
    console.log("🔄 Refreshing all status information...");
    
    try {
        // Run all checks in parallel for better performance
        const promises = [
            checkSystemHealth(),
            checkLastImportInfo()            
        ];
        
        await Promise.allSettled(promises);
        console.log("✅ Status refresh completed");
        
    } catch (error) {
        console.error("❌ Error during status refresh:", error);
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
    console.log("🏥 Checking system health...");
    
    const container = document.getElementById('healthContainer');
    if (!container) {
        console.warn('❌ Health container not found - add id="healthContainer" to your HTML');
        return;
    }
    
    container.innerHTML = `
        <div class="health-item">
            <span class="health-label">System Status</span>
            <span class="health-value">🔄 Checking...</span>
        </div>
    `;
    
    try {
        const response = await fetch('/health');
        const data = await response.json();
        
        console.log("🩺 Health check response:", data);
        
        // ✅ FIXED: Match the actual API response structure
        const isHealthy = response.ok && (data.status === 'ok' || data.status === 'healthy');
        
        // ✅ FIXED: Handle the correct OpenSearch status structure
        let openSearchStatus = '❌ Unknown';
        if (data.components && data.components.opensearch) {
            const osStatus = data.components.opensearch.status;
            if (osStatus === 'connected') {
                openSearchStatus = '✅ Connected';
            } else if (osStatus === 'not configured') {
                openSearchStatus = '⚠️ Not Configured';
            } else {
                openSearchStatus = '❌ Disconnected';
            }
        }
        
        // ✅ FIXED: Handle memory usage if available
        let memoryDisplay = 'Unknown';
        if (data.memory_usage) {
            memoryDisplay = Math.round(data.memory_usage) + '%';
        } else if (data.components && data.components.system && data.components.system.memory_usage) {
            memoryDisplay = Math.round(data.components.system.memory_usage) + '%';
        }
        
        // ✅ FIXED: Add vector search status if available
        let vectorSearchStatus = '';
        if (data.components && data.components.opensearch && data.components.opensearch.vector_search_support !== undefined) {
            const vectorEnabled = data.components.opensearch.vector_search_support;
            vectorSearchStatus = `
                <div class="health-item">
                    <span class="health-label">Vector Search</span>
                    <span class="health-value">${vectorEnabled ? '✅ Enabled' : '❌ Disabled'}</span>
                </div>
            `;
        }
        
        container.innerHTML = `
            <div class="health-item ${isHealthy ? '' : 'unhealthy'}">
                <span class="health-label">System Status</span>
                <span class="health-value">${isHealthy ? '✅ Healthy' : '❌ Unhealthy'}</span>
            </div>
            <div class="health-item">
                <span class="health-label">OpenSearch</span>
                <span class="health-value">${openSearchStatus}</span>
            </div>
            <div class="health-item">
                <span class="health-label">Memory Usage</span>
                <span class="health-value">${memoryDisplay}</span>
            </div>
            ${vectorSearchStatus}
        `;
        
        console.log("✅ Health check completed successfully");
        
    } catch (error) {
        console.error('❌ Health check failed:', error);
        container.innerHTML = `
            <div class="health-item unhealthy">
                <span class="health-label">System Status</span>
                <span class="health-value">❌ Error: ${error.message}</span>
            </div>
        `;
    }
}

async function checkLastImportInfo() {
    console.log("📅 Checking last import info...");
    
    let container = document.getElementById('lastImportInfo');
    
    // ✅ FIXED: Create the container if it doesn't exist
    if (!container) {
        console.log("⚠️ lastImportInfo container not found, creating it...");
        
        // Find a good place to add it (after the health container)
        const healthSection = document.getElementById('healthContainer')?.parentElement;
        if (healthSection) {
            const newContainer = document.createElement('div');
            newContainer.id = 'lastImportInfo';
            newContainer.className = 'health-status';
            newContainer.style.marginTop = '20px';
            
            // Add a header
            const header = document.createElement('h3');
            header.innerHTML = '<span class="emoji">📅</span> Last Import Information';
            healthSection.appendChild(header);
            healthSection.appendChild(newContainer);
            
            container = newContainer;
        } else {
            console.warn('❌ Could not create lastImportInfo container - no health section found');
            return;
        }
    }
    
    container.innerHTML = `
        <div class="health-item">
            <span class="health-label">Last Import</span>
            <span class="health-value">🔄 Checking...</span>
        </div>
    `;
    
    try {
        const response = await fetch('/import_info');
        const data = await response.json();
        
        console.log("📊 Import info response:", data);
        
        if (data.last_import) {
            container.innerHTML = `
                <div class="health-item">
                    <span class="health-label">Last Import</span>
                    <span class="health-value">✅ ${ultraSafeTimestamp(data.last_import.timestamp)}</span>
                </div>
                <div class="health-item">
                    <span class="health-label">Import Type</span>
                    <span class="health-value">${ultraSafeString(data.last_import.type)}</span>
                </div>
                <div class="health-item">
                    <span class="health-label">Status</span>
                    <span class="health-value">${ultraSafeString(data.last_import.status)}</span>
                </div>
            `;
        } else {
            container.innerHTML = `
                <div class="health-item">
                    <span class="health-label">Last Import</span>
                    <span class="health-value">⚠️ No import history found</span>
                </div>
            `;
        }
        
        console.log("✅ Last import info loaded successfully");
        
    } catch (error) {
        console.error('❌ Failed to load import info:', error);
        container.innerHTML = `
            <div class="health-item unhealthy">
                <span class="health-label">Last Import</span>
                <span class="health-value">❌ Error: ${error.message}</span>
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
        evaluation: ultraSafeString(data.evaluation || ''),
        transcript: ultraSafeString(data.transcript || data.transcript_text || data.full_text || '')

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

// NEW: Function to switch between tabs
function switchTab(tabId) {
    // Remove active class from all headers and panels
    document.querySelectorAll('.tab-header').forEach(header => {
        header.classList.remove('active');
    });
    document.querySelectorAll('.tab-panel').forEach(panel => {
        panel.classList.remove('active');
    });
    
    // Add active class to selected tab
    const selectedHeader = document.getElementById(tabId + '-header');
    const selectedPanel = document.getElementById(tabId);
    
    if (selectedHeader) selectedHeader.classList.add('active');
    if (selectedPanel) selectedPanel.classList.add('active');
}

// Make switchTab function globally available
window.switchTab = switchTab;

// NEW: Function to format transcript text
function formatTranscript(transcript) {
    if (!transcript) return '';
    
    // Handle speaker patterns and timestamps
    let formatted = transcript
        // Format speaker headers
        .replace(/Speaker ([AB]) \((\d{2}:\d{2}:\d{2})\):/g, 
                '<div class="speaker-header"><strong>Speaker $1</strong> <span class="timestamp">$2</span></div>')
        // Format regular line breaks
        .replace(/\n/g, '<br>')
        // Format timestamps that appear inline
        .replace(/(\d{2}:\d{2}:\d{2})/g, '<span class="inline-timestamp">$1</span>');
    
    return formatted;

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