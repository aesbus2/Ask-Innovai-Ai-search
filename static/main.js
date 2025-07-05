// Updated main.js for Ask InnovAI Admin Interface
// Compatible with enhanced app.py backend
// Version: 2.0.0

let pollInterval = null;

console.log("✅ Ask InnovAI Admin - Enhanced main.js loaded");

// Auto-refresh status every 10 seconds if not actively importing
setInterval(() => {
    if (!pollInterval) {
        refreshStatus();
    }
}, 10000);

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log("🚀 DOM loaded, initializing admin interface...");
    refreshStatus();
    checkSystemHealth();
    checkLastImportInfo();
    
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
});

// ============================================================================
// IMPORT MANAGEMENT FUNCTIONS
// ============================================================================

async function startImport() {
    const collectionSelect = document.getElementById("collectionSelect");
    const importTypeSelect = document.getElementById("importTypeSelect");
    const maxDocsInput = document.getElementById("maxDocsInput");
    
    const selectedCollection = collectionSelect ? collectionSelect.value : "all";
    const importType = importTypeSelect ? importTypeSelect.value : "full";
    
    // FIXED: Properly handle max documents input
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

    const config = { 
        collection: selectedCollection, 
        import_type: importType
    };
    
    // FIXED: Only add max_docs if it's explicitly set
    if (maxDocs !== null) {
        config.max_docs = maxDocs;
    }
    
    // Enhanced confirmation message
    let modeText;
    if (maxDocs !== null) {
        modeText = `Limiting to ${maxDocs} documents`;
    } else {
        modeText = "Processing ALL available documents";
    }
    
    const importTypeText = importType === "incremental" ? "Incremental (only updated documents)" : "Full (all documents)";

    const confirmMsg = `Start ${importType} import?
    
Collection: ${selectedCollection}
Type: ${importTypeText}
Scope: ${modeText}

This will fetch evaluation data from your API and index it for search and chat.`;

    if (!confirm(confirmMsg)) return;

    // Debug logging
    console.log("🚀 Starting import with config:", config);
    if (maxDocs !== null) {
        console.log(`📊 Max documents limit: ${maxDocs}`);
    } else {
        console.log("📊 No document limit - importing all available");
    }

    try {
        const response = await fetch("/import", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(config)
        });

        const text = await response.text();
        let data;
        try {
            data = JSON.parse(text);
        } catch (e) {
            console.error("Non-JSON response:", text);
            alert("❌ Unexpected response from server. Check console for details.");
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
        } else {
            alert(`❌ Import failed: ${data.detail || data.message || "Unknown error"}`);
        }
    } catch (error) {
        console.error("Import request failed:", error);
        alert(`❌ Import request failed: ${error.message}`);
    }
}

// BONUS: Add input validation for the max documents field
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

// BONUS: Enhanced UI feedback for max documents
function updateMaxDocsDisplay() {
    const maxDocsInput = document.getElementById("maxDocsInput");
    if (maxDocsInput) {
        const displayElement = document.getElementById("maxDocsDisplay");
        const value = maxDocsInput.value.trim();
        
        if (displayElement) {
            if (value === "" || isNaN(value)) {
                displayElement.textContent = "All documents";
                displayElement.style.color = "#666";
            } else {
                displayElement.textContent = `Max: ${parseInt(value).toLocaleString()}`;
                displayElement.style.color = "#6e32a0";
            }
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
                
                // Refresh import info after completion
                if (status.status === 'completed') {
                    setTimeout(() => {
                        checkLastImportInfo();
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
        if (!response.ok) {
            console.error('Status error:', response.status, await response.text());
            return;
        }

        const status = await response.json();
        console.log("📊 Status update:", status);
        updateStatusDisplay(status);
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

    // Add timing information
    if (data.status === 'running' && data.start_time) {
        html += `<div style="margin-top: 10px; font-size: 0.9em; color: #666;">
            Started: ${new Date(data.start_time).toLocaleString()}
        </div>`;
    } else if (data.status === 'completed' && data.end_time) {
        html += `<div style="margin-top: 10px; font-size: 0.9em; color: #666;">
            Completed: ${new Date(data.end_time).toLocaleString()}
        </div>`;
    } else if (data.status === 'failed') {
        html += `<div style="margin-top: 10px; font-size: 0.9em; color: #dc3545;">
            Failed: ${data.end_time ? new Date(data.end_time).toLocaleString() : 'Unknown time'}
        </div>`;
        if (data.error) {
            html += `<div style="margin-top: 10px; padding: 10px; background: #f8d7da; border-radius: 4px; color: #721c24;">
                <strong>Error:</strong> ${data.error}
            </div>`;
        }
    }

    container.innerHTML = html;

    // Show results if completed
    if (data.status === 'completed' && data.results) {
        showResults(data.results);
    }
}

function showResults(results) {
    const section = document.getElementById('resultsSection');
    const grid = document.getElementById('resultsGrid');
    
    if (!section || !grid) return;
    
    section.classList.remove('hidden');
    
    let html = '';
    
    // Define key metrics to display
    const metrics = [
        { key: 'total_documents_processed', label: 'Documents Processed', class: 'success', icon: '📄' },
        { key: 'total_chunks_indexed', label: 'Chunks Indexed', class: 'info', icon: '🧩' },
        { key: 'empty_documents', label: 'Empty Documents', class: 'warning', icon: '📭' },
        { key: 'import_type', label: 'Import Type', class: 'info', icon: '🔄' }
    ];
    
    // Display key metrics
    metrics.forEach(metric => {
        if (results.hasOwnProperty(metric.key)) {
            let value = results[metric.key];
            
            // Format import type
            if (metric.key === 'import_type') {
                value = value.charAt(0).toUpperCase() + value.slice(1);
            }
            
            html += `
                <div class="result-card">
                    <h4>${metric.icon} ${metric.label}</h4>
                    <div class="result-value ${metric.class}">${value}</div>
                </div>
            `;
        }
    });
    
    // Collections processed
    if (results.collections_processed && Array.isArray(results.collections_processed)) {
        html += `
            <div class="result-card">
                <h4>📁 Collections Processed</h4>
                <div class="result-value info">${results.collections_processed.join(', ')}</div>
            </div>
        `;
    }
    
    // Import timestamp
    if (results.new_import_timestamp) {
        try {
            const timestamp = new Date(results.new_import_timestamp).toLocaleString();
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
    
    // Handle empty documents by collection if it's an object
    if (results.empty_documents_by_collection && typeof results.empty_documents_by_collection === 'object') {
        const emptyDocs = results.empty_documents_by_collection;
        if (Object.keys(emptyDocs).length > 0) {
            let emptyDocsText = Object.entries(emptyDocs)
                .map(([collection, count]) => `${collection}: ${count}`)
                .join(', ');
            
            html += `
                <div class="result-card">
                    <h4>📭 Empty by Collection</h4>
                    <div class="result-value warning" style="font-size: 1em;">${emptyDocsText}</div>
                </div>
            `;
        }
    }
    
    grid.innerHTML = html;
    
    console.log("📈 Import results displayed:", results);
}

// ============================================================================
// SYSTEM HEALTH AND MONITORING
// ============================================================================

// Replace your checkSystemHealth function with this safer version:

async function checkSystemHealth() {
    const container = document.getElementById('healthStatus');
    if (!container) return;
    
    container.innerHTML = '<div class="loading"><div class="spinner"></div>Checking system health...</div>';
    
    try {
        const response = await fetch('/health');
        
        // Check if response is ok
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const health = await response.json();
        
        // Check if health data is valid
        if (!health) {
            throw new Error('Health endpoint returned null/empty response');
        }
        
        let html = '';
        
        // Safely check for components
        if (health.components && typeof health.components === 'object') {
            Object.entries(health.components).forEach(([component, info]) => {
                const status = info.status || 'unknown';
                const isHealthy = status === 'connected' || status === 'healthy' || status === 'configured';
                const isWarning = status === 'not configured';
                const cssClass = isHealthy ? '' : (isWarning ? 'warning' : 'unhealthy');
                
                // Add emoji indicators
                let emoji = '✅';
                if (!isHealthy) {
                    emoji = isWarning ? '⚠️' : '❌';
                }
                
                html += `
                    <div class="health-item ${cssClass}">
                        <div class="health-label">${emoji} ${component.toUpperCase()}</div>
                        <div class="health-value">${status}</div>
                    </div>
                `;
            });
        } else {
            // No components found - show basic info
            html += `
                <div class="health-item warning">
                    <div class="health-label">⚠️ SYSTEM</div>
                    <div class="health-value">Components data missing</div>
                </div>
            `;
        }
        
        // Add overall status if available
        if (health.status) {
            const statusEmoji = health.status === 'ok' ? '✅' : '❌';
            html += `
                <div class="health-item">
                    <div class="health-label">${statusEmoji} OVERALL</div>
                    <div class="health-value">${health.status}</div>
                </div>
            `;
        }
        
        // Add last import info if available
        if (health.last_import && health.last_import.timestamp) {
            try {
                const timestamp = new Date(health.last_import.timestamp).toLocaleString();
                html += `
                    <div class="health-item">
                        <div class="health-label">📅 LAST IMPORT</div>
                        <div class="health-value">${timestamp}</div>
                    </div>
                `;
            } catch (e) {
                // Ignore timestamp formatting errors
            }
        }
        
        container.innerHTML = html;
        console.log("🏥 System health checked:", health);
        
    } catch (error) {
        console.error("Health check failed:", error);
        container.innerHTML = `
            <div class="health-item unhealthy">
                <div class="health-label">❌ SYSTEM</div>
                <div class="health-value">Error: ${error.message}</div>
            </div>
        `;
    }
}

// Also update checkLastImportInfo to handle 404 errors:
async function checkLastImportInfo() {
    try {
        const response = await fetch('/last_import_info');
        
        // Handle 404 - endpoint doesn't exist yet
        if (response.status === 404) {
            console.log("📅 Last import info endpoint not available");
            return;
        }
        
        if (!response.ok) {
            console.error(`Last import info error: ${response.status}`);
            return;
        }
        
        const data = await response.json();
        
        if (data.status === 'success' && data.last_import_timestamp) {
            const timestamp = new Date(data.last_import_timestamp).toLocaleString();
            console.log(`📅 Last import: ${timestamp}`);
            
            // You could display this in the UI if there's a container for it
            const infoContainer = document.getElementById("lastImportInfo");
            if (infoContainer) {
                infoContainer.innerHTML = `
                    <div style="background: #e3f2fd; padding: 8px; border-radius: 4px; margin: 8px 0; font-size: 0.9em;">
                        <strong>📅 Last Import:</strong> ${timestamp}
                    </div>
                `;
            }
        }
    } catch (error) {
        console.error("Failed to check last import info:", error);
    }
}

async function checkLastImportInfo() {
    try {
        const response = await fetch('/last_import_info');
        const data = await response.json();
        
        if (data.status === 'success' && data.last_import_timestamp) {
            const timestamp = new Date(data.last_import_timestamp).toLocaleString();
            console.log(`📅 Last import: ${timestamp}`);
            
            // You could display this in the UI if there's a container for it
            const infoContainer = document.getElementById("lastImportInfo");
            if (infoContainer) {
                infoContainer.innerHTML = `
                    <div style="background: #e3f2fd; padding: 8px; border-radius: 4px; margin: 8px 0; font-size: 0.9em;">
                        <strong>📅 Last Import:</strong> ${timestamp}
                    </div>
                `;
            }
        }
    } catch (error) {
        console.error("Failed to check last import info:", error);
    }
}

async function clearImportTimestamp() {
    if (!confirm("Reset import timestamp? The next incremental import will process all documents. Continue?")) {
        return;
    }
    
    try {
        const response = await fetch("/clear_import_timestamp", {
            method: "POST",
            headers: { "Content-Type": "application/json" }
        });
        
        const data = await response.json();
        if (data.status === 'success') {
            alert("✅ Import timestamp reset successfully!");
            checkLastImportInfo(); // Refresh the display
        } else {
            alert(`❌ Error: ${data.error}`);
        }
    } catch (error) {
        alert(`❌ Failed to reset timestamp: ${error.message}`);
    }
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

async function toggleLogs() {
    const container = document.getElementById('logsContainer');
    const content = document.getElementById('logsContent');
    
    if (!container || !content) return;

    if (container.classList.contains('hidden')) {
        try {
            content.textContent = 'Loading logs...';
            container.classList.remove('hidden');
            
            const response = await fetch('/logs');
            const data = await response.json();
            const logs = data.logs || [];
            
            // Show last 50 log entries
            content.textContent = logs.slice(-50).join('\n') || 'No logs available';
        } catch (error) {
            content.textContent = `Error loading logs: ${error.message}`;
        }
    } else {
        container.classList.add('hidden');
    }
}

// Quick action functions that can be called from the UI
async function getImportStatistics() {
    const container = document.getElementById("actionResults");
    if (!container) return;
    
    container.innerHTML = '<div class="loading"><div class="spinner"></div>Loading statistics...</div>';
    
    try {
        const response = await fetch('/import_statistics');
        const data = await response.json();
        
        if (data.status === 'success') {
            const stats = data.statistics;
            let html = `
                <div class="result-card">
                    <h4>📊 Knowledge Base Statistics</h4>
                    <p><strong>Total Documents:</strong> ${stats.total_documents || 0}</p>
                    <p><strong>Total Chunks:</strong> ${stats.total_chunks || 0}</p>
                    <p><strong>Collections:</strong> ${Object.keys(stats.collections || {}).length}</p>
                </div>
            `;
            container.innerHTML = html;
        } else {
            container.innerHTML = `<div class="status failed">❌ Error: ${data.error}</div>`;
        }
    } catch (error) {
        container.innerHTML = `<div class="status failed">❌ Failed: ${error.message}</div>`;
    }
}

async function countDocsByCollectionAndProgram() {
    const container = document.getElementById("actionResults");
    if (!container) return;
    
    container.innerHTML = '<div class="loading"><div class="spinner"></div>Counting documents...</div>';
    
    try {
        const response = await fetch('/count_by_collection_and_program');
        const data = await response.json();
        
        if (data.status === 'success') {
            let html = '<div class="result-card"><h4>📋 Document Counts</h4>';
            
            const counts = data.collection_program_counts;
            if (Object.keys(counts).length === 0) {
                html += '<p>No documents found in any collection.</p>';
            } else {
                for (const [collection, programs] of Object.entries(counts)) {
                    html += `<div style="margin-bottom: 10px;"><strong>${collection}:</strong><ul style="margin-left: 20px;">`;
                    
                    if (Object.keys(programs).length === 0) {
                        html += '<li>No documents found</li>';
                    } else {
                        for (const [program, count] of Object.entries(programs)) {
                            const programDisplay = (!program || program === 'all') ? 
                                '<em>All Programs</em>' : program;
                            html += `<li>${programDisplay}: <strong>${count}</strong> documents</li>`;
                        }
                    }
                    html += '</ul></div>';
                }
            }
            html += '</div>';
            container.innerHTML = html;
        } else {
            container.innerHTML = `<div class="status failed">❌ Error: ${data.error}</div>`;
        }
    } catch (error) {
        container.innerHTML = `<div class="status failed">❌ Failed: ${error.message}</div>`;
    }
}

async function testSearch() {
    const query = prompt('🔍 Enter search query to test:', 'customer service');
    if (!query) return;
    
    const container = document.getElementById("actionResults");
    if (!container) return;
    
    container.innerHTML = '<div class="loading"><div class="spinner"></div>Searching...</div>';
    
    try {
        const response = await fetch(`/search?q=${encodeURIComponent(query)}`);
        const data = await response.json();
        
        if (data.status === 'success') {
            const resultCount = data.results ? data.results.length : 0;
            let html = `
                <div class="result-card">
                    <h4>🔍 Search Results for "${query}"</h4>
                    <p><strong>Found:</strong> ${resultCount} results</p>
            `;
            
            if (resultCount > 0) {
                html += '<div style="max-height: 200px; overflow-y: auto; margin-top: 10px; border-top: 1px solid #eee; padding-top: 10px;">';
                data.results.slice(0, 3).forEach((result, index) => {
                    html += `
                        <div style="margin: 10px 0; padding: 10px; background: #f9f9f9; border-radius: 4px; border-left: 3px solid #6e32a0;">
                            <div style="font-weight: bold; color: #6e32a0; margin-bottom: 5px;">
                                ${index + 1}. ${result.title || 'Untitled'}
                            </div>
                            <div style="font-size: 0.9em; color: #666;">
                                ${(result.text || '').substring(0, 150)}${result.text && result.text.length > 150 ? '...' : ''}
                            </div>
                            ${result.score ? `<div style="font-size: 0.8em; color: #999; margin-top: 5px;">Score: ${Math.round(result.score * 100) / 100}</div>` : ''}
                        </div>
                    `;
                });
                html += '</div>';
                
                if (resultCount > 3) {
                    html += `<div style="margin-top: 10px; font-size: 0.9em; color: #666; text-align: center;">
                        ... and ${resultCount - 3} more results
                    </div>`;
                }
            }
            
            html += '</div>';
            container.innerHTML = html;
        } else {
            container.innerHTML = `<div class="status failed">❌ Search failed: ${data.error}</div>`;
        }
    } catch (error) {
        container.innerHTML = `<div class="status failed">❌ Search failed: ${error.message}</div>`;
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
window.getImportStatistics = getImportStatistics;
window.countDocsByCollectionAndProgram = countDocsByCollectionAndProgram;
window.testSearch = testSearch;

console.log("✅ Ask InnovAI Admin main.js loaded successfully - all functions available");