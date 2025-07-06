// Enhanced main.js for Ask InnovAI Admin Interface with OpenSearch Statistics
// Compatible with enhanced app.py backend
// Version: 2.1.0 - Added comprehensive OpenSearch statistics

let pollInterval = null;

console.log("✅ Ask InnovAI Admin - Enhanced main.js with statistics loaded");

// Auto-refresh status every 10 seconds if not actively importing
setInterval(() => {
    if (!pollInterval) {
        refreshStatus();
    }
}, 10000);

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log("🚀 DOM loaded, initializing enhanced admin interface...");
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
});

// ============================================================================
// OPENSEARCH STATISTICS FUNCTIONS
// ============================================================================

async function loadOpenSearchStats() {
    //Load comprehensive OpenSearch database statistics
    const container = document.getElementById('statisticsContainer');
    if (!container) return;
    
    container.innerHTML = `
        <div class="loading-stats">
            <div class="spinner"></div>
            Loading database statistics...
        </div>
    `;
    
    try {
        console.log("📊 Loading OpenSearch statistics...");
        const response = await fetch('/opensearch_statistics');
        const result = await response.json();
        
        if (result.status === 'success') {
            console.log("✅ Statistics loaded successfully");
            displayStatistics(result.data, result.timestamp);
        } else {
            console.error("❌ Statistics error:", result.error);
            container.innerHTML = `
                <div class="stats-error">
                    <strong>❌ Error loading statistics:</strong> ${result.error}
                    <br><small>Check OpenSearch connection and try again.</small>
                </div>
            `;
        }
    } catch (error) {
        console.error('Failed to load statistics:', error);
        container.innerHTML = `
            <div class="stats-error">
                <strong>❌ Failed to connect to server:</strong> ${error.message}
                <br><small>Verify the backend is running and try refreshing.</small>
            </div>
        `;
    }
}

function displayStatistics(data, timestamp) {
   // Display comprehensive statistics in dashboard format
    const container = document.getElementById('statisticsContainer');
    if (!container) return;
    
    // Calculate some additional metrics
    const avgChunksPerEval = data.total_evaluations > 0 ? 
        (data.total_chunks / data.total_evaluations).toFixed(1) : 0;
    
    const transcriptPercentage = data.total_evaluations > 0 ? 
        ((data.evaluations_with_transcript / data.total_evaluations) * 100).toFixed(1) : 0;
    
    const html = `
        <div class="stats-dashboard">
            <!-- Main Overview Stats -->
            <div class="stats-card">
                <h3>📄 Total Evaluations</h3>
                <div class="stats-number">${data.total_evaluations.toLocaleString()}</div>
                <div class="stats-label">Complete Evaluations</div>
                <div class="stats-breakdown">
                    <div class="breakdown-item">
                        <span class="breakdown-label">📝 With Transcript</span>
                        <span class="breakdown-value">${data.evaluations_with_transcript.toLocaleString()} (${transcriptPercentage}%)</span>
                    </div>
                    <div class="breakdown-item">
                        <span class="breakdown-label">📋 Evaluation Only</span>
                        <span class="breakdown-value">${data.evaluations_without_transcript.toLocaleString()}</span>
                    </div>
                    <div class="breakdown-item">
                        <span class="breakdown-label">📊 Avg Chunks/Eval</span>
                        <span class="breakdown-value">${avgChunksPerEval}</span>
                    </div>
                </div>
            </div>
            
            <div class="stats-card">
                <h3>🧩 Total Chunks</h3>
                <div class="stats-number">${data.total_chunks.toLocaleString()}</div>
                <div class="stats-label">All Content Pieces</div>
                <div class="stats-breakdown">
                    <div class="breakdown-item">
                        <span class="breakdown-label">📝 Evaluation Chunks</span>
                        <span class="breakdown-value">${data.evaluation_chunks.toLocaleString()}</span>
                    </div>
                    <div class="breakdown-item">
                        <span class="breakdown-label">🎙️ Transcript Chunks</span>
                        <span class="breakdown-value">${data.transcript_chunks.toLocaleString()}</span>
                    </div>
                </div>
            </div>
            
            <!-- Template Distribution -->
            <div class="stats-card">
                <h3>📋 Templates</h3>
                <div class="stats-number">${Object.keys(data.template_counts).length}</div>
                <div class="stats-label">Unique Templates</div>
                <div class="stats-breakdown">
                    ${Object.entries(data.template_counts)
                        .sort(([,a], [,b]) => b - a)
                        .slice(0, 5)
                        .map(([template, count]) => `
                            <div class="breakdown-item">
                                <span class="breakdown-label" title="${template}">${template.length > 20 ? template.substring(0, 20) + '...' : template}</span>
                                <span class="breakdown-value">${count.toLocaleString()}</span>
                            </div>
                        `).join('')}
                    ${Object.keys(data.template_counts).length > 5 ? 
                        `<div class="breakdown-item">
                            <span class="breakdown-label">...and ${Object.keys(data.template_counts).length - 5} more</span>
                            <span class="breakdown-value"></span>
                        </div>` : ''}
                </div>
            </div>
            
            <!-- LOB Distribution -->
            <div class="stats-card">
                <h3>🏢 Line of Business</h3>
                <div class="stats-number">${Object.keys(data.lob_counts).length}</div>
                <div class="stats-label">Unique LOBs</div>
                <div class="stats-breakdown">
                    ${Object.entries(data.lob_counts)
                        .sort(([,a], [,b]) => b - a)
                        .map(([lob, count]) => `
                            <div class="breakdown-item">
                                <span class="breakdown-label">${lob}</span>
                                <span class="breakdown-value">${count.toLocaleString()}</span>
                            </div>
                        `).join('')}
                </div>
            </div>
            
            <!-- Partner Distribution -->
            <div class="stats-card">
                <h3>🤝 Partners</h3>
                <div class="stats-number">${Object.keys(data.partner_counts).length}</div>
                <div class="stats-label">Unique Partners</div>
                <div class="stats-breakdown">
                    ${Object.entries(data.partner_counts)
                        .sort(([,a], [,b]) => b - a)
                        .map(([partner, count]) => `
                            <div class="breakdown-item">
                                <span class="breakdown-label">${partner}</span>
                                <span class="breakdown-value">${count.toLocaleString()}</span>
                            </div>
                        `).join('')}
                </div>
            </div>
            
            <!-- Site Distribution -->
            <div class="stats-card">
                <h3>🏢 Sites</h3>
                <div class="stats-number">${Object.keys(data.site_counts).length}</div>
                <div class="stats-label">Unique Sites</div>
                <div class="stats-breakdown">
                    ${Object.entries(data.site_counts)
                        .sort(([,a], [,b]) => b - a)
                        .slice(0, 6)
                        .map(([site, count]) => `
                            <div class="breakdown-item">
                                <span class="breakdown-label">${site}</span>
                                <span class="breakdown-value">${count.toLocaleString()}</span>
                            </div>
                        `).join('')}
                    ${Object.keys(data.site_counts).length > 6 ? 
                        `<div class="breakdown-item">
                            <span class="breakdown-label">...and ${Object.keys(data.site_counts).length - 6} more</span>
                            <span class="breakdown-value"></span>
                        </div>` : ''}
                </div>
            </div>
            
            <!-- Language Distribution -->
            <div class="stats-card">
                <h3>🌐 Languages</h3>
                <div class="stats-number">${Object.keys(data.language_counts).length}</div>
                <div class="stats-label">Languages Used</div>
                <div class="stats-breakdown">
                    ${Object.entries(data.language_counts)
                        .sort(([,a], [,b]) => b - a)
                        .map(([language, count]) => `
                            <div class="breakdown-item">
                                <span class="breakdown-label">${language}</span>
                                <span class="breakdown-value">${count.toLocaleString()}</span>
                            </div>
                        `).join('')}
                </div>
            </div>
            
            <!-- Index Information -->
            <div class="stats-card">
                <h3>💾 Storage</h3>
                <div class="stats-number">${data.indices.length}</div>
                <div class="stats-label">Active Indices</div>
                <div class="stats-breakdown">
                    ${data.indices.slice(0, 4).map(index => `
                        <div class="breakdown-item">
                            <span class="breakdown-label" title="${index.name}">${index.name.length > 15 ? index.name.substring(0, 15) + '...' : index.name}</span>
                            <span class="breakdown-value">${index.size_mb.toFixed(1)}MB</span>
                        </div>
                    `).join('')}
                    ${data.indices.length > 4 ? 
                        `<div class="breakdown-item">
                            <span class="breakdown-label">...and ${data.indices.length - 4} more</span>
                            <span class="breakdown-value"></span>
                        </div>` : ''}
                </div>
            </div>
        </div>
        
        <div class="stats-last-updated">
            📅 Last updated: ${new Date(timestamp).toLocaleString()} | 
            📊 Structure: ${data.structure_info.document_type} | 
            🏷️ Collections: ${data.structure_info.collection_strategy}
        </div>
    `;
    
    container.innerHTML = html;
    console.log("📊 Statistics dashboard updated");
}

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
    
    // Define key metrics to display
    const metrics = [
        { key: 'total_documents_processed', label: 'Documents Processed', class: 'success', icon: '📄' },
        { key: 'total_evaluations_indexed', label: 'Evaluations Indexed', class: 'success', icon: '📋' },
        { key: 'total_chunks_processed', label: 'Chunks Processed', class: 'info', icon: '🧩' },
        { key: 'errors', label: 'Errors', class: 'warning', icon: '❌' },
        { key: 'opensearch_errors', label: 'OpenSearch Errors', class: 'danger', icon: '🔥' },
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
    if (results.template_collections_created && Array.isArray(results.template_collections_created)) {
        html += `
            <div class="result-card">
                <h4>📁 Template Collections</h4>
                <div class="result-value info">${results.template_collections_created.length}</div>
                <div style="font-size: 0.8em; margin-top: 8px; color: #666;">
                    ${results.template_collections_created.slice(0, 3).join(', ')}
                    ${results.template_collections_created.length > 3 ? '...' : ''}
                </div>
            </div>
        `;
    }
    
    // Import timestamp
    if (results.completed_at) {
        try {
            const timestamp = new Date(results.completed_at).toLocaleString();
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
    
    // Success rate
    if (results.success_rate) {
        html += `
            <div class="result-card">
                <h4>📈 Success Rate</h4>
                <div class="result-value success">${results.success_rate}</div>
            </div>
        `;
    }
    
    grid.innerHTML = html;
    
    console.log("📈 Import results displayed:", results);
}

// ============================================================================
// SYSTEM HEALTH AND MONITORING
// ============================================================================

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
        
        // Add enhanced structure info
        if (health.enhancements) {
            html += `
                <div class="health-item">
                    <div class="health-label">🔄 STRUCTURE</div>
                    <div class="health-value">${health.enhancements.document_structure}</div>
                </div>
            `;
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

function openChatInterface() {
    window.open('/chat', '_blank');
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
window.loadOpenSearchStats = loadOpenSearchStats; // NEW: Expose statistics function

console.log("✅ Ask InnovAI Admin enhanced main.js loaded successfully - all functions including statistics available");