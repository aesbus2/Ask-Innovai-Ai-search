let pollInterval = null;

console.log("main.js loaded");

// NEW: Check last import info on page load
async function checkLastImportInfo() {
    try {
        const response = await fetch("/last_import_info");
        const data = await response.json();
        
        if (data.status === 'success' && data.last_import_timestamp) {
            const timestamp = new Date(data.last_import_timestamp).toLocaleString();
            const infoDiv = document.getElementById("lastImportInfo");
            if (infoDiv) {
                infoDiv.innerHTML = `
                    <div style="background: #e3f2fd; padding: 8px; border-radius: 4px; margin: 8px 0; font-size: 0.9em;">
                        <strong>📅 Last Import:</strong> ${timestamp}
                        <button onclick="clearImportTimestamp()" style="margin-left: 10px; padding: 2px 6px; font-size: 0.8em; background: #fff3cd; border: 1px solid #ffc107; border-radius: 3px; cursor: pointer;">
                            Reset
                        </button>
                    </div>
                `;
            }
        }
    } catch (error) {
        console.error("Failed to load last import info:", error);
    }
}

// NEW: Clear import timestamp
async function clearImportTimestamp() {
    if (!confirm("This will reset the import timestamp. The next incremental import will process all documents. Continue?")) {
        return;
    }
    
    try {
        const response = await fetch("/clear_import_timestamp", {
            method: "POST",
            headers: { "Content-Type": "application/json" }
        });
        
        const data = await response.json();
        if (data.status === 'success') {
            alert("Import timestamp reset successfully!");
            checkLastImportInfo(); // Refresh the display
        } else {
            alert(`Error: ${data.error}`);
        }
    } catch (error) {
        alert(`Failed to reset timestamp: ${error.message}`);
    }
}

async function startImport() {
    const collectionSelect = document.getElementById("collectionSelect");
    const maxDocsInput = document.getElementById("maxDocsInput");
    const importTypeSelect = document.getElementById("importTypeSelect"); // NEW
    
    const selectedCollection = collectionSelect ? collectionSelect.value : "all";
    const max_docs = maxDocsInput ? parseInt(maxDocsInput.value) : undefined;
    const import_type = importTypeSelect ? importTypeSelect.value : "full"; // NEW

    const config = { 
        collection: selectedCollection, 
        max_docs: max_docs,
        import_type: import_type  // NEW
    };
    
    const modeText = max_docs ? `Limiting to ${max_docs} documents per collection` : "Importing all documents";
    const importTypeText = import_type === "incremental" ? "Incremental (only updated documents)" : "Full (all documents)"; // NEW

    if (!confirm(`Start ${import_type} import?\nCollection: ${selectedCollection}\n${modeText}\nType: ${importTypeText}`)) return;

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
            alert("Unexpected response from server. Check console.");
            return;
        }

        if (response.ok) {
            alert(`${import_type.charAt(0).toUpperCase() + import_type.slice(1)} import started! (${modeText})`);
            startPolling();
        } else {
            alert(`Error: ${data.detail || data.message || "Unknown error"}`);
        }
    } catch (error) {
        console.error("Import fetch error:", error);
        alert("Import request failed: " + error.message);
    }
}

// Debug function to check API data structure
async function debugAPIData(collection = 'plans') {
    try {
        const response = await fetch(`/debug/metadata_extraction/${collection}`);
        const data = await response.json();
        
        console.log('API Data Debug:', data);
        
        if (data.status === 'success') {
            const summary = data.extraction_summary;
            alert(`Collection: ${collection}\nTotal Documents: ${data.total_documents}\nDocs with Names: ${summary.docs_with_names}\nDocs with URLs: ${summary.docs_with_urls}\nFully Extracted: ${summary.docs_fully_extracted}`);
        } else {
            alert(`API Debug Error: ${data.error}`);
        }
    } catch (error) {
        console.error("API Debug fetch error:", error);
        alert(`API Debug failed: ${error.message}`);
    }
}

function startPolling() {
    if (pollInterval) clearInterval(pollInterval);

    pollInterval = setInterval(async () => {
        await refreshStatus();
        const response = await fetch('/status');
        const status = await response.json();

        if (status.status !== 'running') {
            clearInterval(pollInterval);
            pollInterval = null;
            
            // NEW: Refresh last import info after completion
            if (status.status === 'completed') {
                setTimeout(checkLastImportInfo, 1000);
            }
        }
    }, 2000);
}

async function refreshStatus() {
    let resp = await fetch('/status');
    if (!resp.ok) {
        console.error('Status error:', resp.status, await resp.text());
        return;
    }

    let status = await resp.json();
    console.log("Status from backend:", status);
    updateStatusDisplay(status);
}

function updateStatusDisplay(data) {
    const container = document.getElementById('status-container');
    const statusClass = data.status;

    let html = `<div class="status ${statusClass}">`;
    html += `<h3>Status: ${data.status.toUpperCase()}</h3>`;

    if (data.current_step) {
        html += `<p><strong>Step:</strong> ${data.current_step}</p>`;
        const currentStepDiv = document.getElementById('current-step');
        const stepTextDiv = document.getElementById('step-text');
        if (currentStepDiv) currentStepDiv.style.display = 'block';
        if (stepTextDiv) stepTextDiv.textContent = data.current_step;
    } else {
        const currentStepDiv = document.getElementById('current-step');
        if (currentStepDiv) currentStepDiv.style.display = 'none';
    }

    if (data.status === 'running') {
        html += `<p>Started: ${data.start_time}</p>`;
        // NEW: Show import type
        if (data.import_type) {
            html += `<p><strong>Import Type:</strong> ${data.import_type.charAt(0).toUpperCase() + data.import_type.slice(1)}</p>`;
        }
    } else if (data.status === 'completed') {
        html += `<p>Completed: ${data.end_time}</p>`;
        if (data.results) showResults(data.results);
    } else if (data.status === 'failed') {
        html += `<p>Failed: ${data.end_time}</p>`;
        if (data.error) html += `<p><strong>Error:</strong> ${data.error}</p>`;
    }

    html += '</div>';
    container.innerHTML = html;
}

function showResults(results) {
    const container = document.getElementById('results-container');
    const grid = document.getElementById('results-grid');

    let html = '';
    for (const [key, value] of Object.entries(results)) {
        const displayKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());

        let cardClass = 'result-card';
        let valueColor = '#007bff';  // default blue
        let valueHTML = '';

        // FIXED: Handle different result types properly
        if (key === 'empty_documents_by_collection' && typeof value === 'object') {
            valueColor = '#dc3545';
            if (Object.keys(value).length === 0) {
                valueHTML = '<div style="color: #28a745;">No empty documents found</div>';
            } else {
                const lines = Object.entries(value).map(
                    ([collection, count]) => `<div><strong>${collection}</strong>: ${count} empty</div>`
                );
                valueHTML = lines.join('');
            }
        } else if (key === 'collections_processed' && Array.isArray(value)) {
            valueHTML = `<div style="color: ${valueColor};">${value.join(', ')}</div>`;
        } else if (key === 'documents_per_collection_limit') {
            valueColor = '#ffc107';
            valueHTML = `<div style="color: ${valueColor};">${value || 'No limit'}</div>`;
        } else if (key === 'total_documents_processed') {
            valueColor = '#28a745';
            valueHTML = `<div style="font-size: 1.8em; color: ${valueColor}; font-weight: bold;">${value}</div>`;
        } else if (key === 'total_chunks_indexed') {
            valueColor = '#17a2b8';
            valueHTML = `<div style="font-size: 1.5em; color: ${valueColor};">${value}</div>`;
        } else if (key === 'empty_documents') {
            valueColor = value > 0 ? '#dc3545' : '#28a745';
            valueHTML = `<div style="font-size: 1.2em; color: ${valueColor};">${value}</div>`;
        } else if (key === 'import_type') { // NEW: Display import type
            valueColor = value === 'incremental' ? '#17a2b8' : '#6f42c1';
            valueHTML = `<div style="font-size: 1.2em; color: ${valueColor}; font-weight: bold;">${value.charAt(0).toUpperCase() + value.slice(1)}</div>`;
        } else if (key === 'last_import_timestamp') { // NEW: Display last import timestamp
            if (value) {
                const timestamp = new Date(value).toLocaleString();
                valueHTML = `<div style="font-size: 0.9em; color: #666;">${timestamp}</div>`;
            } else {
                valueHTML = `<div style="font-size: 0.9em; color: #666;">None</div>`;
            }
        } else if (key === 'new_import_timestamp') { // NEW: Display new import timestamp
            if (value) {
                const timestamp = new Date(value).toLocaleString();
                valueColor = '#28a745';
                valueHTML = `<div style="font-size: 0.9em; color: ${valueColor}; font-weight: bold;">${timestamp}</div>`;
            }
        } else {
            valueHTML = `<div style="font-size: 1.2em; color: ${valueColor};">${value}</div>`;
        }

        html += `<div class="${cardClass}">
            <h4>${displayKey}</h4>
            ${valueHTML}
        </div>`;
    }

    grid.innerHTML = html;
    container.style.display = 'block';
}

// ✅ Fix log fetch in toggleLogs()
async function toggleLogs() {
    const container = document.getElementById('logs-container');
    const content = document.getElementById('logs-content');

    if (container.style.display === 'none') {
        try {
            const response = await fetch('/logs');
            const data = await response.json();
            const lines = data.logs || data.lines || [];
            content.textContent = lines.join('\n') || 'No logs available';
            container.style.display = 'block';
        } catch (error) {
            content.textContent = 'Error loading logs: ' + error.message;
            container.style.display = 'block';
        }
    } else {
        container.style.display = 'none';
    }
} 

// ENHANCED: Search function with proper original document names and clickable URLs
async function searchOpenSearch() {
    const query = document.getElementById('searchInput').value.trim();
    const resultsDiv = document.getElementById('searchResults');
    resultsDiv.innerHTML = 'Searching...';

    if (!query) {
        resultsDiv.innerHTML = '<span class="warning">Please enter a search term.</span>';
        return;
    }

    try {
        const response = await fetch(`/search?q=${encodeURIComponent(query)}`);
        const data = await response.json();

        if (data.status === 'success') {
            if (data.results.length === 0) {
                resultsDiv.innerHTML = `
                    <div class="no-results">
                        <h3>No results found for "${query}"</h3>
                        <p>Try different keywords or check spelling.</p>
                    </div>
                `;
                return;
            }

            // Enhanced results display with original document names and clickable URLs
            const html = data.results.map(r => {
                // Extract original document name (this is what user wants to see)
                const originalDocName = r.title || '[Untitled Document]';
                
                // Extract clickable URL to original document
                const originalUrl = r.url || '#';
                const hasClickableUrl = r.has_clickable_url || (originalUrl && originalUrl !== '#');
                
                // Get preview text from chunk
                const previewText = (r.chunk && r.chunk.text) ? 
                    r.chunk.text.slice(0, 250) + '...' : 
                    'No preview available';
                
                // Build the result card
                return `
                    <div class="result-card enhanced-result">
                        <div class="result-header">
                            ${hasClickableUrl ? 
                                `<h4><a href="${originalUrl}" target="_blank" rel="noopener" class="original-doc-link">
                                    ${originalDocName}
                                    <span class="external-link-icon">🔗</span>
                                </a></h4>` :
                                `<h4 class="no-link-title">${originalDocName}</h4>`
                            }
                        </div>
                        
                        <div class="result-metadata">
                            <span class="result-collection">📁 ${r.collection}</span>
                            <span class="result-id">🆔 ${r.id || 'Unknown'}</span>
                            ${r.score ? `<span class="result-score">⭐ ${Math.round(r.score * 100)/100}</span>` : ''}
                        </div>
                        
                        <div class="result-preview">
                            <strong>Content Preview:</strong>
                            <p>${previewText}</p>
                        </div>
                        
                        <div class="result-actions">
                            ${hasClickableUrl ? 
                                `<a href="${originalUrl}" target="_blank" rel="noopener" class="view-original-btn">
                                    📄 View Original Document
                                </a>` :
                                `<span class="no-url-notice">⚠️ Original document link not available</span>`
                            }
                            
                            <button class="use-in-chat-btn" onclick="useInChat('${query}', '${originalDocName.replace(/'/g, "\\'")}')">
                                💬 Ask About This
                            </button>
                        </div>
                        
                        ${r.metadata && r.metadata.program ? 
                            `<div class="result-program">Program: ${r.metadata.program}</div>` : 
                            ''
                        }
                    </div>
                `;
            }).join('');

            // Add summary header
            const summaryHtml = `
                <div class="search-summary">
                    <h3>Search Results for "${query}"</h3>
                    <p>Found ${data.total_hits} documents (from ${data.total_chunks || data.total_hits} chunks)</p>
                </div>
            `;

            resultsDiv.innerHTML = summaryHtml + html;

            // Add back to chat button
            const backBtn = document.createElement('button');
            backBtn.textContent = '← Back to Chat';
            backBtn.className = 'back-to-chat-btn';
            backBtn.onclick = () => {
                document.getElementById('search-results').style.display = 'none';
                document.getElementById('messages').style.display = 'block';
                document.getElementById('searchInput').value = '';
            };
            resultsDiv.appendChild(backBtn);

        } else {
            resultsDiv.innerHTML = `<span class="warning">Search Error: ${data.error || 'Unknown error occurred'}</span>`;
        }
    } catch (err) {
        console.error('Search failed:', err);
        resultsDiv.innerHTML = `<span class="warning">Search failed: ${err.message}</span>`;
    }
}

// Helper function to use search result in chat
function useInChat(query, docName) {
    // Switch back to chat view
    document.getElementById('search-results').style.display = 'none';
    document.getElementById('messages').style.display = 'block';
    
    // Pre-fill chat input with a question about the document
    const chatInput = document.getElementById('chatInput');
    if (chatInput) {
        chatInput.value = `Tell me more about "${docName}" related to ${query}`;
        chatInput.focus();
    }
}

async function countDocsByProgram() {
    const container = document.getElementById("programCounts");
    container.innerHTML = "Counting documents...";

    try {
        const response = await fetch('/count_by_program');
        const data = await response.json();

        if (data.status === 'success') {
            const items = Object.entries(data.program_counts).map(
                ([prog, count]) => `<div><strong>${prog}</strong>: ${count} documents</div>`
            );
            container.innerHTML = `<div class="result-card"><h4>OpenSearch Document Counts</h4>${items.join('')}</div>`;
        } else {
            container.innerHTML = `<span class="warning">Error: ${data.error}</span>`;
        }
    } catch (error) {
        container.innerHTML = `<span class="warning">Failed to retrieve document counts: ${error.message}</span>`;
    }
}

// FIXED: Count by collection and program with better display
async function countDocsByCollectionAndProgram() {
    const container = document.getElementById("programCounts");
    container.innerHTML = "Counting documents by collection and program...";

    try {
        const response = await fetch('/count_by_collection_and_program');
        const data = await response.json();

        if (data.status === 'success') {
            let html = '<div class="result-card"><h4>Document Counts by Collection & Program</h4>';
            
            const counts = data.collection_program_counts;
            if (Object.keys(counts).length === 0) {
                html += '<p>No documents found in any collection.</p>';
            } else {
                for (const [collection, programs] of Object.entries(counts)) {
                    html += `<div style="margin-bottom: 12px;"><strong>${collection}</strong></div><ul style="margin-left: 20px;">`;
                    
                    if (Object.keys(programs).length === 0) {
                        html += '<li>No documents found</li>';
                    } else {
                        for (const [program, count] of Object.entries(programs)) {
                            let programDisplay = (!program || program === 'all') ? 
                                '<span style="color:#28a745; font-style: italic;">All Programs</span>' : 
                                program;
                            html += `<li>${programDisplay}: <strong>${count}</strong> documents</li>`;
                        }
                    }
                    html += `</ul>`;
                }
            }
            html += `</div>`;
            container.innerHTML = html;
        } else {
            container.innerHTML = `<span class="warning">Error: ${data.error}</span>`;
        }
    } catch (error) {
        container.innerHTML = `<span class="warning">Failed to retrieve counts: ${error.message}</span>`;
    }
}

async function checkConfig() {
    try {
        const response = await fetch('/health');
        const health = await response.json();

        let configStatus = "🔧 Configuration Status:\n\n";

        // Embeddings status
        if (health.components && health.components.embeddings) {
            const embeds = health.components.embeddings;
            configStatus += `⚡ Embeddings: ${embeds.status}\n`;
            configStatus += `   Model: ${embeds.model || 'Unknown'}\n`;
            configStatus += `   Dimensions: ${embeds.dimension || 'Unknown'}\n\n`;
        }

        // OpenSearch status
        if (health.components && health.components.opensearch) {
            const os = health.components.opensearch;
            configStatus += `🔍 OpenSearch: ${os.status}\n`;
            if (os.version) configStatus += `   Version: ${os.version}\n\n`;
        } else {
            configStatus += `🔍 OpenSearch: Not configured\n`;
            configStatus += `   Required: OPENSEARCH_HOST, OPENSEARCH_USER, OPENSEARCH_PASS\n\n`;
        }

        //  NEW: GENAI Service status
        if (health.components && health.components.genai) {
            const genai = health.components.genai;
            configStatus += `🤖 GenAI Service: ${genai.status}\n`;
            if (genai.endpoint) configStatus += `   Endpoint: ${genai.endpoint}\n`;
            if (genai.note) configStatus += `   ${genai.note}\n\n`;
        } else {
            configStatus += `🤖 GenAI Service: Not configured\n`;
            configStatus += `   Chat will use search results only\n`;
            configStatus += `   For AI responses: Set GENAI_ENDPOINT, GENAI_ACCESS_KEY\n\n`;
        }

        //  NEW: API Source status
        if (health.components && health.components.api_source) {
            const api = health.components.api_source;
            configStatus += `📡 Data Source API: ${api.status}\n`;
            if (api.base_url) configStatus += `   Base URL: ${api.base_url}\n`;
            if (api.auth_header) configStatus += `   Auth Header: ${api.auth_header}\n`;
            if (api.note) configStatus += `   ${api.note}\n`;
            if (api.missing_variables) {
                configStatus += `   Missing: ${api.missing_variables.join(', ')}\n`;
            }
        } else {
            configStatus += `📡 Data Source API: Not configured\n`;
            configStatus += `   Required: API_BASE_URL, API_AUTH_KEY, API_AUTH_VALUE\n`;
        }

        // NEW: Last import info
        if (health.last_import && health.last_import.timestamp) {
            const timestamp = new Date(health.last_import.timestamp).toLocaleString();
            configStatus += `\n📅 Last Import: ${timestamp} (${health.last_import.type || 'unknown'})\n`;
        } else {
            configStatus += `\n📅 Last Import: None found\n`;
        }

        configStatus += `\n🏗️ Configuration Summary:\n`;
        configStatus += `   • OpenSearch: Content storage and vector search\n`;
        configStatus += `   • GenAI Service: AI-powered chat responses\n`;
        configStatus += `   • API Source: Document import from external API\n`;
        configStatus += `   • Embeddings: Vector similarity search\n`;
        configStatus += `   • Incremental Import: Only process updated documents\n\n`;
        
        configStatus += `🔄 Architecture Flow:\n`;
        configStatus += `   User Query → OpenSearch → Context → GenAI → Response\n`;
        
        alert(configStatus);
    } catch (error) {
        alert('❌ Error checking configuration: ' + error.message);
    }
}

// NEW: Check for document updates in a collection
async function checkDocumentUpdates() {
    const collectionSelect = document.getElementById("collectionSelect");
    const collection = collectionSelect ? collectionSelect.value : "plans";
    
    if (collection === "all") {
        alert("Please select a specific collection to check for updates");
        return;
    }
    
    const container = document.getElementById("documentUpdates");
    container.innerHTML = "Checking for document updates...";
    
    try {
        const response = await fetch(`/check_document_updates/${collection}`);
        const data = await response.json();
        
        if (data.status === 'success') {
            let html = '<div class="result-card"><h4>📋 Document Update Check</h4>';
            html += `<p><strong>Collection:</strong> ${data.collection}</p>`;
            html += `<p><strong>Total Documents:</strong> ${data.total_documents}</p>`;
            html += `<p><strong>Updated Since Last Import:</strong> ${data.updated_documents}</p>`;
            
            if (data.last_import_timestamp) {
                const timestamp = new Date(data.last_import_timestamp).toLocaleString();
                html += `<p><strong>Last Import:</strong> ${timestamp}</p>`;
            } else {
                html += `<p><strong>Last Import:</strong> No previous import found</p>`;
            }
            
            if (data.updates_available) {
                html += `<div style="color: #E20074; font-weight: bold; margin: 10px 0;">✨ Updates Available!</div>`;
                
                if (data.updated_document_list && data.updated_document_list.length > 0) {
                    html += `<div style="margin-top: 15px;"><strong>Recently Updated Documents:</strong></div><ul style="margin-left: 20px;">`;
                    
                    data.updated_document_list.forEach(doc => {
                        const updatedDate = doc.updated ? new Date(doc.updated).toLocaleString() : 'Unknown';
                        html += `<li><strong>${doc.name}</strong> (${doc.id}) - Updated: ${updatedDate}</li>`;
                    });
                    
                    html += `</ul>`;
                    
                    if (data.total_updated_count > data.updated_document_list.length) {
                        html += `<p><em>... and ${data.total_updated_count - data.updated_document_list.length} more documents</em></p>`;
                    }
                }
                
                html += `<button onclick="startIncrementalImport('${collection}')" style="margin-top: 15px; background: #28a745;">🚀 Start Incremental Import</button>`;
            } else {
                html += `<div style="color: #28a745; font-weight: bold; margin: 10px 0;">✅ All documents are up to date</div>`;
            }
            
            html += `</div>`;
            container.innerHTML = html;
        } else {
            container.innerHTML = `<span class="warning">Error: ${data.error}</span>`;
        }
    } catch (error) {
        container.innerHTML = `<span class="warning">Failed to check updates: ${error.message}</span>`;
    }
}

// NEW: Start incremental import for specific collection
async function startIncrementalImport(collection) {
    if (!confirm(`Start incremental import for ${collection}?\nThis will only process documents that have been updated since the last import.`)) {
        return;
    }
    
    try {
        const response = await fetch("/import", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                collection: collection,
                import_type: "incremental"
            })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            alert(`Incremental import started for ${collection}!`);
            startPolling();
        } else {
            alert(`Error: ${data.detail || data.message || "Unknown error"}`);
        }
    } catch (error) {
        alert(`Import request failed: ${error.message}`);
    }
}

// NEW: Get import statistics
async function getImportStatistics() {
    const container = document.getElementById("importStats");
    container.innerHTML = "Loading import statistics...";
    
    try {
        const response = await fetch('/import_statistics');
        const data = await response.json();
        
        if (data.status === 'success') {
            const stats = data.statistics;
            let html = '<div class="result-card"><h4> Knowledge Base Statistics</h4>';
            
            // Overall stats
            html += `<div style="margin-bottom: 15px;">`;
            html += `<p><strong>Total Documents:</strong> ${stats.total_documents || 0}</p>`;
            html += `<p><strong>Total Chunks:</strong> ${stats.total_chunks || 0}</p>`;
            html += `<p><strong>Collections:</strong> ${Object.keys(stats.collections || {}).length}</p>`;
            html += `</div>`;
            
            // Import types
            if (stats.import_types && Object.keys(stats.import_types).length > 0) {
                html += `<div style="margin-bottom: 15px;"><strong>Import Types:</strong><ul style="margin-left: 20px;">`;
                for (const [type, count] of Object.entries(stats.import_types)) {
                    html += `<li>${type}: ${count} chunks</li>`;
                }
                html += `</ul></div>`;
            }
            
            // Date info
            if (stats.oldest_document || stats.newest_document) {
                html += `<div style="margin-bottom: 15px;"><strong>Date Range:</strong>`;
                if (stats.oldest_document) {
                    const oldestDate = new Date(stats.oldest_document).toLocaleString();
                    html += `<br>Oldest: ${oldestDate}`;
                }
                if (stats.newest_document) {
                    const newestDate = new Date(stats.newest_document).toLocaleString();
                    html += `<br>Newest: ${newestDate}`;
                }
                html += `</div>`;
            }
            
            // Last import info
            if (data.last_import_timestamp) {
                const timestamp = new Date(data.last_import_timestamp).toLocaleString();
                html += `<div style="background: #e3f2fd; padding: 10px; border-radius: 4px; margin-top: 15px;">`;
                html += `<strong>📅 Last Import:</strong> ${timestamp}`;
                html += `</div>`;
            }
            
            // Collection breakdown
            if (stats.collections && Object.keys(stats.collections).length > 0) {
                html += `<div style="margin-top: 20px;"><h5>Collection Breakdown:</h5><ul style="margin-left: 20px;">`;
                for (const [collection, collStats] of Object.entries(stats.collections)) {
                    html += `<li><strong>${collection}</strong>: ${collStats.unique_documents} docs, ${collStats.chunks} chunks</li>`;
                }
                html += `</ul></div>`;
            }
            
            html += `</div>`;
            container.innerHTML = html;
        } else {
            container.innerHTML = `<span class="warning">Error: ${data.error}</span>`;
        }
    } catch (error) {
        container.innerHTML = `<span class="warning">Failed to get statistics: ${error.message}</span>`;
    }
}

// NEW: Cleanup old chunks with better safety messaging and preview
async function cleanupOldChunks() {
    // First, get current statistics to show user what will be affected
    try {
        const statsResponse = await fetch('/import_statistics');
        const statsData = await statsResponse.json();
        
        let infoMessage = "🧹 CLEANUP OLD DATA - SAFETY INFORMATION\n\n";
        infoMessage += "This will ONLY remove chunks that are older than the specified days.\n";
        infoMessage += "It will NOT delete your entire database!\n\n";
        
        if (statsData.status === 'success') {
            const stats = statsData.statistics;
            infoMessage += `📊 Current Knowledge Base:\n`;
            infoMessage += `• Total Documents: ${stats.total_documents || 0}\n`;
            infoMessage += `• Total Chunks: ${stats.total_chunks || 0}\n`;
            
            if (stats.oldest_document) {
                const oldestDate = new Date(stats.oldest_document).toLocaleDateString();
                infoMessage += `• Oldest Chunk: ${oldestDate}\n`;
            }
            if (stats.newest_document) {
                const newestDate = new Date(stats.newest_document).toLocaleDateString();
                infoMessage += `• Newest Chunk: ${newestDate}\n`;
            }
        }
        
        infoMessage += "\n⚠️ WHAT GETS DELETED:\n";
        infoMessage += "• Only chunks indexed MORE than X days ago\n";
        infoMessage += "• Orphaned chunks from failed imports\n";
        infoMessage += "• Duplicate chunks from reprocessing\n\n";
        
        infoMessage += "✅ WHAT STAYS SAFE:\n";
        infoMessage += "• All recent chunks (within X days)\n";
        infoMessage += "• All current active documents\n";
        infoMessage += "• All successfully imported content\n\n";
        
        alert(infoMessage);
        
    } catch (error) {
        console.error("Failed to get stats:", error);
    }
    
    const maxAgeDays = prompt("🗓️ Enter maximum age in days for chunks to keep:\n\n(Chunks indexed MORE than this many days ago will be deleted)\n\nRecommended: 30-90 days", "30");
    
    if (!maxAgeDays || isNaN(maxAgeDays)) {
        return;
    }
    
    const ageNum = parseInt(maxAgeDays);
    
    // Preview what would be deleted
    try {
        const previewResponse = await fetch(`/preview_cleanup/${ageNum}`);
        const previewData = await previewResponse.json();
        
        if (previewData.status === 'success') {
            const preview = previewData.preview;
            const cutoffDate = new Date(preview.cutoff_date).toLocaleDateString();
            
            let previewMessage = `📋 CLEANUP PREVIEW\n\n`;
            previewMessage += `🗓️ Will delete chunks indexed before: ${cutoffDate}\n\n`;
            previewMessage += `📊 Impact Analysis:\n`;
            previewMessage += `• Total chunks in database: ${preview.total_chunks}\n`;
            previewMessage += `• Old chunks to DELETE: ${preview.old_chunks_to_delete}\n`;
            previewMessage += `• Recent chunks to KEEP: ${preview.chunks_to_keep}\n`;
            previewMessage += `• Percentage to delete: ${preview.percentage_to_delete}%\n\n`;
            
            if (!preview.is_safe) {
                previewMessage += `⚠️ WARNING: This would delete more than 50% of your data!\n`;
                previewMessage += `Consider using a smaller number of days.\n\n`;
            } else {
                previewMessage += `✅ SAFE: This will preserve most of your recent data.\n\n`;
            }
            
            if (Object.keys(preview.collection_breakdown).length > 0) {
                previewMessage += `📁 Breakdown by Collection:\n`;
                for (const [collection, count] of Object.entries(preview.collection_breakdown)) {
                    previewMessage += `• ${collection}: ${count} old chunks\n`;
                }
                previewMessage += `\n`;
            }
            
            if (preview.old_chunks_to_delete === 0) {
                alert(`✅ No cleanup needed!\n\nAll chunks are newer than ${ageNum} days.`);
                return;
            }
            
            previewMessage += `This action cannot be undone. Proceed with cleanup?`;
            
            if (!confirm(previewMessage)) {
                return;
            }
            
        } else {
            // Fallback if preview fails
            const cutoffDate = new Date();
            cutoffDate.setDate(cutoffDate.getDate() - ageNum);
            
            const confirmMessage = `🧹 CONFIRM CLEANUP OPERATION\n\n` +
                `This will remove chunks indexed BEFORE: ${cutoffDate.toLocaleDateString()}\n\n` +
                `⚠️ Only chunks older than ${ageNum} days will be deleted.\n` +
                `✅ Recent content and active documents will be preserved.\n\n` +
                `This action cannot be undone. Continue?`;
            
            if (!confirm(confirmMessage)) {
                return;
            }
        }
        
    } catch (error) {
        console.error("Preview failed:", error);
        // Continue with basic confirmation
        const cutoffDate = new Date();
        cutoffDate.setDate(cutoffDate.getDate() - ageNum);
        
        if (!confirm(`Remove chunks older than ${ageNum} days (before ${cutoffDate.toLocaleDateString()})?\n\nThis cannot be undone.`)) {
            return;
        }
    }
    
    // Perform the actual cleanup
    try {
        const response = await fetch("/cleanup_old_chunks", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ max_age_days: ageNum })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            alert(`✅ Cleanup completed safely!\n\n` +
                `📊 Results:\n` +
                `• Cleaned up: ${data.chunks_cleaned} old chunks\n` +
                `• Age threshold: ${data.max_age_days} days\n` +
                `• Your active documents are safe\n\n` +
                `💡 This helps optimize search performance by removing stale data.`);
        } else {
            alert(`❌ Cleanup failed: ${data.error}`);
        }
    } catch (error) {
        alert(`Cleanup request failed: ${error.message}`);
    }
}

// NEW: Search for documents to help with reindexing
async function searchDocumentForReindex() {
    const collectionSelect = document.getElementById("collectionSelect");
    const collection = collectionSelect ? collectionSelect.value : null;
    
    if (!collection || collection === "all") {
        alert("Please select a specific collection first");
        return;
    }
    
    const searchTerm = document.getElementById("reindexDocName").value.trim();
    if (!searchTerm) {
        alert("Please enter a document name or keyword to search for");
        return;
    }
    
    try {
        // Use the existing search endpoint to find documents
        const response = await fetch(`/search?q=${encodeURIComponent(searchTerm)}`);
        const data = await response.json();
        
        if (data.status === 'success' && data.results.length > 0) {
            // Create a selection dialog
            let options = "Found documents:\n\n";
            const matchingDocs = data.results.filter(r => r.collection === collection).slice(0, 10);
            
            if (matchingDocs.length === 0) {
                alert(`No documents found in collection "${collection}" matching "${searchTerm}"`);
                return;
            }
            
            matchingDocs.forEach((doc, index) => {
                options += `${index + 1}. ${doc.title} (ID: ${doc.id})\n`;
            });
            
            options += `\nEnter the number (1-${matchingDocs.length}) of the document to reindex:`;
            
            const selection = prompt(options);
            const selectedIndex = parseInt(selection) - 1;
            
            if (selectedIndex >= 0 && selectedIndex < matchingDocs.length) {
                const selectedDoc = matchingDocs[selectedIndex];
                
                // Fill in the form fields
                document.getElementById("reindexDocId").value = selectedDoc.id;
                document.getElementById("reindexDocName").value = selectedDoc.title;
                
                // Ask if they want to proceed with reindexing
                if (confirm(`Reindex document "${selectedDoc.title}" (ID: ${selectedDoc.id})?`)) {
                    reindexDocumentWithInputs();
                }
            }
        } else {
            alert(`No documents found matching "${searchTerm}"`);
        }
    } catch (error) {
        alert(`Search failed: ${error.message}`);
    }
}

// UPDATED: Reindex specific document using form inputs
async function reindexDocument() {
    const documentId = document.getElementById("reindexDocId").value.trim();
    const documentName = document.getElementById("reindexDocName").value.trim();
    
    if (!documentId) {
        alert("Please enter a document ID or use 'Search & Select Document' to find one");
        return;
    }
    
    const collectionSelect = document.getElementById("collectionSelect");
    const collection = collectionSelect ? collectionSelect.value : null;
    
    if (!collection || collection === "all") {
        alert("Please select a specific collection first");
        return;
    }
    
    const displayName = documentName || documentId;
    if (!confirm(`Reindex document "${displayName}" (ID: ${documentId}) in collection ${collection}?\n\nThis will:\n- Delete existing chunks for this document\n- Fetch fresh data from API\n- Reprocess content and generate new embeddings\n- Re-index with current settings`)) {
        return;
    }
    
    reindexDocumentWithInputs();
}

// NEW: Browse documents in collection for reindexing
async function browseDocumentsForReindex() {
    const collectionSelect = document.getElementById("collectionSelect");
    const collection = collectionSelect ? collectionSelect.value : null;
    
    if (!collection || collection === "all") {
        alert("Please select a specific collection first");
        return;
    }
    
    try {
        const response = await fetch(`/list_documents/${collection}?limit=20`);
        const data = await response.json();
        
        if (data.status === 'success' && data.documents.length > 0) {
            // Create a nicer selection interface
            let html = `
                <div style="max-width: 600px; max-height: 400px; overflow-y: auto; padding: 15px; background: white; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
                    <h3 style="margin-top: 0; color: #6E32A0;">📋 Select Document to Reindex</h3>
                    <p style="color: #666; font-size: 0.9em;">Collection: <strong>${collection}</strong> (${data.total_documents} documents)</p>
                    <div style="margin-bottom: 15px;">
                        <input type="text" id="docSearchFilter" placeholder="Filter documents..." style="width: 100%; padding: 8px; border: 1px solid #ddd; border-radius: 4px;" onkeyup="filterDocumentList()">
                    </div>
                    <div id="documentList">
            `;
            
            data.documents.forEach((doc, index) => {
                const updatedDate = doc.updated ? new Date(doc.updated).toLocaleDateString() : 'Unknown';
                const lastIndexed = doc.last_indexed ? new Date(doc.last_indexed).toLocaleDateString() : 'Unknown';
                
                html += `
                    <div class="doc-item" data-name="${doc.name.toLowerCase()}" data-id="${doc.id.toLowerCase()}" style="border: 1px solid #ddd; border-radius: 6px; padding: 12px; margin: 8px 0; cursor: pointer; transition: all 0.2s;" onclick="selectDocumentForReindex('${doc.id}', '${doc.name.replace(/'/g, "\\'")}')">
                        <div style="font-weight: bold; color: #333; margin-bottom: 4px;">${doc.name}</div>
                        <div style="font-size: 0.85em; color: #666;">
                            <strong>ID:</strong> ${doc.id}<br>
                            <strong>Program:</strong> ${doc.program}<br>
                            <strong>Updated:</strong> ${updatedDate}<br>
                            <strong>Chunks:</strong> ${doc.chunk_count}
                        </div>
                        ${doc.url ? `<div style="margin-top: 6px;"><a href="${doc.url}" target="_blank" style="font-size: 0.8em; color: #17a2b8;">🔗 View Original</a></div>` : ''}
                    </div>
                `;
            });
            
            html += `
                    </div>
                    <div style="margin-top: 15px; text-align: center;">
                        <button onclick="closeBrowseDialog()" style="background: #6c757d; color: white; border: none; padding: 8px 16px; border-radius: 4px; cursor: pointer;">Cancel</button>
                    </div>
                </div>
            `;
            
            // Create overlay
            const overlay = document.createElement('div');
            overlay.id = 'browseDocumentsOverlay';
            overlay.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(0,0,0,0.5);
                z-index: 1000;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
                box-sizing: border-box;
            `;
            overlay.innerHTML = html;
            
            document.body.appendChild(overlay);
            
            // Add CSS for hover effects
            const style = document.createElement('style');
            style.textContent = `
                .doc-item:hover {
                    background: #f8f9fa !important;
                    border-color: #6E32A0 !important;
                    transform: translateY(-1px);
                }
            `;
            document.head.appendChild(style);
            
        } else {
            alert(`No documents found in collection "${collection}"`);
        }
    } catch (error) {
        alert(`Failed to browse documents: ${error.message}`);
    }
}

// Helper function to filter document list
function filterDocumentList() {
    const filter = document.getElementById('docSearchFilter').value.toLowerCase();
    const docItems = document.querySelectorAll('.doc-item');
    
    docItems.forEach(item => {
        const name = item.getAttribute('data-name');
        const id = item.getAttribute('data-id');
        
        if (name.includes(filter) || id.includes(filter)) {
            item.style.display = 'block';
        } else {
            item.style.display = 'none';
        }
    });
}

// Helper function to select a document for reindexing
function selectDocumentForReindex(docId, docName) {
    // Fill in the form fields
    document.getElementById("reindexDocId").value = docId;
    document.getElementById("reindexDocName").value = docName;
    
    // Close the browse dialog
    closeBrowseDialog();
    
    // Ask if they want to proceed with reindexing
    if (confirm(`Reindex document "${docName}" (ID: ${docId})?\n\nThis will delete existing chunks and reprocess the document with fresh data from the API.`)) {
        reindexDocumentWithInputs();
    }
}

// Helper function to close browse dialog
function closeBrowseDialog() {
    const overlay = document.getElementById('browseDocumentsOverlay');
    if (overlay) {
        overlay.remove();
    }
}
async function reindexDocumentWithInputs() {
    const documentId = document.getElementById("reindexDocId").value.trim();
    const documentName = document.getElementById("reindexDocName").value.trim();
    const collectionSelect = document.getElementById("collectionSelect");
    const collection = collectionSelect.value;
    
    const displayName = documentName || documentId;
    
    try {
        const response = await fetch("/reindex_document", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                document_id: documentId,
                collection: collection
            })
        });
        
        const data = await response.json();
        
        if (data.status === 'success') {
            alert(`✅ Document "${displayName}" reindexed successfully!\n\nResults:\n- Chunks indexed: ${data.chunks_indexed}\n- Collection: ${data.collection}\n- Document ID: ${data.document_id}`);
            
            // Clear the form fields
            document.getElementById("reindexDocId").value = "";
            document.getElementById("reindexDocName").value = "";
        } else {
            alert(`❌ Reindex failed: ${data.error}`);
        }
    } catch (error) {
        alert(`Reindex request failed: ${error.message}`);
    }
}

// Initial page load behavior
document.addEventListener('DOMContentLoaded', () => {
    console.log("✅ DOM fully loaded, initializing...");

    refreshStatus();
    checkLastImportInfo(); // NEW: Load last import info

    // Server availability check (basic health ping)
    fetch("/ping")
        .then(r => r.json())
        .then(console.log)
        .catch(console.error);

    setInterval(() => {
        if (!pollInterval) refreshStatus();
    }, 10000);
});

document.getElementById('debugToggleBtn').addEventListener('click', async () => {
    const overlay = document.getElementById('debugOverlay');
    const content = document.getElementById('debugContent');

    if (overlay.style.display === 'none') {
        overlay.style.display = 'block';
        content.textContent = 'Loading debug info...';

        try {
            const [ping, status, logs] = await Promise.all([
                fetch('/ping').then(r => r.json()),
                fetch('/status').then(r => r.json()),
                fetch('/logs').then(r => r.json())
            ]);

            content.innerHTML = `
                <strong>Ping:</strong> ${ping.status}<br><br>
                <strong>Status:</strong> ${status.status}<br>
                <strong>Step:</strong> ${status.current_step}<br>
                <strong>Import Type:</strong> ${status.import_type || 'unknown'}<br>
                <strong>Start:</strong> ${status.start_time}<br>
                <strong>End:</strong> ${status.end_time || '—'}<br><br>
                <strong>Last Logs:</strong><br><pre style="max-height:150px;overflow:auto;">${(logs.logs || []).slice(-5).join('\n')}</pre>
            `;
        } catch (err) {
            content.innerHTML = `<span style="color:red;">Debug fetch error: ${err.message}</span>`;
        }
    } else {
        overlay.style.display = 'none';
    }
});

// Enhanced CSS for better search result display
const enhancedSearchCSS = `
<style>
.enhanced-result {
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 16px;
    margin: 12px 0;
    background: white;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    transition: box-shadow 0.2s ease;
}

.enhanced-result:hover {
    box-shadow: 0 4px 8px rgba(0,0,0,0.15);
}

.result-header h4 {
    margin: 0 0 8px 0;
    font-size: 1.2em;
}

.original-doc-link {
    color: #6E32A0;
    text-decoration: none;
    font-weight: bold;
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

.original-doc-link:hover {
    color: #8B4CB8;
    text-decoration: underline;
}

.external-link-icon {
    font-size: 0.8em;
    opacity: 0.7;
}

.no-link-title {
    color: #333;
    margin: 0;
}

.result-metadata {
    display: flex;
    gap: 15px;
    margin: 8px 0;
    font-size: 0.9em;
    color: #666;
    flex-wrap: wrap;
}

.result-metadata span {
    background: #f0f0f0;
    padding: 3px 8px;
    border-radius: 4px;
    font-size: 0.85em;
}

.result-collection { background: #e3f2fd; }
.result-id { background: #f3e5f5; }
.result-score { background: #fff3e0; }

.result-preview {
    margin: 12px 0;
    padding: 10px;
    background: #fafafa;
    border-radius: 4px;
    border-left: 3px solid #6E32A0;
}

.result-preview strong {
    color: #6E32A0;
    font-size: 0.9em;
}

.result-preview p {
    margin: 6px 0 0 0;
    line-height: 1.4;
    color: #555;
}

.result-actions {
    display: flex;
    gap: 10px;
    margin-top: 12px;
    align-items: center;
    flex-wrap: wrap;
}

.view-original-btn {
    background: #10b981;
    color: white;
    padding: 8px 16px;
    border-radius: 4px;
    text-decoration: none;
    font-size: 0.9em;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    transition: background-color 0.2s;
}

.view-original-btn:hover {
    background: #059669;
}

.use-in-chat-btn {
    background: #6E32A0;
    color: white;
    border: none;
    padding: 8px 16px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.9em;
    transition: background-color 0.2s;
}

.use-in-chat-btn:hover {
    background: #8B4CB8;
}

.no-url-notice {
    color: #f59e0b;
    font-size: 0.9em;
    font-style: italic;
}

.result-program {
    margin-top: 8px;
    font-size: 0.85em;
    color: #666;
    font-style: italic;
}

.search-summary {
    background: #f8f9fa;
    padding: 16px;
    border-radius: 6px;
    margin-bottom: 20px;
    border-left: 4px solid #6E32A0;
}

.search-summary h3 {
    margin: 0 0 8px 0;
    color: #6E32A0;
}

.search-summary p {
    margin: 0;
    color: #666;
}

.no-results {
    text-align: center;
    padding: 40px 20px;
    color: #666;
}

.no-results h3 {
    color: #333;
    margin-bottom: 8px;
}

.back-to-chat-btn {
    background: #6c757d;
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 6px;
    cursor: pointer;
    margin-top: 20px;
    width: 100%;
}

.back-to-chat-btn:hover {
    background: #5a6268;
}

@media (max-width: 768px) {
    .result-metadata {
        flex-direction: column;
        gap: 8px;
    }
    
    .result-actions {
        flex-direction: column;
        align-items: stretch;
    }
    
    .view-original-btn,
    .use-in-chat-btn {
        text-align: center;
        justify-content: center;
    }
}
</style>
`;

// Inject enhanced CSS into the page
if (!document.getElementById('enhanced-search-styles')) {
    const styleElement = document.createElement('style');
    styleElement.id = 'enhanced-search-styles';
    styleElement.innerHTML = enhancedSearchCSS;
    document.head.appendChild(styleElement);
}

// FIXED: Explicitly expose all functions to global scope for onclick handlers
window.startImport = startImport;
window.debugAPIData = debugAPIData;
window.refreshStatus = refreshStatus;
window.toggleLogs = toggleLogs;
window.searchOpenSearch = searchOpenSearch;
window.useInChat = useInChat;
window.countDocsByProgram = countDocsByProgram;
window.countDocsByCollectionAndProgram = countDocsByCollectionAndProgram;
window.checkConfig = checkConfig;
window.checkLastImportInfo = checkLastImportInfo;
window.clearImportTimestamp = clearImportTimestamp;
window.checkDocumentUpdates = checkDocumentUpdates;
window.startIncrementalImport = startIncrementalImport;
window.getImportStatistics = getImportStatistics;
window.cleanupOldChunks = cleanupOldChunks;
window.reindexDocument = reindexDocument;
window.searchDocumentForReindex = searchDocumentForReindex;
window.browseDocumentsForReindex = browseDocumentsForReindex;
window.filterDocumentList = filterDocumentList;
window.selectDocumentForReindex = selectDocumentForReindex;
window.closeBrowseDialog = closeBrowseDialog;
window.reindexDocumentWithInputs = reindexDocumentWithInputs;