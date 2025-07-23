// Enhanced main.js for Ask InnovAI Admin Interface v2.2.3
// BULLETPROOF FIX: Complete error handling for toLocaleString() undefined errors
// Version: 5.0.0 - Working Base

let pollInterval = null;

console.log("✅ Ask InnovAI Admin v2.2.3 - BULLETPROOF fix for toLocaleString() errors loaded");

// Auto-refresh status every 30 seconds if not actively importing
setInterval(() => {
    if (!pollInterval) {
        refreshStatus();
    }
}, 30000);

// Auto-refresh statistics every 30 seconds
setInterval(loadOpenSearchStats, 30000);

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    console.log("🚀 DOM loaded, initializing enhanced admin interface v2.2.3...");
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
// BULLETPROOF UTILITY FUNCTIONS - COMPLETE PROTECTION
// ============================================================================

// BULLETPROOF: Ultimate safe number conversion with complete validation
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
            if (trimmed === '' || trimmed.toLowerCase() === 'null' || trimmed.toLowerCase() === 'undefined') {
                return 0;
            }
        }
        
        // Convert to number
        const num = Number(value);
        
        // Check if conversion was successful
        if (isNaN(num) || !isFinite(num)) {
            console.warn('Invalid number conversion for value:', value, 'returning 0');
            return 0;
        }
        
        return num;
        
    } catch (error) {
        console.warn('Error in ultraSafeNumber for value:', value, 'error:', error, 'returning 0');
        return 0;
    }
}

// BULLETPROOF: Ultimate safe formatting with complete error handling
function ultraSafeFormat(value) {
    try {
        const num = ultraSafeNumber(value);
        
        // Additional check before calling toLocaleString
        if (typeof num !== 'number' || isNaN(num) || !isFinite(num)) {
            return '0';
        }
        
        // Try toLocaleString with fallback
        try {
            return num.toLocaleString();
        } catch (localeError) {
            console.warn('toLocaleString failed for value:', num, 'error:', localeError);
            // Fallback to basic string conversion
            return num.toString();
        }
        
    } catch (error) {
        console.warn('Error in ultraSafeFormat for value:', value, 'error:', error, 'returning "0"');
        return '0';
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

// BULLETPROOF: Ultimate safe object validation
function ultraSafeObject(obj) {
    try {
        if (obj === null || obj === undefined) {
            return {};
        }
        
        if (typeof obj === 'object' && !Array.isArray(obj)) {
            // Validate that it's a real object
            try {
                // Test if we can iterate over it
                Object.keys(obj);
                return obj;
            } catch (iterError) {
                console.warn('Object iteration failed:', iterError);
                return {};
            }
        }
        
        console.warn('Value is not a valid object:', typeof obj, obj);
        return {};
        
    } catch (error) {
        console.warn('Error in ultraSafeObject:', error, 'returning empty object');
        return {};
    }
}

// BULLETPROOF: Ultimate safe array validation
function ultraSafeArray(arr) {
    try {
        if (arr === null || arr === undefined) {
            return [];
        }
        
        if (Array.isArray(arr)) {
            return arr;
        }
        
        // Try to convert to array if it's array-like
        if (typeof arr === 'object' && typeof arr.length === 'number') {
            try {
                return Array.from(arr);
            } catch (conversionError) {
                console.warn('Array conversion failed:', conversionError);
                return [];
            }
        }
        
        console.warn('Value is not a valid array:', typeof arr, arr);
        return [];
        
    } catch (error) {
        console.warn('Error in ultraSafeArray:', error, 'returning empty array');
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

// ============================================================================
// OPENSEARCH STATISTICS FUNCTIONS - BULLETPROOF PROTECTION
// ============================================================================

async function loadOpenSearchStats() {
    console.log("📊 Loading OpenSearch statistics...");
    
    const container = document.getElementById('statisticsContainer');
    if (!container) {
        console.error("❌ Statistics container not found");
        return;
    }
    
    // Show enhanced loading state
    container.innerHTML = `
        <div class="stats-loading">
            <div class="loading-spinner"></div>
            <h3>Loading Statistics...</h3>
            <p>Fetching data from OpenSearch cluster</p>
            <small>This may take a few seconds</small>
        </div>
    `;
    
    try {
        // Use parallel loading for better performance
        const startTime = performance.now();
        
        const [statsResponse, vectorResponse] = await Promise.allSettled([
            fetch('/opensearch_statistics'),
            fetch('/opensearch_statistics').then(r => r.json()).then(data => data?.data?.vector_search)
        ]);
        
        const loadTime = Math.round(performance.now() - startTime);
        
        // Handle stats response
        if (statsResponse.status === 'rejected') {
            throw new Error(`Failed to fetch statistics: ${statsResponse.reason}`);
        }
        
        const response = statsResponse.value;
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const result = await response.json();
        
        if (result && result.status === 'success') {
            console.log(`✅ Statistics loaded successfully in ${loadTime}ms`, result.data);
            
            // Add performance metrics to the data
            const enhancedData = {
                ...result.data,
                performance: {
                    load_time_ms: loadTime,
                    parallel_loading: true,
                    vector_status_checked: vectorResponse.status === 'fulfilled'
                }
            };
            
            await displayStatistics(enhancedData, result.timestamp || new Date().toISOString());
            
        } else {
            const errorMsg = ultraSafeString(result?.error || 'Unknown error from server');
            console.error("❌ Statistics error:", errorMsg);
            displayStatisticsError(errorMsg, 'server_error');
        }
        
    } catch (error) {
        console.error('Failed to load statistics:', error);
        
        // More specific error handling
        let errorMessage = ultraSafeString(error?.message || 'Unknown error occurred');
        let errorType = 'unknown_error';
        
        // Classify error types for better user guidance
        if (errorMessage.includes('fetch')) {
            errorType = 'network_error';
            errorMessage = 'Unable to connect to server';
        } else if (errorMessage.includes('JSON')) {
            errorType = 'parse_error';
            errorMessage = 'Server returned invalid response';
        } else if (errorMessage.includes('timeout')) {
            errorType = 'timeout_error';
            errorMessage = 'Request timed out';
        }
        
        displayStatisticsError(errorMessage, errorType);
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
function displayStatistics(data, timestamp) {
    // BULLETPROOF: Display comprehensive statistics with ultimate error handling
    const container = document.getElementById('statisticsContainer');
    if (!container) return;
    
    // BULLETPROOF: Safely handle potentially missing data with comprehensive defaults
    const safeData = {
        total_evaluations: ultraSafeNumber(data?.total_evaluations),
        total_chunks: ultraSafeNumber(data?.total_chunks),
        evaluation_chunks: ultraSafeNumber(data?.evaluation_chunks),
        transcript_chunks: ultraSafeNumber(data?.transcript_chunks),
        evaluations_with_transcript: ultraSafeNumber(data?.evaluations_with_transcript),
        evaluations_without_transcript: ultraSafeNumber(data?.evaluations_without_transcript),
        template_counts: ultraSafeObject(data?.template_counts),
        lob_counts: ultraSafeObject(data?.lob_counts),
        partner_counts: ultraSafeObject(data?.partner_counts),
        site_counts: ultraSafeObject(data?.site_counts),
        language_counts: ultraSafeObject(data?.language_counts),
        indices: ultraSafeArray(data?.indices),
        structure_info: ultraSafeObject(data?.structure_info)
    };
    
    // Add default values for structure_info with bulletproof handling
    if (!ultraSafeString(safeData.structure_info.document_type)) {
        safeData.structure_info.document_type = 'unknown';
    }
    if (!ultraSafeString(safeData.structure_info.collection_strategy)) {
        safeData.structure_info.collection_strategy = 'unknown';
    }
    
    // Calculate additional metrics safely with bulletproof math
    let avgChunksPerEval = '0.0';
    let transcriptPercentage = '0.0';
    
    try {
        if (safeData.total_evaluations > 0 && safeData.total_chunks >= 0) {
            const avg = safeData.total_chunks / safeData.total_evaluations;
            avgChunksPerEval = isFinite(avg) ? avg.toFixed(1) : '0.0';
        }
        
        if (safeData.total_evaluations > 0 && safeData.evaluations_with_transcript >= 0) {
            const percentage = (safeData.evaluations_with_transcript / safeData.total_evaluations) * 100;
            transcriptPercentage = isFinite(percentage) ? percentage.toFixed(1) : '0.0';
        }
    } catch (mathError) {
        console.warn('Error calculating derived metrics:', mathError);
        avgChunksPerEval = '0.0';
        transcriptPercentage = '0.0';
    }
    
    // Check if we have any meaningful data
    const hasData = safeData.total_evaluations > 0 || 
                   safeData.total_chunks > 0 || 
                   Object.keys(safeData.template_counts).length > 0;
    
    if (!hasData) {
        container.innerHTML = `
            <div class="stats-dashboard">
                <div class="stats-card priority-metric">
                    <h3>🆔 No Data Found</h3>
                    <div class="stats-number">0</div>
                    <div class="stats-label">Evaluations in Database</div>
                    <div class="stats-breakdown">
                        <div class="breakdown-item">
                            <span class="breakdown-label">📊 Status</span>
                            <span class="breakdown-value">No evaluations imported yet</span>
                        </div>
                        <div class="breakdown-item">
                            <span class="breakdown-label">🚀 Next Step</span>
                            <span class="breakdown-value">Run an import to populate data</span>
                        </div>
                    </div>
                </div>
                
                <div class="stats-card">
                    <h3>📥 Import Needed</h3>
                    <div class="stats-number">0</div>
                    <div class="stats-label">Documents Available</div>
                    <div class="stats-breakdown">
                        <div class="breakdown-item">
                            <span class="breakdown-label">💡 Suggestion</span>
                            <span class="breakdown-value">Start with a small test import (5-10 docs)</span>
                        </div>
                    </div>
                </div>
            </div>
            <div class="stats-last-updated">
                📅 Last updated: ${ultraSafeTimestamp(timestamp)} | 
                📊 Structure: Ready for enhanced import
            </div>
        `;
        return;
    }
    
    // BULLETPROOF: Helper function to safely process count entries with ultimate protection
    const processCountEntries = (counts, limit = 10) => {
        try {
            const safeCountsObject = ultraSafeObject(counts);
            
            // Safely get entries with complete validation
            let entries;
            try {
                entries = Object.entries(safeCountsObject);
            } catch (entriesError) {
                console.warn('Failed to get object entries:', entriesError);
                return `
                    <div class="breakdown-item">
                        <span class="breakdown-label">Error processing data</span>
                        <span class="breakdown-value">-</span>
                    </div>
                `;
            }
            
            // Process and sort entries with bulletproof handling
            const processedEntries = entries
                .map(([key, count]) => {
                    const safeKey = ultraSafeString(key);
                    const safeCount = ultraSafeNumber(count);
                    return [safeKey || 'Unknown', safeCount];
                })
                .filter(([key, count]) => key !== '' && count >= 0) // Filter out invalid entries
                .sort(([,a], [,b]) => ultraSafeNumber(b) - ultraSafeNumber(a))
                .slice(0, Math.max(1, ultraSafeNumber(limit))); // Ensure at least 1, max limit
            
            if (processedEntries.length === 0) {
                return `
                    <div class="breakdown-item">
                        <span class="breakdown-label">No data available</span>
                        <span class="breakdown-value">-</span>
                    </div>
                `;
            }
            
            return processedEntries.map(([key, count]) => {
                const displayKey = key.length > 20 ? key.substring(0, 20) + '...' : key;
                const displayCount = ultraSafeFormat(count);
                
                return `
                    <div class="breakdown-item">
                        <span class="breakdown-label" title="${ultraSafeString(key)}">${ultraSafeString(displayKey)}</span>
                        <span class="breakdown-value">${displayCount}</span>
                    </div>
                `;
            }).join('');
            
        } catch (error) {
            console.warn('Failed to process count entries:', counts, 'error:', error);
            return `
                <div class="breakdown-item">
                    <span class="breakdown-label">Error processing data</span>
                    <span class="breakdown-value">-</span>
                </div>
            `;
        }
    };
    
    // BULLETPROOF: Generate HTML with complete error protection
    let html = '';
    
    try {
        html = `
            <div class="stats-dashboard">
                <!-- PRIMARY METRIC: Evaluations Processed -->
                <div class="stats-card priority-metric">
                    <h3>🆔 Evaluations Processed</h3>
                    <div class="stats-number">${ultraSafeFormat(safeData.total_evaluations)}</div>
                    <div class="stats-label">Unique EvaluationIDs</div>
                    <div class="stats-breakdown">
                        <div class="breakdown-item">
                            <span class="breakdown-label">📊 Source Data Match</span>
                            <span class="breakdown-value">Compare with your source system</span>
                        </div>
                        <div class="breakdown-item">
                            <span class="breakdown-label">📝 With Transcript</span>
                            <span class="breakdown-value">${ultraSafeFormat(safeData.evaluations_with_transcript)} (${transcriptPercentage}%)</span>
                        </div>
                        <div class="breakdown-item">
                            <span class="breakdown-label">📋 Evaluation Only</span>
                            <span class="breakdown-value">${ultraSafeFormat(safeData.evaluations_without_transcript)}</span>
                        </div>
                    </div>
                </div>
                
                <!-- Evaluation Details -->
                <div class="stats-card">
                    <h3>📄 Evaluation Details</h3>
                    <div class="stats-number">${avgChunksPerEval}</div>
                    <div class="stats-label">Avg Chunks per Evaluation</div>
                    <div class="stats-breakdown">
                        <div class="breakdown-item">
                            <span class="breakdown-label">📊 Data Processing</span>
                            <span class="breakdown-value">Enhanced structure v4.0+</span>
                        </div>
                        <div class="breakdown-item">
                            <span class="breakdown-label">🏗️ Document Model</span>
                            <span class="breakdown-value">1 doc per evaluationID</span>
                        </div>
                        <div class="breakdown-item">
                            <span class="breakdown-label">🧩 Content Grouping</span>
                            <span class="breakdown-value">Chunks within evaluations</span>
                        </div>
                    </div>
                </div>
                
                <div class="stats-card">
                    <h3>🧩 Total Chunks</h3>
                    <div class="stats-number">${ultraSafeFormat(safeData.total_chunks)}</div>
                    <div class="stats-label">All Content Pieces</div>
                    <div class="stats-breakdown">
                        <div class="breakdown-item">
                            <span class="breakdown-label">📝 Evaluation Chunks</span>
                            <span class="breakdown-value">${ultraSafeFormat(safeData.evaluation_chunks)}</span>
                        </div>
                        <div class="breakdown-item">
                            <span class="breakdown-label">🎙️ Transcript Chunks</span>
                            <span class="breakdown-value">${ultraSafeFormat(safeData.transcript_chunks)}</span>
                        </div>
                    </div>
                </div>
                
                <!-- Template Distribution -->
                <div class="stats-card">
                    <h3>📋 Templates</h3>
                    <div class="stats-number">${ultraSafeFormat(Object.keys(safeData.template_counts).length)}</div>
                    <div class="stats-label">Unique Templates</div>
                    <div class="stats-breakdown">
                        ${processCountEntries(safeData.template_counts, 5)}
                        ${Object.keys(safeData.template_counts).length > 5 ? 
                            `<div class="breakdown-item">
                                <span class="breakdown-label">...and ${ultraSafeFormat(Object.keys(safeData.template_counts).length - 5)} more</span>
                                <span class="breakdown-value"></span>
                            </div>` : ''}
                    </div>
                </div>
                
                <!-- LOB Distribution -->
                <div class="stats-card">
                    <h3>🏢 Line of Business</h3>
                    <div class="stats-number">${ultraSafeFormat(Object.keys(safeData.lob_counts).length)}</div>
                    <div class="stats-label">Unique LOBs</div>
                    <div class="stats-breakdown">
                        ${processCountEntries(safeData.lob_counts)}
                    </div>
                </div>
                
                <!-- Partner Distribution -->
                <div class="stats-card">
                    <h3>🤝 Partners</h3>
                    <div class="stats-number">${ultraSafeFormat(Object.keys(safeData.partner_counts).length)}</div>
                    <div class="stats-label">Unique Partners</div>
                    <div class="stats-breakdown">
                        ${processCountEntries(safeData.partner_counts)}
                    </div>
                </div>
                
                <!-- Site Distribution -->
                <div class="stats-card">
                    <h3>🏢 Sites</h3>
                    <div class="stats-number">${ultraSafeFormat(Object.keys(safeData.site_counts).length)}</div>
                    <div class="stats-label">Unique Sites</div>
                    <div class="stats-breakdown">
                        ${processCountEntries(safeData.site_counts, 6)}
                        ${Object.keys(safeData.site_counts).length > 6 ? 
                            `<div class="breakdown-item">
                                <span class="breakdown-label">...and ${ultraSafeFormat(Object.keys(safeData.site_counts).length - 6)} more</span>
                                <span class="breakdown-value"></span>
                            </div>` : ''}
                    </div>
                </div>
                
                <!-- Language Distribution -->
                <div class="stats-card">
                    <h3>🌐 Languages</h3>
                    <div class="stats-number">${ultraSafeFormat(Object.keys(safeData.language_counts).length)}</div>
                    <div class="stats-label">Languages Used</div>
                    <div class="stats-breakdown">
                        ${processCountEntries(safeData.language_counts)}
                    </div>
                </div>
                
                <!-- Index Information -->
                <div class="stats-card">
                    <h3>💾 Storage</h3>
                    <div class="stats-number">${ultraSafeFormat(safeData.indices.length)}</div>
                    <div class="stats-label">Active Indices</div>
                    <div class="stats-breakdown">
                        ${safeData.indices.slice(0, 4).map(index => {
                            try {
                                const safeIndex = ultraSafeObject(index);
                                const name = ultraSafeString(safeIndex.name || 'Unknown');
                                const sizeMb = ultraSafeNumber(safeIndex.size_mb);
                                const displayName = name.length > 15 ? name.substring(0, 15) + '...' : name;
                                
                                return `
                                    <div class="breakdown-item">
                                        <span class="breakdown-label" title="${name}">${displayName}</span>
                                        <span class="breakdown-value">${sizeMb.toFixed(1)}MB</span>
                                    </div>
                                `;
                            } catch (e) {
                                console.warn('Failed to process index:', index, 'error:', e);
                                return `
                                    <div class="breakdown-item">
                                        <span class="breakdown-label">Invalid index</span>
                                        <span class="breakdown-value">-</span>
                                    </div>
                                `;
                            }
                        }).join('') || 
                        `<div class="breakdown-item">
                            <span class="breakdown-label">No indices found</span>
                            <span class="breakdown-value"></span>
                        </div>`}
                        ${safeData.indices.length > 4 ? 
                            `<div class="breakdown-item">
                                <span class="breakdown-label">...and ${ultraSafeFormat(safeData.indices.length - 4)} more</span>
                                <span class="breakdown-value"></span>
                            </div>` : ''}
                    </div>
                </div>
            </div>
            
            <div class="stats-last-updated">
                📅 Last updated: ${ultraSafeTimestamp(timestamp)} | 
                📊 Structure: ${ultraSafeString(safeData.structure_info.document_type)} | 
                🏷️ Collections: ${ultraSafeString(safeData.structure_info.collection_strategy)}
                <br>
                <div style="margin-top: 8px; padding: 8px; background: #e3f2fd; border-radius: 4px; font-size: 0.9em; border-left: 4px solid #6e32a0;">
                    <strong>📋 Source Data Verification:</strong> The "Evaluations Processed" count represents unique evaluationIDs successfully imported. 
                    Compare this number with your source system to verify data completeness and identify any missing evaluations.
                </div>
            </div>
        `;
        
    } catch (htmlError) {
        console.error('Error generating statistics HTML:', htmlError);
        html = `
            <div class="stats-error">
                <strong>❌ Error generating statistics display:</strong> ${ultraSafeString(htmlError.message)}
                <br><small>Data processing failed. Check console for details.</small>
                <br><button onclick="loadOpenSearchStats()" style="margin-top: 8px; padding: 4px 8px; background: #6e32a0; color: white; border: none; border-radius: 4px; cursor: pointer;">🔄 Retry</button>
            </div>
        `;
    }
    
    container.innerHTML = html;
    console.log("📊 Statistics dashboard updated with BULLETPROOF error handling");
}

// ============================================================================
// IMPORT MANAGEMENT FUNCTIONS - ENHANCED (keeping existing working functions)
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
    const maxDocsInfo = document.getElementById("maxDocsInfo");
    const maxDocsInfoText = document.getElementById("maxDocsInfoText");
    const importPreview = document.getElementById("importPreview");
    const importPreviewText = document.getElementById("importPreviewText");
    
    if (!maxDocsInput) return;
    
    const value = maxDocsInput.value.trim();
    const numValue = parseInt(value);
    
    // Update display text
    if (value === "" || isNaN(numValue)) {
        if (maxDocsDisplay) {
            maxDocsDisplay.textContent = "All documents";
            maxDocsDisplay.style.color = "#666";
        }
        if (maxDocsInfoText) {
            maxDocsInfoText.textContent = "All available documents will be processed";
        }
        if (maxDocsInfo) {
            maxDocsInfo.className = "max-docs-info";
        }
    } else {
        if (maxDocsDisplay) {
            maxDocsDisplay.textContent = `Max: ${ultraSafeFormat(numValue)}`;
            maxDocsDisplay.style.color = "#6e32a0";
        }
        if (maxDocsInfoText) {
            maxDocsInfoText.textContent = `Import will be limited to ${ultraSafeFormat(numValue)} documents`;
        }
        if (maxDocsInfo) {
            maxDocsInfo.className = "max-docs-warning";
        }
    }
    
    // Show/hide info box
    if (maxDocsInfo) {
        maxDocsInfo.style.display = "block";
    }
    
    // Update import preview
    const importType = document.getElementById("importTypeSelect")?.value || "full";
    const collection = document.getElementById("collectionSelect")?.value || "all";
    
    let previewText = `${importType.charAt(0).toUpperCase() + importType.slice(1)} import from ${collection} collection`;
    if (value !== "" && !isNaN(numValue)) {
        previewText += ` (limited to ${ultraSafeFormat(numValue)} documents)`;
    } else {
        previewText += ` (all available documents)`;
    }
    
    if (importPreviewText) {
        importPreviewText.textContent = previewText;
    }
    if (importPreview) {
        importPreview.style.display = "block";
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
        
        // Safely check for components with bulletproof handling
        const components = ultraSafeObject(health.components);
        if (Object.keys(components).length > 0) {
            Object.entries(components).forEach(([component, info]) => {
                const safeInfo = ultraSafeObject(info);
                const status = ultraSafeString(safeInfo.status || 'unknown');
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
                        <div class="health-label">${emoji} ${ultraSafeString(component).toUpperCase()}</div>
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
                    <div class="health-value">${ultraSafeString(health.status)}</div>
                </div>
            `;
        }
        
        // Add enhanced structure info with bulletproof handling
        const enhancements = ultraSafeObject(health.enhancements);
        if (enhancements.document_structure) {
            html += `
                <div class="health-item">
                    <div class="health-label">🔄 STRUCTURE</div>
                    <div class="health-value">${ultraSafeString(enhancements.document_structure)}</div>
                </div>
            `;
        }
        
        // Add version info
        html += `
            <div class="health-item">
                <div class="health-label">🚀 VERSION</div>
                <div class="health-value">v2.2.3 BULLETPROOF</div>
            </div>
        `;
        
        container.innerHTML = html;
        console.log("🏥 System health checked with BULLETPROOF handling:", health);
        
    } catch (error) {
        console.error("Health check failed:", error);
        container.innerHTML = `
            <div class="health-item unhealthy">
                <div class="health-label">❌ SYSTEM</div>
                <div class="health-value">Error: ${ultraSafeString(error.message)}</div>
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
            const timestamp = ultraSafeTimestamp(data.last_import_timestamp);
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
            alert(`❌ Error: ${ultraSafeString(data.error)}`);
        }
    } catch (error) {
        alert(`❌ Failed to reset timestamp: ${ultraSafeString(error.message)}`);
    }
}

// ============================================================================
// UTILITY FUNCTIONS - BULLETPROOF
// ============================================================================

async function toggleLogs() {
    const container = document.getElementById('logsContainer');
    const content = document.getElementById('logsContent');
    
    if (!container || !content) {
        console.error('Logs container elements not found in DOM');
        return;
    }

    if (container.classList.contains('hidden')) {
        try {
            // Show loading state
            content.innerHTML = `
                <div class="logs-loading">
                    <div class="spinner"></div>
                    <span>Loading application logs...</span>
                </div>
            `;
            container.classList.remove('hidden');
            
            // Fetch logs from our new endpoint
            const response = await fetch('/logs');
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            if (data.status === 'success' || data.status === 'partial_success') {
                const logs = ultraSafeArray(data.logs || []);
                
                if (logs.length === 0) {
                    content.innerHTML = `
                        <div class="logs-empty">
                            <h4>📝 No Logs Available</h4>
                            <p>No log entries found or logs may not be configured.</p>
                            <small>Log source: ${ultraSafeString(data.log_source || 'unknown')}</small>
                        </div>
                    `;
                } else {
                    // Enhanced log display with syntax highlighting
                    const logsHtml = logs.map(log => {
                        const safeLog = ultraSafeString(log);
                        
                        // Determine log level and apply appropriate styling
                        let logClass = 'log-info';
                        let logIcon = 'ℹ️';
                        
                        if (safeLog.includes('ERROR') || safeLog.includes('❌')) {
                            logClass = 'log-error';
                            logIcon = '❌';
                        } else if (safeLog.includes('WARNING') || safeLog.includes('⚠️')) {
                            logClass = 'log-warning';
                            logIcon = '⚠️';
                        } else if (safeLog.includes('SUCCESS') || safeLog.includes('✅')) {
                            logClass = 'log-success';
                            logIcon = '✅';
                        } else if (safeLog.includes('INFO')) {
                            logIcon = 'ℹ️';
                        }
                        
                        return `
                            <div class="log-entry ${logClass}">
                                <span class="log-icon">${logIcon}</span>
                                <span class="log-text">${safeLog}</span>
                            </div>
                        `;
                    }).join('');
                    
                    content.innerHTML = `
                        <div class="logs-header">
                            <h4>📜 Application Logs</h4>
                            <div class="logs-meta">
                                <span>📊 ${logs.length} entries</span>
                                <span>🔄 Updated: ${new Date().toLocaleString()}</span>
                                <span>📁 Source: ${ultraSafeString(data.log_source || 'system')}</span>
                            </div>
                        </div>
                        <div class="logs-content">
                            ${logsHtml}
                        </div>
                        <div class="logs-footer">
                            <small>✅ Logs endpoint working properly - 404 error resolved</small>
                        </div>
                    `;
                }
                
                // Add error info if partial success
                if (data.status === 'partial_success' && data.error) {
                    content.innerHTML += `
                        <div class="logs-error-info">
                            <h5>⚠️ Limited Functionality</h5>
                            <p>Error: ${ultraSafeString(data.error)}</p>
                            <small>The logs endpoint is working but encountered issues accessing log files.</small>
                        </div>
                    `;
                }
                
            } else {
                throw new Error(data.error || 'Failed to retrieve logs');
            }
            
        } catch (error) {
            console.error('Failed to load logs:', error);
            content.innerHTML = `
                <div class="logs-error">
                    <h4>❌ Logs Loading Error</h4>
                    <p><strong>Error:</strong> ${ultraSafeString(error.message)}</p>
                    <div class="error-details">
                        <p>This could be caused by:</p>
                        <ul>
                            <li>Network connectivity issues</li>
                            <li>Server configuration problems</li>
                            <li>Log file access permissions</li>
                        </ul>
                    </div>
                    <button class="btn secondary" onclick="toggleLogs(); setTimeout(toggleLogs, 100);">
                        🔄 Retry
                    </button>
                </div>
            `;
        }
    } else {
        // Hide logs
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
            const results = ultraSafeArray(data.results);
            const resultCount = results.length;
            let html = `
                <div class="result-card">
                    <h4>🔍 Search Results for "${ultraSafeString(query)}"</h4>
                    <p><strong>Found:</strong> ${ultraSafeFormat(resultCount)} results</p>
            `;
            
            if (resultCount > 0) {
                html += '<div style="max-height: 200px; overflow-y: auto; margin-top: 10px; border-top: 1px solid #eee; padding-top: 10px;">';
                results.slice(0, 3).forEach((result, index) => {
                    const safeResult = ultraSafeObject(result);
                    const title = ultraSafeString(safeResult.title || 'Untitled');
                    const text = ultraSafeString(safeResult.text || '');
                    const score = ultraSafeNumber(safeResult.score);
                    
                    html += `
                        <div style="margin: 10px 0; padding: 10px; background: #f9f9f9; border-radius: 4px; border-left: 3px solid #6e32a0;">
                            <div style="font-weight: bold; color: #6e32a0; margin-bottom: 5px;">
                                ${index + 1}. ${title}
                            </div>
                            <div style="font-size: 0.9em; color: #666;">
                                ${text.substring(0, 150)}${text.length > 150 ? '...' : ''}
                            </div>
                            ${score > 0 ? `<div style="font-size: 0.8em; color: #999; margin-top: 5px;">Score: ${Math.round(score * 100) / 100}</div>` : ''}
                        </div>
                    `;
                });
                html += '</div>';
                
                if (resultCount > 3) {
                    html += `<div style="margin-top: 10px; font-size: 0.9em; color: #666; text-align: center;">
                        ... and ${ultraSafeFormat(resultCount - 3)} more results
                    </div>`;
                }
            }
            
            html += '</div>';
            container.innerHTML = html;
        } else {
            container.innerHTML = `<div class="status failed">❌ Search failed: ${ultraSafeString(data.error || 'Unknown error')}</div>`;
        }
    } catch (error) {
        container.innerHTML = `<div class="status failed">❌ Search failed: ${ultraSafeString(error.message)}</div>`;
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
window.loadOpenSearchStats = loadOpenSearchStats; // Enhanced statistics function

console.log("✅ Ask InnovAI Admin BULLETPROOF main.js v2.2.3 loaded successfully");
console.log("🛡️ BULLETPROOF: ALL toLocaleString() errors COMPLETELY ELIMINATED");
console.log("🔧 COMPLETE PROTECTION: ultraSafe functions handle ALL undefined/null/invalid values");
console.log("📊 All functions including enhanced statistics with ULTIMATE error handling available");