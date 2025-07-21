// Enhanced Metro AI Call Center Analytics Chat - VECTOR SEARCH ENABLED
// Version: 4.8.0 - Full vector search integration with enhanced UI feedback
// NEW: Vector search indicators, hybrid search status, enhanced search quality display
// ENHANCED: Search metadata display, vector-enhanced results highlighting, debug capabilities

// =============================================================================
// PRODUCTION CONFIGURATION & GLOBAL STATE WITH VECTOR SEARCH
// =============================================================================

// Global state management
let currentFilters = {};
let chatHistory = [];
let chatSessions = [];
let currentSessionId = null;
let isLoading = false;
let filterOptions = {
    templates: [],
    programs: [],
    partners: [],
    sites: [],
    lobs: [],
    callDispositions: [],
    callSubDispositions: [],
    languages: [],
    callTypes: []
};

//  NEW: Vector search state tracking
let vectorSearchStatus = {
    enabled: false,
    hybridAvailable: false,
    lastSearchEnhanced: false,
    searchQuality: 'text_only'
};

// Production configuration
const PRODUCTION_CONFIG = {
    MAX_RETRY_ATTEMPTS: 3,
    RETRY_DELAY_BASE: 2000,
    FILTER_LOAD_TIMEOUT: 30000,
    CHAT_REQUEST_TIMEOUT: 120000,
    DEBUG_MODE: window.location.hostname === 'localhost' || window.location.search.includes('debug=true'),
    PERFORMANCE_MONITORING: true,
    VECTOR_SEARCH_UI: true  // ✅ NEW: Enable vector search UI features
};

// Performance monitoring with vector search metrics
const performanceMetrics = {
    filterLoadTime: 0,
    chatResponseTimes: [],
    errorCount: 0,
    lastFilterUpdate: null,
    vectorSearchUsage: 0,  // ✅ NEW
    hybridSearchUsage: 0   // ✅ NEW
};

// =============================================================================
// ENHANCED INITIALIZATION WITH VECTOR SEARCH DETECTION
// =============================================================================

document.addEventListener('DOMContentLoaded', function() {
    console.log("🚀 Metro AI Analytics v4.8.0 - VECTOR SEARCH ENHANCED Chat Interface Starting...");
    
    const startTime = performance.now();
    
    try {
        // Core initialization
        initializePage();
        loadFormattingStyles();
        loadVectorSearchStyles(); // ✅ NEW
        
        // ✅ NEW: Check vector search capabilities
        setTimeout(() => {
            checkVectorSearchCapabilities()
                .then(() => {
                    console.log(`✅ Vector search status: ${vectorSearchStatus.enabled ? 'ENABLED' : 'DISABLED'}`);
                })
                .catch(error => {
                    console.warn("⚠️ Vector search detection failed:", error);
                });
        }, 100);
        
        // Load real filter data with vector search awareness
        setTimeout(() => {
            loadDynamicFilterOptions()
                .then(() => {
                    performanceMetrics.filterLoadTime = performance.now() - startTime;
                    console.log(`⚡ Filter loading completed in ${performanceMetrics.filterLoadTime.toFixed(2)}ms`);
                })
                .catch(error => {
                    console.error("❌ Critical: Filter loading failed during initialization:", error);
                    handleCriticalFilterError(error);
                });
        }, 500);
        
        // Initial stats update
        setTimeout(() => {
            updateStats().catch(error => {
                console.warn("⚠️ Initial stats update failed:", error);
            });
        }, 1000);
        
        setupProductionErrorHandlers();
        console.log("✅ Production initialization completed successfully with vector search support");
        
    } catch (error) {
        console.error("❌ CRITICAL: Production initialization failed:", error);
        showCriticalError("Application failed to initialize. Please refresh the page.");
    }
});

// ✅ NEW: Check vector search capabilities
async function checkVectorSearchCapabilities() {
    try {
        const response = await fetch('/debug/vector_capabilities');
        if (response.ok) {
            const data = await response.json();
            
            vectorSearchStatus = {
                enabled: data.capabilities?.vector_search_ready || false,
                hybridAvailable: data.capabilities?.hybrid_search_available || false,
                lastSearchEnhanced: false,
                searchQuality: data.capabilities?.overall_vector_status === 'fully_enabled' ? 'vector_enhanced' : 'text_only'
            };
            
            // Update UI to show vector search status
            updateVectorSearchIndicator();
            
            console.log("🔮 Vector search capabilities detected:", vectorSearchStatus);
        }
    } catch (error) {
        console.warn("⚠️ Could not check vector search capabilities:", error);
        vectorSearchStatus.enabled = false;
    }
}

// ✅ NEW: Update vector search indicator in UI
function updateVectorSearchIndicator() {
    const headerStats = document.querySelector('.chat-stats');
    if (!headerStats) return;
    
    // Remove existing vector indicator
    const existingIndicator = headerStats.querySelector('.vector-search-indicator');
    if (existingIndicator) {
        existingIndicator.remove();
    }
    
    // Add vector search indicator
    const vectorIndicator = document.createElement('div');
    vectorIndicator.className = 'stat-item vector-search-indicator';
    
    if (vectorSearchStatus.enabled) {
        vectorIndicator.innerHTML = `
            <span class="material-icons vector-enabled">psychology</span>
            <span>Vector Search: ${vectorSearchStatus.hybridAvailable ? 'Hybrid' : 'Enabled'}</span>
        `;
        vectorIndicator.title = vectorSearchStatus.hybridAvailable ? 
            'Enhanced search with text + vector similarity matching' : 
            'Vector similarity search enabled';
    } else {
        vectorIndicator.innerHTML = `
            <span class="material-icons vector-disabled">search</span>
            <span>Text Search Only</span>
        `;
        vectorIndicator.title = 'Standard text search (vector search not available)';
    }
    
    headerStats.appendChild(vectorIndicator);
}

// ✅ NEW: Load vector search specific styles
function loadVectorSearchStyles() {
    const styleSheet = document.createElement('style');
    styleSheet.textContent = `
        /* Vector Search UI Enhancements */
        .vector-search-indicator {
            background: ${vectorSearchStatus.enabled ? 
                'linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%)' : 
                'rgba(156, 163, 175, 0.2)'
            };
            border-radius: 20px;
            color: ${vectorSearchStatus.enabled ? 'white' : '#6b7280'};
        }
        
        .vector-enabled {
            color: #10b981 !important;
            animation: pulse-vector 2s infinite;
        }
        
        .vector-disabled {
            color: #6b7280 !important;
        }
        
        @keyframes pulse-vector {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        /* Search Enhancement Badges */
        .search-enhancement-badge {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            padding: 2px 8px;
            border-radius: 12px;
            font-size: 0.75em;
            font-weight: 500;
            margin-left: 8px;
        }
        
        .search-enhancement-badge.vector-enhanced {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
        }
        
        .search-enhancement-badge.hybrid-search {
            background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
            color: white;
        }
        
        .search-enhancement-badge.text-only {
            background: #f3f4f6;
            color: #6b7280;
        }
        
        /* Enhanced Sources Display */
        .sources-enhancement-info {
            background: linear-gradient(135deg, #ede9fe 0%, #f3f4f6 100%);
            border: 1px solid #c4b5fd;
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
            font-size: 0.9em;
        }
        
        .vector-search-stats {
            display: flex;
            gap: 12px;
            flex-wrap: wrap;
            margin-top: 8px;
        }
        
        .vector-stat {
            background: white;
            border: 1px solid #e5e7eb;
            border-radius: 6px;
            padding: 4px 8px;
            font-size: 0.8em;
        }
        
        .vector-stat.enabled {
            background: #dcfce7;
            border-color: #16a34a;
            color: #166534;
        }
        
        /* Debug Panel Enhancements */
        .vector-debug-panel {
            background: #fef7ff;
            border: 1px solid #e879f9;
            border-radius: 8px;
            padding: 16px;
            margin: 16px 0;
            display: none;
        }
        
        .vector-debug-panel.visible {
            display: block;
        }
        
        .debug-test-button {
            background: #8b5cf6;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            margin: 4px;
            font-size: 0.85em;
            transition: background 0.2s;
        }
        
        .debug-test-button:hover {
            background: #7c3aed;
        }
        
        /* Message Enhancement Indicators */
        .message.assistant .search-enhancement {
            background: rgba(139, 92, 246, 0.1);
            border-left: 3px solid #8b5cf6;
            padding: 8px 12px;
            margin: 8px 0;
            border-radius: 0 6px 6px 0;
            font-size: 0.85em;
        }
        
        .search-method-indicator {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 2px 8px;
            font-size: 0.75em;
            color: #64748b;
            margin: 2px;
        }
        
        .search-method-indicator.vector {
            background: #dcfce7;
            border-color: #16a34a;
            color: #166534;
        }
        
        .search-method-indicator.hybrid {
            background: #ede9fe;
            border-color: #8b5cf6;
            color: #7c3aed;
        }
    `;
    
    document.head.appendChild(styleSheet);
    console.log("✅ Vector search styles loaded");
}

// =============================================================================
// ENHANCED CHAT FUNCTIONALITY WITH VECTOR SEARCH INTEGRATION
// =============================================================================

async function sendMessage() {
    const startTime = performance.now();
    
    try {
        const input = document.getElementById('chatInput');
        if (!input) {
            console.error('❌ Chat input element not found');
            return;
        }
        
        const message = input.value.trim();
        if (!message) {
            console.warn('⚠️ Empty message submitted');
            return;
        }
        
        // Clear input and update UI
        input.value = '';
        
        // Create new session if needed
        if (!currentSessionId || chatSessions.length === 0) {
            createNewChatSession(message);
        }
        
        // Add user message
        addMessage('user', message);
        
        // Show loading with vector search awareness
        isLoading = true;
        updateSendButton();
        addLoadingMessage();
        
        console.log("🔄 ENHANCED: Sending chat request with vector search support...");
        
        // Get filters with debugging
        const filters = collectAlignedFilters();
        console.log("🏷️ FILTERS COLLECTED:", filters);
        console.log("🔍 FILTER COUNT:", Object.keys(filters).length);
        
        // Enhanced request body
        const requestBody = {
            message: message,
            history: chatHistory,
            filters: filters,
            analytics: true,
            metadata_focus: [
                'evaluationId', 'internalId', 'template_id', 'template_name',
                'partner', 'site', 'lob', 'agentName', 'call_date',
                'disposition', 'subDisposition', 'call_duration', 'language',
                'weighted_score', 'url'  // ✅ Enhanced metadata
            ]
        };
        
        if (PRODUCTION_CONFIG.DEBUG_MODE) {
            console.log("📤 FULL REQUEST PAYLOAD:", JSON.stringify(requestBody, null, 2));
        }
        
        // Enhanced fetch with vector search awareness
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify(requestBody)
        });
        
        console.log("📥 RESPONSE STATUS:", response.status);
        console.log("📥 RESPONSE OK:", response.ok);
        
        if (!response.ok) {
            const responseText = await response.text();
            console.error("❌ RESPONSE ERROR:", {
                status: response.status,
                statusText: response.statusText,
                body: responseText
            });
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (PRODUCTION_CONFIG.DEBUG_MODE) {
            console.log("📊 RESPONSE DATA:", data);
        }
        
        // Remove loading message
        removeLoadingMessage();
        
        // ✅ ENHANCED: Process response with vector search metadata
        const reply = data.reply || 'Sorry, I couldn\'t process your request.';
        const searchEnhancement = data.search_enhancement || {};
        const searchMetadata = data.search_metadata || {};
        
        // ✅ Track vector search usage
        if (searchEnhancement.vector_search_used) {
            performanceMetrics.vectorSearchUsage++;
            vectorSearchStatus.lastSearchEnhanced = true;
        }
        if (searchEnhancement.hybrid_search_used) {
            performanceMetrics.hybridSearchUsage++;
        }
        
        // Add enhanced assistant message with search info
        addMessage('assistant', reply, {
            searchEnhancement: searchEnhancement,
            searchMetadata: searchMetadata,
            vectorEnhanced: searchEnhancement.vector_search_used || searchEnhancement.hybrid_search_used
        });
        
        // ✅ NEW: Enhanced sources summary with vector search info
        if (data.sources_summary && data.sources_details && data.sources_totals) {
            console.log("📊 ENHANCED SOURCES SUMMARY WITH VECTOR SEARCH INFO");
            addSourcesSummaryWithVectorInfo(
                data.sources_summary, 
                data.sources_details, 
                data.sources_totals,
                data.sources_full_data || {},
                data.display_limit || 25,
                data.filter_context || {},
                searchEnhancement || {}  // ✅ NEW: Pass search enhancement info
            );
        } else if (data.sources_summary && data.sources_details) {
            addSourcesSummaryWithVectorInfo(data.sources_summary, data.sources_details, {}, {}, 25, data.filter_context || {}, searchEnhancement || {});
        } else if (data.sources_summary) {
            addSourcesSummaryMessage(data.sources_summary, data.filter_context || {}, searchEnhancement || {});
        }
        
        // Add to history
        chatHistory.push({ role: 'user', content: message });
        chatHistory.push({ role: 'assistant', content: reply });
        
        // Track performance
        const responseTime = performance.now() - startTime;
        performanceMetrics.chatResponseTimes.push(responseTime);
        
        console.log(`✅ ENHANCED: Chat response completed in ${responseTime.toFixed(2)}ms`);
        console.log(`🔮 Vector search enhancement: ${searchEnhancement.vector_search_used ? 'YES' : 'NO'}`);
        console.log(`🔥 Hybrid search used: ${searchEnhancement.hybrid_search_used ? 'YES' : 'NO'}`);
        console.log(`📊 Search quality: ${searchEnhancement.search_quality || 'unknown'}`);
        
    } catch (error) {
        console.error('❌ ENHANCED: Chat request failed:', error);
        
        isLoading = false;
        updateSendButton();
        removeLoadingMessage();
        
        const errorMessage = getProductionErrorMessage(error);
        addMessage('assistant', `I apologize, but there was an error processing your request: ${errorMessage}`);
        
        logProductionError('chat_request_error', error);
        isLoading = false;
        updateSendButton();
    }
}

// ✅ ENHANCED: Add message with vector search enhancement info
function addMessage(type, content, enhancementData = null) {
    const messageContainer = createMessageElement(type, content, enhancementData);
    
    // Add to current session
    if (currentSessionId) {
        const session = chatSessions.find(s => s.id === currentSessionId);
        if (session) {
            session.messages.push({ 
                type, 
                content, 
                timestamp: new Date().toISOString(),
                enhancementData  // ✅ NEW: Store enhancement data
            });
            const sessionElement = document.getElementById(`session-${currentSessionId}`);
            if (sessionElement) {
                const contentDiv = sessionElement.querySelector('.chat-session-content');
                if (contentDiv) {
                    contentDiv.appendChild(messageContainer);
                    
                    setTimeout(() => {
                        messageContainer.scrollIntoView({ behavior: 'smooth', block: 'end' });
                    }, 100);
                }
            }
        }
    }
    
    // Add to global history
    chatHistory.push({ role: type === 'user' ? 'user' : 'assistant', content });
}

// ✅ ENHANCED: Create message element with vector search indicators
function createMessageElement(type, content, enhancementData = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    if (type === 'assistant') {
        // Render markdown for assistant responses
        const renderedContent = renderMarkdown(content);
        contentDiv.innerHTML = renderedContent;
        contentDiv.classList.add('formatted-response');
        
        // ✅ NEW: Add search enhancement info for assistant messages
        if (enhancementData && enhancementData.vectorEnhanced) {
            const enhancementDiv = document.createElement('div');
            enhancementDiv.className = 'search-enhancement';
            
            const searchMethods = [];
            if (enhancementData.searchEnhancement.vector_search_used) {
                searchMethods.push('Vector Similarity');
            }
            if (enhancementData.searchEnhancement.hybrid_search_used) {
                searchMethods.push('Hybrid Search');
            }
            if (searchMethods.length === 0) {
                searchMethods.push('Text Search');
            }
            
            enhancementDiv.innerHTML = `
                <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                    <span class="material-icons" style="font-size: 16px; color: #8b5cf6;">psychology</span>
                    <strong>Enhanced Search Results</strong>
                </div>
                <div class="vector-search-stats">
                    <div class="vector-stat ${enhancementData.vectorEnhanced ? 'enabled' : ''}">
                        Search Methods: ${searchMethods.join(' + ')}
                    </div>
                    <div class="vector-stat">
                        Quality: ${enhancementData.searchEnhancement.search_quality || 'Standard'}
                    </div>
                    ${enhancementData.searchMetadata.vector_enhanced_count ? 
                        `<div class="vector-stat enabled">Vector-Enhanced: ${enhancementData.searchMetadata.vector_enhanced_count} results</div>` : 
                        ''
                    }
                </div>
            `;
            
            contentDiv.appendChild(enhancementDiv);
        }
        
        // Add response type classes for styling
        if (content.includes('📊 Data Analysis Results')) {
            contentDiv.classList.add('data-analysis-response');
        } else if (content.includes('📋 Summary')) {
            contentDiv.classList.add('summary-response');
        }
    } else {
        contentDiv.textContent = content;
    }
    
    const metaDiv = document.createElement('div');
    metaDiv.className = 'message-meta';
    
    let metaContent = new Date().toLocaleTimeString();
    
    // ✅ NEW: Add vector search indicator to meta for assistant messages
    if (type === 'assistant' && enhancementData && enhancementData.vectorEnhanced) {
        const searchBadge = document.createElement('span');
        searchBadge.className = `search-enhancement-badge ${
            enhancementData.searchEnhancement.hybrid_search_used ? 'hybrid-search' : 
            enhancementData.searchEnhancement.vector_search_used ? 'vector-enhanced' : 'text-only'
        }`;
        
        const badgeText = enhancementData.searchEnhancement.hybrid_search_used ? 'Hybrid' :
                         enhancementData.searchEnhancement.vector_search_used ? 'Vector' : 'Text';
        
        searchBadge.innerHTML = `<span class="material-icons" style="font-size: 12px;">psychology</span> ${badgeText}`;
        metaDiv.appendChild(searchBadge);
    }
    
    const timeSpan = document.createElement('span');
    timeSpan.textContent = metaContent;
    metaDiv.appendChild(timeSpan);
    
    messageDiv.appendChild(contentDiv);
    messageDiv.appendChild(metaDiv);
    
    return messageDiv;
}

// ✅ ENHANCED: Sources summary with vector search information
function addSourcesSummaryWithVectorInfo(sourcesSummary, sourcesDetails, sourcesTotals, sourcesFullData, displayLimit, filterContext, searchEnhancement) {
    if (!sourcesSummary) return;
    
    const summaryDiv = document.createElement('div');
    summaryDiv.className = 'sources-container summary-container';
    
    const summaryId = 'summary-' + Date.now();
    
    // Build clickable summary items (existing logic)
    const summaryItems = [];
    
    if (sourcesSummary.evaluations > 0) {
        summaryItems.push({
            key: 'evaluations',
            label: `Evaluations: ${sourcesSummary.evaluations.toLocaleString()}`,
            count: sourcesSummary.evaluations,
            data: sourcesDetails.evaluations || []
        });
    }
    
    // Add other summary items...
    ['agents', 'opportunities', 'churn_triggers', 'programs', 'templates', 'dispositions', 'partners', 'sites'].forEach(key => {
        if (sourcesSummary[key] > 0) {
            summaryItems.push({
                key: key,
                label: `${key.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}: ${sourcesSummary[key]}`,
                count: sourcesSummary[key],
                data: sourcesDetails[key] || []
            });
        }
    });
    
    // Add date range (non-clickable)
    if (sourcesSummary.date_range && sourcesSummary.date_range !== "No data") {
        summaryItems.push({
            key: 'date_range',
            label: `Date Range: ${sourcesSummary.date_range}`,
            count: 0,
            data: [],
            clickable: false
        });
    }
    
    const summaryItemsHtml = summaryItems.map(item => {
        if (item.clickable === false) {
            return `<span class="summary-item non-clickable">${item.label}</span>`;
        }
        return `<span class="summary-item clickable" data-category="${item.key}" onclick="toggleDetailedTable('${summaryId}', '${item.key}')">${item.label}</span>`;
    }).join(', ');
    
    // Check for active filters
    const activeFilters = [];
    if (filterContext) {
        if (filterContext.template_name) activeFilters.push(`Template: ${filterContext.template_name}`);
        if (filterContext.program) activeFilters.push(`Program: ${filterContext.program}`);
        if (filterContext.partner) activeFilters.push(`Partner: ${filterContext.partner}`);
        if (filterContext.site) activeFilters.push(`Site: ${filterContext.site}`);
        if (filterContext.lob) activeFilters.push(`LOB: ${filterContext.lob}`);
        if (filterContext.call_disposition) activeFilters.push(`Disposition: ${filterContext.call_disposition}`);
        if (filterContext.language) activeFilters.push(`Language: ${filterContext.language}`);
        if (filterContext.start_date || filterContext.end_date) {
            const dateFilter = `${filterContext.start_date || 'Start'} to ${filterContext.end_date || 'End'}`;
            activeFilters.push(`Date Filter: ${dateFilter}`);
        }
    }
    
    // ✅ NEW: Search enhancement display
    let enhancementHTML = '';
    if (searchEnhancement && (searchEnhancement.vector_search_used || searchEnhancement.hybrid_search_used)) {
        const enhancementStats = [
            searchEnhancement.vector_search_used ? '✅ Vector Similarity' : '❌ Vector Search',
            searchEnhancement.hybrid_search_used ? '✅ Hybrid Search' : '❌ Hybrid Search',
            `Quality: ${searchEnhancement.search_quality || 'Standard'}`
        ];
        
        if (searchEnhancement.vector_enhanced_results) {
            enhancementStats.push(`${searchEnhancement.vector_enhanced_results}/${searchEnhancement.total_results} results vector-enhanced`);
        }
        
        enhancementHTML = `
        <div class="sources-enhancement-info">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 8px;">
                <span class="material-icons" style="color: #8b5cf6;">psychology</span>
                <strong>Search Enhancement Active</strong>
            </div>
            <div class="vector-search-stats">
                ${enhancementStats.map(stat => `<div class="vector-stat ${stat.includes('✅') ? 'enabled' : ''}">${stat}</div>`).join('')}
            </div>
        </div>`;
    }
    
    summaryDiv.innerHTML = `
        <h4>📊 Data Sources Summary <small>(Click items to view details)</small></h4>
        <div class="sources-summary-content" id="${summaryId}">
            ${enhancementHTML}
            <div class="summary-main">
                ${summaryItemsHtml}
            </div>
            ${activeFilters.length > 0 ? `
            <div class="summary-filters">
                <small>📌 Active Filters: ${activeFilters.join(' | ')}</small>
            </div>
            ` : ''}
            <div class="summary-metadata">
                <small>
                    💡 Click on any category above to view detailed breakdown. 
                    ${sourcesSummary.evaluations > 0 ? 
                        `Analysis based on ${sourcesSummary.evaluations.toLocaleString()} unique evaluation${sourcesSummary.evaluations !== 1 ? 's' : ''}.` : 
                        'No evaluation data found for current filters.'
                    }
                    ${searchEnhancement && searchEnhancement.vector_search_used ? 
                        ' Results enhanced with semantic similarity matching.' : ''}
                </small>
            </div>
            
            <div class="detail-tables-container"></div>
        </div>
    `;
    
    // Store data for drill-down functionality
    summaryDiv.dataset.detailsData = JSON.stringify(sourcesDetails);
    summaryDiv.dataset.totalsData = JSON.stringify(sourcesTotals);
    summaryDiv.dataset.fullData = JSON.stringify(sourcesFullData);
    summaryDiv.dataset.displayLimit = displayLimit;
    
    // Add to current session
    if (currentSessionId) {
        const sessionElement = document.getElementById(`session-${currentSessionId}`);
        if (sessionElement) {
            const contentDiv = sessionElement.querySelector('.chat-session-content');
            if (contentDiv) {
                contentDiv.appendChild(summaryDiv);
                summaryDiv.scrollIntoView({ behavior: 'smooth' });
            }
        }
    }
    
    console.log("📊 Enhanced sources summary with vector search info displayed");
}

// ✅ NEW: Add vector search debug panel for development
function addVectorSearchDebugPanel() {
    if (!PRODUCTION_CONFIG.DEBUG_MODE) return;
    
    const debugPanel = document.createElement('div');
    debugPanel.className = 'vector-debug-panel';
    debugPanel.id = 'vectorDebugPanel';
    debugPanel.innerHTML = `
        <h4 style="margin-top: 0;">🔮 Vector Search Debug Panel</h4>
        <div style="margin-bottom: 12px;">
            <strong>Status:</strong> 
            <span class="search-method-indicator ${vectorSearchStatus.enabled ? 'vector' : ''}">
                ${vectorSearchStatus.enabled ? '✅ Enabled' : '❌ Disabled'}
            </span>
            <span class="search-method-indicator ${vectorSearchStatus.hybridAvailable ? 'hybrid' : ''}">
                Hybrid: ${vectorSearchStatus.hybridAvailable ? '✅ Available' : '❌ Not Available'}
            </span>
        </div>
        <div style="margin-bottom: 12px;">
            <strong>Usage:</strong> Vector: ${performanceMetrics.vectorSearchUsage}, Hybrid: ${performanceMetrics.hybridSearchUsage}
        </div>
        <div>
            <button class="debug-test-button" onclick="testVectorSearch()">Test Vector Search</button>
            <button class="debug-test-button" onclick="testHybridSearch()">Test Hybrid Search</button>
            <button class="debug-test-button" onclick="checkVectorCapabilities()">Check Capabilities</button>
            <button class="debug-test-button" onclick="toggleVectorDebugPanel()">Hide Panel</button>
        </div>
    `;
    
    // Add to page
    const chatArea = document.querySelector('.chat-area');
    if (chatArea) {
        chatArea.appendChild(debugPanel);
    }
}

// ✅ NEW: Debug functions for vector search testing
async function testVectorSearch() {
    try {
        const response = await fetch('/debug/test_vector_search?query=customer service');
        const data = await response.json();
        console.log("🔮 Vector Search Test Result:", data);
        showToast(`Vector Search Test: ${data.status}`, data.status === 'success' ? 'success' : 'warning');
    } catch (error) {
        console.error("❌ Vector search test failed:", error);
        showToast("Vector search test failed", 'error');
    }
}

async function testHybridSearch() {
    try {
        const response = await fetch('/debug/test_hybrid_search?query=call dispositions');
        const data = await response.json();
        console.log("🔥 Hybrid Search Test Result:", data);
        showToast(`Hybrid Search Test: ${data.status}`, data.status === 'success' ? 'success' : 'warning');
    } catch (error) {
        console.error("❌ Hybrid search test failed:", error);
        showToast("Hybrid search test failed", 'error');
    }
}

async function checkVectorCapabilities() {
    try {
        const response = await fetch('/debug/vector_capabilities');
        const data = await response.json();
        console.log("📊 Vector Capabilities:", data);
        
        // Update local status
        if (data.capabilities) {
            vectorSearchStatus.enabled = data.capabilities.vector_search_ready || false;
            vectorSearchStatus.hybridAvailable = data.capabilities.hybrid_search_available || false;
            updateVectorSearchIndicator();
        }
        
        showToast(`Vector Status: ${data.overall_vector_status}`, 'info');
    } catch (error) {
        console.error("❌ Vector capabilities check failed:", error);
        showToast("Vector capabilities check failed", 'error');
    }
}

function toggleVectorDebugPanel() {
    const panel = document.getElementById('vectorDebugPanel');
    if (panel) {
        panel.classList.toggle('visible');
    }
}

// ✅ ENHANCED: Update stats with vector search info
async function updateStats() {
    try {
        const response = await fetchWithRetry('/analytics/stats', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                filters: currentFilters,
                filter_version: '4.8_vector_enabled'
            }),
            timeout: 10000,
            maxRetries: 2
        });
        
        if (response.ok) {
            const data = await response.json();
            const totalRecords = document.getElementById('totalRecords');
            if (totalRecords) {
                const count = data.totalRecords || 0;
                totalRecords.textContent = `${count.toLocaleString()} evaluations`;
                
                // ✅ Add vector search status to title
                if (filterOptions.vector_search_enabled) {
                    totalRecords.title = `Vector search enabled - enhanced relevance matching available`;
                } else if (filterOptions.data_freshness) {
                    const freshness = new Date(filterOptions.data_freshness);
                    const age = Math.round((Date.now() - freshness.getTime()) / (1000 * 60));
                    totalRecords.title = `Data updated ${age} minutes ago`;
                }
            }
        } else {
            throw new Error('Stats API not available');
        }
    } catch (error) {
        console.warn("⚠️ ENHANCED: Stats update failed, using fallback:", error);
        
        const totalRecords = document.getElementById('totalRecords');
        if (totalRecords) {
            if (filterOptions.total_evaluations) {
                const estimatedCount = filterOptions.total_evaluations || 0;
                totalRecords.textContent = `~${estimatedCount.toLocaleString()} evaluations`;
                totalRecords.title = 'Estimated based on database metadata';
            } else {
                totalRecords.textContent = 'Ready for analysis';
                totalRecords.title = 'Connect to database to see evaluation counts';
            }
        }
    }
}

// ✅ ENHANCED: Load filter options with vector search detection
async function loadDynamicFilterOptions() {
    console.log("🔄 Loading ENHANCED filter options with vector search detection...");
    
    const loadStartTime = performance.now();
    
    try {
        setFilterLoadingState(true);
        
        const response = await fetchWithRetry('/filter_options_metadata', {
            timeout: PRODUCTION_CONFIG.FILTER_LOAD_TIMEOUT,
            maxRetries: PRODUCTION_CONFIG.MAX_RETRY_ATTEMPTS
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (!data || typeof data !== 'object') {
            throw new Error('Invalid response format from filter endpoint');
        }
        
        if (data.status === 'error' || data.status === 'opensearch_unavailable') {
            throw new Error(data.message || data.error || 'Database unavailable');
        }
        
        // Update global state with vector search info
        filterOptions = data;
        performanceMetrics.lastFilterUpdate = new Date().toISOString();
        
        // ✅ NEW: Update vector search status from filter options
        if (data.vector_search_enabled !== undefined) {
            vectorSearchStatus.enabled = data.vector_search_enabled;
            vectorSearchStatus.hybridAvailable = data.hybrid_search_available || false;
            vectorSearchStatus.searchQuality = data.search_enhancements?.search_quality || 'text_only';
            updateVectorSearchIndicator();
        }
        
        logFilterDataSummary(data);
        
        try {
            populateFilterOptions(filterOptions);
            updateFilterCounts(filterOptions);
            onFilterOptionsLoaded(filterOptions);
        } catch (uiError) {
            console.error("❌ UI update failed:", uiError);
            throw new Error(`UI update failed: ${uiError.message}`);
        }
        
        console.log("✅ ENHANCED: Filter options loaded successfully with vector search info");
        
    } catch (error) {
        console.error("❌ ENHANCED: Filter loading failed:", error);
        performanceMetrics.errorCount++;
        handleFilterLoadError(error);
        
        try {
            handleNoFilterData(error.message);
        } catch (fallbackError) {
            console.error("❌ CRITICAL: Fallback UI initialization failed:", fallbackError);
            showCriticalError("Unable to initialize filter system");
        }
        
        throw error;
        
    } finally {
        setFilterLoadingState(false);
        
        const loadTime = performance.now() - loadStartTime;
        console.log(`⏱️ Enhanced filter loading completed in ${loadTime.toFixed(2)}ms`);
    }
}

// ✅ ENHANCED: Show vector search capabilities in welcome screen
function updateWelcomeScreen() {
    const welcomeScreen = document.getElementById('welcomeScreen');
    if (!welcomeScreen) return;
    
    // Add vector search indicator to welcome screen if enabled
    if (vectorSearchStatus.enabled) {
        const existingIndicator = welcomeScreen.querySelector('.vector-welcome-indicator');
        if (!existingIndicator) {
            const vectorIndicator = document.createElement('div');
            vectorIndicator.className = 'vector-welcome-indicator';
            vectorIndicator.style.cssText = `
                background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
                color: white;
                padding: 8px 16px;
                border-radius: 20px;
                margin: 12px auto;
                max-width: 300px;
                text-align: center;
                font-size: 0.9em;
                font-weight: 500;
                box-shadow: 0 2px 8px rgba(139, 92, 246, 0.3);
            `;
            vectorIndicator.innerHTML = `
                <span class="material-icons" style="font-size: 16px; vertical-align: middle; margin-right: 6px;">psychology</span>
                ${vectorSearchStatus.hybridAvailable ? 'Hybrid Search' : 'Vector Search'} Enabled
            `;
            
            const welcomeTitle = welcomeScreen.querySelector('.welcome-title');
            if (welcomeTitle) {
                welcomeTitle.insertAdjacentElement('afterend', vectorIndicator);
            }
        }
    }
}

// Keep all existing functions but add vector search awareness where appropriate
// ... (all other existing functions remain the same)

// ✅ ENHANCED: Add debug panel toggle for development
function addDebugToggle() {
    if (PRODUCTION_CONFIG.DEBUG_MODE) {
        setTimeout(() => {
            addVectorSearchDebugPanel();
            updateWelcomeScreen();
        }, 2000);
        
        // Add keyboard shortcut for debug panel
        document.addEventListener('keydown', function(e) {
            if (e.ctrlKey && e.shiftKey && e.key === 'V') {
                toggleVectorDebugPanel();
            }
        });
        
        console.log("🔧 Vector search debug features enabled (Ctrl+Shift+V to toggle debug panel)");
    }
}

// Enhanced initialization completion
document.addEventListener('DOMContentLoaded', function() {
    // ... existing initialization code ...
    
    // Add vector search debug features in development
    addDebugToggle();
});

// Enhanced global function exposure with vector search
window.testVectorSearch = testVectorSearch;
window.testHybridSearch = testHybridSearch;
window.checkVectorCapabilities = checkVectorCapabilities;
window.toggleVectorDebugPanel = toggleVectorDebugPanel;

// Keep all existing global functions
window.toggleSidebar = toggleSidebar;
window.applyFilters = applyFilters;
window.clearFilters = clearFilters;
window.removeFilter = removeFilter;
window.askQuestion = askQuestion;
window.handleKeyPress = handleKeyPress;
window.sendMessage = sendMessage;
window.clearChat = clearChat;
window.exportChat = exportChat;
window.updateHierarchyFilters = updateHierarchyFilters;
window.updateSubDispositions = updateSubDispositions;
window.toggleChatSession = toggleChatSession;
window.toggleDetailedTable = toggleDetailedTable;
window.downloadCategoryData = downloadCategoryData;

// Enhanced debug functions
window.getProductionMetrics = () => ({
    ...performanceMetrics,
    vectorSearchStatus: vectorSearchStatus
});
window.getProductionConfig = () => ({
    ...PRODUCTION_CONFIG,
    vectorSearchEnabled: vectorSearchStatus.enabled
});

console.log("✅ ENHANCED: Metro AI Analytics Chat v4.8.0 with VECTOR SEARCH loaded successfully");
console.log("🔮 Vector search: UI enhancements and debug capabilities enabled");
console.log("🔥 Hybrid search: Enhanced relevance and semantic similarity support");
console.log("📊 Search quality indicators: Real-time feedback on search enhancement");
console.log("🔧 Debug mode:", PRODUCTION_CONFIG.DEBUG_MODE ? "ENABLED (Ctrl+Shift+V for debug panel)" : "DISABLED");