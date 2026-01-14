// Enhanced Metro AI Call Center Analytics Chat - VECTOR SEARCH ENABLED
// Version:1-13-26.2 - Full Width Layout Update

// Updating for transcript search

// =============================================================================
// PRODUCTION CONFIGURATION & GLOBAL STATE
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

// Global state for transcript search
let transcriptSearchMode = false;
let lastTranscriptResults = [];

// Vector search state tracking
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
    VECTOR_SEARCH_UI: true
};

//Configuration for default evaluation inclusion
const EVALUATION_DEFAULTS = {
    INCLUDE_ALL_FILTERED: true,     // Default to all filtered evaluations
    DEFAULT_MAX_RESULTS: null,      // null = no limit
    DEFAULT_CHAT_LIMIT: null,       // null = no limit for chat
    DEFAULT_TRANSCRIPT_LIMIT: null, // null = no limit for transcript search
    ALLOW_USER_OVERRIDE: true       // Allow users to specify limits
};

// Performance monitoring
const performanceMetrics = {
    filterLoadTime: 0,
    chatResponseTimes: [],
    errorCount: 0,
    lastFilterUpdate: null,
    vectorSearchUsage: 0,
    hybridSearchUsage: 0
};

// STRICT METADATA CONFIGURATION - Only display these fields
const ALLOWED_API_FIELDS = new Set([
    'evaluationId',      // Primary evaluation identifier
    'weighted_score',    // Your scoring metric
    'url',              // Evaluation URL
    'partner',          // Partner name
    'site',             // Site location
    'lob',              // Line of business
    'agentName',        // Agent full name
    'agentId',          // Agent identifier
    'disposition',      // Primary disposition
    'subDisposition',   // Secondary disposition
    'created_on',       // Creation timestamp
    'call_date',        // Call date
    'call_duration',    // Duration in seconds
    'language',         // Call language
    'evaluation'        // Evaluation content
]);

// Fields that should NEVER be displayed in UI
const FORBIDDEN_INTERNAL_FIELDS = new Set([
    '_score', '_id', '_index', 'score', 'search_type', 'match_count',
    'chunk_id', 'vector_score', 'text_score', 'hybrid_score',
    'highlighted_snippets', 'search_words', 'template_name', 'template_id',
    'program', 'internalId', 'vector_dimension', 'best_matching_chunks',
    'vector_enhanced', 'Type', 'Template', 'Program'
]);

// =============================================================================
// CORE UTILITY FUNCTIONS - UPDATED FOR FULL WIDTH LAYOUT
// =============================================================================

function toggleSidebar() {
    const leftPanel = document.querySelector('.left-panel');
    const sidebarTab = document.getElementById('sidebarTab');
    
    if (!leftPanel) {
        console.error("❌ Left panel element not found");
        return;
    }
    
    // Toggle the collapsed state
    leftPanel.classList.toggle('collapsed');
    
    // Update sidebar tab appearance
    if (sidebarTab) {
        const isCollapsed = leftPanel.classList.contains('collapsed');
        const icon = sidebarTab.querySelector('.material-icons');
        
        if (icon) {
            icon.textContent = isCollapsed ? 'menu' : 'menu_open';
        }
        
        // Update active state
        if (isCollapsed) {
            sidebarTab.classList.remove('active');
        } else {
            sidebarTab.classList.add('active');
        }
    }
    
    const isCollapsed = leftPanel.classList.contains('collapsed');
    console.log("🔄 Sidebar toggled:", isCollapsed ? 'COLLAPSED' : 'EXPANDED');
}

// Updated apply filters and close function
function applyFiltersAndClose() {
    console.log("🔍 Applying filters and maintaining sidebar state");
    
    if (typeof applyFilters === 'function') {
        applyFilters();
    } else {
        console.warn("⚠️ applyFilters function not available yet");
    }
    
    // Note: In the new layout, we don't close the sidebar - it stays visible
    console.log("📌 Sidebar remains visible in new layout");
}

// Updated cancel filters function
function cancelFilters() {
    console.log("❌ Canceling filter changes");
    
    // Clear any pending changes
    // You might want to reset form values here
    
    // Note: In the new layout, we don't close the sidebar
    console.log("📌 Sidebar remains visible in new layout");
}

// Remove backdrop functionality (not needed in new layout)
function removeBackdrop() {
    const backdrop = document.querySelector('.sidebar-backdrop');
    if (backdrop) {
        backdrop.remove();
        console.log("🗑️ Backdrop removed");
    }
}

function showCriticalError(message) {
    console.error("🚨 CRITICAL ERROR:", message);
    
    // Remove existing error overlays
    const existingOverlay = document.getElementById('criticalErrorOverlay');
    if (existingOverlay) {
        existingOverlay.remove();
    }
    
    // Create error overlay
    const errorOverlay = document.createElement('div');
    errorOverlay.id = 'criticalErrorOverlay';
    errorOverlay.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.8);
        z-index: 10000;
        display: flex;
        align-items: center;
        justify-content: center;
    `;
    
    const errorBox = document.createElement('div');
    errorBox.style.cssText = `
        background: white;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.3);
        max-width: 500px;
        width: 90%;
        text-align: center;
        font-family: 'Segoe UI', system-ui, sans-serif;
    `;
    
    errorBox.innerHTML = `
        <div style="color: #dc3545; font-size: 48px; margin-bottom: 20px;">⚠️</div>
        <h2 style="color: #dc3545; margin-bottom: 15px;">Critical Error</h2>
        <p style="color: #666; margin-bottom: 20px; line-height: 1.5;">${message}</p>
        <button onclick="location.reload()" style="
            background: linear-gradient(135deg, #6e32a0 0%, #e20074 100%);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 6px;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            transition: transform 0.2s;
        " onmouseover="this.style.transform='translateY(-1px)'" onmouseout="this.style.transform='translateY(0)'">
            Reload Application
        </button>
    `;
    
    errorOverlay.appendChild(errorBox);
    document.body.appendChild(errorOverlay);
}

async function withRetry(operation, context = "operation", maxAttempts = PRODUCTION_CONFIG.MAX_RETRY_ATTEMPTS) {
    let lastError;
    
    for (let attempt = 1; attempt <= maxAttempts; attempt++) {
        try {
            console.log(`🔄 ${context} - Attempt ${attempt}/${maxAttempts}`);
            return await operation();
        } catch (error) {
            lastError = error;
            console.warn(`⚠️ ${context} failed on attempt ${attempt}:`, error.message);
            
            if (attempt < maxAttempts) {
                const delay = PRODUCTION_CONFIG.RETRY_DELAY_BASE * Math.pow(2, attempt - 1);
                console.log(`⏳ Waiting ${delay}ms before retry...`);
                await new Promise(resolve => setTimeout(resolve, delay));
            }
        }
    }
    
    performanceMetrics.errorCount++;
    throw new Error(`${context} failed after ${maxAttempts} attempts. Last error: ${lastError.message}`);
}

function sanitizeApiData(data) {
    if (!data || typeof data !== 'object') return data;
    
    const sanitized = {};
    Object.keys(data).forEach(key => {
        if (ALLOWED_API_FIELDS.has(key) && !FORBIDDEN_INTERNAL_FIELDS.has(key)) {
            sanitized[key] = data[key];
        }
    });
    
    return sanitized;
}

function formatEvaluationResults(evaluations) {
    if (!Array.isArray(evaluations)) return [];
    
    return evaluations.map(evaluation => {
        const sanitized = sanitizeApiData(evaluation);
        
        // Ensure required formatting
        if (sanitized.call_date) {
            sanitized.call_date = new Date(sanitized.call_date).toLocaleDateString();
        }
        
        if (sanitized.weighted_score && typeof sanitized.weighted_score === 'number') {
            sanitized.weighted_score = Math.round(sanitized.weighted_score * 10) / 10;
        }
        
        return sanitized;
    });
}

// =============================================================================
// FILTER MANAGEMENT SYSTEM
// =============================================================================

async function loadFilterOptions() {
    const startTime = Date.now();
    console.log("🔄 Loading filter options...");
    
    try {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), PRODUCTION_CONFIG.FILTER_LOAD_TIMEOUT);

        const response = await withRetry(
            () => fetch('/filter_options_metadata', {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                },
                signal: controller.signal
            }),
            "Loading filter options"
        );
        
        clearTimeout(timeoutId);
        
        if (!response.ok) {
            throw new Error(`Filter API returned ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        console.log("📊 Raw filter options received:", data);
        
        // Update global filter options
        filterOptions = {
            templates: data.templates || [],
            programs: data.programs || [],
            partners: data.partners || [],
            sites: data.sites || [],
            lobs: data.lobs || [],
            callDispositions: data.callDispositions || [],
            callSubDispositions: data.callSubDispositions || [],
            languages: data.languages || [],
            callTypes: data.callTypes || []
        };
        
        // Populate UI dropdowns
        populateFilterDropdowns();
        
        performanceMetrics.filterLoadTime = Date.now() - startTime;
        performanceMetrics.lastFilterUpdate = new Date();
        
        console.log(`✅ Filter options loaded successfully in ${performanceMetrics.filterLoadTime}ms`);
        console.log('📊 Filter counts:', {
            templates: filterOptions.templates.length,
            programs: filterOptions.programs.length,
            partners: filterOptions.partners.length,
            sites: filterOptions.sites.length,
            lobs: filterOptions.lobs.length,
            dispositions: filterOptions.callDispositions.length
        });
        
        // Update status indicators
        updateDataStatusIndicators('success');
        
        return data;
        
    } catch (error) {
        console.error("❌ Failed to load filter options:", error);
        updateDataStatusIndicators('error');
        
        showCriticalError(`
            Unable to load filter options from the server.
            This may indicate a network issue or server problem.
            Please check your connection and try again.
        `);
        throw error;
    }
}

function populateFilterDropdowns() {
    console.log("🔧 Populating filter dropdowns...");
    
    const dropdownMappings = {
        templates: { elementId: 'templateFilter', countElementId: 'templateCount' },
        programs: { elementId: 'programFilter', countElementId: 'programCount' },
        partners: { elementId: 'partnerFilter', countElementId: 'partnerCount' },
        sites: { elementId: 'siteFilter', countElementId: 'siteCount' },
        lobs: { elementId: 'lobFilter', countElementId: 'lobCount' },
        callDispositions: { elementId: 'dispositionFilter', countElementId: 'dispositionCount' },
        callSubDispositions: { elementId: 'subDispositionFilter', countElementId: 'subDispositionCount' },
        languages: { elementId: 'languageFilter', countElementId: 'languageCount' }
    };
    
    Object.entries(dropdownMappings).forEach(([dataKey, { elementId, countElementId }]) => {
        const element = document.getElementById(elementId);
        const countElement = document.getElementById(countElementId);
        
        if (element && filterOptions[dataKey]) {
            // Clear existing options (keep "All" option)
            const allOption = element.querySelector('option[value=""]');
            element.innerHTML = '';
            if (allOption) element.appendChild(allOption);
            
            // Add new options
            filterOptions[dataKey].forEach(item => {
                const option = document.createElement('option');
                option.value = typeof item === 'string' ? item : item.value || item.name;
                option.textContent = typeof item === 'string' ? item : item.label || item.name || item.value;
                element.appendChild(option);
            });
            
            // Update count indicator
            if (countElement) {
                countElement.textContent = `(${filterOptions[dataKey].length})`;
            }
            
            console.log(`✅ Populated ${elementId} with ${filterOptions[dataKey].length} options`);
        } else {
            console.warn(`⚠️ Element ${elementId} not found or no data for ${dataKey}`);
        }
    });
}

function updateDataStatusIndicators(status) {
    const indicators = ['hierarchyDataStatus', 'callMetaDataStatus'];
    
    indicators.forEach(indicatorId => {
        const indicator = document.getElementById(indicatorId);
        if (indicator) {
            indicator.className = `data-status data-status-${status}`;
            
            switch (status) {
                case 'success':
                    indicator.textContent = '✅ Ready';
                    break;
                case 'loading':
                    indicator.textContent = '⏳ Loading...';
                    break;
                case 'error':
                    indicator.textContent = '❌ Error';
                    break;
            }
        }
    });
}

function updateHierarchyFilters(changedFilter) {
    console.log(`🔄 Updating hierarchy filters after ${changedFilter} change`);
    
    // This would contain logic to cascade filter changes
    // For now, we'll just log the change
    console.log("📋 Current filters:", getCurrentFilters());
}

function getCurrentFilters() {
    const filters = {};
    
    // Date filters
    const startDate = document.getElementById('startCallDate')?.value;
    const endDate = document.getElementById('endCallDate')?.value;
    if (startDate) filters.startCallDate = startDate;
    if (endDate) filters.endCallDate = endDate;
    
    // Hierarchy filters
    const hierarchyFilters = {
        template: 'templateFilter',
        program: 'programFilter', 
        partner: 'partnerFilter',
        site: 'siteFilter',
        lob: 'lobFilter'
    };
    
    Object.entries(hierarchyFilters).forEach(([key, elementId]) => {
        const value = document.getElementById(elementId)?.value;
        if (value) filters[key] = value;
    });
    
    // Call identifier filters
    const identifierFilters = {
        evaluationId: 'evaluationIdFilter',
        agentName: 'agentNameFilter',
        agentId: 'agentIdFilter'
    };
    
    Object.entries(identifierFilters).forEach(([key, elementId]) => {
        const value = document.getElementById(elementId)?.value?.trim();
        if (value) filters[key] = value;
    });
    
    // Call metadata filters
    const callMetadataFilters = {
        disposition: 'dispositionFilter',
        subDisposition: 'subDispositionFilter',
        language: 'languageFilter'
    };
    
    Object.entries(callMetadataFilters).forEach(([key, elementId]) => {
        const value = document.getElementById(elementId)?.value;
        if (value) filters[key] = value;
    });
    
    // Duration filters
    const minDuration = document.getElementById('minDurationFilter')?.value;
    const maxDuration = document.getElementById('maxDurationFilter')?.value;
    if (minDuration) filters.minDuration = parseInt(minDuration);
    if (maxDuration) filters.maxDuration = parseInt(maxDuration);
    
    // Score filters
    const minScore = document.getElementById('minScoreFilter')?.value;
    const maxScore = document.getElementById('maxScoreFilter')?.value;
    if (minScore) filters.minScore = parseFloat(minScore);
    if (maxScore) filters.maxScore = parseFloat(maxScore);
    
    return filters;
}

function clearFilters() {
    console.log("🔄 Clearing all filters...");
    
    // Clear all form elements
    const inputs = document.querySelectorAll('#sidebar input, #sidebar select');
    inputs.forEach(input => {
        if (input.type === 'date' || input.type === 'text' || input.type === 'number') {
            input.value = '';
        } else if (input.tagName === 'SELECT') {
            input.selectedIndex = 0;
        }
    });
    
    // Reset global state
    currentFilters = {};
    
    // Update UI
    updateActiveFiltersDisplay();
    updateChatStats(0, 0);
    
    console.log("✅ All filters cleared");
}

async function applyFilters() {
    console.log("🔍 Applying filters...");
    
    const filters = getCurrentFilters();
    console.log("📋 Filters to apply:", filters);
    
    currentFilters = filters;
    
    // Update UI immediately
    updateActiveFiltersDisplay();
    
    try {
        // Get stats for the current filters
        const stats = await getFilteredStats(filters);
        updateChatStats(Object.keys(filters).length, stats.totalRecords || 0);
        
        console.log("✅ Filters applied successfully");
        console.log("📊 Stats:", stats);
        
    } catch (error) {
        console.error("❌ Failed to apply filters:", error);
        showErrorMessage("Failed to apply filters. Please try again.");
    }
}

async function getFilteredStats(filters) {
    try {
        const response = await fetch('/analytics/stats', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ filters })
        });
        
        if (!response.ok) {
            throw new Error(`Stats API returned ${response.status}: ${response.statusText}`);
        }
        
        return await response.json();
    } catch (error) {
        console.error("❌ Failed to get filtered stats:", error);
        throw error;
    }
}

function updateActiveFiltersDisplay() {
    const container = document.getElementById('activeFiltersDisplay');
    if (!container) return;
    
    container.innerHTML = '';
    
    Object.entries(currentFilters).forEach(([key, value]) => {
        if (value && value.toString().trim()) {
            const filterTag = document.createElement('div');
            filterTag.className = 'filter-tag';
            filterTag.innerHTML = `
                <span class="filter-name">${formatFilterName(key)}:</span>
                <span class="filter-value">${formatFilterValue(value)}</span>
                <button class="remove-filter" onclick="removeFilter('${key}')" title="Remove filter">×</button>
            `;
            container.appendChild(filterTag);
        }
    });
}

function formatFilterName(key) {
    const nameMap = {
        startCallDate: 'Start Date',
        endCallDate: 'End Date',
        template: 'Template',
        program: 'Program',
        partner: 'Partner',
        site: 'Site',
        lob: 'LOB',
        evaluationId: 'Evaluation ID',
        agentName: 'Agent Name',
        agentId: 'Agent ID',
        disposition: 'Disposition',
        subDisposition: 'Sub-Disposition',
        language: 'Language',
        minDuration: 'Min Duration',
        maxDuration: 'Max Duration',
        minScore: 'Min Score',
        maxScore: 'Max Score'
    };
    
    return nameMap[key] || key;
}

function formatFilterValue(value) {
    if (typeof value === 'string' && value.length > 20) {
        return value.substring(0, 17) + '...';
    }
    return value.toString();
}

function removeFilter(key) {
    delete currentFilters[key];
    
    // Clear the corresponding UI element
    const elementMappings = {
        startCallDate: 'startCallDate',
        endCallDate: 'endCallDate',
        template: 'templateFilter',
        program: 'programFilter',
        partner: 'partnerFilter',
        site: 'siteFilter',
        lob: 'lobFilter',
        evaluationId: 'evaluationIdFilter',
        agentName: 'agentNameFilter',
        agentId: 'agentIdFilter',
        disposition: 'dispositionFilter',
        subDisposition: 'subDispositionFilter',
        language: 'languageFilter',
        minDuration: 'minDurationFilter',
        maxDuration: 'maxDurationFilter',
        minScore: 'minScoreFilter',
        maxScore: 'maxScoreFilter'
    };
    
    const elementId = elementMappings[key];
    const element = document.getElementById(elementId);
    if (element) {
        if (element.tagName === 'SELECT') {
            element.selectedIndex = 0;
        } else {
            element.value = '';
        }
    }
    
    // Update displays
    updateActiveFiltersDisplay();
    applyFilters(); // Reapply remaining filters
}

function updateChatStats(filterCount, totalRecords) {
    const activeFiltersElement = document.getElementById('activeFiltersCount');
    const totalRecordsElement = document.getElementById('totalRecords');
    
    if (activeFiltersElement) {
        activeFiltersElement.textContent = `${filterCount} filter${filterCount !== 1 ? 's' : ''}`;
    }
    
    if (totalRecordsElement) {
        totalRecordsElement.textContent = `${totalRecords.toLocaleString()} evaluations`;
    }
}

// =============================================================================
// CHAT FUNCTIONALITY
// =============================================================================

function sendMessage(message = null) {
    const input = document.getElementById('chatInput');
    const sendButton = document.getElementById('sendButton');
    
    const messageText = message || input.value.trim();
    
    if (!messageText) {
        console.warn("⚠️ Empty message, not sending");
        return;
    }
    
    if (isLoading) {
        console.warn("⚠️ Already processing a request");
        return;
    }
    
    console.log("💬 Sending message:", messageText);
    
    // Clear input and disable button
    input.value = '';
    sendButton.disabled = true;
    isLoading = true;
    
    // Hide welcome screen
    const welcomeScreen = document.getElementById('welcomeScreen');
    if (welcomeScreen && !welcomeScreen.classList.contains('hidden')) {
        welcomeScreen.classList.add('hidden');
    }
    
    // Add user message to chat
    addMessageToChat('user', messageText);
    
    // Add loading message
    const loadingId = addLoadingMessage();
    
    // Send to backend
    processMessage(messageText, loadingId)
        .finally(() => {
            sendButton.disabled = false;
            isLoading = false;
        });
}

function addMessageToChat(type, content) {
    const messagesContainer = document.getElementById('messagesContainer');
    if (!messagesContainer) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    const timestamp = new Date().toLocaleTimeString();
    
    messageDiv.innerHTML = `
        <div class="message-header">
            <span class="message-type">${type === 'user' ? 'You' : 'Metro AI Assistant'}</span>
            <span class="message-time">${timestamp}</span>
        </div>
        <div class="message-content">${formatMessageContent(content)}</div>
    `;
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function formatMessageContent(content) {
    // Basic formatting for message content
    return content
        .replace(/\n/g, '<br>')
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>');
}

function addLoadingMessage() {
    const messagesContainer = document.getElementById('messagesContainer');
    if (!messagesContainer) return null;
    
    const loadingId = 'loading-' + Date.now();
    const loadingDiv = document.createElement('div');
    loadingDiv.id = loadingId;
    loadingDiv.className = 'message loading-message';
    
    loadingDiv.innerHTML = `
        <div class="message-header">
            <span class="message-type">Metro AI Assistant</span>
            <span class="message-time">${new Date().toLocaleTimeString()}</span>
        </div>
        <div class="message-content">
            <div class="loading-indicator">
                <div class="loading-dots">
                    <div class="dot"></div>
                    <div class="dot"></div>
                    <div class="dot"></div>
                </div>
                <span>Analyzing your request...</span>
            </div>
        </div>
    `;
    
    messagesContainer.appendChild(loadingDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    return loadingId;
}

function removeLoadingMessage(loadingId) {
    const loadingElement = document.getElementById(loadingId);
    if (loadingElement) {
        loadingElement.remove();
    }
}

async function processMessage(message, loadingId) {
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message,
                filters: currentFilters,
                context: {
                    sessionId: currentSessionId,
                    previousMessages: chatHistory.slice(-5) // Last 5 messages for context
                }
            }),
            timeout: PRODUCTION_CONFIG.CHAT_REQUEST_TIMEOUT
        });
        
        if (!response.ok) {
            throw new Error(`Chat API returned ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // Remove loading message
        removeLoadingMessage(loadingId);
        
        // Add response to chat
        addMessageToChat('assistant', data.response || 'No response received');
        
        // Update chat history
        chatHistory.push(
            { role: 'user', content: message, timestamp: new Date() },
            { role: 'assistant', content: data.response, timestamp: new Date() }
        );
        
        console.log("✅ Message processed successfully");
        
    } catch (error) {
        console.error("❌ Failed to process message:", error);
        
        // Remove loading message
        removeLoadingMessage(loadingId);
        
        // Show error message
        addErrorMessage("Sorry, I encountered an error while processing your request. Please try again.");
        
        performanceMetrics.errorCount++;
    }
}

function addErrorMessage(message) {
    const messagesContainer = document.getElementById('messagesContainer');
    if (!messagesContainer) return;
    
    const errorDiv = document.createElement('div');
    errorDiv.className = 'message error-message';
    
    errorDiv.innerHTML = `
        <div class="message-header">
            <span class="message-type">Error</span>
            <span class="message-time">${new Date().toLocaleTimeString()}</span>
        </div>
        <div class="message-content">
            <div class="error-content">${message}</div>
        </div>
    `;
    
    messagesContainer.appendChild(errorDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function clearChat() {
    console.log("🔄 Clearing chat...");
    
    const messagesContainer = document.getElementById('messagesContainer');
    const welcomeScreen = document.getElementById('welcomeScreen');
    
    if (messagesContainer) {
        // Clear all messages
        const messages = messagesContainer.querySelectorAll('.message');
        messages.forEach(message => message.remove());
    }
    
    if (welcomeScreen) {
        welcomeScreen.classList.remove('hidden');
    }
    
    // Reset chat state
    chatHistory = [];
    currentSessionId = null;
    
    console.log("✅ Chat cleared");
}

function exportChat() {
    console.log("📥 Exporting chat...");
    
    if (chatHistory.length === 0) {
        alert("No chat history to export.");
        return;
    }
    
    try {
        const exportData = {
            timestamp: new Date().toISOString(),
            filters: currentFilters,
            messages: chatHistory,
            metadata: {
                version: "6.0.0",
                totalMessages: chatHistory.length,
                sessionId: currentSessionId
            }
        };
        
        const dataStr = JSON.stringify(exportData, null, 2);
        const dataBlob = new Blob([dataStr], { type: 'application/json' });
        
        const link = document.createElement('a');
        link.href = URL.createObjectURL(dataBlob);
        link.download = `metro-ai-chat-${new Date().toISOString().slice(0, 19).replace(/:/g, '-')}.json`;
        
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        console.log("✅ Chat exported successfully");
        
    } catch (error) {
        console.error("❌ Failed to export chat:", error);
        alert("Failed to export chat. Please try again.");
    }
}

// =============================================================================
// INITIALIZATION AND EVENT HANDLERS
// =============================================================================

function initializeApp() {
    console.log("🚀 Initializing Metro AI Call Center Analytics...");
    
    try {
        // Load filter options
        loadFilterOptions();
        
        // Set up event listeners
        setupEventListeners();
        
        // Initialize UI state
        updateChatStats(0, 0);
        updateActiveFiltersDisplay();
        
        // Remove any existing backdrop
        removeBackdrop();
        
        console.log("✅ Application initialized successfully");
        
    } catch (error) {
        console.error("❌ Failed to initialize application:", error);
        showCriticalError("Failed to initialize the application. Please reload the page.");
    }
}

function setupEventListeners() {
    console.log("🔧 Setting up event listeners...");
    
    // Chat input handling
    const chatInput = document.getElementById('chatInput');
    const sendButton = document.getElementById('sendButton');
    
    if (chatInput) {
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage();
            }
        });
        
        // Auto-resize textarea
        chatInput.addEventListener('input', () => {
            chatInput.style.height = 'auto';
            chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
        });
    }
    
    if (sendButton) {
        sendButton.addEventListener('click', () => sendMessage());
    }
    
    // Mobile responsiveness
    window.addEventListener('resize', () => {
        const leftPanel = document.querySelector('.left-panel');
        if (leftPanel && window.innerWidth <= 768) {
            // On mobile, collapse sidebar by default
            leftPanel.classList.add('collapsed');
        } else if (leftPanel && window.innerWidth > 768) {
            // On desktop, expand sidebar by default
            leftPanel.classList.remove('collapsed');
        }
    });
    
    console.log("✅ Event listeners set up successfully");
}

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Small delay to ensure all resources are loaded
    setTimeout(initializeApp, 100);
});

// Export functions to global scope for HTML onclick handlers
window.toggleSidebar = toggleSidebar;
window.applyFiltersAndClose = applyFiltersAndClose;
window.cancelFilters = cancelFilters;
window.sendMessage = sendMessage;
window.clearChat = clearChat;
window.exportChat = exportChat;
window.applyFilters = applyFilters;
window.clearFilters = clearFilters;
window.removeFilter = removeFilter;
window.updateHierarchyFilters = updateHierarchyFilters;

// Debug functions for development
window.debugGetCurrentFilters = getCurrentFilters;
window.debugGetChatHistory = () => chatHistory;
window.debugGetPerformanceMetrics = () => performanceMetrics;

console.log("🎯 Metro AI Chat - Full Width Layout loaded successfully");

// =============================================================================
// COMPREHENSIVE SEARCH TOGGLE - Added for v6.0.0
// =============================================================================

function getComprehensiveToggleState() {
    const toggle = document.getElementById('comprehensiveToggle');
    return toggle ? toggle.checked : false;
}

function updateComprehensiveMode() {
    const toggle = document.getElementById('comprehensiveToggle');
    const description = document.getElementById('searchModeDescription');
    
    if (toggle && description) {
        if (toggle.checked) {
            // Comprehensive mode ON
            description.textContent = '(Comprehensive - searches all data)';
            description.style.color = '#6e32a0';
            description.style.fontWeight = 'bold';
            console.log('🔍 COMPREHENSIVE MODE: ON - Will search full dataset');
        } else {
            // Smart detection mode
            description.textContent = '(Smart detection)';
            description.style.color = '#999';
            description.style.fontWeight = 'normal';
            console.log('⚡ SMART MODE: ON - Will use automatic detection');
        }
    }
}
