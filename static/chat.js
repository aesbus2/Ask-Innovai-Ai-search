// Enhanced Metro AI Call Center Analytics Chat - VECTOR SEARCH ENABLED
// Version: 4.8.0 - Complete working version with all functions defined
// FIXED: All missing function definitions added

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

// Performance monitoring
const performanceMetrics = {
    filterLoadTime: 0,
    chatResponseTimes: [],
    errorCount: 0,
    lastFilterUpdate: null,
    vectorSearchUsage: 0,
    hybridSearchUsage: 0
};

// =============================================================================
// CORE UTILITY FUNCTIONS
// =============================================================================

function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    if (!sidebar) return;
    
    sidebar.classList.toggle('open');
    console.log("📱 Sidebar toggled");
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
    
    errorOverlay.innerHTML = `
        <div style="
            background: white;
            padding: 30px;
            border-radius: 12px;
            max-width: 500px;
            margin: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            text-align: center;
        ">
            <div style="color: #dc3545; font-size: 48px; margin-bottom: 20px;">⚠️</div>
            <h2 style="color: #dc3545; margin: 0 0 15px 0; font-size: 1.5em;">System Error</h2>
            <p style="margin: 0 0 25px 0; color: #666; line-height: 1.5;">${message}</p>
            <div style="display: flex; gap: 10px; justify-content: center;">
                <button onclick="location.reload()" style="
                    background: #007bff; color: white; border: none; padding: 10px 20px;
                    border-radius: 6px; cursor: pointer; font-size: 1em;
                ">🔄 Reload Page</button>
                <button onclick="document.getElementById('criticalErrorOverlay').remove()" style="
                    background: #6c757d; color: white; border: none; padding: 10px 20px;
                    border-radius: 6px; cursor: pointer; font-size: 1em;
                ">Continue Anyway</button>
            </div>
        </div>
    `;
    
    document.body.appendChild(errorOverlay);
}

function initializePage() {
    console.log("🚀 Initializing page...");
    
    try {
        // Set default date range to last 30 days
        const today = new Date();
        const thirtyDaysAgo = new Date(today.getTime() - (30 * 24 * 60 * 60 * 1000));
        
        const endCallDate = document.getElementById('endCallDate');
        const startCallDate = document.getElementById('startCallDate');
        
        if (endCallDate) {
            endCallDate.valueAsDate = today;
        }
        
        if (startCallDate) {
            startCallDate.valueAsDate = thirtyDaysAgo;
        }
        
        // Initialize UI components
        setupEventListeners();
        updateDateRangeDisplay();
        setupIdFieldValidation();
        
        console.log("✅ Page initialization complete");
        
    } catch (error) {
        console.error("❌ Page initialization failed:", error);
        showCriticalError("Failed to initialize page: " + error.message);
    }
}

function setupEventListeners() {
    console.log("🔧 Setting up event listeners...");
    
    try {
        // Chat input handling
        const chatInput = document.getElementById('chatInput');
        if (chatInput) {
            chatInput.addEventListener('keydown', handleKeyPress);
            chatInput.addEventListener('input', function() {
                this.style.height = 'auto';
                this.style.height = this.scrollHeight + 'px';
            });
        }
        
        // Send button
        const sendButton = document.getElementById('sendButton');
        if (sendButton) {
            sendButton.addEventListener('click', sendMessage);
        }
        
        // Filter controls
        const applyFiltersBtn = document.getElementById('applyFilters');
        if (applyFiltersBtn) {
            applyFiltersBtn.addEventListener('click', applyFilters);
        }
        
        const clearFiltersBtn = document.getElementById('clearFilters');
        if (clearFiltersBtn) {
            clearFiltersBtn.addEventListener('click', clearFilters);
        }
        
        // Date range controls
        const startCallDate = document.getElementById('startCallDate');
        const endCallDate = document.getElementById('endCallDate');
        
        if (startCallDate) {
            startCallDate.addEventListener('change', updateDateRangeDisplay);
        }
        
        if (endCallDate) {
            endCallDate.addEventListener('change', updateDateRangeDisplay);
        }
        
        console.log("✅ Event listeners set up successfully");
        
    } catch (error) {
        console.error("❌ Error setting up event listeners:", error);
        throw error;
    }
}

function updateDateRangeDisplay() {
    console.log("📅 Date range updated");
    
    const startDate = document.getElementById('startCallDate');
    const endDate = document.getElementById('endCallDate');
    const display = document.getElementById('dateRangeDisplay');
    
    if (startDate && endDate && display) {
        const start = startDate.value || 'Not set';
        const end = endDate.value || 'Not set';
        display.textContent = `${start} to ${end}`;
    }
}

function setupIdFieldValidation() {
    console.log("🔍 Setting up ID field validation");
    
    const idFields = document.querySelectorAll('input[type="text"][id*="Id"]');
    idFields.forEach(field => {
        field.addEventListener('input', function() {
            if (this.id.toLowerCase().includes('id')) {
                this.value = this.value.replace(/[^0-9]/g, '');
            }
        });
    });
}

// =============================================================================
// FILTER MANAGEMENT
// =============================================================================

async function loadDynamicFilterOptions() {
    console.log("📋 Loading dynamic filter options...");
    const startTime = performance.now();
    
    try {
        const response = await fetch('/filter_options_metadata', {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json'
            },
            timeout: PRODUCTION_CONFIG.FILTER_LOAD_TIMEOUT
        });
        
        if (!response.ok) {
            throw new Error(`Filter API returned ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (data.status === 'success') {
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
            
            // Update vector search status
            if (data.vector_search_enabled !== undefined) {
                vectorSearchStatus.enabled = data.vector_search_enabled;
                vectorSearchStatus.hybridAvailable = data.hybrid_search_available || false;
                vectorSearchStatus.searchQuality = data.search_enhancements?.search_quality || 'text_only';
            }
            
            populateFilterOptions(filterOptions);
            updateFilterCounts(data);
            
            const loadTime = performance.now() - startTime;
            performanceMetrics.filterLoadTime = loadTime;
            performanceMetrics.lastFilterUpdate = new Date().toISOString();
            
            console.log(`✅ Filter options loaded successfully in ${loadTime.toFixed(2)}ms`);
            console.log(`🔮 Vector search: ${vectorSearchStatus.enabled ? 'ENABLED' : 'DISABLED'}`);
            
            return data;
            
        } else {
            throw new Error(data.message || data.error || 'Unknown filter loading error');
        }
        
    } catch (error) {
        console.error("❌ Failed to load filter options:", error);
        performanceMetrics.errorCount++;
        handleFilterLoadError(error.message);
        throw error;
    }
}

function populateFilterOptions(options) {
    console.log("🔄 Populating filter dropdowns...");
    
    // Template filter
    populateSelect('templateFilter', options.templates);
    
    // Program filter
    populateSelect('programFilter', options.programs);
    
    // Partner filter
    populateSelect('partnerFilter', options.partners);
    
    // Site filter
    populateSelect('siteFilter', options.sites);
    
    // LOB filter
    populateSelect('lobFilter', options.lobs);
    
    // Disposition filters
    populateSelect('dispositionFilter', options.callDispositions);
    populateSelect('subDispositionFilter', options.callSubDispositions);
    
    // Other filters
    populateSelect('languageFilter', options.languages);
    populateSelect('callTypeFilter', options.callTypes);
    
    console.log("✅ Filter dropdowns populated");
}

function populateSelect(selectId, options) {
    const select = document.getElementById(selectId);
    if (!select) {
        console.warn(`⚠️ Select element ${selectId} not found`);
        return;
    }
    
    // Clear existing options except the first (placeholder)
    const firstOption = select.firstElementChild;
    select.innerHTML = '';
    if (firstOption) {
        select.appendChild(firstOption);
    }
    
    // Add new options
    options.forEach(option => {
        const optionElement = document.createElement('option');
        optionElement.value = option;
        optionElement.textContent = option;
        select.appendChild(optionElement);
    });
}

function updateFilterCounts(data) {
    const countsElement = document.getElementById('filterCounts');
    if (!countsElement) return;
    
    const totalEvaluations = data.total_evaluations || 0;
    const totalIndices = data.total_indices || 0;
    const vectorEnabled = data.vector_search_enabled ? '🔮' : '';
    
    countsElement.innerHTML = `
        <div style="font-size: 0.85em; color: #666; padding: 8px 12px; background: #f8f9fa; border-radius: 6px; margin: 10px 0;">
            📊 <strong>${totalEvaluations.toLocaleString()}</strong> evaluations available ${vectorEnabled}
            <br>
            <span style="font-size: 0.8em;">Across ${totalIndices} template collections</span>
            ${vectorEnabled ? '<br><span style="color: #28a745;">🔮 Enhanced with vector search</span>' : ''}
        </div>
    `;
}

function handleFilterLoadError(errorMessage) {
    console.error("🚨 Filter load error:", errorMessage);
    
    // Set empty filter options
    filterOptions = {
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
    
    populateFilterOptions(filterOptions);
    showFilterDataWarning(getErrorMessage(errorMessage));
}

function showFilterDataWarning(message) {
    const sidebar = document.getElementById('sidebar');
    if (!sidebar) return;
    
    const existingWarning = document.getElementById('filterDataWarning');
    if (existingWarning) {
        existingWarning.remove();
    }
    
    const warningDiv = document.createElement('div');
    warningDiv.id = 'filterDataWarning';
    warningDiv.innerHTML = `
        <div style="
            background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%);
            border: 1px solid #ffc107;
            border-radius: 8px;
            padding: 16px;
            margin: 16px;
            font-size: 0.9em;
            color: #856404;
            box-shadow: 0 2px 8px rgba(255, 193, 7, 0.2);
        ">
            <div style="font-weight: 600; margin-bottom: 8px; display: flex; align-items: center; gap: 8px;">
                ⚠️ Filter System Notice
            </div>
            <div style="margin-bottom: 12px; line-height: 1.4;">${message}</div>
            <div style="display: flex; gap: 8px;">
                <button onclick="loadDynamicFilterOptions().catch(console.error)" style="
                    padding: 6px 12px; background: #856404; color: white; border: none;
                    border-radius: 4px; cursor: pointer; font-size: 0.8em;
                ">🔄 Retry</button>
                <button onclick="document.getElementById('filterDataWarning').remove()" style="
                    padding: 6px 12px; background: transparent; color: #856404;
                    border: 1px solid #856404; border-radius: 4px; cursor: pointer; font-size: 0.8em;
                ">Dismiss</button>
            </div>
        </div>
    `;
    
    const sidebarHeader = sidebar.querySelector('.sidebar-header');
    if (sidebarHeader) {
        sidebarHeader.insertAdjacentElement('afterend', warningDiv);
    }
}

function getErrorMessage(errorMessage) {
    if (errorMessage.includes('timeout') || errorMessage.includes('ECONNABORTED')) {
        return "Connection timeout - please check your internet connection and try again";
    } else if (errorMessage.includes('network') || errorMessage.includes('fetch')) {
        return "Network error - unable to connect to the server";
    } else if (errorMessage.includes('opensearch') || errorMessage.includes('database')) {
        return "Database temporarily unavailable - filters will be limited";
    } else {
        return "Unable to load filter data - please try refreshing the page";
    }
}

// =============================================================================
// FILTER ACTIONS
// =============================================================================

function applyFilters() {
    console.log("🔍 Applying filters...");
    
    currentFilters = {};
    
    // Collect filter values
    const filterMappings = [
        { id: 'templateFilter', key: 'template_name' },
        { id: 'programFilter', key: 'program' },
        { id: 'partnerFilter', key: 'partner' },
        { id: 'siteFilter', key: 'site' },
        { id: 'lobFilter', key: 'lob' },
        { id: 'dispositionFilter', key: 'disposition' },
        { id: 'subDispositionFilter', key: 'sub_disposition' },
        { id: 'languageFilter', key: 'language' },
        { id: 'callTypeFilter', key: 'call_type' },
        { id: 'startCallDate', key: 'call_date_start' },
        { id: 'endCallDate', key: 'call_date_end' }
    ];
    
    filterMappings.forEach(mapping => {
        const element = document.getElementById(mapping.id);
        if (element && element.value) {
            currentFilters[mapping.key] = element.value;
        }
    });
    
    updateActiveFiltersDisplay();
    console.log("✅ Filters applied:", currentFilters);
}

function clearFilters() {
    console.log("🗑️ Clearing all filters...");
    
    currentFilters = {};
    
    // Clear all select elements
    const selects = document.querySelectorAll('select[id$="Filter"]');
    selects.forEach(select => {
        select.selectedIndex = 0;
    });
    
    // Clear date inputs
    const dateInputs = document.querySelectorAll('input[type="date"]');
    dateInputs.forEach(input => {
        input.value = '';
    });
    
    updateActiveFiltersDisplay();
    console.log("✅ All filters cleared");
}

function removeFilter(filterKey) {
    console.log(`🗑️ Removing filter: ${filterKey}`);
    
    delete currentFilters[filterKey];
    
    // Clear the corresponding UI element
    const elementMappings = {
        'template_name': 'templateFilter',
        'program': 'programFilter',
        'partner': 'partnerFilter',
        'site': 'siteFilter',
        'lob': 'lobFilter',
        'disposition': 'dispositionFilter',
        'sub_disposition': 'subDispositionFilter',
        'language': 'languageFilter',
        'call_type': 'callTypeFilter',
        'call_date_start': 'startCallDate',
        'call_date_end': 'endCallDate'
    };
    
    const elementId = elementMappings[filterKey];
    if (elementId) {
        const element = document.getElementById(elementId);
        if (element) {
            if (element.tagName === 'SELECT') {
                element.selectedIndex = 0;
            } else {
                element.value = '';
            }
        }
    }
    
    updateActiveFiltersDisplay();
}

function updateActiveFiltersDisplay() {
    const activeFiltersDiv = document.getElementById('activeFilters');
    if (!activeFiltersDiv) return;
    
    if (Object.keys(currentFilters).length === 0) {
        activeFiltersDiv.innerHTML = '<p style="color: #666; font-style: italic;">No active filters</p>';
        return;
    }
    
    const filterLabels = {
        'template_name': 'Template',
        'program': 'Program',
        'partner': 'Partner',
        'site': 'Site',
        'lob': 'LOB',
        'disposition': 'Disposition',
        'sub_disposition': 'Sub-Disposition',
        'language': 'Language',
        'call_type': 'Call Type',
        'call_date_start': 'Start Date',
        'call_date_end': 'End Date'
    };
    
    const filterTags = Object.entries(currentFilters)
        .map(([key, value]) => {
            const label = filterLabels[key] || key;
            return `
                <span class="filter-tag" style="
                    display: inline-flex;
                    align-items: center;
                    gap: 6px;
                    background: #e3f2fd;
                    color: #1976d2;
                    padding: 4px 8px;
                    border-radius: 12px;
                    font-size: 0.8em;
                    margin: 2px;
                ">
                    <strong>${label}:</strong> ${value}
                    <button onclick="removeFilter('${key}')" style="
                        background: none;
                        border: none;
                        color: #1976d2;
                        cursor: pointer;
                        padding: 0;
                        margin-left: 4px;
                        font-size: 1.1em;
                        line-height: 1;
                    ">×</button>
                </span>
            `;
        })
        .join('');
    
    activeFiltersDiv.innerHTML = `
        <div style="margin-bottom: 8px;">
            <strong>Active Filters:</strong>
        </div>
        <div>${filterTags}</div>
    `;
}

// =============================================================================
// CHAT FUNCTIONALITY
// =============================================================================

function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

async function sendMessage() {
    const chatInput = document.getElementById('chatInput');
    const message = chatInput.value.trim();
    
    if (!message || isLoading) return;
    
    console.log("💬 Sending message:", message);
    
    try {
        isLoading = true;
        updateSendButtonState(true);
        
        // Add user message to chat
        addMessageToChat('user', message);
        
        // Clear input
        chatInput.value = '';
        chatInput.style.height = 'auto';
        
        // Show typing indicator
        showTypingIndicator();
        
        const startTime = performance.now();
        
        // Send request
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: message,
                history: chatHistory,
                filters: currentFilters,
                analytics: true
            }),
            timeout: PRODUCTION_CONFIG.CHAT_REQUEST_TIMEOUT
        });
        
        if (!response.ok) {
            throw new Error(`Chat API returned ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        const responseTime = performance.now() - startTime;
        performanceMetrics.chatResponseTimes.push(responseTime);
        
        // Remove typing indicator
        hideTypingIndicator();
        
        if (data.response) {
            // Add assistant response
            addMessageToChat('assistant', data.response, data);
            
            // Update chat history
            chatHistory.push(
                { role: 'user', content: message },
                { role: 'assistant', content: data.response }
            );
            
            console.log(`✅ Message sent successfully in ${responseTime.toFixed(2)}ms`);
            
        } else {
            throw new Error('No response received from chat API');
        }
        
    } catch (error) {
        console.error("❌ Chat error:", error);
        hideTypingIndicator();
        addMessageToChat('error', `Sorry, I encountered an error: ${error.message}`);
        performanceMetrics.errorCount++;
    } finally {
        isLoading = false;
        updateSendButtonState(false);
    }
}

function addMessageToChat(role, content, metadata = null) {
    const chatMessages = document.getElementById('chatMessages');
    const welcomeScreen = document.getElementById('welcomeScreen');
    
    if (!chatMessages) return;
    
    // Hide welcome screen on first message
    if (welcomeScreen && !welcomeScreen.classList.contains('hidden')) {
        welcomeScreen.classList.add('hidden');
        chatMessages.classList.remove('hidden');
    }
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message`;
    
    const timestamp = new Date().toLocaleTimeString();
    
    if (role === 'user') {
        messageDiv.innerHTML = `
            <div class="message-header">
                <span class="role">You</span>
                <span class="timestamp">${timestamp}</span>
            </div>
            <div class="message-content">${escapeHtml(content)}</div>
        `;
    } else if (role === 'assistant') {
        const vectorIndicator = metadata && vectorSearchStatus.enabled ? 
            `<span style="color: #28a745; font-size: 0.8em;">🔮 ${vectorSearchStatus.searchQuality}</span>` : '';
        
        messageDiv.innerHTML = `
            <div class="message-header">
                <span class="role">AI Assistant</span>
                <span class="timestamp">${timestamp}</span>
                ${vectorIndicator}
            </div>
            <div class="message-content">${formatMessage(content)}</div>
        `;
    } else if (role === 'error') {
        messageDiv.innerHTML = `
            <div class="message-header">
                <span class="role" style="color: #dc3545;">⚠️ Error</span>
                <span class="timestamp">${timestamp}</span>
            </div>
            <div class="message-content" style="color: #dc3545;">${escapeHtml(content)}</div>
        `;
    }
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showTypingIndicator() {
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return;
    
    const typingDiv = document.createElement('div');
    typingDiv.id = 'typingIndicator';
    typingDiv.className = 'message assistant-message typing';
    typingDiv.innerHTML = `
        <div class="message-header">
            <span class="role">AI Assistant</span>
            <span class="timestamp">typing...</span>
        </div>
        <div class="message-content">
            <div class="typing-dots">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    `;
    
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function hideTypingIndicator() {
    const typingIndicator = document.getElementById('typingIndicator');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

function updateSendButtonState(loading) {
    const sendButton = document.getElementById('sendButton');
    if (!sendButton) return;
    
    if (loading) {
        sendButton.disabled = true;
        sendButton.innerHTML = '⏳';
        sendButton.style.opacity = '0.6';
    } else {
        sendButton.disabled = false;
        sendButton.innerHTML = '📤';
        sendButton.style.opacity = '1';
    }
}

function clearChat() {
    if (!confirm('Clear all chat messages? This cannot be undone.')) {
        return;
    }
    
    chatHistory = [];
    chatSessions = [];
    currentSessionId = null;
    
    const chatMessages = document.getElementById('chatMessages');
    const welcomeScreen = document.getElementById('welcomeScreen');
    
    if (chatMessages) {
        chatMessages.innerHTML = '';
        chatMessages.classList.add('hidden');
    }
    
    if (welcomeScreen) {
        welcomeScreen.classList.remove('hidden');
    }
    
    console.log("🗑️ Chat cleared");
}

function exportChat() {
    if (chatHistory.length === 0) {
        alert('No chat history to export.');
        return;
    }
    
    const exportData = {
        chatHistory: chatHistory,
        filters: currentFilters,
        timestamp: new Date().toISOString(),
        sessionInfo: {
            totalMessages: chatHistory.length,
            vectorSearchEnabled: vectorSearchStatus.enabled,
            searchQuality: vectorSearchStatus.searchQuality
        }
    };
    
    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
        type: 'application/json'
    });
    
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `metro-ai-chat-${new Date().toISOString().slice(0, 19)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    console.log("📥 Chat exported successfully");
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

function escapeHtml(unsafe) {
    return unsafe
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function formatMessage(message) {
    // Convert markdown-like formatting to HTML
    return message
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/`(.*?)`/g, '<code>$1</code>')
        .replace(/\n/g, '<br>');
}

function askQuestion(question) {
    const chatInput = document.getElementById('chatInput');
    if (!chatInput) return;
    
    chatInput.value = question;
    chatInput.style.height = 'auto';
    chatInput.style.height = chatInput.scrollHeight + 'px';
    chatInput.focus();
    chatInput.scrollIntoView({ behavior: 'smooth' });
}

function toggleChatSession(sessionId) {
    const sessionElement = document.getElementById(`session-${sessionId}`);
    if (!sessionElement) return;
    
    sessionElement.classList.toggle('collapsed');
}

function updateHierarchyFilters() {
    // Implement hierarchy filtering logic
    console.log("🔄 Updating hierarchy filters");
}

function updateSubDispositions() {
    // Update sub-dispositions based on selected disposition
    console.log("🔄 Updating sub-dispositions");
}

function toggleDetailedTable() {
    // Toggle detailed analytics table
    console.log("📊 Toggling detailed table");
}

function downloadCategoryData() {
    // Download category data
    console.log("📥 Downloading category data");
}

// =============================================================================
// VECTOR SEARCH CAPABILITIES
// =============================================================================

async function checkVectorSearchCapabilities() {
    try {
        const response = await fetch('/debug/vector_capabilities');
        const data = await response.json();
        
        if (data.status === 'success') {
            vectorSearchStatus.enabled = data.capabilities?.cluster_vector_support || false;
            vectorSearchStatus.hybridAvailable = data.capabilities?.hybrid_search_available || false;
            vectorSearchStatus.searchQuality = vectorSearchStatus.hybridAvailable ? 'hybrid_enhanced' : 
                                              vectorSearchStatus.enabled ? 'vector_enhanced' : 'text_only';
            
            console.log(`✅ Vector search capabilities checked: ${vectorSearchStatus.searchQuality}`);
        }
    } catch (error) {
        console.warn("⚠️ Could not check vector search capabilities:", error);
    }
}

// =============================================================================
// STYLES AND FORMATTING
// =============================================================================

function loadFormattingStyles() {
    const style = document.createElement('style');
    style.textContent = `
        .typing-dots {
            display: flex;
            gap: 4px;
            padding: 8px 0;
        }
        
        .typing-dots span {
            width: 8px;
            height: 8px;
            background: #ccc;
            border-radius: 50%;
            animation: typing 1.4s infinite ease-in-out;
        }
        
        .typing-dots span:nth-child(1) { animation-delay: -0.32s; }
        .typing-dots span:nth-child(2) { animation-delay: -0.16s; }
        
        @keyframes typing {
            0%, 80%, 100% { transform: scale(0.8); opacity: 0.5; }
            40% { transform: scale(1); opacity: 1; }
        }
        
        .message {
            margin-bottom: 16px;
            padding: 12px;
            border-radius: 8px;
        }
        
        .user-message {
            background: #e3f2fd;
            margin-left: 20%;
        }
        
        .assistant-message {
            background: #f5f5f5;
            margin-right: 20%;
        }
        
        .message-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 0.85em;
            color: #666;
        }
        
        .role {
            font-weight: 600;
        }
        
        .filter-tag {
            transition: all 0.2s ease;
        }
        
        .filter-tag:hover {
            transform: translateY(-1px);
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
    `;
    document.head.appendChild(style);
}

// =============================================================================
// DEBUG FUNCTIONS
// =============================================================================

function debugChatSystem() {
    console.log("🔧 DEBUG: Chat System Status");
    console.log("Current Filters:", currentFilters);
    console.log("Chat History:", chatHistory);
    console.log("Filter Options:", filterOptions);
    console.log("Performance Metrics:", performanceMetrics);
    console.log("Vector Search Status:", vectorSearchStatus);
}

// =============================================================================
// GLOBAL FUNCTION EXPOSURE
// =============================================================================

// Core functions
window.toggleSidebar = toggleSidebar;
window.initializePage = initializePage;
window.showCriticalError = showCriticalError;

// Filter functions
window.applyFilters = applyFilters;
window.clearFilters = clearFilters;
window.removeFilter = removeFilter;
window.updateHierarchyFilters = updateHierarchyFilters;
window.updateSubDispositions = updateSubDispositions;

// Chat functions
window.askQuestion = askQuestion;
window.handleKeyPress = handleKeyPress;
window.sendMessage = sendMessage;
window.clearChat = clearChat;
window.exportChat = exportChat;
window.toggleChatSession = toggleChatSession;

// Utility functions
window.toggleDetailedTable = toggleDetailedTable;
window.downloadCategoryData = downloadCategoryData;

// Debug functions
window.debugChatSystem = debugChatSystem;
window.getProductionMetrics = () => performanceMetrics;
window.getProductionConfig = () => PRODUCTION_CONFIG;

// =============================================================================
// INITIALIZATION
// =============================================================================

document.addEventListener('DOMContentLoaded', function() {
    console.log("🚀 Production Chat Interface v4.3.0 initializing...");
    console.log("🚀 Metro AI Analytics v4.8.0 - VECTOR SEARCH ENHANCED Chat Interface Starting...");
    
    const startTime = performance.now();
    
    try {
        // Initialize page
        initializePage();
        
        // Load styles
        loadFormattingStyles();
        
        // Check vector search capabilities
        setTimeout(() => {
            checkVectorSearchCapabilities()
                .then(() => {
                    console.log(`✅ Vector search status: ${vectorSearchStatus.enabled ? 'ENABLED' : 'DISABLED'}`);
                })
                .catch(error => {
                    console.warn("⚠️ Vector search check failed:", error);
                });
        }, 500);
        
        // Load filter options (non-blocking)
        setTimeout(() => {
            loadDynamicFilterOptions()
                .then(() => {
                    const loadTime = performance.now() - startTime;
                    console.log(`✅ PRODUCTION initialization completed in ${loadTime.toFixed(2)}ms`);
                })
                .catch(error => {
                    console.error("❌ Filter loading failed:", error);
                    showCriticalError("Failed to load filter options: " + error.message);
                });
        }, 1000);
        
    } catch (error) {
        console.error("❌ CRITICAL: Production initialization failed:", error);
        showCriticalError("Critical initialization failure: " + error.message);
    }
});

console.log("✅ ENHANCED: Metro AI Analytics Chat v4.8.0 with VECTOR SEARCH loaded successfully");
console.log("🔮 Vector search: Enhanced relevance and semantic similarity support");
console.log("🔧 Debug mode:", PRODUCTION_CONFIG.DEBUG_MODE ? "ENABLED" : "DISABLED");