// Enhanced Metro AI Call Center Analytics Chat - VECTOR SEARCH ENABLED
// Version: 6.0.0 - Working Base - FIXED
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
// CORE UTILITY FUNCTIONS
// =============================================================================

function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    if (!sidebar) {
        console.error("❌ Sidebar element not found");
        return;
    }
    
    // Toggle the sidebar visibility
    sidebar.classList.toggle('open');
    
    // Update button state
    const toggleBtn = document.getElementById('toggleSidebarBtn');
    if (toggleBtn) {
        const icon = toggleBtn.querySelector('.material-icons');
        if (icon) {
            icon.textContent = sidebar.classList.contains('open') ? 'close' : 'tune';
        }
    }
    
    console.log("🔄 Sidebar toggled:", sidebar.classList.contains('open') ? 'OPEN' : 'CLOSED');
}

// Fixed updateEvaluationScope function with proper error handling
function updateEvaluationScope() {
    const toggle = document.getElementById('includeAllFilteredToggle');
    if (toggle) {
        EVALUATION_DEFAULTS.INCLUDE_ALL_FILTERED = toggle.checked;
        console.log(`🔧 Evaluation scope updated: ${toggle.checked ? 'ALL filtered' : 'LIMITED'}`);
        
        // Show feedback
        if (typeof showToast === 'function') {
            showToast(toggle.checked ? 'Using all filtered evaluations' : 'Using limited evaluations', 'info');
        }
    } else {
        console.warn("⚠️ includeAllFilteredToggle element not found");
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

// =============================================================================
// FILTER MANAGEMENT FUNCTIONS
// =============================================================================

async function loadDynamicFilterOptions() {
    const startTime = performance.now();
    console.log("🔄 Loading dynamic filter options from API...");
    
    try {
        const response = await fetch('/filter_options_metadata', {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            },
            signal: AbortSignal.timeout(PRODUCTION_CONFIG.FILTER_LOAD_TIMEOUT)
        });
        
        if (!response.ok) {
            throw new Error(`Filter API responded with ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        if (!data || typeof data !== 'object') {
            throw new Error('Invalid filter data received from API');
        }
        
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
        
        // Populate filter dropdowns
        populateFilterDropdowns();
        
        // Update filter counts
        updateFilterCounts();
        
        const loadTime = performance.now() - startTime;
        performanceMetrics.filterLoadTime = loadTime;
        performanceMetrics.lastFilterUpdate = new Date();
        
        console.log(`✅ Filter options loaded successfully in ${loadTime.toFixed(2)}ms`);
        console.log('📊 Filter counts:', {
            templates: filterOptions.templates.length,
            programs: filterOptions.programs.length,
            partners: filterOptions.partners.length,
            sites: filterOptions.sites.length,
            lobs: filterOptions.lobs.length,
            dispositions: filterOptions.callDispositions.length
        });
        
    } catch (error) {
        console.error("❌ Failed to load filter options:", error);
        performanceMetrics.errorCount++;
        
        // Show user-friendly error message
        showCriticalError(`Failed to load filter options: ${error.message}. Please check your internet connection and try refreshing the page.`);
        
        throw error;
    }
}

function populateFilterDropdowns() {
    console.log("🔄 Populating filter dropdowns...");
    
    // Define dropdown mappings
    const dropdowns = [
        { id: 'templateFilter', options: filterOptions.templates, countId: 'templateCount' },
        { id: 'programFilter', options: filterOptions.programs, countId: 'programCount' },
        { id: 'partnerFilter', options: filterOptions.partners, countId: 'partnerCount' },
        { id: 'siteFilter', options: filterOptions.sites, countId: 'siteCount' },
        { id: 'lobFilter', options: filterOptions.lobs, countId: 'lobCount' },
        { id: 'callDispositionFilter', options: filterOptions.callDispositions, countId: 'dispositionCount' },
        { id: 'languageFilter', options: filterOptions.languages, countId: 'languageCount' },
        { id: 'callTypeFilter', options: filterOptions.callTypes, countId: 'callTypeCount' }
    ];
    
    dropdowns.forEach(({ id, options, countId }) => {
        const dropdown = document.getElementById(id);
        const countElement = document.getElementById(countId);
        
        if (!dropdown) {
            console.warn(`⚠️ Dropdown ${id} not found in DOM`);
            return;
        }
        
        // Clear existing options (except first "All" option)
        const firstOption = dropdown.querySelector('option');
        dropdown.innerHTML = '';
        if (firstOption) {
            dropdown.appendChild(firstOption);
        }
        
        // Add new options
        options.forEach(option => {
            const optionElement = document.createElement('option');
            optionElement.value = option;
            optionElement.textContent = option;
            dropdown.appendChild(optionElement);
        });
        
        // Update count
        if (countElement) {
            countElement.textContent = `(${options.length})`;
        }
        
        // Remove loading state
        dropdown.classList.remove('loading-filter');
        dropdown.disabled = false;
    });
    
    console.log("✅ Filter dropdowns populated successfully");
}

function updateFilterCounts() {
    const counts = {
        templates: filterOptions.templates.length,
        programs: filterOptions.programs.length,
        partners: filterOptions.partners.length,
        sites: filterOptions.sites.length,
        lobs: filterOptions.lobs.length,
        dispositions: filterOptions.callDispositions.length,
        languages: filterOptions.languages.length,
        callTypes: filterOptions.callTypes.length
    };
    
    // Update count displays
    Object.entries(counts).forEach(([key, count]) => {
        const countElement = document.getElementById(`${key.replace('s', '')}Count`);
        if (countElement) {
            countElement.textContent = `(${count})`;
        }
    });
}

function updateHierarchyFilters(changedFilter) {
    console.log(`🔄 Updating hierarchy filters after ${changedFilter} change`);
    
    // Get current selections
    const selections = {
        template: document.getElementById('templateFilter')?.value || '',
        program: document.getElementById('programFilter')?.value || '',
        partner: document.getElementById('partnerFilter')?.value || '',
        site: document.getElementById('siteFilter')?.value || ''
    };
    
    // Update dependent filters based on hierarchy
    // This would typically involve API calls to get filtered options
    // For now, we'll just log the change
    console.log('📊 Current selections:', selections);
}

function updateSubDispositions() {
    const dispositionFilter = document.getElementById('callDispositionFilter');
    const subDispositionFilter = document.getElementById('callSubDispositionFilter');
    
    if (!dispositionFilter || !subDispositionFilter) return;
    
    const selectedDisposition = dispositionFilter.value;
    
    // Clear sub-dispositions
    subDispositionFilter.innerHTML = '<option value="">All Sub-Dispositions</option>';
    
    // Filter sub-dispositions based on selected disposition
    const relevantSubDispositions = filterOptions.callSubDispositions.filter(sub => 
        !selectedDisposition || sub.startsWith(selectedDisposition)
    );
    
    relevantSubDispositions.forEach(sub => {
        const option = document.createElement('option');
        option.value = sub;
        option.textContent = sub;
        subDispositionFilter.appendChild(option);
    });
    
    console.log(`🔄 Updated sub-dispositions for: ${selectedDisposition || 'all'}`);
}

// =============================================================================
// FILTER APPLICATION FUNCTIONS
// =============================================================================

function applyFilters() {
    console.log("🔍 Applying filters...");
    
    // Collect filter values
    currentFilters = {
        evaluationId: document.getElementById('evaluationIdFilter')?.value || '',
        phoneNumber: document.getElementById('phoneNumberFilter')?.value || '',
        contactId: document.getElementById('contactIdFilter')?.value || '',
        ucid: document.getElementById('ucidFilter')?.value || '',
        startCallDate: document.getElementById('startCallDate')?.value || '',
        endCallDate: document.getElementById('endCallDate')?.value || '',
        template: document.getElementById('templateFilter')?.value || '',
        program: document.getElementById('programFilter')?.value || '',
        partner: document.getElementById('partnerFilter')?.value || '',
        site: document.getElementById('siteFilter')?.value || '',
        lob: document.getElementById('lobFilter')?.value || '',
        callDisposition: document.getElementById('callDispositionFilter')?.value || '',
        callSubDisposition: document.getElementById('callSubDispositionFilter')?.value || '',
        language: document.getElementById('languageFilter')?.value || '',
        callType: document.getElementById('callTypeFilter')?.value || ''
    };
    
    // Remove empty filters
    Object.keys(currentFilters).forEach(key => {
        if (!currentFilters[key]) {
            delete currentFilters[key];
        }
    });
    
    console.log("📊 Applied filters:", currentFilters);
    
    // Update UI to show active filters
    updateActiveFiltersDisplay();
    
    // Update stats
    refreshAnalyticsStats();
}

function clearFilters() {
    console.log("🧹 Clearing all filters...");
    
    // Reset filter form
    const filterForm = document.querySelector('.filter-panel');
    if (filterForm) {
        const inputs = filterForm.querySelectorAll('input, select');
        inputs.forEach(input => {
            if (input.type === 'date' || input.type === 'text') {
                input.value = '';
            } else if (input.tagName === 'SELECT') {
                input.selectedIndex = 0;
            }
        });
    }
    
    // Clear current filters
    currentFilters = {};
    
    // Update UI
    updateActiveFiltersDisplay();
    
    // Update stats
    refreshAnalyticsStats();
}

function updateActiveFiltersDisplay() {
    const filtersContainer = document.getElementById('chatHeaderFilters');
    const activeFiltersCount = document.getElementById('activeFiltersCount');
    
    if (!filtersContainer) return;
    
    // Clear existing filter tags
    filtersContainer.innerHTML = '';
    
    const filterCount = Object.keys(currentFilters).length;
    
    if (activeFiltersCount) {
        activeFiltersCount.textContent = `${filterCount} filter${filterCount !== 1 ? 's' : ''}`;
    }
    
    if (filterCount === 0) return;
    
    // Create filter tags
    Object.entries(currentFilters).forEach(([key, value]) => {
        const filterTag = document.createElement('div');
        filterTag.className = 'filter-tag';
        filterTag.innerHTML = `
            <span class="filter-name">${key}:</span>
            <span class="filter-value">${value}</span>
            <button onclick="removeFilter('${key}')" class="remove-filter">×</button>
        `;
        filtersContainer.appendChild(filterTag);
    });
}

function removeFilter(filterKey) {
    delete currentFilters[filterKey];
    
    // Clear the form field
    const fieldMap = {
        evaluationId: 'evaluationIdFilter',
        phoneNumber: 'phoneNumberFilter',
        contactId: 'contactIdFilter',
        ucid: 'ucidFilter',
        startCallDate: 'startCallDate',
        endCallDate: 'endCallDate',
        template: 'templateFilter',
        program: 'programFilter',
        partner: 'partnerFilter',
        site: 'siteFilter',
        lob: 'lobFilter',
        callDisposition: 'callDispositionFilter',
        callSubDisposition: 'callSubDispositionFilter',
        language: 'languageFilter',
        callType: 'callTypeFilter'
    };
    
    const fieldId = fieldMap[filterKey];
    if (fieldId) {
        const field = document.getElementById(fieldId);
        if (field) {
            if (field.type === 'date' || field.type === 'text') {
                field.value = '';
            } else if (field.tagName === 'SELECT') {
                field.selectedIndex = 0;
            }
        }
    }
    
    updateActiveFiltersDisplay();
    refreshAnalyticsStats();
}

// =============================================================================
// CHAT FUNCTIONS
// =============================================================================

function askQuestion(question) {
    const chatInput = document.getElementById('chatInput');
    if (chatInput) {
        chatInput.value = question;
        sendMessage();
    }
}

function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

async function sendMessage() {
    console.log("🔄 sendMessage called");
    
    const chatInput = document.getElementById('chatInput');
    if (!chatInput) {
        console.error("❌ chatInput element not found");
        return;
    }
    
    const message = chatInput.value.trim();
    if (!message) {
        console.log("⚠️ No message to send");
        return;
    }
    
    if (isLoading) {
        console.log("⚠️ Already loading, ignoring request");
        return;
    }
    
    console.log("💬 Sending message:", message);
    
    // Clear input and show loading
    chatInput.value = '';
    isLoading = true;
    
    // Show user message
    addMessageToChat('user', message);
    
    // Show loading indicator
    const loadingId = addLoadingMessage();
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                history: chatHistory || [],
                filters: currentFilters,
                analytics: true,
                metadata_focus: []
            })
        });
        
        if (!response.ok) {
            throw new Error(`Chat API error: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Remove loading message
        removeLoadingMessage(loadingId);
        
        // Show response - handle both 'reply' and 'response' fields from backend
        const responseText = data.reply || data.response || 'No response received';
        addMessageToChat('assistant', responseText);
        
        // Update session ID if provided
        if (data.sessionId) {
            currentSessionId = data.sessionId;
        }
        
    } catch (error) {
        console.error("❌ Chat error:", error);
        removeLoadingMessage(loadingId);
        addMessageToChat('error', `Sorry, I encountered an error: ${error.message}`);
    } finally {
        isLoading = false;
        console.log("✅ sendMessage completed");
    }
}

function addMessageToChat(type, content) {
    const messagesContainer = document.getElementById('chatMessages');
    const welcomeScreen = document.getElementById('welcomeScreen');
    
    if (!messagesContainer) return;
    
    // Hide welcome screen and show chat
    if (welcomeScreen) {
        welcomeScreen.classList.add('hidden');
    }
    messagesContainer.classList.remove('hidden');
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    const timestamp = new Date().toLocaleTimeString();
    
    // Format content based on type
    let formattedContent = content;
    
    if (type === 'assistant') {
        formattedContent = formatAssistantMessage(content);
    } else if (type === 'user') {
        formattedContent = escapeHtml(content);
    } else if (type === 'error') {
        formattedContent = `<div class="error-content">${escapeHtml(content)}</div>`;
    }
    
    messageDiv.innerHTML = `
        <div class="message-header">
            <span class="message-type">${type === 'user' ? '👤 You' : type === 'assistant' ? '🤖 Assistant' : '❌ Error'}</span>
            <span class="message-time">${timestamp}</span>
        </div>
        <div class="message-content">${formattedContent}</div>
    `;
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function formatAssistantMessage(content) {
    if (!content) return '';
    
    // If content is already HTML (contains tags), return as-is but clean it
    if (content.includes('<') && content.includes('>')) {
        return cleanHtml(content);
    }
    
    // Otherwise, convert plain text to formatted HTML
    let formatted = escapeHtml(content);
    
    // Convert line breaks to proper paragraphs
    formatted = formatted
        .split(/\n\s*\n/) // Split on double line breaks for paragraphs
        .map(paragraph => {
            if (paragraph.trim()) {
                return `<p>${paragraph.replace(/\n/g, '<br>')}</p>`;
            }
            return '';
        })
        .filter(p => p) // Remove empty paragraphs
        .join('');
    
    // If no paragraphs were created, wrap in a single paragraph
    if (!formatted.includes('<p>')) {
        formatted = `<p>${formatted.replace(/\n/g, '<br>')}</p>`;
    }
    
    // Convert markdown-style formatting
    formatted = convertBasicMarkdown(formatted);
    
    return formatted;
}

function convertBasicMarkdown(text) {
    // Convert **bold** and __bold__
    text = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/__(.*?)__/g, '<strong>$1</strong>');
    
    // Convert *italic* and _italic_
    text = text.replace(/\*(.*?)\*/g, '<em>$1</em>');
    text = text.replace(/_(.*?)_/g, '<em>$1</em>');
    
    // Convert `code`
    text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    // Convert numbered lists (basic)
    text = text.replace(/^(\d+\.)\s+(.+)$/gm, '<ol><li>$2</li></ol>');
    
    // Convert bullet points (basic)
    text = text.replace(/^[•\-\*]\s+(.+)$/gm, '<ul><li>$1</li></ul>');
    
    // Merge consecutive list items
    text = text.replace(/<\/ol>\s*<ol>/g, '');
    text = text.replace(/<\/ul>\s*<ul>/g, '');
    
    return text;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function cleanHtml(html) {
    // Basic HTML cleaning - remove script tags and dangerous attributes
    let cleaned = html
        .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
        .replace(/javascript:/gi, '')
        .replace(/on\w+\s*=/gi, '');
    
    return cleaned;
}

function addLoadingMessage() {
    const messagesContainer = document.getElementById('chatMessages');
    if (!messagesContainer) return null;
    
    const loadingId = 'loading-' + Date.now();
    const loadingDiv = document.createElement('div');
    loadingDiv.id = loadingId;
    loadingDiv.className = 'message assistant-message loading-message';
    loadingDiv.innerHTML = `
        <div class="message-header">
            <span class="message-type">🤖 Assistant</span>
            <span class="message-time">Thinking...</span>
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
    if (!loadingId) return;
    const loadingElement = document.getElementById(loadingId);
    if (loadingElement) {
        loadingElement.remove();
    }
}

function clearChat() {
    const messagesContainer = document.getElementById('chatMessages');
    const welcomeScreen = document.getElementById('welcomeScreen');
    
    if (messagesContainer) {
        messagesContainer.innerHTML = '';
        messagesContainer.classList.add('hidden');
    }
    
    if (welcomeScreen) {
        welcomeScreen.classList.remove('hidden');
    }
    
    currentSessionId = null;
    chatHistory = [];
    
    console.log("🧹 Chat cleared");
}

function exportChat() {
    const messages = document.querySelectorAll('.message:not(.loading-message)');
    if (messages.length === 0) {
        alert('No chat history to export.');
        return;
    }
    
    let exportData = 'Metro AI Call Center Analytics - Chat Export\n';
    exportData += '=' .repeat(50) + '\n';
    exportData += `Export Date: ${new Date().toLocaleString()}\n`;
    exportData += `Total Messages: ${messages.length}\n\n`;
    
    messages.forEach((message, index) => {
        const type = message.querySelector('.message-type')?.textContent || '';
        const time = message.querySelector('.message-time')?.textContent || '';
        const content = message.querySelector('.message-content')?.textContent || '';
        
        exportData += `${index + 1}. ${type} [${time}]\n`;
        exportData += content + '\n\n';
    });
    
    // Create and download file
    const blob = new Blob([exportData], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `chat-export-${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// =============================================================================
// ANALYTICS FUNCTIONS
// =============================================================================

async function refreshAnalyticsStats() {
    console.log("📊 Refreshing analytics stats...");
    
    try {
        const response = await fetch('/analytics/stats', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                filters: currentFilters
            })
        });
        
        if (!response.ok) {
            throw new Error(`Stats API error: ${response.status}`);
        }
        
        const stats = await response.json();
        
        // Update total records display
        const totalRecords = document.getElementById('totalRecords');
        if (totalRecords && stats.total !== undefined) {
            totalRecords.textContent = `${stats.total.toLocaleString()} transcripts`;
        }
        
        // Update active filters count if filters are applied
        const activeFiltersCount = document.getElementById('activeFiltersCount');
        const filterCount = Object.keys(currentFilters).length;
        if (activeFiltersCount) {
            if (filterCount > 0 && stats.total !== undefined) {
                activeFiltersCount.textContent = `${filterCount} filter${filterCount !== 1 ? 's' : ''} (${stats.total.toLocaleString()} results)`;
            } else {
                activeFiltersCount.textContent = `${filterCount} filter${filterCount !== 1 ? 's' : ''}`;
            }
        }
        
        console.log("✅ Analytics stats updated:", stats);
        
    } catch (error) {
        console.error("❌ Failed to refresh analytics stats:", error);
        const totalRecords = document.getElementById('totalRecords');
        if (totalRecords) {
            totalRecords.textContent = 'Stats unavailable';
        }
    }
}

// =============================================================================
// INITIALIZATION FUNCTIONS
// =============================================================================

function initializePage() {
    console.log("🔄 Initializing page components...");
    
    // Set up sidebar toggle functionality
    const sidebar = document.getElementById('sidebar');
    if (sidebar) {
        // Ensure sidebar starts closed on mobile
        if (window.innerWidth <= 768) {
            sidebar.classList.remove('open');
        }
    }
    
    // Set up responsive behavior
    window.addEventListener('resize', () => {
        const sidebar = document.getElementById('sidebar');
        if (sidebar && window.innerWidth > 768) {
            sidebar.classList.add('open');
        }
    });
    
    // Initialize send button functionality
    initializeSendButton();
    
    console.log("✅ Page initialization complete");
}

function initializeSendButton() {
    console.log("🔄 Initializing send button...");
    
    const sendBtn = document.getElementById('sendBtn');
    const chatInput = document.getElementById('chatInput');
    
    if (!sendBtn) {
        console.error("❌ Send button not found!");
        return;
    }
    
    if (!chatInput) {
        console.error("❌ Chat input not found!");
        return;
    }
    
    // Ensure send button is enabled
    sendBtn.disabled = false;
    
    // Add event listener as backup to onclick attribute
    sendBtn.addEventListener('click', function(event) {
        event.preventDefault();
        console.log("🔄 Send button clicked via event listener");
        sendMessage();
    });
    
    // Add input event listener for Enter key
    chatInput.addEventListener('keydown', function(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            console.log("🔄 Enter key pressed in chat input");
            sendMessage();
        }
    });
    
    console.log("✅ Send button initialized successfully");
}

function addEvaluationScopeToggle() {
    const chatHeaderFilters = document.getElementById('chatHeaderFilters');
    if (!chatHeaderFilters) {
        console.warn("⚠️ chatHeaderFilters not found, skipping scope toggle");
        return;
    }
    
    // Check if toggle already exists
    if (document.getElementById('evaluationScopeToggle')) {
        console.log("ℹ️ Evaluation scope toggle already exists");
        return;
    }
    
    const scopeToggle = document.createElement('div');
    scopeToggle.id = 'evaluationScopeToggle';
    scopeToggle.innerHTML = `
        <label class="toggle-switch header-toggle">
            <input type="checkbox" id="includeAllFilteredToggle" checked onchange="updateEvaluationScope()">
            <span class="toggle-slider"></span>
            <span class="toggle-label">All Filtered Evaluations</span>
        </label>
    `;
    
    chatHeaderFilters.appendChild(scopeToggle);
    console.log("✅ Evaluation scope toggle added");
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

function getAppliedFilters() {
    return { ...currentFilters };
}

function toggleDetailedTable() {
    // Placeholder for detailed table functionality
    console.log("🔄 Toggling detailed table view");
}

function downloadCategoryData() {
    // Placeholder for category data download
    console.log("⬇️ Downloading category data");
}

function testSendButton() {
    console.log("🔧 Testing send button functionality");
    
    // Check if elements exist
    const sendBtn = document.getElementById('sendBtn');
    const chatInput = document.getElementById('chatInput');
    
    console.log("Send button found:", !!sendBtn);
    console.log("Chat input found:", !!chatInput);
    
    // Check if functions are available globally
    console.log("sendMessage function:", typeof window.sendMessage);
    console.log("handleKeyPress function:", typeof window.handleKeyPress);
    
    // Check onclick attribute
    if (sendBtn) {
        console.log("Send button onclick:", sendBtn.getAttribute('onclick'));
        console.log("Send button disabled:", sendBtn.disabled);
    }
    
    // Test direct call
    if (chatInput) {
        const originalValue = chatInput.value;
        chatInput.value = "test message";
        console.log("Testing direct sendMessage call...");
        try {
            if (window.sendMessage) {
                window.sendMessage();
            } else {
                console.error("❌ sendMessage not available on window");
            }
        } catch (error) {
            console.error("❌ Error calling sendMessage:", error);
        }
        chatInput.value = originalValue;
    }
}

function debugChatSystem() {
    console.log("🔧 DEBUG: Chat System Status");
    console.log("Current Filters:", currentFilters);
    console.log("Chat History:", chatHistory);
    console.log("Filter Options:", filterOptions);
    console.log("Performance Metrics:", performanceMetrics);
    console.log("Vector Search Status:", vectorSearchStatus);
    
    // Debug last message formatting
    const lastMessage = document.querySelector('.message:last-child .message-content');
    if (lastMessage) {
        console.log("🔧 DEBUG: Last message content:", lastMessage.innerHTML);
        console.log("🔧 DEBUG: Last message text:", lastMessage.textContent);
    }
    
    // Test send button
    testSendButton();
}

// =============================================================================
// VECTOR SEARCH FUNCTIONS (Placeholder)
// =============================================================================

async function checkVectorSearchCapabilities() {
    try {
        const response = await fetch('/vector_status_simple');
        if (response.ok) {
            const status = await response.json();
            vectorSearchStatus = { ...vectorSearchStatus, ...status };
            console.log("📍 Vector search capabilities:", vectorSearchStatus);
        }
    } catch (error) {
        console.warn("⚠️ Could not check vector search capabilities:", error);
    }
}

function loadFormattingStyles() {
    // Add any dynamic formatting styles
    console.log("🎨 Loading formatting styles");
}

// =============================================================================
// TRANSCRIPT SEARCH FUNCTIONS (Placeholder)
// =============================================================================

function initializeEnhancedTranscriptSearch() {
    console.log("🎯 Enhanced transcript search initialization placeholder");
}

function sendMessageWithTranscriptSearch() {
    console.log("🔍 Transcript search message handling");
    return sendMessage();
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
window.updateFilterCounts = updateFilterCounts;
window.updateHierarchyFilters = updateHierarchyFilters;
window.updateSubDispositions = updateSubDispositions;
window.updateEvaluationScope = updateEvaluationScope;

// Analytics function
window.refreshAnalyticsStats = refreshAnalyticsStats;

// Chat functions
window.askQuestion = askQuestion;
window.handleKeyPress = handleKeyPress;
window.sendMessage = sendMessage;
window.clearChat = clearChat;
window.exportChat = exportChat;

// Utility functions
window.toggleDetailedTable = toggleDetailedTable;
window.downloadCategoryData = downloadCategoryData;

// Debug functions
window.debugChatSystem = debugChatSystem;
window.testSendButton = testSendButton;
window.getProductionMetrics = () => performanceMetrics;
window.getProductionConfig = () => PRODUCTION_CONFIG;

// Export configuration
window.EVALUATION_DEFAULTS = EVALUATION_DEFAULTS;
window.getAppliedFilters = getAppliedFilters;

// =============================================================================
// INITIALIZATION WITH ANALYTICS STATS
// =============================================================================

document.addEventListener('DOMContentLoaded', function() {
    console.log("🚀 Metro AI Call Center Analytics v6.0.0 - Production initializing...");
    console.log("🚀 Fixed Version - All JavaScript Errors Resolved");
    
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
                    
                    // Load initial analytics stats on startup
                    refreshAnalyticsStats();
                })
                .catch(error => {
                    console.error("❌ Filter loading failed:", error);
                    // Don't show critical error for filter loading failure - app can still function
                    console.log("⚠️ Continuing with empty filter options");
                });
        }, 1000);
        
    } catch (error) {
        console.error("❌ CRITICAL: Production initialization failed:", error);
        showCriticalError("Critical initialization failure: " + error.message);
    }

    // Initialize enhanced transcript search
    setTimeout(initializeEnhancedTranscriptSearch, 1000);
    setTimeout(() => {
        addEvaluationScopeToggle();
    }, 1500);

    console.log("✅ Default evaluation scope: ALL filtered evaluations");    
    console.log("✅ Metro AI Call Center Analytics v6.0.0 Fixed version loaded successfully");
    console.log("🔧 FIXED: All JavaScript errors resolved, proper function definitions");
    console.log("📱 FIXED: Sidebar toggle functionality improved");
    console.log("🔍 FIXED: Filter population with error handling");
    console.log("🗑️ REMOVED: chat-data-filter functionality (no longer needed)");
    console.log("🔧 Debug mode:", PRODUCTION_CONFIG.DEBUG_MODE ? "ENABLED" : "DISABLED");
});

console.log("🎯 Fixed chat.js loaded successfully - All issues resolved");
