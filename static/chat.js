// Enhanced Metro AI Call Center Analytics Chat - VECTOR SEARCH ENABLED
// Version: 6.0.0 - Working Base
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



// Initialize transcript search functionality
function initializeTranscriptSearch() {
    console.log("🎯 Initializing transcript search functionality...");
    
    // First, try to add toggle to header if it's missing
    addToggleToHeader();
    
    // Add transcript search results container
    addTranscriptResultsContainer();
    
    // Verify everything is working
    setTimeout(() => {
        debugTranscriptToggle();
    }, 200);
}


function updateChatInterfaceForTranscriptMode(isTranscriptMode) {
    console.log(`🔄 FIXED: Updating interface for transcript mode: ${isTranscriptMode}`);
    
    const chatInput = document.getElementById('chatInput') || 
                     document.querySelector('input[type="text"]') ||
                     document.querySelector('textarea');
    
    const sendButton = document.getElementById('sendButton') || document.querySelector('.send-btn');
    const quickQuestions = document.getElementById('quickQuestions') || document.querySelector('.quick-questions');
    const welcomeExamples = document.querySelector('.example-grid') || document.querySelector('.example-cards');
    
    if (isTranscriptMode) {
        // TRANSCRIPT SEARCH MODE
        console.log("🎯 Enabling transcript search mode");
        
        // Update input placeholder and styling
        if (chatInput) {
            chatInput.placeholder = "🔍 Enter words or phrases to find in call transcripts (e.g., 'billing issue', 'cancel account', 'refund')";
            chatInput.style.borderColor = "#6e32a0";
            chatInput.style.background = "#f8f4ff";
            chatInput.classList.add('transcript-mode');
        }
        
        // Update send button
        if (sendButton) {
            sendButton.innerHTML = '🔍 Search Transcripts';
            sendButton.style.background = "linear-gradient(135deg, #6e32a0 0%, #8b4cb8 100%)";
        }
        
        // Hide regular chat elements
        if (quickQuestions) {
            quickQuestions.style.display = 'none';
        }
        
        if (welcomeExamples) {
            welcomeExamples.style.display = 'none';
        }
        
        // Show transcript-specific guidance
        addTranscriptSearchGuidance();
        
        // Hide any existing chat messages/results that aren't transcript results
        const chatMessages = document.getElementById('chatMessages');
        if (chatMessages && !chatMessages.classList.contains('transcript-results')) {
            chatMessages.style.display = 'none';
        }
        
    } else {
        // REGULAR SEARCH MODE
        console.log("💬 Enabling regular chat mode");
        
        // Restore input
        if (chatInput) {
            chatInput.placeholder = "Ask a question about the evaluation data...";
            chatInput.style.borderColor = "";
            chatInput.style.background = "";
            chatInput.classList.remove('transcript-mode');
        }
        
        // Restore send button
        if (sendButton) {
            sendButton.innerHTML = '↗ Send';
            sendButton.style.background = "";
        }
        
        // Show regular chat elements
        if (quickQuestions) {
            quickQuestions.style.display = 'block';
        }
        
        if (welcomeExamples) {
            welcomeExamples.style.display = 'grid';
        }
        
        // Remove transcript guidance
        removeTranscriptSearchGuidance();
        
        // Show chat messages
        const chatMessages = document.getElementById('chatMessages');
        if (chatMessages) {
            chatMessages.style.display = 'block';
        }
        
        // Clear any transcript results when switching back
        clearTranscriptResults();
    }
}


function addTranscriptSearchGuidance() {
    // Check if guidance already exists
    if (document.getElementById('transcriptSearchGuidance')) {
        return;
    }
    
    const chatInput = document.getElementById('chatInput');
    if (!chatInput) return;
    
    const guidance = document.createElement('div');
    guidance.id = 'transcriptSearchGuidance';
    guidance.innerHTML = `
        <div style="
            background: linear-gradient(135deg, #f3e5f5 0%, #e1bee7 100%);
            border: 1px solid #ce93d8;
            border-radius: 8px;
            padding: 12px;
            margin: 8px 0;
            font-size: 0.85rem;
            color: #4a148c;
        ">
            <div style="font-weight: 600; margin-bottom: 6px;">🎯 Transcript Search Tips:</div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 8px;">
                <div>• Use quotes for exact phrases: <code>"billing issue"</code></div>
                <div>• Multiple words: <code>cancel account refund</code></div>
                <div>• Single keywords: <code>frustrated</code></div>
                <div>• Agent responses: <code>policy procedure</code></div>
            </div>
        </div>
    `;
    
    // Insert after chat input
    chatInput.parentNode.insertBefore(guidance, chatInput.nextSibling);
}

function removeTranscriptSearchGuidance() {
    const guidance = document.getElementById('transcriptSearchGuidance');
    if (guidance) {
        guidance.remove();
    }
}

function handleTranscriptToggleChange() {
    const toggleSwitch = document.getElementById('transcriptSearchToggle') || 
                        document.querySelector('input[type="checkbox"][id*="transcript"]');
    
    if (!toggleSwitch) {
        console.warn("⚠️ Transcript toggle not found");
        return;
    }
    
    const isTranscriptMode = toggleSwitch.checked;
    console.log(`🔄 FIXED: Transcript toggle changed: ${isTranscriptMode}`);
    
    // Update the entire interface
    updateChatInterfaceForTranscriptMode(isTranscriptMode);
    
    // Update the send function
    updateSendFunction(isTranscriptMode);
    
    // Clear any existing results when switching modes
    if (isTranscriptMode) {
        // Clear regular chat results
        const chatMessages = document.getElementById('chatMessages');
        if (chatMessages) {
            chatMessages.innerHTML = '';
        }
    } else {
        // Clear transcript results
        clearTranscriptResults();
    }
    
    // Focus the input
    const chatInput = document.getElementById('chatInput');
    if (chatInput) {
        chatInput.focus();
    }
}console.log("✅ Complete transcript search UI functions added to chat.js");

function updateSendFunction(isTranscriptMode) {
    const sendButton = document.getElementById('sendButton') || document.querySelector('.send-btn');
    const chatInput = document.getElementById('chatInput');
    
    if (!sendButton || !chatInput) {
        console.warn("⚠️ Send button or chat input not found");
        return;
    }
    
    // Remove existing event listeners by cloning the elements
    const newSendButton = sendButton.cloneNode(true);
    sendButton.parentNode.replaceChild(newSendButton, sendButton);
    
    const newChatInput = chatInput.cloneNode(true);
    chatInput.parentNode.replaceChild(newChatInput, chatInput);
    
    if (isTranscriptMode) {
        // Add transcript search functionality
        newSendButton.addEventListener('click', sendMessageWithTranscriptSearch);
        newChatInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessageWithTranscriptSearch();
            }
        });
        console.log("✅ FIXED: Transcript search functions attached");
    } else {
        // Add regular chat functionality
        newSendButton.addEventListener('click', sendMessage); // Assuming sendMessage exists
        newChatInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage(); // Assuming sendMessage exists
            }
        });
        console.log("✅ FIXED: Regular chat functions attached");
    }
}

function cleanMetadataForDisplay(data) {
    /**
     * Cleans any data object to only include allowed fields
     * @param {Object} data - Raw data object
     * @returns {Object} - Cleaned data with only allowed fields
     */
    if (!data || typeof data !== 'object') {
        return {};
    }
    
    const cleaned = {};
    
    // Only include allowed fields
    for (const field of ALLOWED_API_FIELDS) {
        if (data[field] !== undefined && data[field] !== null && 
            data[field] !== '' && data[field] !== 'Unknown') {
            cleaned[field] = data[field];
        }
        
        // Check in metadata subdictionary if it exists
        if (data.metadata && data.metadata[field] !== undefined && 
            data.metadata[field] !== null && data.metadata[field] !== '') {
            cleaned[field] = data.metadata[field];
        }
    }
    
    // Remove any forbidden fields that might have slipped through
    for (const forbidden of FORBIDDEN_INTERNAL_FIELDS) {
        delete cleaned[forbidden];
    }
    
    return cleaned;
}


// =============================================================================
// Initialize the enhanced system
// =============================================================================
function initializeEnhancedTranscriptSearch() {
    console.log("🚀 FIXED: Initializing enhanced transcript search system...");
    
    // STEP 1: Create the toggle (this was missing!)
    addToggleToHeader();
    
    // STEP 2: Add transcript search results container
    addTranscriptResultsContainer();
    
    // STEP 3: Wait a moment for DOM to update, then find and setup the toggle
    setTimeout(() => {
        // Look for the actual toggle ID that gets created by addToggleToHeader()
        const toggle = document.getElementById('transcriptSearchToggleInput') || 
                      document.getElementById('transcriptSearchToggle') || 
                      document.querySelector('input[type="checkbox"][id*="transcript"]');
        
        if (toggle) {
            console.log("✅ Found transcript toggle:", toggle.id);
            
            // Remove any existing event listeners and add our enhanced handler
            toggle.removeEventListener('change', toggleTranscriptSearchMode); // Remove original
            toggle.addEventListener('change', handleEnhancedTranscriptToggleChange);
            
            // Set initial state based on toggle
            const isTranscriptMode = toggle.checked;
            updateChatInterfaceForTranscriptMode(isTranscriptMode);
            updateSendFunction(isTranscriptMode);
            
            console.log("✅ FIXED: Enhanced transcript search initialized with toggle");
        } else {
            console.warn("⚠️ Transcript toggle still not found after creation attempt");
            // Try to debug what's in the header
            const header = document.getElementById('chatHeaderFilters');
            if (header) {
                console.log("Header contents:", header.innerHTML);
            }
        }
    }, 200);
    
    // STEP 4: Verify everything is working (like the original did)
    setTimeout(() => {
        debugTranscriptToggle();
    }, 400);
}

function handleEnhancedTranscriptToggleChange() {
    // Get the toggle (use the actual ID from addToggleToHeader)
    const toggle = document.getElementById('transcriptSearchToggleInput') || 
                  document.getElementById('transcriptSearchToggle') || 
                  document.querySelector('input[type="checkbox"][id*="transcript"]');
    
    if (!toggle) {
        console.warn("⚠️ Transcript toggle not found in change handler");
        return;
    }
    
    const isTranscriptMode = toggle.checked;
    console.log(`🔄 ENHANCED: Transcript toggle changed: ${isTranscriptMode}`);
    
    // Update the interface (same as before)
    updateChatInterfaceForTranscriptMode(isTranscriptMode);
    updateSendFunction(isTranscriptMode);
    
    // Handle the comprehensive option visibility (from original logic)
    const comprehensiveOption = document.getElementById('comprehensiveOption');
    if (comprehensiveOption) {
        comprehensiveOption.style.display = isTranscriptMode ? 'inline-flex' : 'none';
    }
    
    // Clear results when switching modes
    if (isTranscriptMode) {
        // Clear regular chat results
        const chatMessages = document.getElementById('chatMessages');
        if (chatMessages) {
            chatMessages.innerHTML = '';
        }
    } else {
        // Clear transcript results
        clearTranscriptResults();
    }
    
    // Show feedback toast (from original logic)
    if (typeof showToast === 'function') {
        showToast(
            isTranscriptMode ? 
            '🎯 Transcript Search Mode Enabled - Searching call transcripts only' : 
            '💬 Normal Chat Mode Enabled - Full system search available',
            isTranscriptMode ? 'info' : 'success'
        );
    }
    
    // Focus the input
    const chatInput = document.getElementById('chatInput');
    if (chatInput) {
        chatInput.focus();
    }
}

async function sendMessageWithTranscriptSearch() {
    console.log("🎯 FIXED: Starting transcript search with increased limits...");
    
    const chatInput = document.getElementById('chatInput');
    const sendButton = document.getElementById('sendButton') || document.querySelector('.send-btn');
    
    if (!chatInput) {
        console.error("❌ Chat input not found");
        return;
    }
    
    const message = chatInput.value?.trim();
    if (!message) {
        console.log("⚠️ Empty search query");
        return;
    }
    
    // Store the search query
    window.lastSearchQuery = message;
    
    // Get current filters
    const currentFilters = (typeof getAppliedFilters === 'function') ? getAppliedFilters() : {};
    const useComprehensive = document.getElementById('comprehensiveToggle')?.checked || false;
    
    try {
        // Disable inputs during search
        chatInput.disabled = true;
        if (sendButton) sendButton.disabled = true;
        
        // Show loading state
        if (useComprehensive) {
            showTranscriptSearchLoading(message, true);
        } else {
            showTranscriptSearchLoading(message, false);
        }
        
        // FIXED: Use relative URLs like the regular sendMessage function
        const endpoint = useComprehensive ? 
            '/search_transcripts_comprehensive' : 
            '/search_transcripts';
        
        const requestBody = {
            query: message,
            filters: currentFilters || {},
            display_size: useComprehensive ? 1000 : 1000,  
            size: useComprehensive ? 1000 : 1000,          
            max_scan: useComprehensive ? 25000 : 10000, 
            highlight: true
        };
        
        console.log("🔍 FIXED: Sending transcript search with increased limits:", {
            endpoint,
            query: message,
            comprehensive: useComprehensive,
            display_size: requestBody.display_size,
            max_scan: requestBody.max_scan,
            filtersCount: Object.keys(currentFilters || {}).length
        });
        
        // Execute search with timeout
        const abortController = new AbortController();
        const timeoutId = setTimeout(() => {
            abortController.abort();
            console.warn("⏰ Request timeout after 45 seconds");
        }, 45000); // Increased timeout for larger searches
        
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify(requestBody),
            signal: abortController.signal
        });
        
        clearTimeout(timeoutId);
        
        if (!response.ok) {
            let errorText;
            try {
                errorText = await response.text();
                console.error("API Error Response:", errorText);
            } catch (e) {
                errorText = `HTTP ${response.status} ${response.statusText}`;
            }
            throw new Error(`Search failed: ${errorText}`);
        }
        
        const data = await response.json();
        console.log("✅ FIXED: Transcript search response:", data);
        
        // Display results with the appropriate function
        if (useComprehensive) {
            displayComprehensiveTranscriptResults(data, message);
        } else {
            displayTranscriptSearchResults(data, message);
        }
        
        // Clear the input
        chatInput.value = '';
        chatInput.style.height = 'auto';
        
    } catch (error) {
        console.error("❌ Transcript search error:", error);
        showTranscriptSearchError(error.message || 'Search failed');
    } finally {
        // Re-enable inputs
        chatInput.disabled = false;
        if (sendButton) sendButton.disabled = false;
        chatInput.focus();
    }
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
    populateSelect('callDispositionFilter', options.callDispositions);
    populateSelect('callSubDispositionFilter', options.callSubDispositions);
    
    // Other filters
    populateSelect('languageFilter', options.languages);    
    
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

// =============================================================================
// 🔧 FIXED: SINGLE updateFilterCounts FUNCTION - NO DUPLICATES
// Removed all conflicting versions, this is the only one
// =============================================================================

function toggleTranscriptSearchMode() {
    console.log("🎯 Toggling transcript search mode...");
    
    // Use the correct IDs from the HTML
    const toggle = document.getElementById('transcriptSearchToggleInput');
    const toggleContainer = document.getElementById('transcriptToggle');  
    const comprehensiveOption = document.getElementById('comprehensiveOption');
    const chatInput = document.getElementById('chatInput');
    
    if (!toggle) {
        console.error("❌ Transcript search toggle not found! Check HTML file.");
        return;
    }
    
    // Update global state
    transcriptSearchMode = toggle.checked;
    
    console.log(`🎯 Transcript search mode: ${transcriptSearchMode ? 'ENABLED' : 'DISABLED'}`);
    
    // Update visual states
    if (transcriptSearchMode) {
        // Activate header toggle styling
        if (toggleContainer) {
            toggleContainer.classList.add('active');
        }
        
        // Show comprehensive search option in header
        if (comprehensiveOption) {
            comprehensiveOption.style.display = 'inline-flex';
        }
        
        // Update chat input
        if (chatInput) {
            chatInput.classList.add('transcript-mode');
            chatInput.placeholder = "Enter words or phrases to find in call transcripts...";
        }
        
        console.log("✅ Transcript search mode activated");
    } else {
        // Deactivate header toggle styling
        if (toggleContainer) {
            toggleContainer.classList.remove('active');
        }
        
        // Hide comprehensive search option
        if (comprehensiveOption) {
            comprehensiveOption.style.display = 'none';
        }
        
        // Reset chat input
        if (chatInput) {
            chatInput.classList.remove('transcript-mode');
            chatInput.placeholder = "Ask specific questions about calls, dispositions, site performance, quality metrics, or any evaluation data from your live database...";
        }
        
        // Clear any existing transcript results
        clearTranscriptResults();
        
        console.log("✅ Normal chat mode activated");
    }
    
    // Show feedback toast (if showToast function exists)
    if (typeof showToast === 'function') {
        showToast(
            transcriptSearchMode ? 
            '🎯 Transcript Search Mode Enabled - Searching call transcripts only' : 
            '💬 Normal Chat Mode Enabled - Full system search available',
            transcriptSearchMode ? 'info' : 'success'
        );
    }
}

function updateFilterCounts(data) {
    console.log("📊 Updating filter counts with data:", data);
    
    // PART 1: Update overall stats display in sidebar
    updateSidebarStats(data);
    
    // PART 2: Update individual filter count indicators 
    updateIndividualFilterCounts(data);
    
    // PART 3: 🔧 FIX - Update totalRecords in chat header (WAS MISSING!)
    updateChatHeaderStats(data);
    
    // PART 4: Update data status indicators
    updateDataStatusIndicators(data);
    
    // PART 5: Remove loading states and enable filters
    removeLoadingStates();
    
    console.log("✅ Filter counts update completed with chat-stats integration");
}

// =============================================================================
// 🔧 FIXED: HELPER FUNCTIONS - NO LONGER NESTED OR DUPLICATED
// Moved all helper functions outside to prevent conflicts
// =============================================================================

function updateSidebarStats(data) {
    const countsElement = document.getElementById('filterCounts');
    if (countsElement) {
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
}

function updateIndividualFilterCounts(data) {
    const counts = [
        { id: 'templateCount', data: data.templates, label: 'templates' },
        { id: 'programCount', data: data.programs, label: 'programs' },
        { id: 'partnerCount', data: data.partners, label: 'partners' },
        { id: 'siteCount', data: data.sites, label: 'sites' },
        { id: 'lobCount', data: data.lobs, label: 'LOBs' },
        { id: 'dispositionCount', data: data.callDispositions, label: 'dispositions' },
        { id: 'subDispositionCount', data: data.callSubDispositions, label: 'sub-dispositions' },
        { id: 'languageCount', data: data.languages, label: 'languages' }
    ];

    counts.forEach(({ id, data: itemData, label }) => {
        const element = document.getElementById(id);
        if (element) {
            const count = Array.isArray(itemData) ? itemData.length : 0;
            if (count > 0) {
                element.textContent = `(${count})`;
                element.className = 'count-indicator data-status-ok';
                element.title = `${count} ${label} found in database`;
            } else {
                element.textContent = '(0)';
                element.className = 'count-indicator data-status-warning';
                element.title = `No ${label} found in database`;
            }
        }
    });
}

// 🔧 NEW FUNCTION - This was completely missing!
function updateChatHeaderStats(data) {
    // Update totalRecords element in chat header
    const totalRecordsElement = document.getElementById('totalRecords');
    if (totalRecordsElement) {
        const totalEvaluations = data.total_evaluations || 0;
        if (totalEvaluations > 0) {
            totalRecordsElement.textContent = `${totalEvaluations.toLocaleString()} evaluations`;
            totalRecordsElement.title = `${totalEvaluations} evaluations available for analysis`;
        } else {
            totalRecordsElement.textContent = 'No data available';
            totalRecordsElement.title = 'No evaluation data found';
        }
        console.log(`📊 Updated totalRecords: ${totalRecordsElement.textContent}`);
    } else {
        console.warn("⚠️ totalRecords element not found in chat header");
    }
    
    // Update active filters count in header
    const activeFiltersCountElement = document.getElementById('activeFiltersCount');
    if (activeFiltersCountElement) {
        const filterCount = Object.keys(currentFilters).length;
        activeFiltersCountElement.textContent = `${filterCount} filters`;
    }
}

function updateDataStatusIndicators(data) {
    const statusElements = [
        { id: 'hierarchyDataStatus', categories: ['templates', 'programs', 'partners', 'sites', 'lobs'] },
        { id: 'callDataStatus', categories: ['callDispositions', 'callSubDispositions'] },
        { id: 'languageDataStatus', categories: ['languages'] }
    ];

    statusElements.forEach(({ id, categories }) => {
        const element = document.getElementById(id);
        if (!element) return;

        const totalCategories = categories.length;
        const populatedCategories = categories.filter(cat =>
            data[cat] && data[cat].length > 0
        ).length;

        if (populatedCategories === totalCategories) {
            element.textContent = '✅ All data loaded';
            element.className = 'data-status data-status-ok';
        } else if (populatedCategories > 0) {
            element.textContent = `⚠️ ${populatedCategories}/${totalCategories} loaded`;
            element.className = 'data-status data-status-warning';
        } else {
            element.textContent = '❌ No data found';
            element.className = 'data-status data-status-error';
        }
    });
}

function removeLoadingStates() {
    const selects = document.querySelectorAll('.filter-select, .filter-input');
    selects.forEach(select => {
        select.classList.remove('loading-filter');
        select.disabled = false;
    });
}

function handleFilterLoadError(errorMessage) {
    console.error("🚨 Filter load error:", errorMessage);
    
    // Set empty filter options to prevent crashes
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
    
    // Update UI with error state
    const countsElement = document.getElementById('filterCounts');
    if (countsElement) {
        countsElement.innerHTML = `
            <div style="font-size: 0.85em; color: #c33; padding: 8px 12px; background: #fee; border-radius: 6px; margin: 10px 0; border: 1px solid #fcc;">
                ❌ <strong>Filter Loading Failed</strong><br>
                <span style="font-size: 0.8em;">${getErrorMessage(errorMessage)}</span>
            </div>
        `;
    }
    
    // Show error in totalRecords
    const totalRecordsElement = document.getElementById('totalRecords');
    if (totalRecordsElement) {
        totalRecordsElement.textContent = 'Filter error';
        totalRecordsElement.title = `Filter loading failed: ${errorMessage}`;
    }
    
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
// 🔧 FIXED: ANALYTICS STATS INTEGRATION
// This function connects to /analytics/stats endpoint
// =============================================================================

async function refreshAnalyticsStats() {
    console.log("📊 Refreshing analytics stats with filters:", currentFilters);
    
    try {
        const response = await fetch('/analytics/stats', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filters: currentFilters
            })
        });
        
        if (!response.ok) {
            throw new Error(`Analytics API returned ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        console.log("📊 Analytics stats response:", data);
        
        if (data.status === 'success') {
            // Update totalRecords in chat header
            const totalRecordsElement = document.getElementById('totalRecords');
            if (totalRecordsElement) {
                const totalRecords = data.totalRecords || 0;
                totalRecordsElement.textContent = `${totalRecords.toLocaleString()} evaluations`;
                totalRecordsElement.title = `${totalRecords} evaluations match current filters`;
            }
            
            // Update active filters count
            const activeFiltersCountElement = document.getElementById('activeFiltersCount');
            if (activeFiltersCountElement) {
                const filterCount = Object.keys(currentFilters).length;
                activeFiltersCountElement.textContent = `${filterCount} filters`;
            }
            
            // Update performance metrics
            performanceMetrics.lastStatsUpdate = new Date().toISOString();
            
        } else {
            console.error("Analytics stats error:", data.error);
            
            // Show error state in UI
            const totalRecordsElement = document.getElementById('totalRecords');
            if (totalRecordsElement) {
                totalRecordsElement.textContent = 'Stats unavailable';
                totalRecordsElement.title = 'Unable to load evaluation statistics';
            }
        }
        
    } catch (error) {
        console.error("❌ Failed to refresh analytics stats:", error);
        performanceMetrics.errorCount++;
        
        // Show error state in UI
        const totalRecordsElement = document.getElementById('totalRecords');
        if (totalRecordsElement) {
            totalRecordsElement.textContent = 'Stats error';
            totalRecordsElement.title = 'Error loading evaluation statistics';
        }
    }
}

// =============================================================================
// 🔧 FIXED: FILTER ACTIONS WITH ANALYTICS INTEGRATION
// Added refreshAnalyticsStats() calls to connect filter changes to stats
// =============================================================================

function applyFilters() {
    console.log("🔍 Applying filters...");
    
    const filters = {};
    
    // Collect filter values
    const filterMappings = {
        'templateFilter': 'template_name',
        'programFilter': 'program', 
        'partnerFilter': 'partner',
        'siteFilter': 'site',
        'lobFilter': 'lob',
        'callDispositionFilter': 'disposition',
        'callSubDispositionFilter': 'subDisposition',
        'languageFilter': 'language',
        'startCallDate': 'call_date_start',
        'endCallDate': 'call_date_end'
    };
    
    Object.entries(filterMappings).forEach(([elementId, filterKey]) => {
        const element = document.getElementById(elementId);
        if (element && element.value && element.value !== '') {
            filters[filterKey] = element.value;
        }
    });
    
    currentFilters = filters;
    updateActiveFiltersDisplay();
    
    // 🔧 FIX: Refresh analytics stats after filter changes
    refreshAnalyticsStats();
    
    console.log("✅ Filters applied:", filters);
}

function clearFilters() {
    console.log("🗑️ Clearing all filters");
    
    // Clear filter state
    currentFilters = {};
    
    // Clear UI elements
    const filterElements = [
        'templateFilter', 'programFilter', 'partnerFilter', 'siteFilter', 
        'lobFilter', 'callDispositionFilter', 'callSubDispositionFilter',
        'languageFilter', 'startCallDate', 'endCallDate'
    ];
    
    filterElements.forEach(elementId => {
        const element = document.getElementById(elementId);
        if (element) {
            if (element.tagName === 'SELECT') {
                element.selectedIndex = 0;
            } else {
                element.value = '';
            }
        }
    });
    
    // Update displays
    updateActiveFiltersDisplay();
    
    // 🔧 FIX: Refresh stats after clearing filters
    refreshAnalyticsStats();
    
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
        'disposition': 'callDispositionFilter',
        'subDisposition': 'callSubDispositionFilter',
        'language': 'languageFilter',
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
    
    // 🔧 FIX: Refresh stats after removing filter
    refreshAnalyticsStats();
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
        'subDisposition': 'Sub-Disposition',
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
        
        if (data.reply) {

            // Clean the response data to remove internal fields
            const cleanedData = {
                reply: data.reply,
                sources: data.sources ? data.sources.map(cleanMetadataForDisplay) : [],
                sources_summary: data.sources_summary ? {
                    evaluations: data.sources_summary.evaluations || 0,
                    agents: data.sources_summary.agents || 0,
                    partners: data.sources_summary.partners || 0,
                    sites: data.sources_summary.sites || 0,
                    dispositions: data.sources_summary.dispositions || 0
                } : null
            };

            if (cleanedData.sources) {
                cleanedData.sources = cleanedData.sources.map(source => {
                    // Remove internal fields
                    delete source._score;
                    delete source.score;
                    delete source.search_type;
                    delete source.template_name;
                    delete source.template_id;
                    delete source.program;
                    delete source.vector_enhanced;
                    return source;
                });
            }
            addMessageToChat('assistant', cleanedData.reply, cleanedData);
            chatHistory.push(
                { role: 'user', content: message },
                { role: 'assistant', content: cleanedData.reply }
            );

            if (cleanedData.sources && cleanedData.sources.length > 0) {
                displayEvaluationSources(cleanedData.sources);
            }
            
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

function displayEvaluationSources(sources) {
    /**
     * Display clean evaluation sources below chat response
     * Only shows allowed API fields
     */
    if (!sources || sources.length === 0) return;
    
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) return;
    
    // Create a clean display of sources
    const sourcesDiv = document.createElement('div');
    sourcesDiv.className = 'evaluation-sources';
    sourcesDiv.style.cssText = `
        margin: 12px 0;
        padding: 12px;
        background: linear-gradient(135deg, #f5f5f5 0%, #e0e0e0 100%);
        border-radius: 8px;
        border-left: 3px solid #1976d2;
        font-size: 0.85rem;
    `;
    
    const displaySources = sources.slice(0, 3); // Show first 3
    
    sourcesDiv.innerHTML = `
        <div style="font-weight: 600; margin-bottom: 8px; color: #333;">
            📊 Related Evaluations
        </div>
        ${displaySources.map(source => {
            // Build display with ONLY allowed fields
            let sourceHTML = `<div style="background: white; padding: 8px; margin: 4px 0; border-radius: 4px;">`;
            
            // Display allowed fields only
            if (source.evaluationId) {
                sourceHTML += `<div><strong>ID:</strong> ${source.evaluationId}</div>`;
            }
            if (source.partner) {
                sourceHTML += `<div><strong>Partner:</strong> ${source.partner}</div>`;
            }
            if (source.disposition) {
                sourceHTML += `<div><strong>Disposition:</strong> ${source.disposition}</div>`;
            }
            if (source.agentName) {
                sourceHTML += `<div><strong>Agent:</strong> ${source.agentName}</div>`;
            }
            if (source.weighted_score !== undefined && source.weighted_score !== null) {
                sourceHTML += `<div><strong>Score:</strong> ${source.weighted_score}</div>`;
            }
            
            sourceHTML += `</div>`;
            return sourceHTML;
        }).join('')}
        ${sources.length > 3 ? `
            <div style="margin-top: 8px; color: #666;">
                ... and ${sources.length - 3} more evaluations
            </div>
        ` : ''}
    `;
    
    chatMessages.appendChild(sourcesDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
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

// ===================================================================
// MISSING DISPLAY FUNCTIONS - Add these to your chat.js file
// These functions handle displaying the transcript search results
// ===================================================================

// Function to display standard transcript search results
// =============================================================================
// FIXED: displayTranscriptSearchResults function
// Replace this function in your chat.js
// =============================================================================

function displayTranscriptSearchResults(data, query) {
    console.log("🎯 Displaying transcript search results in chat container:", data);
    
    // Hide quick questions
    hideQuickQuestions();
    
    // Hide welcome screen
    const welcomeScreen = document.getElementById('welcomeScreen') || 
                         document.querySelector('.welcome-screen');
    if (welcomeScreen) {
        welcomeScreen.style.display = 'none';
    }
    
    // Use the chat messages container instead of separate container
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) {
        console.error("❌ Chat messages container not found!");
        return;
    }
    
    // Show chat messages container and clear existing content
    chatMessages.classList.remove('hidden');
    chatMessages.style.display = 'block';
    chatMessages.innerHTML = '';
    
    // Extract data from response
    const results = data.display_results || data.results || [];
    const summary = data.comprehensive_summary || data.summary || {};
    const totalResults = results.length;
    const totalEvaluationsScanned = summary.total_evaluations_searched || 0;
    
    // Store results globally for export (FIX: Use consistent variable name)
    window.lastTranscriptResults = results;
    
    console.log(`🎨 BALANCED: Will highlight search terms and variations from: "${query}"`);
    console.log(`📊 Stored ${results.length} results for export`);
    
    // Add summary to input container
    addSummaryToInputContainer(query, totalResults, totalEvaluationsScanned);
    
    // Create results in chat messages container
    if (totalResults === 0) {
        chatMessages.innerHTML = `
            <div style="text-align: center; padding: 40px; color: #666;">
                <div style="font-size: 3rem; margin-bottom: 16px; opacity: 0.5;">🔍</div>
                <h3>No matches found for "${query}"</h3>
                <p>Try different search terms or check your filters.</p>
            </div>
        `;
    } else {
        // Create individual result cards in chat container
        const resultsHTML = results.map((result, index) => {
            const evaluationId = result.evaluationId || result.evaluation_id || result.id;
            const metadata = result.metadata || {};
            const transcript = result.transcript || result.highlighted_text || result.text || '';
            const score = result._score || result.score || 0;
            
            // BALANCED HIGHLIGHTING: Search terms and variations
            const highlightedTranscript = highlightSearchTerms(transcript, query);
            const truncatedTranscript = highlightedTranscript.length > 400 ? 
                                      highlightedTranscript.substring(0, 400) + '...' : 
                                      highlightedTranscript;
            
            return `
                <div class="message assistant-message" style="margin-bottom: 16px;">
                    <div class="message-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <span class="role" style="font-weight: 600; color: #1976d2;">Result ${index + 1}</span>
                            <span style="color: #666; font-size: 0.85rem;">ID: ${evaluationId}</span>
                            ${metadata.partner ? `<span style="background: #e3f2fd; color: #1565c0; padding: 2px 6px; border-radius: 12px; font-size: 0.7rem;">${metadata.partner}</span>` : ''}
                            ${score > 0 ? `<span style="background: #4caf50; color: white; padding: 2px 6px; border-radius: 8px; font-size: 0.7rem;">Score: ${(score * 100).toFixed(1)}%</span>` : ''}
                        </div>
                        <div style="display: flex; gap: 6px;">
                            <button onclick="copyTranscriptToClipboard('${transcript.replace(/'/g, "\\'")}')'" style="
                                background: #6e32a0; color: white; border: none; 
                                padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 0.7rem;
                            ">📋 Copy</button>
                        </div>
                    </div>
                    
                    <div class="message-content" style="
                        background: #f8f9fa; 
                        border-left: 3px solid #1976d2; 
                        padding: 12px; 
                        border-radius: 0 8px 8px 0;
                        font-size: 0.9rem; 
                        line-height: 1.5;
                    ">
                        ${truncatedTranscript}
                    </div>
                    
                    ${metadata.call_date || metadata.program || metadata.disposition ? `
                        <div style="margin-top: 8px; padding: 8px; background: #f0f0f0; border-radius: 6px; font-size: 0.8rem; color: #666;">
                            ${metadata.call_date ? `📅 ${metadata.call_date}` : ''}
                            ${metadata.program ? ` • 📋 ${metadata.program}` : ''}
                            ${metadata.disposition ? ` • 📞 ${metadata.disposition}` : ''}
                        </div>
                    ` : ''}
                </div>
            `;
        }).join('');
        
        chatMessages.innerHTML = resultsHTML;
    }
    
    // Scroll to top of results
    chatMessages.scrollTop = 0;
    
    console.log(`✅ Displayed ${totalResults} results in chat container with balanced highlighting`);
}

function hideQuickQuestions() {
    const quickQuestions = document.getElementById('quickQuestions') || 
                          document.querySelector('.quick-questions');
    if (quickQuestions) {
        quickQuestions.style.display = 'none';
        console.log("✅ Quick questions hidden during transcript search");
    }
}

function showQuickQuestions() {
    const quickQuestions = document.getElementById('quickQuestions') || 
                          document.querySelector('.quick-questions');
    if (quickQuestions) {
        quickQuestions.style.display = 'block';
        console.log("✅ Quick questions restored");
    }
}

// =============================================================================
// MISSING FUNCTION: highlightSearchTerms - Add this to your chat.js
// =============================================================================

function highlightSearchTerms(text, query) {
    if (!text || !query) return text;
    
    const searchTerm = query.trim();
    if (!searchTerm) return text;
    
    console.log("🎨 BALANCED: Highlighting search terms and variations:", {
        originalQuery: query,
        searchTerm: searchTerm
    });
    
    // Handle quoted phrases vs individual words
    const isQuoted = searchTerm.startsWith('"') && searchTerm.endsWith('"');
    
    if (isQuoted) {
        // EXACT PHRASE: Only highlight the complete phrase inside quotes
        const phrase = searchTerm.slice(1, -1).trim();
        if (!phrase) return text;
        
        console.log("🎯 Highlighting exact phrase:", phrase);
        const regex = new RegExp(`(${escapeRegex(phrase)})`, 'gi');
        return text.replace(regex, '<mark style="background: #ffeb3b; color: #333; padding: 2px 4px; border-radius: 3px; font-weight: 600;">$1</mark>');
        
    } else {
        // INDIVIDUAL WORDS: Highlight search terms and their variations, but be selective
        let highlightedText = text;
        
        // Split the search term into individual words and clean them
        const searchWords = searchTerm
            .toLowerCase()
            .split(/\s+/)
            .filter(word => word.length > 0)
            .map(word => word.replace(/[^\w]/g, '')) // Remove punctuation
            .filter(word => word.length >= 3); // Only highlight words with 3+ characters
        
        console.log("🎯 Highlighting these search words and variations:", searchWords);
        
        // Highlight each search word and its variations
        searchWords.forEach(word => {
            if (word.length >= 3) {
                // Create flexible regex that catches the word and its variations
                // This will match the word at the beginning of other words (like "bill" in "billing")
                const flexibleRegex = new RegExp(`\\b(${escapeRegex(word)}\\w*)`, 'gi');
                
                highlightedText = highlightedText.replace(flexibleRegex, '<mark style="background: #ffeb3b; color: #333; padding: 2px 4px; border-radius: 3px; font-weight: 600;">$1</mark>');
            }
        });
        
        console.log(`✅ BALANCED: Highlighted ${searchWords.length} search terms and their variations`);
        return highlightedText;
    }
}

// Helper function to escape regex special characters
function escapeRegex(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// Helper function to identify common words that shouldn't be highlighted
function isCommonWord(word) {
    const commonWords = [
        'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy', 'did', 'does', 'each', 'she', 'they', 'them', 'been', 'have', 'that', 'this', 'will', 'with', 'were', 'said', 'what', 'when', 'where', 'why', 'how', 'would', 'could', 'should'
    ];
    return commonWords.includes(word.toLowerCase());
}
// Function to display comprehensive transcript search results with enhanced analytics
function displayComprehensiveTranscriptResults(data, query) {
    console.log("🎯 Displaying comprehensive results in chat container:", data);
    
    // Hide quick questions and welcome screen
    hideQuickQuestions();
    const welcomeScreen = document.getElementById('welcomeScreen') || 
                         document.querySelector('.welcome-screen');
    if (welcomeScreen) {
        welcomeScreen.style.display = 'none';
    }
    
    // Use chat messages container
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) {
        console.error("❌ Chat messages container not found!");
        return;
    }
    
    chatMessages.classList.remove('hidden');
    chatMessages.style.display = 'block';
    chatMessages.innerHTML = '';
    
    // Extract comprehensive data
    const results = data.display_results || data.results || [];
    const summary = data.comprehensive_summary || {};
    const totalTranscriptsFound = results.length;
    const totalEvaluationsScanned = summary.total_evaluations_searched || 0;
    const matchPercentage = summary.match_percentage || 0;
    
    // Store for export
    window.lastTranscriptResults = results;
    
    // Add enhanced summary for comprehensive search
    const inputContainer = document.querySelector('.chat-input-area') || 
                          document.querySelector('.input-container') ||
                          document.getElementById('chatInput')?.parentElement;
    
    if (inputContainer) {
        const existingSummary = document.getElementById('transcriptSearchSummary');
        if (existingSummary) {
            existingSummary.remove();
        }
        
        const summaryDiv = document.createElement('div');
        summaryDiv.id = 'transcriptSearchSummary';
        summaryDiv.innerHTML = `
            <div style="
                background: linear-gradient(135deg, #e8f5e8 0%, #c8e6c9 100%);
                border: 1px solid #2e7d32;
                border-radius: 8px;
                padding: 8px 12px;
                margin-bottom: 8px;
                font-size: 0.85rem;
            ">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                    <strong style="color: #2e7d32;">📊 Comprehensive: "${query}" - ${totalTranscriptsFound} matches</strong>
                    <div style="display: flex; gap: 8px;">
                        <button onclick="downloadTranscriptResults('csv', '${query}')" style="
                            background: #2e7d32; color: white; border: none; 
                            padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 0.75rem;
                        ">📊 Export CSV</button>
                        <button onclick="clearTranscriptSearchResults()" style="
                            background: #f44336; color: white; border: none; 
                            padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 0.75rem;
                        ">✕ Clear</button>
                    </div>
                </div>
                <div style="color: #666; font-size: 0.8rem;">
                    ${totalEvaluationsScanned} transcripts scanned • ${matchPercentage.toFixed(1)}% hit rate
                </div>
            </div>
        `;
        
        inputContainer.parentNode.insertBefore(summaryDiv, inputContainer);
    }
    
    // Display results (same as regular search)
    if (totalTranscriptsFound === 0) {
        chatMessages.innerHTML = `
            <div style="text-align: center; padding: 40px; color: #666;">
                <div style="font-size: 3rem; margin-bottom: 16px; opacity: 0.5;">🔍</div>
                <h3>No matches found in comprehensive search</h3>
                <p>Searched ${totalEvaluationsScanned} transcripts for "${query}"</p>
            </div>
        `;
    } else {
        // Use same result format as regular search
        const resultsHTML = results.map((result, index) => {
            const evaluationId = result.evaluationId || result.evaluation_id || result.id;
            const metadata = result.metadata || {};
            const transcript = result.transcript || result.highlighted_text || result.text || '';
            const score = result._score || result.score || 0;
            
            const highlightedTranscript = highlightSearchTerms(transcript, query);
            const truncatedTranscript = highlightedTranscript.length > 400 ? 
                                      highlightedTranscript.substring(0, 400) + '...' : 
                                      highlightedTranscript;
            
            return `
                <div class="message assistant-message" style="margin-bottom: 16px;">
                    <div class="message-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <span class="role" style="font-weight: 600; color: #2e7d32;">Match ${index + 1}</span>
                            <span style="color: #666; font-size: 0.85rem;">ID: ${evaluationId}</span>
                            ${metadata.partner ? `<span style="background: #e8f5e8; color: #2e7d32; padding: 2px 6px; border-radius: 12px; font-size: 0.7rem;">${metadata.partner}</span>` : ''}
                            ${score > 0 ? `<span style="background: #4caf50; color: white; padding: 2px 6px; border-radius: 8px; font-size: 0.7rem;">Score: ${(score * 100).toFixed(1)}%</span>` : ''}
                        </div>
                        <button onclick="copyTranscriptToClipboard('${transcript.replace(/'/g, "\\'")}')'" style="
                            background: #6e32a0; color: white; border: none; 
                            padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 0.7rem;
                        ">📋 Copy</button>
                    </div>
                    
                    <div class="message-content" style="
                        background: #f8f9fa; 
                        border-left: 3px solid #2e7d32; 
                        padding: 12px; 
                        border-radius: 0 8px 8px 0;
                        font-size: 0.9rem; 
                        line-height: 1.5;
                    ">
                        ${truncatedTranscript}
                    </div>
                    
                    ${metadata.call_date || metadata.program || metadata.disposition ? `
                        <div style="margin-top: 8px; padding: 8px; background: #f0f0f0; border-radius: 6px; font-size: 0.8rem; color: #666;">
                            ${metadata.call_date ? `📅 ${metadata.call_date}` : ''}
                            ${metadata.program ? ` • 📋 ${metadata.program}` : ''}
                            ${metadata.disposition ? ` • 📞 ${metadata.disposition}` : ''}
                        </div>
                    ` : ''}
                </div>
            `;
        }).join('');
        
        chatMessages.innerHTML = resultsHTML;
    }
    
    chatMessages.scrollTop = 0;
    console.log(`✅ Displayed ${totalTranscriptsFound} comprehensive results in chat container`);
}

console.log("✅ IMPROVED TRANSCRIPT DISPLAY LOADED:");
console.log("   📊 Fixed CSV export - removed JSON option");
console.log("   📍 Summary moved to input container");
console.log("   💬 Results display in chat messages container");
console.log("   🔧 Consistent variable naming for export");
// Helper function to create focused highlights for comprehensive search
function createFocusedHighlights(transcript, query) {
    if (!transcript || !query) return '<div style="color: #666; font-style: italic;">No content to highlight</div>';
    
    // Clean up the query
    const cleanQuery = query.trim().toLowerCase();
    const isQuotedPhrase = cleanQuery.startsWith('"') && cleanQuery.endsWith('"');
    
    let searchTerms = [];
    if (isQuotedPhrase) {
        // For quoted phrases, search for the exact phrase
        searchTerms = [cleanQuery.slice(1, -1)]; // Remove quotes
    } else {
        // For regular queries, search for individual words
        searchTerms = cleanQuery.split(/\s+/).filter(term => term.length > 2);
    }
    
    if (searchTerms.length === 0) return '<div style="color: #666; font-style: italic;">No valid search terms</div>';
    
    const snippets = [];
    const contextSize = 100; // Characters before and after match
    
    searchTerms.forEach(term => {
        const regex = new RegExp(`(.{0,${contextSize}})\\b(${escapeRegex(term)})\\b(.{0,${contextSize}})`, 'gi');
        let match;
        
        while ((match = regex.exec(transcript)) !== null && snippets.length < 5) {
            const before = match[1].length === contextSize ? '...' + match[1] : match[1];
            const matchedWord = match[2];
            const after = match[3].length === contextSize ? match[3] + '...' : match[3];
            
            const snippet = `${before}<mark style="background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); border: 1px solid #f1c40f; border-radius: 3px; padding: 2px 4px; font-weight: 600; color: #8b4513;">${matchedWord}</mark>${after}`;
            
            // Avoid duplicate snippets
            if (!snippets.some(existing => existing.includes(matchedWord))) {
                snippets.push(snippet);
            }
        }
    });
    
    if (snippets.length === 0) {
        return '<div style="color: #666; font-style: italic;">No highlighted matches found in transcript</div>';
    }
    
    return snippets
        .map(snippet => `<div style="
            background: white;
            border: 1px solid #e1e8ed;
            border-radius: 6px;
            padding: 12px;
            margin-bottom: 8px;
            line-height: 1.6;
            font-size: 0.9rem;
            color: #333;
        ">${snippet}</div>`)
        .join('');
}

// Helper function to escape regex special characters
function escapeRegex(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

console.log("✅ Final transcript search functions loaded - ready to replace in chat.js");


function generateResultsHTML(results, query) {
    if (results.length === 0) {
        return `
            <div style="
                text-align: center;
                padding: 40px 20px;
                color: #666;
            ">
                <div style="font-size: 3rem; margin-bottom: 16px; opacity: 0.5;">🔍</div>
                <h3 style="margin-bottom: 8px; color: #333;">No matches found</h3>
                <p>No transcript content matched your search for "<strong>${query}</strong>"</p>
                <div style="
                    background: #fff3cd;
                    border: 1px solid #ffeaa7;
                    border-radius: 6px;
                    padding: 16px;
                    margin-top: 16px;
                    text-align: left;
                    max-width: 400px;
                    margin-left: auto;
                    margin-right: auto;
                ">
                    <h5 style="color: #8b4513; margin-bottom: 8px;">Try searching for:</h5>
                    <ul style="color: #8b4513; margin: 0; padding-left: 20px;">
                        <li>Different keywords or phrases</li>
                        <li>Broader search terms</li>
                        <li>Check spelling and try synonyms</li>
                        <li>Remove quotes for broader matching</li>
                    </ul>
                </div>
            </div>
        `;
    }
    
    return results.map((result, index) => {
        const evaluationId = result.evaluationId || result.evaluation_id || 'Unknown';
        const score = result._score || result.score || 0;
        const matchCount = result.match_count || 1;
        
        // Enhanced metadata display
        const partner = result.metadata?.partner || result.partner || 'N/A';
        const program = result.metadata?.program || result.program || 'N/A';
        const disposition = result.disposition || result.metadata?.disposition || 'N/A';
        const callDate = result.metadata?.call_date || result.call_date || 'N/A';
        
        // Get highlighted content
        let highlightedContent = '';
        if (result.highlighted_snippets && result.highlighted_snippets.length > 0) {
            highlightedContent = result.highlighted_snippets
                .map(snippet => `<div style="
                    background: white;
                    border: 1px solid #e1e8ed;
                    border-radius: 6px;
                    padding: 12px;
                    margin-bottom: 8px;
                    line-height: 1.6;
                    position: relative;
                ">${snippet}</div>`)
                .join('');
        } else if (result.transcript) {
            highlightedContent = createFocusedHighlights(result.transcript, query);
        } else {
            highlightedContent = '<div style="color: #666; font-style: italic; padding: 20px; text-align: center;">No transcript content available</div>';
        }
        
        return `
            <div style="
                border: 1px solid #e1e8ed;
                border-radius: 12px;
                margin-bottom: 16px;
                background: white;
                overflow: hidden;
                transition: all 0.3s ease;
            " onmouseover="this.style.boxShadow='0 4px 12px rgba(0,0,0,0.1)'; this.style.transform='translateY(-2px)'" 
               onmouseout="this.style.boxShadow=''; this.style.transform=''">
                
                <!-- Header -->
                <div style="
                    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
                    padding: 16px 20px;
                    border-bottom: 1px solid #e1e8ed;
                ">
                    <div style="display: flex; align-items: center; gap: 12px; flex-wrap: wrap;">
                        <span style="font-size: 1.2rem; color: #2196f3;">📞</span>
                        <strong style="color: #333;">Call ${evaluationId}</strong>
                        ${score > 0 ? `<span style="
                            background: linear-gradient(135deg, #4caf50 0%, #45a049 100%);
                            color: white;
                            padding: 3px 8px;
                            border-radius: 12px;
                            font-size: 0.75rem;
                            font-weight: 500;
                        ">Score: ${(score * 100).toFixed(1)}%</span>` : ''}
                        <span style="
                            background: linear-gradient(135deg, #ff9800 0%, #f57c00 100%);
                            color: white;
                            padding: 3px 8px;
                            border-radius: 12px;
                            font-size: 0.75rem;
                            font-weight: 500;
                        ">${matchCount} match${matchCount !== 1 ? 'es' : ''}</span>
                    </div>
                </div>
                
                <!-- Metadata -->
                <div style="padding: 16px 20px; border-bottom: 1px solid #f0f0f0;">
                    <div style="
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                        gap: 12px;
                    ">
                        <div>
                            <div style="font-size: 0.75rem; color: #666; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;">Partner</div>
                            <div style="font-size: 0.85rem; color: #333; font-weight: 500;">${partner}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.75rem; color: #666; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;">Program</div>
                            <div style="font-size: 0.85rem; color: #333; font-weight: 500;">${program}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.75rem; color: #666; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;">Disposition</div>
                            <div style="font-size: 0.85rem; color: #333; font-weight: 500;">${disposition}</div>
                        </div>
                        <div>
                            <div style="font-size: 0.75rem; color: #666; font-weight: 500; text-transform: uppercase; letter-spacing: 0.5px;">Date</div>
                            <div style="font-size: 0.85rem; color: #333; font-weight: 500;">${callDate}</div>
                        </div>
                    </div>
                </div>
                
                <!-- Highlighted Content -->
                <div style="
                    background: #f8f9fa;
                    border-left: 4px solid #2196f3;
                    padding: 16px;
                    margin: 12px;
                    border-radius: 0 6px 6px 0;
                ">
                    ${highlightedContent}
                </div>
                
                <!-- Actions -->
                <div style="
                    padding: 16px 20px;
                    border-top: 1px solid #f0f0f0;
                    display: flex;
                    gap: 8px;
                    flex-wrap: wrap;
                ">
                    <button onclick="copyTranscriptToClipboard('${result.transcript ? result.transcript.replace(/'/g, "\\'") : ''}')" style="
                        background: linear-gradient(135deg, #6e32a0 0%, #8b4cb8 100%);
                        color: white;
                        border: none;
                        padding: 8px 16px;
                        border-radius: 20px;
                        cursor: pointer;
                        font-size: 0.8rem;
                        font-weight: 500;
                    ">📋 Copy Text</button>
                    <button onclick="analyzeCall('${evaluationId}')" style="
                        background: #f8f9fa;
                        color: #666;
                        border: 1px solid #e1e8ed;
                        padding: 8px 16px;
                        border-radius: 20px;
                        cursor: pointer;
                        font-size: 0.8rem;
                        font-weight: 500;
                    ">🔍 Analyze Call</button>
                    <button onclick="applyQuickFilter('partner', '${partner}')" style="
                        background: linear-gradient(135deg, #2196f3 0%, #1976d2 100%);
                        color: white;
                        border: none;
                        padding: 8px 16px;
                        border-radius: 20px;
                        cursor: pointer;
                        font-size: 0.8rem;
                        font-weight: 500;
                    ">🔧 Filter by Partner</button>
                </div>
            </div>
        `;
    }).join('');
}

function addSummaryToInputContainer(query, totalResults, totalEvaluationsScanned) {
    // Find the input container (where the chat input and send button are)
    const inputContainer = document.querySelector('.chat-input-area') || 
                          document.querySelector('.input-container') ||
                          document.getElementById('chatInput')?.parentElement;
    
    if (!inputContainer) {
        console.warn("⚠️ Input container not found for summary");
        return;
    }
    
    // Remove any existing summary
    const existingSummary = document.getElementById('transcriptSearchSummary');
    if (existingSummary) {
        existingSummary.remove();
    }
    
    // Create compact summary element
    const summaryDiv = document.createElement('div');
    summaryDiv.id = 'transcriptSearchSummary';
    summaryDiv.innerHTML = `
        <div style="
            background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%);
            border: 1px solid #1976d2;
            border-radius: 8px;
            padding: 8px 12px;
            margin-bottom: 8px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 0.85rem;
        ">
            <div style="display: flex; align-items: center; gap: 12px;">
                <strong style="color: #1565c0;">🔍 "${query}": ${totalResults} matches</strong>
                <span style="color: #666;">(${totalEvaluationsScanned} scanned)</span>
            </div>
            <div style="display: flex; gap: 8px;">
                <button onclick="downloadTranscriptResults('csv', '${query}')" style="
                    background: #1976d2; color: white; border: none; 
                    padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 0.75rem;
                ">📊 Export CSV</button>
                <button onclick="clearTranscriptSearchResults()" style="
                    background: #f44336; color: white; border: none; 
                    padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 0.75rem;
                ">✕ Clear</button>
            </div>
        </div>
    `;
    
    // Insert summary before the input container
    inputContainer.parentNode.insertBefore(summaryDiv, inputContainer);
    
    console.log("✅ Summary added to input container");
}

function clearTranscriptSearchResults() {
    // Clear chat messages
    const chatMessages = document.getElementById('chatMessages');
    if (chatMessages) {
        chatMessages.innerHTML = '';
        chatMessages.classList.add('hidden');
    }
    
    // Remove summary from input container
    const summary = document.getElementById('transcriptSearchSummary');
    if (summary) {
        summary.remove();
    }
    
    // Show welcome screen
    const welcomeScreen = document.getElementById('welcomeScreen') || 
                         document.querySelector('.welcome-screen');
    if (welcomeScreen) {
        welcomeScreen.style.display = 'block';
    }
    
    // Show quick questions
    showQuickQuestions();
    
    // Clear stored results
    window.lastTranscriptResults = [];
    
    console.log("🧹 Transcript search results cleared, welcome screen restored");
}

// Helper function to download transcript search results
function downloadTranscriptResults(format, query) {
    // Fix: Use window.lastTranscriptResults consistently
    if (!window.lastTranscriptResults || window.lastTranscriptResults.length === 0) {
        showToast('❌ No results to download', 'error');
        console.log("❌ No results found. window.lastTranscriptResults:", window.lastTranscriptResults);
        return;
    }
    
    console.log(`📥 Downloading ${window.lastTranscriptResults.length} transcript results as CSV`);
    
    if (format === 'csv') {
        // Enhanced CSV headers with better field mapping
        const headers = [
            'Evaluation ID',
            'Internal ID', 
            'Partner',
            'Program',
            'Call Date',
            'Disposition',
            'Sub Disposition',
            'Search Query',
            'Match Score',
            'Transcript Preview',
            'Full Transcript'
        ];
        const csvRows = [headers.join(',')];
        
        window.lastTranscriptResults.forEach(result => {
            // Handle both possible data structures
            const metadata = result.metadata || {};
            const evaluationId = result.evaluationId || result.evaluation_id || result.id || '';
            const internalId = result.internalId || result.evaluationId || '';
            const partner = metadata.partner || result.partner || '';
            const program = metadata.program || result.program || '';
            const callDate = metadata.call_date || result.call_date || metadata.callDate || '';
            const disposition = result.disposition || metadata.disposition || '';
            const subDisposition = result.sub_disposition || metadata.subDisposition || '';
            const score = result._score || result.score || 0;
            
            // Get transcript content from various possible fields
            const fullTranscript = result.transcript || 
                                 result.highlighted_text || 
                                 result.text || 
                                 result.content || 
                                 '';
            
            // Create preview (first 200 chars)
            const transcriptPreview = fullTranscript.length > 200 ? 
                                    fullTranscript.substring(0, 200) + '...' : 
                                    fullTranscript;
            
            // Properly escape CSV fields
            const escapeCSVField = (field) => {
                if (field === null || field === undefined) return '';
                const str = String(field);
                if (str.includes(',') || str.includes('"') || str.includes('\n')) {
                    return `"${str.replace(/"/g, '""')}"`;
                }
                return str;
            };
            
            const row = [
                escapeCSVField(evaluationId),
                escapeCSVField(internalId),
                escapeCSVField(partner),
                escapeCSVField(program),
                escapeCSVField(callDate),
                escapeCSVField(disposition),
                escapeCSVField(subDisposition),
                escapeCSVField(query),
                escapeCSVField(score.toFixed(3)),
                escapeCSVField(transcriptPreview),
                escapeCSVField(fullTranscript)
            ];
            csvRows.push(row.join(','));
        });
        
        const content = csvRows.join('\n');
        const filename = `transcript_search_${query.replace(/[^a-zA-Z0-9]/g, '_')}_${new Date().toISOString().split('T')[0]}.csv`;
        
        // Create and trigger download
        const blob = new Blob([content], { type: 'text/csv' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        showToast(`📥 Downloaded ${filename} with ${window.lastTranscriptResults.length} results`, 'success');
    }
}

// Helper function to apply quick filters from search results
function applyQuickFilter(filterType, filterValue) {
    console.log(`🔧 Applying quick filter: ${filterType} = ${filterValue}`);
    
    // Map filter types to actual filter IDs
    const filterMappings = {
        'partner': 'partnerFilter',
        'program': 'programFilter',
        'template': 'templateFilter',
        'disposition': 'callDispositionFilter'
    };
    
    const filterId = filterMappings[filterType];
    if (!filterId) {
        console.warn(`Unknown filter type: ${filterType}`);
        return;
    }
    
    const filterSelect = document.getElementById(filterId);
    if (!filterSelect) {
        console.warn(`Filter element not found: ${filterId}`);
        return;
    }
    
    // Set the filter value
    filterSelect.value = filterValue;
    
    // Apply the filters
    if (typeof applyFilters === 'function') {
        applyFilters();
        showToast(`🔧 Applied ${filterType} filter: ${filterValue}`, 'info');
    } else {
        console.warn('applyFilters function not available');
    }
}

console.log("✅ Transcript search display functions added successfully");
console.log("🎯 Functions added: displayTranscriptSearchResults, displayComprehensiveTranscriptResults, downloadTranscriptResults, applyQuickFilter");

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

function showTranscriptSearchLoading(query, isComprehensive = false) {
    try {
        const resultsContainer = document.getElementById('transcriptSearchResults') || 
                               document.querySelector('.transcript-search-results');
        
        if (!resultsContainer) {
            console.warn("⚠️ No results container found for loading state");
            return;
        }
        
        const loadingHTML = `
            <div class="transcript-search-loading">
                <div class="loading-content">
                    <div class="loading-spinner"></div>
                    <div class="loading-text">
                        <h4>🔍 Searching transcripts${isComprehensive ? ' (comprehensive)' : ''}...</h4>
                        <p>Looking for: <strong>"${escapeHtml(query)}"</strong></p>
                        <div class="loading-progress">
                            <div class="progress-bar"></div>
                        </div>
                        <small class="loading-tip">
                            ${isComprehensive ? 
                                'Scanning all available transcripts for matches...' : 
                                'Searching recent transcripts for quick results...'
                            }
                        </small>
                    </div>
                </div>
            </div>
        `;
        
        resultsContainer.innerHTML = loadingHTML;
        resultsContainer.style.display = 'block';
        
        // Add CSS for loading animation if not already present
        if (!document.getElementById('transcript-loading-styles')) {
            const loadingStyles = document.createElement('style');
            loadingStyles.id = 'transcript-loading-styles';
            loadingStyles.textContent = `
                .transcript-search-loading {
                    padding: 40px 20px;
                    text-align: center;
                    background: white;
                    border-radius: 8px;
                    margin: 20px 0;
                }
                
                .loading-content {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    gap: 16px;
                }
                
                .loading-spinner {
                    width: 32px;
                    height: 32px;
                    border: 3px solid #f3f3f3;
                    border-top: 3px solid #2196f3;
                    border-radius: 50%;
                    animation: spin 1s linear infinite;
                }
                
                .loading-text h4 {
                    margin: 0;
                    color: #333;
                    font-size: 1.1rem;
                }
                
                .loading-text p {
                    margin: 8px 0;
                    color: #666;
                }
                
                .loading-progress {
                    width: 200px;
                    height: 4px;
                    background: #f0f0f0;
                    border-radius: 2px;
                    overflow: hidden;
                }
                
                .progress-bar {
                    height: 100%;
                    background: linear-gradient(90deg, #2196f3, #21cbf3);
                    border-radius: 2px;
                    animation: progress 2s ease-in-out infinite;
                }
                
                .loading-tip {
                    color: #888;
                    font-style: italic;
                }
                
                @keyframes progress {
                    0% { width: 0%; }
                    50% { width: 70%; }
                    100% { width: 100%; }
                }
            `;
            document.head.appendChild(loadingStyles);
        }
        
    } catch (error) {
        console.error("❌ Error showing loading state:", error);
    }
}

// =============================================================================
// UPDATED: clearTranscriptResults function with showQuickQuestions
// =============================================================================
function clearTranscriptResults() {
    const resultsContainer = document.getElementById('transcriptSearchResults');
    const resultsList = document.getElementById('transcriptResultsList');
    const resultsSummary = document.getElementById('transcriptResultsSummary');
    
    if (resultsContainer) {
        resultsContainer.classList.add('hidden');
        resultsContainer.style.display = 'none';
    }
    
    if (resultsList) {
        resultsList.innerHTML = '';
    }
    
    if (resultsSummary) {
        resultsSummary.innerHTML = '';
    }
    
    // Show welcome screen again when clearing results
    const welcomeScreen = document.getElementById('welcomeScreen') || 
                         document.querySelector('.welcome-screen');
    if (welcomeScreen) {
        welcomeScreen.style.display = 'block';
        console.log("✅ Welcome screen restored");
    }
    
    // Show quick questions again (use direct approach)
    const quickQuestions = document.getElementById('quickQuestions') || 
                          document.querySelector('.quick-questions');
    if (quickQuestions) {
        quickQuestions.style.display = 'block';
        console.log("✅ Quick questions restored");
    }
    
    // Clear stored results
    if (typeof window.lastTranscriptResults !== 'undefined') {
        window.lastTranscriptResults = [];
    }
    
    console.log("🧹 Transcript search results cleared, welcome screen and quick questions restored");
}

function addTranscriptResultsContainer() {
    console.log("🎯 Adding transcript search results container...");
    
    // Check if container already exists
    if (document.getElementById('transcriptSearchResults')) {
        console.log("✅ Transcript results container already exists");
        // MAKE SURE it's visible and properly structured
        const existing = document.getElementById('transcriptSearchResults');
        if (!document.getElementById('transcriptResultsSummary') || !document.getElementById('transcriptResultsList')) {
            console.log("🔧 Fixing incomplete container structure...");
            existing.innerHTML = `
                <div class="transcript-results-header" style="background: #f8f9fa; border-bottom: 1px solid #e9ecef; padding: 16px 20px; display: flex; justify-content: space-between; align-items: center;">
                    <h3 style="margin: 0; color: #333; font-size: 1.2rem;">🎯 Transcript Search Results</h3>
                    <button onclick="clearTranscriptResults()" style="background: none; border: none; color: #666; cursor: pointer; font-size: 18px; padding: 4px; border-radius: 50%; width: 28px; height: 28px;">✕</button>
                </div>
                <div id="transcriptResultsSummary" class="results-summary" style="background: #e8f5e8; border-bottom: 1px solid #e9ecef; padding: 16px 20px;"></div>
                <div id="transcriptResultsList" class="results-list" style="max-height: 600px; overflow-y: auto; padding: 0 20px 20px 20px;"></div>
            `;
        }
        return existing;
    }
    
    // Create the results container with INLINE STYLES to ensure it works
    const resultsContainer = document.createElement('div');
    resultsContainer.id = 'transcriptSearchResults';
    resultsContainer.className = 'transcript-search-results';
    // REMOVE 'hidden' class - we'll control visibility with display style
    resultsContainer.style.cssText = `
        background: white;
        border: 1px solid #e1e8ed;
        border-radius: 12px;
        margin: 20px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        display: none;
    `;
    
    resultsContainer.innerHTML = `
        <div class="transcript-results-header" style="background: #f8f9fa; border-bottom: 1px solid #e9ecef; padding: 16px 20px; display: flex; justify-content: space-between; align-items: center;">
            <h3 style="margin: 0; color: #333; font-size: 1.2rem;">🎯 Transcript Search Results</h3>
            <button onclick="clearTranscriptResults()" style="background: none; border: none; color: #666; cursor: pointer; font-size: 18px; padding: 4px; border-radius: 50%; width: 28px; height: 28px;">✕</button>
        </div>
        <div id="transcriptResultsSummary" class="results-summary" style="background: #e8f5e8; border-bottom: 1px solid #e9ecef; padding: 16px 20px;"></div>
        <div id="transcriptResultsList" class="results-list" style="max-height: 600px; overflow-y: auto; padding: 0 20px 20px 20px;"></div>
    `;
    
    // IMPROVED: Better container finding with your actual HTML structure
    let insertLocation = null;
    let insertBefore = null;
    
    // Try to find the best location in your chat interface
    const locations = [
        // Option 1: Your main chat area
        { container: document.querySelector('.chat-area'), before: document.querySelector('.chat-input-area') },
        // Option 2: Main layout area
        { container: document.querySelector('.main-layout'), before: null },
        // Option 3: Messages container parent
        { container: document.querySelector('.messages-container')?.parentNode, before: document.querySelector('.chat-input-area') },
        // Option 4: Fallback to body
        { container: document.body, before: null }
    ];
    
    for (const location of locations) {
        if (location.container) {
            insertLocation = location.container;
            insertBefore = location.before;
            console.log("📍 Found insertion location:", location.container.className || location.container.tagName);
            break;
        }
    }
    
    // Insert the container
    if (insertBefore) {
        insertLocation.insertBefore(resultsContainer, insertBefore);
        console.log("✅ Transcript results container inserted before chat input");
    } else {
        insertLocation.appendChild(resultsContainer);
        console.log("✅ Transcript results container appended to container");
    }
    
    return resultsContainer;
}



function retryLastTranscriptSearch() {
    const chatInput = document.getElementById('chatInput');
    if (chatInput && window.lastSearchQuery) {
        chatInput.value = window.lastSearchQuery;
        sendMessageWithTranscriptSearch();
    } else {
        showToast('❌ No previous search to retry', 'warning');
    }
}

/**
 * UTILITY: Clear transcript results
 */
function clearTranscriptResults() {
    const resultsContainer = document.getElementById('transcriptSearchResults');
    const resultsList = document.getElementById('transcriptResultsList');
    const resultsSummary = document.getElementById('transcriptResultsSummary');
    
    if (resultsContainer) {
        resultsContainer.classList.add('hidden');
        resultsContainer.style.display = 'none';
    }
    
    if (resultsList) {
        resultsList.innerHTML = '';
    }
    
    if (resultsSummary) {
        resultsSummary.innerHTML = '';
    }
    
    // Show welcome screen again when clearing results
    const welcomeScreen = document.getElementById('welcomeScreen') || 
                         document.querySelector('.welcome-screen');
    if (welcomeScreen) {
        welcomeScreen.style.display = 'block';
        console.log("✅ Welcome screen restored");
    }
    
    // Clear stored results
    if (typeof lastTranscriptResults !== 'undefined') {
        lastTranscriptResults = [];
    }
    
    console.log("🧹 Transcript search results cleared and welcome screen restored");
}

// Display transcript search results
function displayTranscriptSearchResults(data, query) {
    console.log("🎯 Displaying transcript search results in chat container:", data);
    
    // Hide quick questions
    hideQuickQuestions();
    
    // Hide welcome screen
    const welcomeScreen = document.getElementById('welcomeScreen') || 
                         document.querySelector('.welcome-screen');
    if (welcomeScreen) {
        welcomeScreen.style.display = 'none';
    }
    
    // Use the chat messages container instead of separate container
    const chatMessages = document.getElementById('chatMessages');
    if (!chatMessages) {
        console.error("❌ Chat messages container not found!");
        return;
    }
    
    // Show chat messages container and clear existing content
    chatMessages.classList.remove('hidden');
    chatMessages.style.display = 'block';
    chatMessages.innerHTML = '';
    
    // Extract data from response
    const results = data.display_results || data.results || [];
    const summary = data.comprehensive_summary || data.summary || {};
    const totalResults = results.length;
    const totalEvaluationsScanned = summary.total_evaluations_searched || 0;
    
    // Store results globally for export (FIX: Use consistent variable name)
    window.lastTranscriptResults = results;
    
    console.log(`🎨 BALANCED: Will highlight search terms and variations from: "${query}"`);
    console.log(`📊 Stored ${results.length} results for export`);
    
    // Add summary to input container
    addSummaryToInputContainer(query, totalResults, totalEvaluationsScanned);
    
    // Create results in chat messages container
    if (totalResults === 0) {
        chatMessages.innerHTML = `
            <div style="text-align: center; padding: 40px; color: #666;">
                <div style="font-size: 3rem; margin-bottom: 16px; opacity: 0.5;">🔍</div>
                <h3>No matches found for "${query}"</h3>
                <p>Try different search terms or check your filters.</p>
            </div>
        `;
    } else {
        // Create individual result cards in chat container
        const resultsHTML = results.map((result, index) => {
            const evaluationId = result.evaluationId || result.evaluation_id || result.id;
            const metadata = result.metadata || {};
            const transcript = result.transcript || result.highlighted_text || result.text || '';
            const score = result._score || result.score || 0;
            
            // BALANCED HIGHLIGHTING: Search terms and variations
            const highlightedTranscript = highlightSearchTerms(transcript, query);
            const truncatedTranscript = highlightedTranscript.length > 400 ? 
                                      highlightedTranscript.substring(0, 400) + '...' : 
                                      highlightedTranscript;
            
            return `
                <div class="message assistant-message" style="margin-bottom: 16px;">
                    <div class="message-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <div style="display: flex; align-items: center; gap: 12px;">
                            <span class="role" style="font-weight: 600; color: #1976d2;">Result ${index + 1}</span>
                            <span style="color: #666; font-size: 0.85rem;">ID: ${evaluationId}</span>
                            ${metadata.partner ? `<span style="background: #e3f2fd; color: #1565c0; padding: 2px 6px; border-radius: 12px; font-size: 0.7rem;">${metadata.partner}</span>` : ''}
                            ${score > 0 ? `<span style="background: #4caf50; color: white; padding: 2px 6px; border-radius: 8px; font-size: 0.7rem;">Score: ${(score * 100).toFixed(1)}%</span>` : ''}
                        </div>
                        <div style="display: flex; gap: 6px;">
                            <button onclick="copyTranscriptToClipboard('${transcript.replace(/'/g, "\\'")}')'" style="
                                background: #6e32a0; color: white; border: none; 
                                padding: 4px 8px; border-radius: 4px; cursor: pointer; font-size: 0.7rem;
                            ">📋 Copy</button>
                        </div>
                    </div>
                    
                    <div class="message-content" style="
                        background: #f8f9fa; 
                        border-left: 3px solid #1976d2; 
                        padding: 12px; 
                        border-radius: 0 8px 8px 0;
                        font-size: 0.9rem; 
                        line-height: 1.5;
                    ">
                        ${truncatedTranscript}
                    </div>
                    
                    ${metadata.call_date || metadata.program || metadata.disposition ? `
                        <div style="margin-top: 8px; padding: 8px; background: #f0f0f0; border-radius: 6px; font-size: 0.8rem; color: #666;">
                            ${metadata.call_date ? `📅 ${metadata.call_date}` : ''}
                            ${metadata.program ? ` • 📋 ${metadata.program}` : ''}
                            ${metadata.disposition ? ` • 📞 ${metadata.disposition}` : ''}
                        </div>
                    ` : ''}
                </div>
            `;
        }).join('');
        
        chatMessages.innerHTML = resultsHTML;
    }
    
    // Scroll to top of results
    chatMessages.scrollTop = 0;
    
    console.log(`✅ Displayed ${totalResults} results in chat container with balanced highlighting`);
}

// Show loading state for transcript search
function showTranscriptSearchLoading(query) {
    const resultsContainer = document.getElementById('transcriptSearchResults');
    const resultsList = document.getElementById('transcriptResultsList');
    const resultsSummary = document.getElementById('transcriptResultsSummary');
    
    if (!resultsContainer || !resultsList || !resultsSummary) {
        console.warn("⚠️ Transcript results containers not found for loading state");
        return;
    }
    
    // Show container
    resultsContainer.classList.remove('hidden');
    
    // Show loading in summary
    resultsSummary.innerHTML = `
        <div class="loading-summary">
            <div class="loading-spinner"></div>
            <span>Searching transcripts for "<strong>${query}</strong>"...</span>
        </div>
    `;
    
    // Show loading in results
    resultsList.innerHTML = `
        <div class="transcript-search-loading">
            <div class="loading-spinner"></div>
            <p>Analyzing call transcripts...</p>
        </div>
    `;
    
    console.log(`🔄 Showing transcript search loading for query: "${query}"`);
}

// Placeholder functions for result actions
function viewFullTranscript(callId) {
    console.log(`📄 Viewing full transcript for call: ${callId}`);
    // Implement full transcript viewing logic
    if (typeof showToast === 'function') {
        showToast(`Opening full transcript for call ${callId}`, 'info');
    }
}

function analyzeCall(callId) {
    console.log(`📊 Analyzing call: ${callId}`);
    // Implement call analysis logic
    if (typeof showToast === 'function') {
        showToast(`Starting analysis for call ${callId}`, 'info');
    }
}

function ensureTranscriptContainersReady() {
    console.log("🔧 Ensuring transcript containers are ready...");
    
    // Make sure the main container exists
    const mainContainer = addTranscriptResultsContainer();
    
    // Verify the sub-containers exist
    const summary = document.getElementById('transcriptResultsSummary');
    const list = document.getElementById('transcriptResultsList');
    
    if (!summary || !list) {
        console.error("❌ Sub-containers missing, recreating...");
        addTranscriptResultsContainer(); // This will fix the structure
    }
    
    console.log("✅ All transcript containers verified and ready");
    return { container: mainContainer, summary, list };
}



// Store last search query for retry functionality
let originalSendMessageWithTranscriptSearch = sendMessageWithTranscriptSearch;
sendMessageWithTranscriptSearch = function() {
    const chatInput = document.getElementById('chatInput');
    if (chatInput && chatInput.value?.trim()) {
        window.lastSearchQuery = chatInput.value.trim();
    }
    return originalSendMessageWithTranscriptSearch();
};

console.log("✅ Fully optimized sendMessageWithTranscriptSearch function loaded");
function showTranscriptSearchError(errorMessage) {
    try {
        const resultsContainer = document.getElementById('transcriptSearchResults') || 
                               document.querySelector('.transcript-search-results');
        
        if (!resultsContainer) {
            // Fallback: show toast notification
            if (typeof showToast === 'function') {
                showToast(`❌ Transcript Search Error: ${errorMessage}`, 'error');
            } else {
                console.error("❌ Transcript Search Error:", errorMessage);
                alert(`Transcript Search Error: ${errorMessage}`);
            }
            return;
        }
        
        const errorHTML = `
            <div class="transcript-search-error">
                <div class="error-content">
                    <div class="error-icon">❌</div>
                    <h4>Search Error</h4>
                    <p class="error-message">${escapeHtml(errorMessage)}</p>
                    <div class="error-actions">
                        <button class="transcript-btn transcript-btn--primary" onclick="retryLastTranscriptSearch()">
                            🔄 Try Again
                        </button>
                        <button class="transcript-btn transcript-btn--secondary" onclick="clearTranscriptResults()">
                            ✕ Clear
                        </button>
                    </div>
                    <details class="error-help">
                        <summary>💡 Troubleshooting Tips</summary>
                        <ul>
                            <li>Try a simpler search term (single words work best)</li>
                            <li>Check your internet connection</li>
                            <li>Avoid special characters in search queries</li>
                            <li>Try refreshing the page if errors persist</li>
                        </ul>
                    </details>
                </div>
            </div>
        `;
        
        resultsContainer.innerHTML = errorHTML;
        resultsContainer.style.display = 'block';
        
    } catch (displayError) {
        console.error("❌ Error displaying error message:", displayError);
        // Ultimate fallback
        alert(`Error: ${errorMessage}`);
    }
}



/**
 * UTILITY: Escape HTML to prevent XSS
 */
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}


// Utility function to copy text to clipboard
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        // Show temporary feedback
        showToast(`📋 Copied: ${text}`, 'success');
    }).catch(err => {
        console.error('Failed to copy text: ', err);
        showToast('Failed to copy to clipboard', 'error');
    });
}

// Enhanced toast notification system
function showToast(message, type = 'info') {
    // Remove any existing toast
    const existingToast = document.querySelector('.transcript-toast');
    if (existingToast) {
        existingToast.remove();
    }
    
    const toast = document.createElement('div');
    toast.className = `transcript-toast toast-${type}`;
    toast.innerHTML = `
        <span class="toast-message">${message}</span>
        <button onclick="this.parentElement.remove()" class="toast-close">×</button>
    `;
    
    document.body.appendChild(toast);
    
    // Auto-remove after 3 seconds
    setTimeout(() => {
        if (toast.parentElement) {
            toast.remove();
        }
    }, 3000);
}

function truncateText(text, maxLength) {
    if (!text || text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

// =============================================================================
// STYLES AND FORMATTING
// =============================================================================

function addToggleToHeader() {
    console.log("🔧 Adding transcript toggle to header via JavaScript...");
    
    const chatHeaderFilters = document.getElementById('chatHeaderFilters');
    
    if (!chatHeaderFilters) {
        console.error("❌ chatHeaderFilters not found");
        return;
    }
    
    // Check if toggle already exists
    if (document.getElementById('transcriptToggle')) {
        console.log("✅ Toggle already exists");
        return;
    }
    
    // Create the toggle HTML
    const toggleDiv = document.createElement('div');
    toggleDiv.id = 'transcriptToggle';
    toggleDiv.className = 'header-transcript-toggle';
    toggleDiv.innerHTML = `
        <label class="toggle-switch header-toggle">
            <input type="checkbox" id="transcriptSearchToggleInput" onchange="toggleTranscriptSearchMode()">
            <span class="toggle-slider"></span>
            <span class="toggle-label">Search Transcripts Only</span>
        </label>
        <div class="comprehensive-option" id="comprehensiveOption" style="display: none;">
            <label class="toggle-switch secondary">
                <input type="checkbox" id="comprehensiveSearchToggle" checked>
                <span class="toggle-slider small"></span>
                <span class="toggle-label small">Comprehensive Analysis</span>
            </label>
        </div>
    `;
    
    // Add to the beginning of chatHeaderFilters
    chatHeaderFilters.insertBefore(toggleDiv, chatHeaderFilters.firstChild);
    
    console.log("✅ Toggle added to header!");
    
    // Verify it worked
    setTimeout(() => {
        const toggle = document.getElementById('transcriptSearchToggleInput');
        if (toggle) {
            console.log("🎯 Toggle verification: SUCCESS");
        } else {
            console.log("❌ Toggle verification: FAILED");
        }
    }, 100);
}


function loadFormattingStyles() {
    const style = document.createElement('style');
    style.textContent = `
        .typing-dots {
            display: flex;
            color: #E20074
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
            background: #9646c3
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

function debugTranscriptToggle() {
    console.log("🔍 Debugging transcript toggle...");
    
    const elements = {
        'transcriptSearchToggleInput': document.getElementById('transcriptSearchToggleInput'),
        'transcriptToggle': document.getElementById('transcriptToggle'),
        'comprehensiveOption': document.getElementById('comprehensiveOption'),
        'chatHeaderFilters': document.getElementById('chatHeaderFilters'),
        'chatInput': document.getElementById('chatInput')
    };
    
    Object.entries(elements).forEach(([name, element]) => {
        if (element) {
            console.log(`✅ ${name}: Found`);
        } else {
            console.log(`❌ ${name}: NOT FOUND`);
        }
    });
    
    // Check if CSS is loaded
    const testElement = document.createElement('div');
    testElement.className = 'header-transcript-toggle';
    document.body.appendChild(testElement);
    const styles = window.getComputedStyle(testElement);
    const hasStyles = styles.display !== 'block'; // Default div display
    document.body.removeChild(testElement);
    
    console.log(`🎨 CSS loaded: ${hasStyles ? 'YES' : 'NO'}`);
    
    return elements;
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

// Analytics function - NEW!
window.refreshAnalyticsStats = refreshAnalyticsStats;

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
// 🔧 FIXED: INITIALIZATION WITH ANALYTICS STATS
// Added refreshAnalyticsStats() call to load initial stats on page startup
// =============================================================================

document.addEventListener('DOMContentLoaded', function() {
    console.log("🚀 Metro AI Call Center Analytics v4.4.0 - Production initializing...");
    console.log("🚀 Chat-Stats Integration FIXED - Vector Search ENABLED");
    
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
                    
                    // 🔧 FIX: Load initial analytics stats on startup
                    refreshAnalyticsStats();
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

    // 🔧 UPDATED: Use enhanced transcript search initialization
    setTimeout(initializeEnhancedTranscriptSearch, 1000);

    console.log("✅ Metro AI Call Center Analytics v4.9.0 production loaded successfully");
    console.log("🔧 FIXED: Chat-stats integration, duplicate functions removed, analytics connected");
    console.log("🔮 Vector search: Enhanced relevance and semantic similarity support");
    console.log("🔧 Debug mode:", PRODUCTION_CONFIG.DEBUG_MODE ? "ENABLED" : "DISABLED");
    console.log("🎯 Enhanced transcript search: Initialization scheduled with fixes for limits, highlighting, and interface switching");
});

// =============================================================================
// TRANSCRIPT SEARCH MESSAGE OVERRIDE (ADD THIS AFTER YOUR DOMContentLoaded)
// =============================================================================

// Override the existing sendMessage function to handle transcript search
const originalSendMessage = window.sendMessage;
window.sendMessage = function() {
    if (transcriptSearchMode) {
        return sendMessageWithTranscriptSearch();
    } else {
        return originalSendMessage ();
    }
};

console.log("🎯 Transcript search: sendMessage override applied");