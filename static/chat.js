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
    
    // The toggle is now in the HTML, so we DON'T call addTranscriptSearchToggle()
    // Just add the results container
    addTranscriptResultsContainer();
    
    // Verify the toggle exists in HTML
    const toggle = document.getElementById('transcriptSearchToggleInput');
    if (toggle) {
        console.log("✅ Transcript toggle found in HTML");
    } else {
        console.warn("⚠️ Transcript toggle NOT found in HTML - check your HTML file");
    }
}


function updateChatInterfaceForTranscriptMode(isTranscriptMode) {
    const chatInput = document.getElementById('chatInput') || 
                     document.querySelector('input[type="text"]') ||
                     document.querySelector('textarea');
    
    if (chatInput) {
        if (isTranscriptMode) {
            // Clear, specific placeholder with examples
            chatInput.placeholder = "Enter words or phrases to find in call transcripts (e.g., cancel, billing issue, refund request)";
            chatInput.classList.add('transcript-mode');
            
            // Add helpful guidance text below input
            addTranscriptSearchGuidance();
        } else {
            chatInput.placeholder = "Ask a question about the evaluation data...";
            chatInput.classList.remove('transcript-mode');
            chatInput.style.borderColor = "";
            chatInput.style.boxShadow = "";
            
            // Remove guidance text
            removeTranscriptSearchGuidance();
        }
    }
}

// ADD this new function to your chat.js
function addTranscriptSearchGuidance() {
    // Check if guidance already exists
    if (document.getElementById('transcriptSearchGuidance')) {
        return;
    }
    
    const chatInput = document.getElementById('chatInput');
    if (!chatInput) return;
    
    const guidanceDiv = document.createElement('div');
    guidanceDiv.id = 'transcriptSearchGuidance';
    guidanceDiv.className = 'transcript-search-guidance';
    guidanceDiv.innerHTML = `
        <div class="guidance-content">
            <div class="guidance-header">
                <span class="guidance-icon">💡</span>
                <strong>Transcript Search Tips:</strong>
            </div>
            <div class="guidance-examples">
                <div class="example-row">
                    <span class="example-label">Single words:</span>
                    <code>cancel</code>, <code>billing</code>, <code>refund</code>
                </div>
                <div class="example-row">
                    <span class="example-label">Exact phrases:</span>
                    <code>"cancel my service"</code>, <code>"billing issue"</code>
                </div>
                <div class="example-row">
                    <span class="example-label">Multiple terms:</span>
                    <code>account password reset</code>
                </div>
            </div>
            <div class="guidance-note">
                <small>💡 Enter the exact words you want to find - no need for full sentences</small>
            </div>
        </div>
    `;
    
    // Insert guidance after the chat input
    if (chatInput.parentNode) {
        chatInput.parentNode.insertBefore(guidanceDiv, chatInput.nextSibling);
    }
}

// ADD this new function to your chat.js
function removeTranscriptSearchGuidance() {
    const guidanceDiv = document.getElementById('transcriptSearchGuidance');
    if (guidanceDiv) {
        guidanceDiv.remove();
    }
}

console.log("✅ Complete transcript search UI functions added to chat.js");

async function sendMessageWithTranscriptSearch() {
    // PERFORMANCE: Use requestAnimationFrame to prevent UI blocking
    return new Promise((resolve) => {
        requestAnimationFrame(async () => {
            const performanceStart = performance.now();
            console.log("🔍 DEBUG: sendMessageWithTranscriptSearch called");
            
            try {
                // STEP 1: Fast validation and setup
                const chatInput = document.getElementById('chatInput');
                if (!chatInput) {
                    console.error("🔍 DEBUG: #chatInput not found!");
                    resolve();
                    return;
                }
                
                // PERFORMANCE: Capture message immediately before any async operations
                const message = chatInput.value?.trim();
                console.log("🔍 DEBUG: Message captured:", message);
                
                // Early exit validations
                if (!message) {
                    console.log("🔍 DEBUG: No message provided");
                    resolve();
                    return;
                }
                
                if (window.isLoading) {
                    console.log("🔍 DEBUG: Already loading, preventing duplicate request");
                    resolve();
                    return;
                }
                
                if (!transcriptSearchMode) {
                    console.log("🔍 DEBUG: Not in transcript mode, calling original sendMessage");
                    resolve(originalSendMessage?.());
                    return;
                }
                
                // STEP 2: Set loading state immediately
                window.isLoading = true;
                
                // PERFORMANCE: Clear input and update UI synchronously
                chatInput.value = '';
                chatInput.style.height = 'auto';
                
                // Determine search type
                const comprehensiveToggle = document.getElementById('comprehensiveSearchToggle');
                const useComprehensive = comprehensiveToggle?.checked ?? false;
                
                console.log(`🎯 Performing transcript search for: "${message}" (comprehensive: ${useComprehensive})`);
                
                // STEP 3: Show loading state asynchronously (non-blocking)
                setTimeout(() => {
                    showTranscriptSearchLoading(message, useComprehensive);
                }, 0);
                
                // STEP 4: Prepare request configuration
                const BASE_URL = 'https://ask-innovai-g8jbr.ondigitalocean.app';
                const endpoint = useComprehensive ? 
                    `${BASE_URL}/search_transcripts_comprehensive` : 
                    `${BASE_URL}/search_transcripts`;
                
                const requestBody = {
                    query: message,
                    filters: currentFilters || {},
                    display_size: 20,
                    max_scan: 10000,
                    highlight: true
                };
                
                console.log("🔍 Sending transcript search request:", {
                    endpoint,
                    query: message,
                    comprehensive: useComprehensive,
                    filtersCount: Object.keys(currentFilters || {}).length
                });
                
                // STEP 5: Create abort controller for timeout handling
                const abortController = new AbortController();
                const timeoutId = setTimeout(() => {
                    abortController.abort();
                    console.warn("⏰ Request timeout after 30 seconds");
                }, 30000);
                
                try {
                    // STEP 6: Execute fetch request
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
                    
                    // STEP 7: Handle response
                    if (!response.ok) {
                        let errorText;
                        try {
                            errorText = await response.text();
                            console.error("API Error Response:", errorText);
                        } catch (e) {
                            errorText = `HTTP ${response.status} ${response.statusText}`;
                        }
                        
                        throw new Error(`Search failed: ${response.status} ${response.statusText}${errorText ? ` - ${errorText}` : ''}`);
                    }
                    
                    const data = await response.json();
                    console.log("🔍 Search response received:", {
                        status: data.status,
                        resultCount: data.results?.length || 0,
                        totalMatches: data.summary?.total_word_occurrences || 0
                    });
                    
                    // STEP 8: Process successful response
                    if (data.status === 'success') {
                        // PERFORMANCE: Update UI asynchronously
                        setTimeout(() => {
                            try {
                                if (useComprehensive) {
                                    displayComprehensiveTranscriptResults(data, message);
                                } else {
                                    displayTranscriptSearchResults(data, message);
                                }
                                
                                // Cache results for potential reuse
                                lastTranscriptResults = data.display_results || data.results || [];
                                
                                // Show success toast
                                if (typeof showToast === 'function') {
                                    const resultCount = lastTranscriptResults.length;
                                    const totalMatches = data.summary?.total_word_occurrences || 0;
                                    showToast(
                                        `✅ Found ${resultCount} transcripts with ${totalMatches} total matches for "${message}"`,
                                        'success'
                                    );
                                }
                                
                            } catch (displayError) {
                                console.error("❌ Error displaying results:", displayError);
                                showTranscriptSearchError(`Display error: ${displayError.message}`);
                            }
                        }, 0);
                        
                    } else {
                        // Handle API-level errors
                        const errorMessage = data.error || 'Unknown error occurred';
                        console.error("🔍 Search API error:", errorMessage);
                        setTimeout(() => showTranscriptSearchError(errorMessage), 0);
                    }
                    
                } catch (fetchError) {
                    clearTimeout(timeoutId);
                    
                    // STEP 9: Handle fetch errors
                    console.error('❌ Transcript search request failed:', fetchError);
                    
                    let userErrorMessage;
                    if (fetchError.name === 'AbortError') {
                        userErrorMessage = 'Search request timed out. Please try again with a shorter query.';
                    } else if (fetchError.message.includes('Failed to fetch')) {
                        userErrorMessage = 'Network error. Please check your connection and try again.';
                    } else if (fetchError.message.includes('500')) {
                        userErrorMessage = 'Server error. Please try again or contact support if the issue persists.';
                    } else {
                        userErrorMessage = `Search failed: ${fetchError.message}`;
                    }
                    
                    setTimeout(() => showTranscriptSearchError(userErrorMessage), 0);
                }
                
            } catch (criticalError) {
                console.error('❌ CRITICAL: Unexpected error in sendMessageWithTranscriptSearch:', criticalError);
                
                // STEP 10: Handle critical errors
                setTimeout(() => {
                    showTranscriptSearchError('A critical error occurred. Please refresh the page and try again.');
                    
                    // Optional: Show more detailed error in development
                    if (PRODUCTION_CONFIG?.DEBUG_MODE) {
                        console.error('DEBUG: Critical error details:', criticalError.stack);
                    }
                }, 0);
                
            } finally {
                // STEP 11: Cleanup (always executed)
                setTimeout(() => {
                    window.isLoading = false;
                    
                    // Performance logging
                    const performanceEnd = performance.now();
                    const duration = performanceEnd - performanceStart;
                    console.log(`🚀 sendMessageWithTranscriptSearch completed in ${duration.toFixed(2)}ms`);
                    
                    // Track performance metrics if monitoring is enabled
                    if (typeof performanceMetrics !== 'undefined') {
                        performanceMetrics.chatResponseTimes.push(duration);
                        // Keep only last 50 measurements
                        if (performanceMetrics.chatResponseTimes.length > 50) {
                            performanceMetrics.chatResponseTimes.shift();
                        }
                    }
                    
                }, 100); // Small delay to ensure UI updates complete
                
                resolve();
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
        'callSubDispositionFilter': 'sub_disposition',
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
        'sub_disposition': 'callSubDispositionFilter',
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
        
        if (data.reply) {
            addMessageToChat('assistant', data.reply, data);
            chatHistory.push(
                { role: 'user', content: message },
                { role: 'assistant', content: data.reply }
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

// ===================================================================
// MISSING DISPLAY FUNCTIONS - Add these to your chat.js file
// These functions handle displaying the transcript search results
// ===================================================================

// Function to display standard transcript search results
function displayTranscriptSearchResults(data, query) {
    console.log("🎯 Displaying standard transcript search results:", data);
    
    const resultsContainer = document.getElementById('transcriptSearchResults');
    const resultsList = document.getElementById('transcriptResultsList');
    const resultsSummary = document.getElementById('transcriptResultsSummary');
    
    if (!resultsContainer || !resultsList || !resultsSummary) {
        console.error("❌ Transcript results containers not found!");
        return;
    }
    
    // Show container
    resultsContainer.classList.remove('hidden');
    
    // Reset loading state
    window.isLoading = false;
    
    const results = data.display_results || data.results || [];
    const totalMatches = data.total_matches || results.length;
    const searchTime = data.search_time_ms || 0;
    
    // Update summary
    resultsSummary.innerHTML = `
        <div class="search-summary">
            <h4>🎯 Standard Search Results</h4>
            <div class="summary-stats">
                <span class="stat">
                    <strong>${totalMatches}</strong> matches found
                </span>
                <span class="stat">
                    Query: <strong>"${query}"</strong>
                </span>
                <span class="stat">
                    ${searchTime}ms
                </span>
            </div>
        </div>
    `;
    
    // Display results
    if (results.length === 0) {
        resultsList.innerHTML = `
            <div class="no-results">
                <h4>🔍 No matches found</h4>
                <p>No transcript content matched your search for "<strong>${query}</strong>"</p>
                <p>Try:</p>
                <ul>
                    <li>Different keywords or phrases</li>
                    <li>Broader search terms</li>
                    <li>Check spelling</li>
                </ul>
            </div>
        `;
        return;
    }
    
    let resultsHtml = '';
    results.forEach((result, index) => {
        const evaluationId = result.evaluationId || result.evaluation_id || 'Unknown';
        const transcript = result.highlighted_text || result.highlighted_snippets?.join(' ') || result.transcript || '';
        const score = result._score || result.score || 0;
        const partner = result.metadata?.partner || result.partner || 'N/A';
        const program = result.metadata?.program || result.program || 'N/A';
        const callDate = result.metadata?.call_date || result.call_date || 'N/A';
        const matchCount = result.match_count || 1;
        
        resultsHtml += `
            <div class="search-result-item">
                <div class="result-header">
                    <div class="result-title">
                        <strong>Evaluation ${evaluationId}</strong>
                        ${score > 0 ? `<span class="score-badge">Score: ${(score * 100).toFixed(1)}%</span>` : ''}
                        <span class="match-badge">${matchCount} match${matchCount !== 1 ? 'es' : ''}</span>
                    </div>
                </div>
                <div class="result-meta">
                    <div class="reference-grid">
                        <div class="ref-item">
                            <span class="ref-label">Partner:</span>
                            <span class="ref-value">${partner}</span>
                        </div>
                        <div class="ref-item">
                            <span class="ref-label">Program:</span>
                            <span class="ref-value">${program}</span>
                        </div>
                        <div class="ref-item">
                            <span class="ref-label">Date:</span>
                            <span class="ref-value">${callDate}</span>
                        </div>
                    </div>
                </div>
                <div class="result-content">
                    <div class="highlighted-snippet">${transcript}</div>
                </div>
                <div class="result-actions">
                    <button class="btn-primary" onclick="copyToClipboard('${transcript.replace(/'/g, "\\'")}')">
                        📋 Copy Text
                    </button>
                    <button class="btn-secondary" onclick="askQuestion('Tell me more about evaluation ${evaluationId}')">
                        🔍 Analyze This
                    </button>
                </div>
            </div>
        `;
    });
    
    resultsList.innerHTML = resultsHtml;
}

// Function to display comprehensive transcript search results with enhanced analytics
function displayComprehensiveTranscriptResults(data, query) {
    console.log("🎯 Displaying comprehensive transcript search results:", data);
   
    const containers = ensureTranscriptContainersReady();
    containers.container.style.display = 'block';
    
    const resultsContainer = document.getElementById('transcriptSearchResults');
    const resultsList = document.getElementById('transcriptResultsList');
    const resultsSummary = document.getElementById('transcriptResultsSummary');
    
    if (!resultsContainer || !resultsList || !resultsSummary) {
        console.error("❌ Transcript results containers not found!");
        return;
    }
    
    // Show container
    resultsContainer.classList.remove('hidden');
    
    // Reset loading state
    window.isLoading = false;
    
    const results = data.display_results || [];
    const summary = data.comprehensive_summary || {};
    const searchTime = data.search_time_ms || 0;
    
    // Update comprehensive summary
    resultsSummary.innerHTML = `
        <div class="comprehensive-summary">
            <h4>📊 Comprehensive Analysis for "${query}"</h4>
            
            <div class="summary-grid">
                <div class="stat">
                    <strong>${summary.total_evaluations_searched || 0}</strong>
                    <span>Total Searched</span>
                </div>
                <div class="stat">
                    <strong>${summary.evaluations_with_matches || 0}</strong>
                    <span>With Matches</span>
                </div>
                <div class="stat">
                    <strong>${summary.match_percentage || 0}%</strong>
                    <span>Match Rate</span>
                </div>
                <div class="stat">
                    <strong>${summary.total_document_matches || 0}</strong>
                    <span>Total Matches</span>
                </div>
                <div class="stat">
                    <strong>${summary.unique_templates || 0}</strong>
                    <span>Templates</span>
                </div>
                <div class="stat">
                    <strong>${searchTime}ms</strong>
                    <span>Search Time</span>
                </div>
            </div>
            
            ${summary.top_patterns && summary.top_patterns.length > 0 ? `
                <div class="pattern-analysis">
                    <h5>🔍 Most Common Patterns:</h5>
                    <div class="pattern-list">
                        ${summary.top_patterns.slice(0, 5).map(pattern => 
                            `<span class="pattern-tag">${pattern.phrase || pattern} (${pattern.count || ''}x)</span>`
                        ).join('')}
                    </div>
                </div>
            ` : ''}
            
            <div class="download-section">
                <h5>📥 Export Options:</h5>
                <div class="download-buttons">
                    <button class="btn-download" onclick="downloadTranscriptResults('csv', '${query}')">
                        📊 Download CSV
                    </button>
                    <button class="btn-download" onclick="downloadTranscriptResults('json', '${query}')">
                        🔧 Download JSON
                    </button>
                </div>
            </div>
        </div>
    `;
    
    // Display results using the same format as standard search
    if (results.length === 0) {
        resultsList.innerHTML = `
            <div class="no-results">
                <h4>🔍 No matches found in comprehensive scan</h4>
                <p>No transcript content matched your search for "<strong>${query}</strong>" across ${summary.total_evaluations_searched || 0} evaluations.</p>
                <p>Try:</p>
                <ul>
                    <li>Different keywords or phrases</li>
                    <li>Broader search terms</li>
                    <li>Check spelling and try synonyms</li>
                </ul>
            </div>
        `;
        return;
    }
    
    let resultsHtml = '';
    results.forEach((result, index) => {
        const evaluationId = result.evaluationId || result.evaluation_id || 'Unknown';
        const highlights = result.highlighted_snippets || [];
        const transcript = highlights.length > 0 ? highlights.join('<br><br>') : result.transcript || '';
        const score = result._score || result.score || 0;
        const matchCount = result.match_count || highlights.length || 1;
        
        // Enhanced metadata display
        const partner = result.metadata?.partner || result.partner || 'N/A';
        const program = result.metadata?.program || result.program || 'N/A';
        const disposition = result.disposition || result.metadata?.disposition || 'N/A';
        const subDisposition = result.sub_disposition || result.metadata?.sub_disposition || 'N/A';
        const callDate = result.metadata?.call_date || result.call_date || 'N/A';
        
        resultsHtml += `
            <div class="search-result-item">
                <div class="result-header">
                    <div class="result-title">
                        <strong>Evaluation ${evaluationId}</strong>
                        ${score > 0 ? `<span class="score-badge">Score: ${(score * 100).toFixed(1)}%</span>` : ''}
                        <span class="match-badge">${matchCount} match${matchCount !== 1 ? 'es' : ''}</span>
                    </div>
                </div>
                <div class="result-meta">
                    <div class="reference-grid">
                        <div class="ref-item">
                            <span class="ref-label">Partner:</span>
                            <span class="ref-value">${partner}</span>
                        </div>
                        <div class="ref-item">
                            <span class="ref-label">Program:</span>
                            <span class="ref-value">${program}</span>
                        </div>
                        <div class="ref-item">
                            <span class="ref-label">Date:</span>
                            <span class="ref-value">${callDate}</span>
                        </div>
                        <div class="ref-item">
                            <span class="ref-label">Disposition:</span>
                            <span class="ref-value disposition-tag">${disposition}</span>
                        </div>
                        <div class="ref-item">
                            <span class="ref-label">Sub-Disp:</span>
                            <span class="ref-value">${subDisposition}</span>
                        </div>
                    </div>
                </div>
                <div class="result-content">
                    ${highlights.length > 0 ? 
                        highlights.map(highlight => `<div class="highlighted-snippet">${highlight}</div>`).join('') :
                        `<div class="transcript-preview">${transcript.substring(0, 300)}${transcript.length > 300 ? '...' : ''}</div>`
                    }
                </div>
                <div class="result-actions">
                    <button class="btn-primary" onclick="copyToClipboard('${transcript.replace(/'/g, "\\'")}')">
                        📋 Copy Text
                    </button>
                    <button class="btn-secondary" onclick="askQuestion('Analyze evaluation ${evaluationId} in detail')">
                        🔍 Deep Analysis
                    </button>
                    <button class="btn-filter" onclick="applyQuickFilter('partner', '${partner}')">
                        🔧 Filter by Partner
                    </button>
                </div>
            </div>
        `;
    });
    
    resultsList.innerHTML = resultsHtml;
}

// Helper function to download transcript search results
function downloadTranscriptResults(format, query) {
    if (!lastTranscriptResults || lastTranscriptResults.length === 0) {
        showToast('❌ No results to download', 'error');
        return;
    }
    
    console.log(`📥 Downloading transcript results as ${format.toUpperCase()}`);
    
    let content, filename, mimeType;
    
    if (format === 'csv') {
        // Create CSV content
        const headers = ['Evaluation ID', 'Partner', 'Program', 'Call Date', 'Disposition', 'Sub Disposition', 'Match Count', 'Match Text', 'Score'];
        const csvRows = [headers.join(',')];
        
        lastTranscriptResults.forEach(result => {
            const row = [
                result.evaluationId || result.evaluation_id || '',
                result.metadata?.partner || result.partner || '',
                result.metadata?.program || result.program || '',
                result.metadata?.call_date || result.call_date || '',
                result.disposition || result.metadata?.disposition || '',
                result.sub_disposition || result.metadata?.sub_disposition || '',
                result.match_count || 1,
                `"${(result.transcript || result.highlighted_text || '').replace(/"/g, '""')}"`, // Escape quotes
                result._score || result.score || 0
            ];
            csvRows.push(row.join(','));
        });
        
        content = csvRows.join('\n');
        filename = `transcript_search_${query.replace(/[^a-zA-Z0-9]/g, '_')}_${new Date().toISOString().split('T')[0]}.csv`;
        mimeType = 'text/csv';
        
    } else if (format === 'json') {
        content = JSON.stringify({
            query: query,
            timestamp: new Date().toISOString(),
            total_results: lastTranscriptResults.length,
            search_type: 'transcript_search',
            results: lastTranscriptResults
        }, null, 2);
        filename = `transcript_search_${query.replace(/[^a-zA-Z0-9]/g, '_')}_${new Date().toISOString().split('T')[0]}.json`;
        mimeType = 'application/json';
    }
    
    // Create and trigger download
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    showToast(`📥 Downloaded ${filename}`, 'success');
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

function clearTranscriptResults() {
    const resultsContainer = document.getElementById('transcriptSearchResults');
    const resultsList = document.getElementById('transcriptResultsList');
    const resultsSummary = document.getElementById('transcriptResultsSummary');
    
    resultsContainer.classList.add('hidden');
    resultsList.innerHTML = '';
    resultsSummary.innerHTML = '';
    lastTranscriptResults = [];
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
        // Use display instead of addClass('hidden') for better control
        resultsContainer.style.display = 'none';
    }
    
    if (resultsList) {
        resultsList.innerHTML = '';
    }
    
    if (resultsSummary) {
        resultsSummary.innerHTML = '';
    }
    
    // Clear cached results
    if (typeof lastTranscriptResults !== 'undefined') {
        lastTranscriptResults = [];
    }
    
    console.log("🧹 Transcript results cleared");
}

// Display transcript search results
function displayTranscriptSearchResults(data, query) {
    console.log("🎯 Displaying transcript search results:", data);
    
    const resultsContainer = document.getElementById('transcriptSearchResults');
    const resultsList = document.getElementById('transcriptResultsList');
    const resultsSummary = document.getElementById('transcriptResultsSummary');
    
    if (!resultsContainer || !resultsList || !resultsSummary) {
        console.error("❌ Transcript results containers not found!");
        return;
    }
    
    // Show results container
    resultsContainer.classList.remove('hidden');
    
    // Display summary
    const totalResults = data.results ? data.results.length : 0;
    const totalMatches = data.total_matches || totalResults;
    
    resultsSummary.innerHTML = `
        <div class="summary-stats">
            <div class="stat-item">
                <span class="stat-number">${totalResults}</span>
                <span class="stat-label">calls found</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">${totalMatches}</span>
                <span class="stat-label">total matches</span>
            </div>
            <div class="stat-item">
                <span class="stat-number">"${query}"</span>
                <span class="stat-label">search query</span>
            </div>
        </div>
    `;
    
    // Display results
    if (totalResults === 0) {
        resultsList.innerHTML = `
            <div class="no-results">
                <div class="no-results-icon">🔍</div>
                <h3>No matches found</h3>
                <p>No call transcripts contain the search term "<strong>${query}</strong>"</p>
                <div class="search-suggestions">
                    <h4>Try searching for:</h4>
                    <ul>
                        <li>Different keywords or phrases</li>
                        <li>Common call center terms like "billing", "cancel", "refund"</li>
                        <li>Broader terms instead of specific phrases</li>
                    </ul>
                </div>
            </div>
        `;
    } else {
        const resultsHTML = data.results.map((result, index) => {
            const highlights = result.highlights || [];
            const metadata = result.metadata || {};
            
            return `
                <div class="transcript-result-item">
                    <div class="result-header">
                        <div class="result-title">
                            <span class="result-icon">📞</span>
                            Call ${metadata.call_id || result.call_id || `#${index + 1}`}
                        </div>
                        <div class="result-meta">
                            <span class="result-date">${metadata.date || result.date || 'Unknown date'}</span>
                            <span class="result-score">Score: ${(result.score || 0).toFixed(2)}</span>
                        </div>
                    </div>
                    
                    <div class="result-content">
                        <div class="result-text">
                            ${highlights.length > 0 ? highlights.join('... ') : (result.content || result.text || 'No content available')}
                        </div>
                        
                        ${metadata.agent_id ? `<div class="result-agent">Agent: ${metadata.agent_id}</div>` : ''}
                        ${metadata.disposition ? `<div class="result-disposition">Disposition: ${metadata.disposition}</div>` : ''}
                    </div>
                    
                    <div class="result-actions">
                        <button class="result-btn" onclick="viewFullTranscript('${result.call_id || index}')">
                            <span class="material-icons">description</span>
                            View Full Transcript
                        </button>
                        <button class="result-btn" onclick="analyzeCall('${result.call_id || index}')">
                            <span class="material-icons">analytics</span>
                            Analyze Call
                        </button>
                    </div>
                </div>
            `;
        }).join('');
        
        resultsList.innerHTML = resultsHTML;
    }
    
    // Scroll results into view
    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
    
    console.log(`✅ Displayed ${totalResults} transcript search results`);
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

        setTimeout(initializeTranscriptSearch, 1000);

        console.log("✅ Metro AI Call Center Analytics v4.9.0 production loaded successfully");
        console.log("🔧 FIXED: Chat-stats integration, duplicate functions removed, analytics connected");
        console.log("🔮 Vector search: Enhanced relevance and semantic similarity support");
        console.log("🔧 Debug mode:", PRODUCTION_CONFIG.DEBUG_MODE ? "ENABLED" : "DISABLED");
        console.log("🎯 Transcript search: Initialization scheduled"); // ADD THIS LINE TOO
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