// Enhanced Metro AI Call Center Analytics Chat
// Version: 4.0.0 - Aligned with Call Detail Metadata Structure

// Global state management
let currentFilters = {};
let chatHistory = [];
let isLoading = false;
let filterOptions = {
    programs: [],
    partners: [],
    sites: [],
    lobs: [],
    callDispositions: [],
    callSubDispositions: [],
    agentDispositions: [],
    agentSubDispositions: [],
    agentNames: [],
    languages: [],
    callTypes: []
};

// Hierarchical filter cache
let hierarchyCache = {
    program_partners: {},
    partner_sites: {},
    site_lobs: {},
    disposition_subdispositions: {}
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Metro AI Call Center Analytics v4.0 initializing...');
    try {
        initializePage();
        loadDynamicFilterOptions();
        updateStats();
    } catch (error) {
        console.error('❌ Initialization error:', error);
    }
});

// =============================================================================
// INITIALIZATION FUNCTIONS
// =============================================================================

function initializePage() {
    console.log('📋 Initializing page with aligned metadata structure...');
    
    // Set default date range to last 30 days
    const today = new Date();
    const thirtyDaysAgo = new Date(today.getTime() - (30 * 24 * 60 * 60 * 1000));
    
    const endCallDate = document.getElementById('endCallDate');
    const startCallDate = document.getElementById('startCallDate');
    
    if (endCallDate) endCallDate.value = today.toISOString().split('T')[0];
    if (startCallDate) startCallDate.value = thirtyDaysAgo.toISOString().split('T')[0];

    // Auto-resize textarea
    const chatInput = document.getElementById('chatInput');
    if (chatInput) {
        chatInput.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = this.scrollHeight + 'px';
        });
    }

    // Initialize event listeners for new filter structure
    setupEventListeners();
    
    // Update date range display
    updateDateRangeDisplay();
    
    console.log('✅ Page initialization complete with metadata alignment');
}

function setupEventListeners() {
    // Handle date changes
    const startCallDate = document.getElementById('startCallDate');
    const endCallDate = document.getElementById('endCallDate');
    
    if (startCallDate) {
        startCallDate.addEventListener('change', updateDateRangeDisplay);
    }
    if (endCallDate) {
        endCallDate.addEventListener('change', updateDateRangeDisplay);
    }

    // Handle ID field validation
    setupIdFieldValidation();
    
    // Handle agent name autocomplete
    setupAgentNameAutocomplete();
}

function setupIdFieldValidation() {
    // Phone number validation
    const phoneInput = document.getElementById('phoneNumberFilter');
    if (phoneInput) {
        phoneInput.addEventListener('input', function(e) {
            // Allow only numbers and common phone formats
            this.value = this.value.replace(/[^\d\-\(\)\+\s]/g, '');
        });
    }

    // Contact ID validation (numeric)
    const contactIdInput = document.getElementById('contactIdFilter');
    if (contactIdInput) {
        contactIdInput.addEventListener('input', function(e) {
            this.value = this.value.replace(/[^\d]/g, '');
        });
    }

    // UCID validation (alphanumeric)
    const ucidInput = document.getElementById('ucidFilter');
    if (ucidInput) {
        ucidInput.addEventListener('input', function(e) {
            this.value = this.value.replace(/[^a-zA-Z0-9]/g, '');
        });
    }
}

function setupAgentNameAutocomplete() {
    const agentNameInput = document.getElementById('agentNameFilter');
    const datalist = document.getElementById('agentNamesList');
    
    if (agentNameInput && datalist) {
        agentNameInput.addEventListener('input', function(e) {
            const value = this.value.toLowerCase();
            
            // Filter agent names based on input
            const filteredAgents = filterOptions.agentNames.filter(name => 
                name.toLowerCase().includes(value)
            );
            
            // Update datalist options
            datalist.innerHTML = '';
            filteredAgents.slice(0, 10).forEach(agent => {
                const option = document.createElement('option');
                option.value = agent;
                datalist.appendChild(option);
            });
        });
    }
}

function updateDateRangeDisplay() {
    const startDate = document.getElementById('startCallDate')?.value;
    const endDate = document.getElementById('endCallDate')?.value;
    const dateRangeDisplay = document.getElementById('dateRange');
    
    if (dateRangeDisplay) {
        if (startDate && endDate) {
            const start = new Date(startDate).toLocaleDateString();
            const end = new Date(endDate).toLocaleDateString();
            dateRangeDisplay.textContent = `${start} - ${end}`;
        } else if (startDate) {
            dateRangeDisplay.textContent = `From ${new Date(startDate).toLocaleDateString()}`;
        } else if (endDate) {
            dateRangeDisplay.textContent = `Until ${new Date(endDate).toLocaleDateString()}`;
        } else {
            dateRangeDisplay.textContent = 'All dates';
        }
    }
}

// =============================================================================
// DYNAMIC FILTER LOADING
// =============================================================================

async function loadDynamicFilterOptions() {
    console.log('📊 Loading dynamic filter options from metadata...');
    
    try {
        // Show loading indicators
        showFilterLoadingState();
        
        // Load filter options from API endpoint
        const response = await fetch('/filter_options_metadata');
        if (response.ok) {
            const data = await response.json();
            filterOptions = data;
            hierarchyCache = data.hierarchy || {};
            console.log('✅ Dynamic filter options loaded:', filterOptions);
        } else {
            throw new Error('API not available');
        }
    } catch (error) {
        console.warn('⚠️ Could not load filter options from API, using sample data:', error);
        
        // Enhanced fallback data based on call details structure
        filterOptions = {
            programs: [
                'Ai Corporate SPTR - TEST',
                'Customer Service Quality',
                'Technical Support QA',
                'Billing Specialist Review'
            ],
            partners: [
                'iQor', 'Teleperformance', 'Concentrix', 'Alorica', 'Sitel'
            ],
            sites: [
                'Dasma', 'Manila', 'Cebu', 'Davao', 'Iloilo', 'Bacolod',
                'Quezon City', 'Makati', 'Taguig', 'Pasig'
            ],
            lobs: [
                'WNP', 'Prepaid', 'Postpaid', 'Business', 'Enterprise'
            ],
            callDispositions: [
                'Account', 'Technical Support', 'Billing', 'Port Out',
                'Service Inquiry', 'Complaint', 'Equipment', 'Rate Plan'
            ],
            callSubDispositions: [
                'Rate Plan Or Plan Fit Analysis',
                'Port Out - Questions/pin/acct #',
                'Account - Profile Update',
                'Billing - Payment Plan',
                'Technical - Device Setup',
                'Equipment - Troubleshooting'
            ],
            agentDispositions: [
                'Equipment', 'Account Management', 'Technical Support',
                'Customer Service', 'Billing Support'
            ],
            agentSubDispositions: [
                'NA', 'Resolved', 'Escalated', 'Follow-up Required', 'Transferred'
            ],
            agentNames: [
                'Rey Mendoza', 'Maria Garcia', 'John Smith', 'Sarah Johnson',
                'Ana Rodriguez', 'David Chen', 'Lisa Wang', 'Carlos Martinez'
            ],
            languages: [
                'English', 'Spanish', 'Tagalog', 'Cebuano'
            ],
            callTypes: [
                'Direct Connect', 'Transfer', 'Inbound', 'Outbound'
            ]
        };
    }
    
    populateFilterOptions(filterOptions);
    hideFilterLoadingState();
}

function showFilterLoadingState() {
    // Show loading in program dropdown
    const programLoading = document.getElementById('programLoading');
    const dispositionLoading = document.getElementById('dispositionLoading');
    
    if (programLoading) programLoading.style.display = 'block';
    if (dispositionLoading) dispositionLoading.style.display = 'block';
}

function hideFilterLoadingState() {
    const programLoading = document.getElementById('programLoading');
    const dispositionLoading = document.getElementById('dispositionLoading');
    
    if (programLoading) programLoading.style.display = 'none';
    if (dispositionLoading) dispositionLoading.style.display = 'none';
}

function populateFilterOptions(data) {
    console.log('🔧 Populating filter UI with metadata structure...');
    
    try {
        // Populate hierarchical dropdowns
        populateSelectOptions('programFilter', data.programs);
        populateSelectOptions('partnerFilter', data.partners);
        populateSelectOptions('siteFilter', data.sites);
        populateSelectOptions('lobFilter', data.lobs);
        
        // Populate call classification dropdowns
        populateSelectOptions('callDispositionFilter', data.callDispositions);
        populateSelectOptions('callSubDispositionFilter', data.callSubDispositions);
        populateSelectOptions('agentDispositionFilter', data.agentDispositions);
        populateSelectOptions('agentSubDispositionFilter', data.agentSubDispositions);
        populateSelectOptions('callTypeFilter', data.callTypes);
        populateSelectOptions('languageFilter', data.languages);
        
        // Populate agent names datalist
        populateDatalistOptions('agentNamesList', data.agentNames);
        
        console.log('✅ All filter options populated with metadata alignment');
    } catch (error) {
        console.error('❌ Error populating filter options:', error);
    }
}

function populateSelectOptions(selectId, options) {
    const select = document.getElementById(selectId);
    if (!select || !options) return;
    
    // Clear existing options (except the first "All" option)
    const firstOption = select.firstElementChild;
    select.innerHTML = '';
    if (firstOption) select.appendChild(firstOption);
    
    options.forEach(option => {
        const optionElement = document.createElement('option');
        optionElement.value = option;
        optionElement.textContent = option;
        select.appendChild(optionElement);
    });
}

function populateDatalistOptions(datalistId, options) {
    const datalist = document.getElementById(datalistId);
    if (!datalist || !options) return;
    
    datalist.innerHTML = '';
    options.forEach(option => {
        const optionElement = document.createElement('option');
        optionElement.value = option;
        datalist.appendChild(optionElement);
    });
}

// =============================================================================
// HIERARCHICAL FILTERING
// =============================================================================

function updateHierarchyFilters(changedLevel) {
    console.log(`🔄 Updating hierarchy filters from level: ${changedLevel}`);
    
    const program = document.getElementById('programFilter')?.value;
    const partner = document.getElementById('partnerFilter')?.value;
    const site = document.getElementById('siteFilter')?.value;
    
    try {
        switch (changedLevel) {
            case 'program':
                updatePartnerOptions(program);
                clearDownstreamFilters(['partner', 'site', 'lob']);
                break;
            case 'partner':
                updateSiteOptions(program, partner);
                clearDownstreamFilters(['site', 'lob']);
                break;
            case 'site':
                updateLobOptions(program, partner, site);
                clearDownstreamFilters(['lob']);
                break;
        }
    } catch (error) {
        console.error('❌ Error updating hierarchy filters:', error);
    }
}

function updatePartnerOptions(selectedProgram) {
    const partnerSelect = document.getElementById('partnerFilter');
    if (!partnerSelect) return;
    
    // If hierarchy cache exists, use it; otherwise show all partners
    let availablePartners = filterOptions.partners;
    
    if (hierarchyCache.program_partners && selectedProgram) {
        availablePartners = hierarchyCache.program_partners[selectedProgram] || filterOptions.partners;
    }
    
    populateSelectOptions('partnerFilter', availablePartners);
    console.log(`🔗 Updated partners for program: ${selectedProgram}`);
}

function updateSiteOptions(selectedProgram, selectedPartner) {
    const siteSelect = document.getElementById('siteFilter');
    if (!siteSelect) return;
    
    let availableSites = filterOptions.sites;
    
    if (hierarchyCache.partner_sites && selectedPartner) {
        availableSites = hierarchyCache.partner_sites[selectedPartner] || filterOptions.sites;
    }
    
    populateSelectOptions('siteFilter', availableSites);
    console.log(`🔗 Updated sites for partner: ${selectedPartner}`);
}

function updateLobOptions(selectedProgram, selectedPartner, selectedSite) {
    const lobSelect = document.getElementById('lobFilter');
    if (!lobSelect) return;
    
    let availableLobs = filterOptions.lobs;
    
    if (hierarchyCache.site_lobs && selectedSite) {
        availableLobs = hierarchyCache.site_lobs[selectedSite] || filterOptions.lobs;
    }
    
    populateSelectOptions('lobFilter', availableLobs);
    console.log(`🔗 Updated LOBs for site: ${selectedSite}`);
}

function updateSubDispositions() {
    const disposition = document.getElementById('callDispositionFilter')?.value;
    const subDispositionSelect = document.getElementById('callSubDispositionFilter');
    
    if (!subDispositionSelect) return;
    
    let availableSubDispositions = filterOptions.callSubDispositions;
    
    if (hierarchyCache.disposition_subdispositions && disposition) {
        availableSubDispositions = hierarchyCache.disposition_subdispositions[disposition] || filterOptions.callSubDispositions;
    }
    
    populateSelectOptions('callSubDispositionFilter', availableSubDispositions);
    console.log(`🔗 Updated sub-dispositions for disposition: ${disposition}`);
}

function clearDownstreamFilters(levels) {
    levels.forEach(level => {
        let selectId;
        switch (level) {
            case 'partner':
                selectId = 'partnerFilter';
                break;
            case 'site':
                selectId = 'siteFilter';
                break;
            case 'lob':
                selectId = 'lobFilter';
                break;
        }
        
        const select = document.getElementById(selectId);
        if (select) {
            select.value = '';
        }
    });
}

// =============================================================================
// FILTER MANAGEMENT
// =============================================================================

function applyFilters() {
    console.log('🔍 Applying aligned metadata filters...');
    
    currentFilters = collectAlignedFilters();
    updateActiveFilters();
    updateStats();
    
    console.log('📊 Active filters with metadata alignment:', currentFilters);
    
    // If there are messages, refresh the analysis
    if (chatHistory.length > 0) {
        addMessage('system', '🔄 Filters updated. Your analysis will now use the aligned filter criteria for more targeted insights.');
    }
}

function collectAlignedFilters() {
    const filters = {};

    try {
        // Date range filters
        const startCallDate = document.getElementById('startCallDate')?.value;
        const endCallDate = document.getElementById('endCallDate')?.value;
        const startCreatedDate = document.getElementById('startCreatedDate')?.value;
        const endCreatedDate = document.getElementById('endCreatedDate')?.value;
        
        if (startCallDate) filters.call_date_start = startCallDate;
        if (endCallDate) filters.call_date_end = endCallDate;
        if (startCreatedDate) filters.created_date_start = startCreatedDate;
        if (endCreatedDate) filters.created_date_end = endCreatedDate;

        // Organizational hierarchy filters
        const program = document.getElementById('programFilter')?.value;
        const partner = document.getElementById('partnerFilter')?.value;
        const site = document.getElementById('siteFilter')?.value;
        const lob = document.getElementById('lobFilter')?.value;
        
        if (program) filters.program = program;
        if (partner) filters.partner = partner;
        if (site) filters.site = site;
        if (lob) filters.lob = lob;

        // Call identifier filters
        const phoneNumber = document.getElementById('phoneNumberFilter')?.value?.trim();
        const contactId = document.getElementById('contactIdFilter')?.value?.trim();
        const ucid = document.getElementById('ucidFilter')?.value?.trim();
        const userId = document.getElementById('userIdFilter')?.value?.trim();
        
        if (phoneNumber) filters.phone_number = phoneNumber;
        if (contactId) filters.contact_id = contactId;
        if (ucid) filters.ucid = ucid;
        if (userId) filters.user_id = userId;

        // Call classification filters
        const callDisposition = document.getElementById('callDispositionFilter')?.value;
        const callSubDisposition = document.getElementById('callSubDispositionFilter')?.value;
        const agentDisposition = document.getElementById('agentDispositionFilter')?.value;
        const agentSubDisposition = document.getElementById('agentSubDispositionFilter')?.value;
        const callType = document.getElementById('callTypeFilter')?.value;
        
        if (callDisposition) filters.call_disposition = callDisposition;
        if (callSubDisposition) filters.call_sub_disposition = callSubDisposition;
        if (agentDisposition) filters.agent_disposition = agentDisposition;
        if (agentSubDisposition) filters.agent_sub_disposition = agentSubDisposition;
        if (callType) filters.call_type = callType;

        // Agent performance filters
        const agentName = document.getElementById('agentNameFilter')?.value?.trim();
        if (agentName) filters.agent_name = agentName;

        // Call characteristics
        const minDuration = document.getElementById('minDuration')?.value;
        const maxDuration = document.getElementById('maxDuration')?.value;
        const language = document.getElementById('languageFilter')?.value;
        
        if (minDuration) filters.min_duration = parseInt(minDuration);
        if (maxDuration) filters.max_duration = parseInt(maxDuration);
        if (language) filters.call_language = language;

        // Evaluation metadata filters
        const evaluationId = document.getElementById('evaluationIdFilter')?.value?.trim();
        const internalId = document.getElementById('internalIdFilter')?.value?.trim();
        const templateId = document.getElementById('templateIdFilter')?.value?.trim();
        
        if (evaluationId) filters.evaluation_id = evaluationId;
        if (internalId) filters.internal_id = internalId;
        if (templateId) filters.template_id = templateId;

    } catch (error) {
        console.error('❌ Error collecting aligned filters:', error);
    }

    return filters;
}

function updateActiveFilters() {
    const activeFiltersDiv = document.getElementById('activeFilters');
    if (!activeFiltersDiv) return;
    
    activeFiltersDiv.innerHTML = '';

    const filterCount = Object.keys(currentFilters).length;
    const activeFiltersCount = document.getElementById('activeFiltersCount');
    if (activeFiltersCount) {
        activeFiltersCount.textContent = `${filterCount} filters`;
    }

    Object.entries(currentFilters).forEach(([key, value]) => {
        const tag = document.createElement('span');
        tag.className = 'filter-tag';
        
        let displayValue = value;
        let displayKey = key;
        
        // Format display names for metadata alignment
        const keyMap = {
            'call_date_start': 'Call From',
            'call_date_end': 'Call To',
            'created_date_start': 'Created From',
            'created_date_end': 'Created To',
            'program': 'Program',
            'partner': 'Partner',
            'site': 'Site',
            'lob': 'LOB',
            'phone_number': 'Phone',
            'contact_id': 'Contact ID',
            'ucid': 'UCID',
            'user_id': 'User ID',
            'call_disposition': 'Call Disposition',
            'call_sub_disposition': 'Call Sub-Disposition',
            'agent_disposition': 'Agent Disposition',
            'agent_sub_disposition': 'Agent Sub-Disposition',
            'call_type': 'Call Type',
            'agent_name': 'Agent',
            'min_duration': 'Min Duration',
            'max_duration': 'Max Duration',
            'call_language': 'Language',
            'evaluation_id': 'Eval ID',
            'internal_id': 'Internal ID',
            'template_id': 'Template ID'
        };
        
        displayKey = keyMap[key] || key;
        
        if (key.includes('date')) {
            displayValue = new Date(value).toLocaleDateString();
        } else if (key.includes('duration')) {
            displayValue = `${value}s`;
        }
        
        tag.innerHTML = `
            ${displayKey}: ${displayValue}
            <span class="material-icons remove" onclick="removeFilter('${key}')">close</span>
        `;
        activeFiltersDiv.appendChild(tag);
    });
}

function removeFilter(filterKey) {
    console.log(`🗑️ Removing filter: ${filterKey}`);
    delete currentFilters[filterKey];
    
    // Clear the corresponding UI element
    const fieldMap = {
        'call_date_start': 'startCallDate',
        'call_date_end': 'endCallDate',
        'created_date_start': 'startCreatedDate',
        'created_date_end': 'endCreatedDate',
        'program': 'programFilter',
        'partner': 'partnerFilter',
        'site': 'siteFilter',
        'lob': 'lobFilter',
        'phone_number': 'phoneNumberFilter',
        'contact_id': 'contactIdFilter',
        'ucid': 'ucidFilter',
        'user_id': 'userIdFilter',
        'call_disposition': 'callDispositionFilter',
        'call_sub_disposition': 'callSubDispositionFilter',
        'agent_disposition': 'agentDispositionFilter',
        'agent_sub_disposition': 'agentSubDispositionFilter',
        'call_type': 'callTypeFilter',
        'agent_name': 'agentNameFilter',
        'min_duration': 'minDuration',
        'max_duration': 'maxDuration',
        'call_language': 'languageFilter',
        'evaluation_id': 'evaluationIdFilter',
        'internal_id': 'internalIdFilter',
        'template_id': 'templateIdFilter'
    };
    
    const fieldId = fieldMap[filterKey];
    if (fieldId) {
        const element = document.getElementById(fieldId);
        if (element) {
            element.value = '';
        }
    }
    
    updateActiveFilters();
    updateStats();
    updateDateRangeDisplay();
}

function clearFilters() {
    console.log('🧹 Clearing all aligned filters...');
    
    currentFilters = {};
    updateActiveFilters();
    updateStats();
    updateDateRangeDisplay();
    
    // Reset all form elements
    const elementsToReset = [
        'startCallDate', 'endCallDate', 'startCreatedDate', 'endCreatedDate',
        'programFilter', 'partnerFilter', 'siteFilter', 'lobFilter',
        'phoneNumberFilter', 'contactIdFilter', 'ucidFilter', 'userIdFilter',
        'callDispositionFilter', 'callSubDispositionFilter', 'agentDispositionFilter',
        'agentSubDispositionFilter', 'callTypeFilter', 'agentNameFilter',
        'minDuration', 'maxDuration', 'languageFilter',
        'evaluationIdFilter', 'internalIdFilter', 'templateIdFilter'
    ];
    
    elementsToReset.forEach(id => {
        const element = document.getElementById(id);
        if (element) element.value = '';
    });
    
    console.log('✅ All aligned filters cleared');
}

// =============================================================================
// STATISTICS AND DATA MANAGEMENT
// =============================================================================

async function updateStats() {
    console.log('📊 Updating statistics with aligned filters...');
    
    try {
        const response = await fetch('/analytics/stats', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filters: currentFilters,
                filter_version: '4.0'
            })
        });
        
        if (response.ok) {
            const data = await response.json();
            const totalRecords = document.getElementById('totalRecords');
            if (totalRecords) {
                totalRecords.textContent = `${data.totalRecords || 0} evaluations`;
            }
            console.log('✅ Statistics updated with aligned filters:', data);
        } else {
            throw new Error('Stats API not available');
        }
    } catch (error) {
        console.warn('⚠️ Could not fetch real stats, using simulated data:', error);
        
        // Fallback to simulated stats
        const recordCount = Math.floor(Math.random() * 1000) + 100;
        const totalRecords = document.getElementById('totalRecords');
        if (totalRecords) {
            totalRecords.textContent = `${recordCount} evaluations`;
        }
    }
}

// =============================================================================
// CHAT FUNCTIONALITY (Enhanced for metadata alignment)
// =============================================================================

function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    if (sidebar) {
        sidebar.classList.toggle('open');
    }
}

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
    const input = document.getElementById('chatInput');
    if (!input) return;
    
    const message = input.value.trim();
    
    if (!message || isLoading) return;
    
    input.value = '';
    input.style.height = 'auto';
    
    console.log('💬 Sending analytics message with aligned metadata:', message);
    console.log('🔍 With aligned filters:', currentFilters);
    
    // Hide welcome screen, show chat
    const welcomeScreen = document.getElementById('welcomeScreen');
    const chatMessages = document.getElementById('chatMessages');
    
    if (welcomeScreen) welcomeScreen.classList.add('hidden');
    if (chatMessages) chatMessages.classList.remove('hidden');
    
    // Add user message
    addMessage('user', message);
    
    // Show loading
    isLoading = true;
    updateSendButton();
    addLoadingMessage();
    
    try {
        // Make API call with aligned metadata and filters
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                history: chatHistory,
                filters: currentFilters,
                analytics: true,
                filter_version: '4.0',
                metadata_focus: [
                    'evaluationId', 'internalId', 'template_id', 'template_name',
                    'partner', 'site', 'lob', 'agentName', 'call_date',
                    'call_disposition', 'call_sub_disposition', 'agent_disposition',
                    'agent_sub_disposition', 'call_duration', 'call_language',
                    'call_type', 'phone_number', 'contact_id', 'ucid'
                ]
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // Remove loading message
        removeLoadingMessage();
        
        // Add assistant response
        const reply = data.reply || 'Sorry, I couldn\'t process your request.';
        addMessage('assistant', reply);
        
        // If there are related evaluations, show them
        if (data.sources && data.sources.length > 0) {
            addSourcesMessage(data.sources);
        }
        
        console.log('✅ Analytics message sent successfully with metadata alignment');
        
    } catch (error) {
        console.error('❌ Error sending message:', error);
        removeLoadingMessage();
        addMessage('assistant', 'Sorry, there was an error processing your analytics request. Please try again.');
    } finally {
        isLoading = false;
        updateSendButton();
    }
}

function addMessage(sender, content) {
    const messagesContainer = document.getElementById('chatMessages');
    if (!messagesContainer) return;
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    const timestamp = new Date().toLocaleTimeString();
    
    // Format content based on sender
    let formattedContent = content;
    if (sender === 'assistant') {
        formattedContent = formatAssistantMessage(content);
    }
    
    messageDiv.innerHTML = `
        <div class="message-content">
            ${formattedContent}
        </div>
        <div class="message-meta">
            <span>${timestamp}</span>
            ${sender === 'assistant' ? '<span class="material-icons">smart_toy</span>' : ''}
            ${sender === 'user' ? '<span class="material-icons">person</span>' : ''}
            ${sender === 'system' ? '<span class="material-icons">info</span>' : ''}
        </div>
    `;
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
    
    // Add to history (but not system messages)
    if (sender !== 'system') {
        chatHistory.push({
            role: sender === 'user' ? 'user' : 'assistant',
            content: content
        });
    }
}

function formatAssistantMessage(content) {
    // Enhanced formatting for assistant messages
    return content
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\*(.*?)\*/g, '<em>$1</em>')
        .replace(/\n/g, '<br>');
}

function addSourcesMessage(sources) {
    const messagesContainer = document.getElementById('chatMessages');
    if (!messagesContainer) return;
    
    const sourceDiv = document.createElement('div');
    sourceDiv.className = 'message assistant';
    
    let sourcesHtml = '<div class="sources-container"><h4>📚 Related Call Evaluations:</h4>';
    
    sources.forEach((source, index) => {
        const metadata = source.metadata || {};
        const evaluationId = metadata.evaluationId || metadata.evaluation_id || 'Unknown';
        const internalId = metadata.internalId || metadata.internal_id || 'Unknown';
        
        // Build evaluation URL
        const evalUrl = metadata.url || 
            `https://innovai-demo.metrocare-agent.com/evaluation/view/${evaluationId}`;
        
        sourcesHtml += `
            <div class="source-item">
                <div class="source-header">
                    <div>
                        <div class="source-title">
                            ${metadata.agentName || metadata.agent_name || 'Unknown Agent'} - ${metadata.call_disposition || metadata.disposition || 'Call'}
                        </div>
                        <div class="source-meta">
                            <strong>Call Date:</strong> ${metadata.call_date ? new Date(metadata.call_date).toLocaleDateString() : 'Unknown'} | 
                            <strong>Duration:</strong> ${metadata.call_duration || 'Unknown'}s | 
                            <strong>Language:</strong> ${metadata.call_language || metadata.language || 'Unknown'}<br>
                            <strong>Partner:</strong> ${metadata.partner || 'Unknown'} | 
                            <strong>Site:</strong> ${metadata.site || 'Unknown'} | 
                            <strong>LOB:</strong> ${metadata.lob || 'Unknown'}<br>
                            <strong>Call Type:</strong> ${metadata.call_type || 'Unknown'} | 
                            <strong>Sub-Disposition:</strong> ${metadata.call_sub_disposition || metadata.subDisposition || 'None'}<br>
                            ${metadata.phone_number ? `<strong>Phone:</strong> ${metadata.phone_number} | ` : ''}
                            ${metadata.contact_id ? `<strong>Contact ID:</strong> ${metadata.contact_id} | ` : ''}
                            <strong>Internal ID:</strong> ${internalId}
                        </div>
                    </div>
                    <div class="source-actions">
                        <a href="${evalUrl}" target="_blank" class="source-link">
                            <span class="material-icons">open_in_new</span>
                            View Evaluation
                        </a>
                    </div>
                </div>
                <span class="source-text">${source.text?.substring(0, 300) || 'No text available'}...</span>
            </div>
        `;
    });
    
    sourcesHtml += '</div>';
    
    sourceDiv.innerHTML = `
        <div class="message-content">
            ${sourcesHtml}
        </div>
        <div class="message-meta">
            <span>${new Date().toLocaleTimeString()}</span>
            <span class="material-icons">source</span>
        </div>
    `;
    
    messagesContainer.appendChild(sourceDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function addLoadingMessage() {
    const messagesContainer = document.getElementById('chatMessages');
    if (!messagesContainer) return;
    
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message assistant';
    loadingDiv.id = 'loadingMessage';
    loadingDiv.innerHTML = `
        <div class="message-content">
            <div class="loading-indicator">
                <div class="spinner"></div>
                <div>Analyzing call center data with aligned metadata filters...</div>
                <div style="font-size: 0.8rem; opacity: 0.7; margin-top: 4px;">
                    Processing evaluations, call details, dispositions, and performance metrics
                </div>
            </div>
        </div>
    `;
    messagesContainer.appendChild(loadingDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

function removeLoadingMessage() {
    const loadingMessage = document.getElementById('loadingMessage');
    if (loadingMessage) {
        loadingMessage.remove();
    }
}

function updateSendButton() {
    const sendBtn = document.getElementById('sendBtn');
    if (!sendBtn) return;
    
    sendBtn.disabled = isLoading;
    sendBtn.innerHTML = isLoading ? 
        '<div class="spinner"></div> Analyzing...' : 
        '<span class="material-icons">analytics</span> Analyze';
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

function clearChat() {
    console.log('🧹 Clearing chat history...');
    
    chatHistory = [];
    const chatMessages = document.getElementById('chatMessages');
    const welcomeScreen = document.getElementById('welcomeScreen');
    
    if (chatMessages) {
        chatMessages.innerHTML = '';
        chatMessages.classList.add('hidden');
    }
    
    if (welcomeScreen) {
        welcomeScreen.classList.remove('hidden');
    }
    
    console.log('✅ Chat cleared');
}

function exportChat() {
    if (chatHistory.length === 0) {
        alert('No chat history to export');
        return;
    }
    
    console.log('📁 Exporting comprehensive chat history with metadata alignment...');
    
    // Create detailed export content
    let exportContent = `Metro AI Call Center Analytics Export (v4.0)\n`;
    exportContent += `${'='.repeat(55)}\n`;
    exportContent += `Generated: ${new Date().toISOString()}\n`;
    exportContent += `Total Messages: ${chatHistory.length}\n`;
    exportContent += `Filter Version: Aligned with Call Detail Metadata\n\n`;
    
    // Add comprehensive filter information
    if (Object.keys(currentFilters).length > 0) {
        exportContent += `Applied Filters (Metadata Aligned):\n`;
        exportContent += `${'-'.repeat(35)}\n`;
        Object.entries(currentFilters).forEach(([key, value]) => {
            let displayValue = value;
            if (key.includes('date')) {
                displayValue = new Date(value).toLocaleDateString();
            } else if (key.includes('duration')) {
                displayValue = `${value} seconds`;
            }
            exportContent += `  ${key}: ${displayValue}\n`;
        });
        exportContent += '\n';
    }
    
    // Add chat history
    exportContent += `Analytics Conversation:\n`;
    exportContent += `${'-'.repeat(30)}\n\n`;
    
    chatHistory.forEach((msg, index) => {
        exportContent += `${index + 1}. ${msg.role.toUpperCase()}:\n`;
        exportContent += `${msg.content}\n\n`;
    });
    
    // Add footer
    exportContent += `\n${'-'.repeat(55)}\n`;
    exportContent += `Metro AI Call Center Analytics v4.0\n`;
    exportContent += `Advanced evaluation analysis with aligned metadata filtering\n`;
    
    // Create and download file
    const blob = new Blob([exportContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `metro-ai-analytics-v4-${new Date().toISOString().split('T')[0]}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    
    console.log('✅ Comprehensive chat history exported with metadata alignment');
}

// =============================================================================
// GLOBAL FUNCTION EXPOSURE - MUST BE AT END
// =============================================================================

// Expose functions to global scope for HTML event handlers
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

// Enhanced debugging interface
window.chatDebug = {
    getCurrentFilters: () => currentFilters,
    getChatHistory: () => chatHistory,
    getFilterOptions: () => filterOptions,
    getHierarchyCache: () => hierarchyCache,
    isLoading: () => isLoading,
    testFunctions: () => {
        const functions = ['toggleSidebar', 'applyFilters', 'clearFilters', 'askQuestion', 'sendMessage'];
        const results = {};
        functions.forEach(func => {
            results[func] = typeof window[func] === 'function' ? '✅ Available' : '❌ Missing';
        });
        console.log('🔧 Function availability:', results);
        return results;
    },
    showFilterStats: () => {
        console.log('📊 Filter Statistics (v4.0):');
        console.log('Current Filters:', currentFilters);
        console.log('Available Options:', filterOptions);
        console.log('Hierarchy Cache:', hierarchyCache);
        console.log('Active Filter Count:', Object.keys(currentFilters).length);
    },
    testFilters: () => {
        console.log('🧪 Testing aligned filter collection...');
        const testFilters = collectAlignedFilters();
        console.log('Collected Filters:', testFilters);
        return testFilters;
    },
    validateMetadataAlignment: () => {
        console.log('🔍 Validating metadata alignment...');
        const expectedFields = [
            'call_date_start', 'call_date_end', 'program', 'partner', 'site', 'lob',
            'phone_number', 'contact_id', 'ucid', 'call_disposition', 'call_sub_disposition',
            'agent_disposition', 'call_type', 'call_language', 'agent_name'
        ];
        
        const currentFields = Object.keys(currentFilters);
        const alignmentStatus = {
            total_possible: expectedFields.length,
            currently_used: currentFields.length,
            aligned_fields: currentFields.filter(field => expectedFields.includes(field)),
            missing_fields: expectedFields.filter(field => !currentFields.includes(field))
        };
        
        console.log('📋 Metadata Alignment Status:', alignmentStatus);
        return alignmentStatus;
    }
};

console.log('✅ Metro AI Call Center Analytics Chat v4.0 loaded successfully');
console.log('🔧 Enhanced debugging tools available at window.chatDebug');
console.log('📊 Metadata-aligned filtering and hierarchical organization ready');
console.log('📋 Call detail structure: Phone, Contact ID, UCID, Dispositions, Duration, Language, etc.');

// Test function availability on load
setTimeout(() => {
    if (window.chatDebug && window.chatDebug.testFunctions) {
        window.chatDebug.testFunctions();
    }
}, 1000);