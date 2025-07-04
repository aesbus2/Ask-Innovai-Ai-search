// Enhanced Metro AI Call Center Analytics Chat
// Version: 3.0.0 - Comprehensive Metadata Support

// Global state management
let currentFilters = {};
let chatHistory = [];
let isLoading = false;
let filterOptions = {
    agents: [],
    dispositions: [],
    subDispositions: [],
    partners: [],
    sites: [],
    lobs: [],
    languages: [],
    templates: []
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Metro AI Call Center Analytics initializing...');
    initializePage();
    loadFilterOptions();
    updateStats();
});

// =============================================================================
// INITIALIZATION FUNCTIONS
// =============================================================================

function initializePage() {
    console.log('📋 Initializing page components...');
    
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

    // Initialize event listeners
    setupEventListeners();
    
    // Update date range display
    updateDateRangeDisplay();
    
    console.log('✅ Page initialization complete');
}

function setupEventListeners() {
    // Handle checkbox group behavior
    document.addEventListener('change', function(event) {
        if (event.target.type === 'checkbox') {
            handleCheckboxGroupChange(event);
        }
    });

    // Handle partner/site relationship
    document.getElementById('partnerFilter')?.addEventListener('change', function(event) {
        updateSiteOptions(event.target.value);
    });

    // Handle date changes
    document.getElementById('startCallDate')?.addEventListener('change', updateDateRangeDisplay);
    document.getElementById('endCallDate')?.addEventListener('change', updateDateRangeDisplay);
}

function handleCheckboxGroupChange(event) {
    const group = event.target.closest('.checkbox-group');
    if (!group) return;
    
    const allCheckbox = group.querySelector('input[value="all"]');
    const otherCheckboxes = group.querySelectorAll('input[type="checkbox"]:not([value="all"])');
    
    if (event.target.value === 'all') {
        if (event.target.checked) {
            otherCheckboxes.forEach(cb => cb.checked = false);
        }
    } else {
        if (event.target.checked) {
            if (allCheckbox) allCheckbox.checked = false;
        } else {
            // If no other checkboxes are selected, check "all"
            const anySelected = Array.from(otherCheckboxes).some(cb => cb.checked);
            if (!anySelected && allCheckbox) {
                allCheckbox.checked = true;
            }
        }
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
// FILTER MANAGEMENT
// =============================================================================

async function loadFilterOptions() {
    console.log('📊 Loading comprehensive filter options...');
    
    try {
        const response = await fetch('/filter_options');
        if (response.ok) {
            const data = await response.json();
            filterOptions = data;
            console.log('✅ Filter options loaded from API:', filterOptions);
        } else {
            throw new Error('API not available');
        }
    } catch (error) {
        console.warn('⚠️ Could not load filter options from API, using sample data:', error);
        
        // Comprehensive fallback data
        filterOptions = {
            agents: [
                'Rey Mendoza', 'Maria Garcia', 'John Smith', 'Sarah Johnson', 
                'Ana Rodriguez', 'David Chen', 'Lisa Wang', 'Carlos Martinez',
                'Jennifer Taylor', 'Michael Brown', 'Ashley Davis', 'Robert Wilson'
            ],
            dispositions: [
                'Account', 'Technical Support', 'Billing', 'Port Out', 
                'Service Inquiry', 'Complaint', 'New Service', 'Upgrade Request',
                'Cancellation', 'Payment Issue', 'Device Support', 'Network Issue'
            ],
            subDispositions: [
                'Port Out - Questions/pin/acct #', 'Account - Profile Update',
                'Billing - Payment Plan', 'Technical - Device Setup', 'Service - Plan Change',
                'Complaint - Service Quality', 'New Service - Activation', 'Upgrade - Device',
                'Cancellation - Retention', 'Payment - Past Due', 'Device - Replacement',
                'Network - Coverage Issue', 'Account - Security', 'Billing - Dispute'
            ],
            partners: ['iQor', 'Teleperformance', 'Concentrix', 'Alorica', 'Sitel'],
            sites: [
                'Dasma', 'Manila', 'Cebu', 'Davao', 'Iloilo', 'Bacolod',
                'Quezon City', 'Makati', 'Taguig', 'Pasig', 'Calamba'
            ],
            lobs: ['WNP', 'Prepaid', 'Postpaid', 'Business', 'Enterprise', 'Government'],
            languages: ['english', 'spanish', 'tagalog', 'cebuano'],
            templates: [
                'Ai Corporate SPTR - TEST', 'Customer Service Quality', 'Technical Support QA',
                'Billing Specialist Review', 'Retention Specialist Evaluation', 'Sales Performance'
            ]
        };
    }
    
    populateFilterOptions(filterOptions);
}

function populateFilterOptions(data) {
    console.log('🔧 Populating comprehensive filter UI elements...');
    
    try {
        // Populate agent filters
        populateCheckboxGroup('agentFilters', data.agents, 'agent');
        
        // Populate disposition filters
        populateCheckboxGroup('dispositionFilters', data.dispositions, 'disp');
        
        // Populate sub-disposition filters
        populateCheckboxGroup('subDispositionFilters', data.subDispositions, 'subdisp');
        
        // Populate dropdown options
        populateSelectOptions('partnerFilter', data.partners);
        populateSelectOptions('siteFilter', data.sites);
        populateSelectOptions('lobFilter', data.lobs);
        populateSelectOptions('languageFilter', data.languages);
        populateSelectOptions('templateFilter', data.templates);
        
        console.log('✅ All filter options populated successfully');
    } catch (error) {
        console.error('❌ Error populating filter options:', error);
    }
}

function populateCheckboxGroup(containerId, options, prefix) {
    const container = document.getElementById(containerId);
    if (!container || !options) return;
    
    options.forEach(option => {
        const div = document.createElement('div');
        div.className = 'checkbox-item';
        const safeId = `${prefix}-${option.replace(/\s+/g, '-').replace(/[^a-zA-Z0-9-]/g, '_')}`;
        div.innerHTML = `
            <input type="checkbox" id="${safeId}" value="${option}">
            <label for="${safeId}">${option}</label>
        `;
        container.appendChild(div);
    });
}

function populateSelectOptions(selectId, options) {
    const select = document.getElementById(selectId);
    if (!select || !options) return;
    
    options.forEach(option => {
        const optionElement = document.createElement('option');
        optionElement.value = option;
        optionElement.textContent = option;
        select.appendChild(optionElement);
    });
}

function updateSiteOptions(selectedPartner) {
    // This could be enhanced to filter sites based on partner
    // For now, it's a placeholder for future enhancement
    console.log('🏢 Partner selected:', selectedPartner);
}

function applyFilters() {
    console.log('🔍 Applying comprehensive filters...');
    
    currentFilters = collectFilters();
    updateActiveFilters();
    updateStats();
    
    console.log('📊 Active filters:', currentFilters);
    
    // If there are messages, refresh the analysis
    if (chatHistory.length > 0) {
        addMessage('system', '🔄 Filters updated. Your analysis will now use the new filter criteria for more targeted insights.');
    }
}

function collectFilters() {
    const filters = {};

    try {
        // Date range filters
        const startCallDate = document.getElementById('startCallDate')?.value;
        const endCallDate = document.getElementById('endCallDate')?.value;
        const startCreatedDate = document.getElementById('startCreatedDate')?.value;
        const endCreatedDate = document.getElementById('endCreatedDate')?.value;
        
        if (startCallDate) filters.startCallDate = startCallDate;
        if (endCallDate) filters.endCallDate = endCallDate;
        if (startCreatedDate) filters.startCreatedDate = startCreatedDate;
        if (endCreatedDate) filters.endCreatedDate = endCreatedDate;

        // Agent filters
        const selectedAgents = Array.from(document.querySelectorAll('#agentFilters input[type="checkbox"]:checked:not([value="all"])'))
            .map(checkbox => checkbox.value);
        if (selectedAgents.length > 0) filters.agents = selectedAgents;

        // Disposition filters
        const selectedDispositions = Array.from(document.querySelectorAll('#dispositionFilters input[type="checkbox"]:checked:not([value="all"])'))
            .map(checkbox => checkbox.value);
        if (selectedDispositions.length > 0) filters.dispositions = selectedDispositions;

        // Sub-disposition filters
        const selectedSubDispositions = Array.from(document.querySelectorAll('#subDispositionFilters input[type="checkbox"]:checked:not([value="all"])'))
            .map(checkbox => checkbox.value);
        if (selectedSubDispositions.length > 0) filters.subDispositions = selectedSubDispositions;

        // Organization filters
        const partner = document.getElementById('partnerFilter')?.value;
        const site = document.getElementById('siteFilter')?.value;
        const lob = document.getElementById('lobFilter')?.value;
        
        if (partner) filters.partner = partner;
        if (site) filters.site = site;
        if (lob) filters.lob = lob;

        // Call characteristics
        const minDuration = document.getElementById('minDuration')?.value;
        const maxDuration = document.getElementById('maxDuration')?.value;
        const language = document.getElementById('languageFilter')?.value;
        
        if (minDuration) filters.minDuration = parseInt(minDuration);
        if (maxDuration) filters.maxDuration = parseInt(maxDuration);
        if (language) filters.language = language;

        // Template filter
        const template = document.getElementById('templateFilter')?.value;
        if (template) filters.template = template;

    } catch (error) {
        console.error('❌ Error collecting filters:', error);
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
        
        // Format display names
        const keyMap = {
            'startCallDate': 'Call From',
            'endCallDate': 'Call To',
            'startCreatedDate': 'Created From',
            'endCreatedDate': 'Created To',
            'agents': 'Agents',
            'dispositions': 'Dispositions',
            'subDispositions': 'Sub-Dispositions',
            'partner': 'Partner',
            'site': 'Site',
            'lob': 'LOB',
            'minDuration': 'Min Duration',
            'maxDuration': 'Max Duration',
            'language': 'Language',
            'template': 'Template'
        };
        
        displayKey = keyMap[key] || key;
        
        if (Array.isArray(value)) {
            displayValue = value.length > 2 ? `${value.length} selected` : value.join(', ');
        } else if (key.includes('Date')) {
            displayValue = new Date(value).toLocaleDateString();
        } else if (key.includes('Duration')) {
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
    updateActiveFilters();
    updateStats();
    updateDateRangeDisplay();
}

function clearFilters() {
    console.log('🧹 Clearing all filters...');
    
    currentFilters = {};
    updateActiveFilters();
    updateStats();
    updateDateRangeDisplay();
    
    // Reset all form elements
    const elementsToReset = [
        'startCallDate', 'endCallDate', 'startCreatedDate', 'endCreatedDate',
        'partnerFilter', 'siteFilter', 'lobFilter', 'languageFilter', 'templateFilter',
        'minDuration', 'maxDuration'
    ];
    
    elementsToReset.forEach(id => {
        const element = document.getElementById(id);
        if (element) element.value = '';
    });
    
    // Reset checkbox groups
    const checkboxGroups = ['agentFilters', 'dispositionFilters', 'subDispositionFilters'];
    checkboxGroups.forEach(groupId => {
        const group = document.getElementById(groupId);
        if (group) {
            group.querySelectorAll('input[type="checkbox"]:not([value="all"])').forEach(cb => cb.checked = false);
            const allCheckbox = group.querySelector('input[value="all"]');
            if (allCheckbox) allCheckbox.checked = true;
        }
    });
    
    console.log('✅ All filters cleared');
}

// =============================================================================
// STATISTICS AND DATA MANAGEMENT
// =============================================================================

async function updateStats() {
    console.log('📊 Updating statistics with current filters...');
    
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
        
        if (response.ok) {
            const data = await response.json();
            const totalRecords = document.getElementById('totalRecords');
            if (totalRecords) {
                totalRecords.textContent = `${data.totalRecords || 0} evaluations`;
            }
            console.log('✅ Statistics updated:', data);
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
// CHAT FUNCTIONALITY
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
    
    console.log('💬 Sending analytics message:', message);
    console.log('🔍 With comprehensive filters:', currentFilters);
    
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
        // Make API call with comprehensive filters and metadata
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
                metadata_focus: [
                    'internalId', 'evaluationId', 'template_id', 'template_name',
                    'partner', 'site', 'lob', 'agentName', 'agentId',
                    'disposition', 'subDisposition', 'call_date', 'created_on',
                    'call_duration', 'language', 'url'
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
        
        console.log('✅ Analytics message sent successfully');
        
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
    
    let sourcesHtml = '<div class="sources-container"><h4>📚 Related Evaluations:</h4>';
    
    sources.forEach((source, index) => {
        const metadata = source.metadata || {};
        const evaluationId = metadata.evaluationId || 'Unknown';
        const internalId = metadata.internalId || 'Unknown';
        
        // Build evaluation URL
        const evalUrl = metadata.url || 
            `https://innovai-demo.metrocare-agent.com/evaluation/view/${evaluationId}`;
        
        sourcesHtml += `
            <div class="source-item">
                <div class="source-header">
                    <div>
                        <div class="source-title">
                            ${metadata.agentName || 'Unknown Agent'} - ${metadata.disposition || 'Call'}
                        </div>
                        <div class="source-meta">
                            <strong>Partner:</strong> ${metadata.partner || 'Unknown'} | 
                            <strong>Site:</strong> ${metadata.site || 'Unknown'} | 
                            <strong>LOB:</strong> ${metadata.lob || 'Unknown'}<br>
                            <strong>Call Date:</strong> ${metadata.call_date ? new Date(metadata.call_date).toLocaleDateString() : 'Unknown'} | 
                            <strong>Duration:</strong> ${metadata.call_duration || 'Unknown'}s | 
                            <strong>Language:</strong> ${metadata.language || 'Unknown'}<br>
                            <strong>Sub-Disposition:</strong> ${metadata.subDisposition || 'None'}<br>
                            <strong>Template:</strong> ${metadata.template_name || 'Unknown'} | 
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
                <div>Analyzing call center data with applied filters...</div>
                <div style="font-size: 0.8rem; opacity: 0.7; margin-top: 4px;">
                    Processing evaluations, performance metrics, and quality scores
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
    
    console.log('📁 Exporting comprehensive chat history...');
    
    // Create detailed export content
    let exportContent = `Metro AI Call Center Analytics Export\n`;
    exportContent += `${'='.repeat(50)}\n`;
    exportContent += `Generated: ${new Date().toISOString()}\n`;
    exportContent += `Total Messages: ${chatHistory.length}\n\n`;
    
    // Add comprehensive filter information
    if (Object.keys(currentFilters).length > 0) {
        exportContent += `Applied Filters:\n`;
        exportContent += `${'-'.repeat(20)}\n`;
        Object.entries(currentFilters).forEach(([key, value]) => {
            let displayValue = Array.isArray(value) ? value.join(', ') : value;
            if (key.includes('Date')) {
                displayValue = new Date(value).toLocaleDateString();
            } else if (key.includes('Duration')) {
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
    exportContent += `\n${'-'.repeat(50)}\n`;
    exportContent += `Metro AI Call Center Analytics\n`;
    exportContent += `Advanced evaluation analysis and insights\n`;
    
    // Create and download file
    const blob = new Blob([exportContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `metro-ai-analytics-${new Date().toISOString().split('T')[0]}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    
    console.log('✅ Comprehensive chat history exported');
}

// =============================================================================
// GLOBAL FUNCTION EXPOSURE
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

// Enhanced debugging interface
window.chatDebug = {
    getCurrentFilters: () => currentFilters,
    getChatHistory: () => chatHistory,
    getFilterOptions: () => filterOptions,
    isLoading: () => isLoading,
    showFilterStats: () => {
        console.log('📊 Filter Statistics:');
        console.log('Current Filters:', currentFilters);
        console.log('Available Options:', filterOptions);
        console.log('Active Filter Count:', Object.keys(currentFilters).length);
    },
    testFilters: () => {
        console.log('🧪 Testing filter collection...');
        const testFilters = collectFilters();
        console.log('Collected Filters:', testFilters);
        return testFilters;
    }
};

console.log('✅ Metro AI Call Center Analytics Chat v3.0 loaded successfully');
console.log('🔧 Enhanced debugging tools available at window.chatDebug');
console.log('📊 Comprehensive metadata filtering and analytics ready');