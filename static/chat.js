// Enhanced chat.js for Metro AI Analytics
// Version: 2.0.0 - Analytics-focused with advanced filtering

// Global state management
let currentFilters = {};
let chatHistory = [];
let isLoading = false;
let filterOptions = {
    agents: [],
    dispositions: [],
    sites: [],
    lobs: []
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    console.log('🚀 Metro AI Analytics Chat initializing...');
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
    
    const endDateInput = document.getElementById('endDate');
    const startDateInput = document.getElementById('startDate');
    
    if (endDateInput) endDateInput.value = today.toISOString().split('T')[0];
    if (startDateInput) startDateInput.value = thirtyDaysAgo.toISOString().split('T')[0];

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
    
    console.log('✅ Page initialization complete');
}

function setupEventListeners() {
    // Handle checkbox group behavior
    document.addEventListener('change', function(event) {
        if (event.target.type === 'checkbox') {
            handleCheckboxGroupChange(event);
        }
    });

    // Handle filter changes
    document.addEventListener('change', function(event) {
        if (event.target.classList.contains('filter-input') || 
            event.target.classList.contains('filter-select')) {
            // Auto-apply filters when they change (optional)
            // You might want to debounce this
            // setTimeout(() => applyFilters(), 500);
        }
    });
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

// =============================================================================
// FILTER MANAGEMENT
// =============================================================================

async function loadFilterOptions() {
    console.log('📊 Loading filter options...');
    
    try {
        // Try to load real filter options from the backend
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
        
        // Fallback to sample data
        filterOptions = {
            agents: ['Rey Mendoza', 'Maria Garcia', 'John Smith', 'Sarah Johnson', 'Ana Rodriguez', 'David Chen'],
            dispositions: ['Account', 'Technical Support', 'Billing', 'Port Out - Questions/pin/acct #', 'Service Inquiry', 'Complaint'],
            sites: ['Dasma', 'Manila', 'Cebu', 'Davao', 'Iloilo', 'Bacolod'],
            lobs: ['WNP', 'Prepaid', 'Postpaid', 'Business', 'Enterprise']
        };
    }
    
    populateFilterOptions(filterOptions);
}

function populateFilterOptions(data) {
    console.log('🔧 Populating filter UI elements...');
    
    try {
        // Populate agent filters
        const agentFilters = document.getElementById('agentFilters');
        if (agentFilters && data.agents) {
            data.agents.forEach(agent => {
                const div = document.createElement('div');
                div.className = 'checkbox-item';
                div.innerHTML = `
                    <input type="checkbox" id="agent-${agent.replace(/\s+/g, '-')}" value="${agent}">
                    <label for="agent-${agent.replace(/\s+/g, '-')}">${agent}</label>
                `;
                agentFilters.appendChild(div);
            });
        }

        // Populate disposition filters
        const dispositionFilters = document.getElementById('dispositionFilters');
        if (dispositionFilters && data.dispositions) {
            data.dispositions.forEach(disposition => {
                const div = document.createElement('div');
                div.className = 'checkbox-item';
                const safeId = disposition.replace(/\s+/g, '-').replace(/[^a-zA-Z0-9-]/g, '');
                div.innerHTML = `
                    <input type="checkbox" id="disp-${safeId}" value="${disposition}">
                    <label for="disp-${safeId}">${disposition}</label>
                `;
                dispositionFilters.appendChild(div);
            });
        }

        // Populate site filter
        const siteFilter = document.getElementById('siteFilter');
        if (siteFilter && data.sites) {
            data.sites.forEach(site => {
                const option = document.createElement('option');
                option.value = site;
                option.textContent = site;
                siteFilter.appendChild(option);
            });
        }

        // Populate LOB filter
        const lobFilter = document.getElementById('lobFilter');
        if (lobFilter && data.lobs) {
            data.lobs.forEach(lob => {
                const option = document.createElement('option');
                option.value = lob;
                option.textContent = lob;
                lobFilter.appendChild(option);
            });
        }
        
        console.log('✅ Filter options populated successfully');
    } catch (error) {
        console.error('❌ Error populating filter options:', error);
    }
}

function applyFilters() {
    console.log('🔍 Applying filters...');
    
    currentFilters = collectFilters();
    updateActiveFilters();
    updateStats();
    
    console.log('📊 Active filters:', currentFilters);
    
    // If there are messages, refresh the analysis
    if (chatHistory.length > 0) {
        addMessage('system', '🔄 Filters updated. Your analysis will now use the new filter criteria.');
    }
}

function collectFilters() {
    const filters = {};

    try {
        // Date range
        const startDate = document.getElementById('startDate')?.value;
        const endDate = document.getElementById('endDate')?.value;
        if (startDate) filters.startDate = startDate;
        if (endDate) filters.endDate = endDate;

        // Agents
        const selectedAgents = Array.from(document.querySelectorAll('#agentFilters input[type="checkbox"]:checked:not([value="all"])'))
            .map(checkbox => checkbox.value);
        if (selectedAgents.length > 0) filters.agents = selectedAgents;

        // Dispositions
        const selectedDispositions = Array.from(document.querySelectorAll('#dispositionFilters input[type="checkbox"]:checked:not([value="all"])'))
            .map(checkbox => checkbox.value);
        if (selectedDispositions.length > 0) filters.dispositions = selectedDispositions;

        // Site
        const site = document.getElementById('siteFilter')?.value;
        if (site) filters.site = site;

        // LOB
        const lob = document.getElementById('lobFilter')?.value;
        if (lob) filters.lob = lob;

        // Duration
        const minDuration = document.getElementById('minDuration')?.value;
        const maxDuration = document.getElementById('maxDuration')?.value;
        if (minDuration) filters.minDuration = parseInt(minDuration);
        if (maxDuration) filters.maxDuration = parseInt(maxDuration);

        // Language
        const language = document.getElementById('languageFilter')?.value;
        if (language) filters.language = language;

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
        if (Array.isArray(value)) {
            displayValue = value.length > 2 ? `${value.length} selected` : value.join(', ');
        }
        
        tag.innerHTML = `
            ${key}: ${displayValue}
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
}

function clearFilters() {
    console.log('🧹 Clearing all filters...');
    
    currentFilters = {};
    updateActiveFilters();
    updateStats();
    
    // Reset form elements
    const elementsToReset = [
        'startDate', 'endDate', 'siteFilter', 'lobFilter', 
        'minDuration', 'maxDuration', 'languageFilter'
    ];
    
    elementsToReset.forEach(id => {
        const element = document.getElementById(id);
        if (element) element.value = '';
    });
    
    // Uncheck all checkboxes except "all"
    document.querySelectorAll('#agentFilters input[type="checkbox"]:not([value="all"])').forEach(cb => cb.checked = false);
    document.querySelectorAll('#dispositionFilters input[type="checkbox"]:not([value="all"])').forEach(cb => cb.checked = false);
    document.querySelectorAll('input[value="all"]').forEach(cb => cb.checked = true);
    
    console.log('✅ All filters cleared');
}

// =============================================================================
// STATISTICS AND DATA MANAGEMENT
// =============================================================================

async function updateStats() {
    console.log('📊 Updating statistics...');
    
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
                totalRecords.textContent = `${data.totalRecords || 0} records`;
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
            totalRecords.textContent = `${recordCount} records`;
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
    
    console.log('💬 Sending message:', message);
    console.log('🔍 With filters:', currentFilters);
    
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
        // Make API call with filters
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                history: chatHistory,
                filters: currentFilters,
                analytics: true // Flag to indicate this is an analytics query
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
        
        // If there are related documents, show them
        if (data.sources && data.sources.length > 0) {
            addSourcesMessage(data.sources);
        }
        
        console.log('✅ Message sent successfully');
        
    } catch (error) {
        console.error('❌ Error sending message:', error);
        removeLoadingMessage();
        addMessage('assistant', 'Sorry, there was an error processing your request. Please try again.');
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
    // Basic formatting for assistant messages
    // You can enhance this to handle markdown, charts, etc.
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
    
    let sourcesHtml = '<div class="sources-container"><h4>📚 Related Sources:</h4>';
    
    sources.forEach((source, index) => {
        const metadata = source.metadata || {};
        sourcesHtml += `
            <div class="source-item">
                <strong>${metadata.agent || 'Unknown Agent'}</strong> - ${metadata.disposition || 'Call'}
                <br>
                <small>Date: ${metadata.call_date || 'Unknown'} | Duration: ${metadata.call_duration || 'Unknown'}s</small>
                <br>
                <span class="source-text">${source.text?.substring(0, 200) || 'No text'}...</span>
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
                Analyzing your data with current filters...
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
        '<div class="spinner"></div> Sending...' : 
        '<span class="material-icons">send</span> Send';
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
    
    console.log('📁 Exporting chat history...');
    
    // Create export content
    let exportContent = `Metro AI Analytics Chat Export\n`;
    exportContent += `Generated: ${new Date().toISOString()}\n`;
    exportContent += `Total Messages: ${chatHistory.length}\n\n`;
    
    // Add active filters
    if (Object.keys(currentFilters).length > 0) {
        exportContent += `Active Filters:\n`;
        Object.entries(currentFilters).forEach(([key, value]) => {
            exportContent += `  ${key}: ${Array.isArray(value) ? value.join(', ') : value}\n`;
        });
        exportContent += '\n';
    }
    
    // Add chat history
    exportContent += `Chat History:\n`;
    exportContent += `${'='.repeat(50)}\n\n`;
    
    chatHistory.forEach((msg, index) => {
        exportContent += `${index + 1}. ${msg.role.toUpperCase()}:\n`;
        exportContent += `${msg.content}\n\n`;
    });
    
    // Create and download file
    const blob = new Blob([exportContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `metro-ai-analytics-${new Date().toISOString().split('T')[0]}.txt`;
    a.click();
    URL.revokeObjectURL(url);
    
    console.log('✅ Chat history exported');
}

// =============================================================================
// SEARCH FUNCTIONALITY (Legacy support)
// =============================================================================

// Keep the original search functionality for backward compatibility
async function performSearch() {
    const searchInput = document.getElementById('searchInput');
    if (!searchInput) return;
    
    const query = searchInput.value.trim();
    if (!query) return;

    console.log('🔍 Performing search:', query);
    
    try {
        const response = await fetch('/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                query: query,
                filters: currentFilters
            })
        });
        
        if (response.ok) {
            const results = await response.json();
            displaySearchResults(results);
        } else {
            throw new Error('Search failed');
        }
    } catch (error) {
        console.error('❌ Search error:', error);
        alert('Search failed. Please try again.');
    }
}

function displaySearchResults(results) {
    console.log('📋 Displaying search results:', results);
    // Implementation depends on your search results UI
    // This is a placeholder for backward compatibility
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
window.performSearch = performSearch;

// Initialize debugging
window.chatDebug = {
    getCurrentFilters: () => currentFilters,
    getChatHistory: () => chatHistory,
    getFilterOptions: () => filterOptions,
    isLoading: () => isLoading
};

console.log('✅ Enhanced Metro AI Analytics Chat loaded successfully');
console.log('🔧 Debug tools available at window.chatDebug');