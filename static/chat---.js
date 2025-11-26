// Metro AI Analytics - Optimized Chat Interface
// Version 6.0 - Clean, modular implementation

const ChatUI = (function() {
    'use strict';

    // ==================== Configuration ====================
    const CONFIG = {
        API_ENDPOINTS: {
            CHAT: '/api/chat',
            FILTERS: '/api/filters',
            EXPORT: '/api/export'
        },
        TIMEOUTS: {
            API_CALL: 30000,
            TYPING_DELAY: 500
        },
        MESSAGES: {
            MAX_HISTORY: 50,
            AUTOSAVE: true
        },
        VERSION: '6.0'
    };

    // ==================== State Management ====================
    const state = {
        messages: [],
        filters: {},
        isLoading: false,
        currentSessionId: generateSessionId(),
        filterOptions: {
            templates: [],
            programs: [],
            partners: [],
            sites: [],
            lobs: [],
            dispositions: [],
            subDispositions: [],
            languages: [],
            callTypes: []
        }
    };

    // ==================== Utility Functions ====================
    function generateSessionId() {
        return 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function formatTimestamp(date = new Date()) {
        return date.toLocaleTimeString('en-US', { 
            hour: '2-digit', 
            minute: '2-digit' 
        });
    }

    function formatMessage(content) {
        if (!content) return '';
        
        // Process markdown-style formatting
        let formatted = content
            // Bold
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            // Italic
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            // Code blocks
            .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
            // Inline code
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            // Links
            .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank">$1</a>')
            // Line breaks
            .replace(/\n/g, '<br>');

        // Process lists
        formatted = formatted
            .replace(/^\* (.+)$/gm, '<li>$1</li>')
            .replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>')
            .replace(/^\d+\. (.+)$/gm, '<li>$1</li>')
            .replace(/(<li>.*<\/li>)/s, function(match) {
                if (!match.includes('<ul>')) {
                    return '<ol>' + match + '</ol>';
                }
                return match;
            });

        return formatted;
    }

    function debounce(func, wait) {
        let timeout;
        return function executedFunction(...args) {
            const later = () => {
                clearTimeout(timeout);
                func(...args);
            };
            clearTimeout(timeout);
            timeout = setTimeout(later, wait);
        };
    }

    // ==================== DOM Manipulation ====================
    const DOM = {
        get chatMessages() { return document.getElementById('chatMessages'); },
        get chatInput() { return document.getElementById('chatInput'); },
        get sendBtn() { return document.getElementById('sendBtn'); },
        get welcomeScreen() { return document.getElementById('welcomeScreen'); },
        get messagesContainer() { return document.getElementById('messagesContainer'); },
        get sidebar() { return document.getElementById('sidebar'); },
        get activeFilters() { return document.getElementById('activeFilters'); },
        get statusText() { return document.getElementById('statusText'); }
    };

    // ==================== Message Management ====================
    function addMessage(role, content, metadata = {}) {
        // Hide welcome screen if visible
        if (DOM.welcomeScreen && !DOM.welcomeScreen.classList.contains('hidden')) {
            DOM.welcomeScreen.classList.add('hidden');
            DOM.chatMessages.classList.remove('hidden');
        }

        // Create message element
        const messageEl = document.createElement('div');
        messageEl.className = `message message-${role}`;
        messageEl.dataset.messageId = metadata.id || Date.now();
        messageEl.dataset.timestamp = new Date().toISOString();

        // Build message content based on role
        let messageContent = '';
        
        switch(role) {
            case 'user':
                messageContent = `
                    <div class="message-content">
                        <div class="message-text">${escapeHtml(content)}</div>
                        <div class="message-time">${formatTimestamp()}</div>
                    </div>
                `;
                break;
                
            case 'assistant':
                messageContent = `
                    <div class="message-content">
                        <div class="message-text">${formatMessage(content)}</div>
                        <div class="message-time">${formatTimestamp()}</div>
                        ${metadata.sources ? renderSources(metadata.sources) : ''}
                    </div>
                `;
                break;
                
            case 'error':
                messageContent = `
                    <div class="message-content error">
                        <span class="material-icons">error_outline</span>
                        <div class="message-text">${escapeHtml(content)}</div>
                    </div>
                `;
                break;
                
            case 'system':
                messageContent = `
                    <div class="message-content system">
                        <div class="message-text">${formatMessage(content)}</div>
                    </div>
                `;
                break;
        }

        messageEl.innerHTML = messageContent;
        DOM.chatMessages.appendChild(messageEl);
        
        // Smooth scroll to bottom
        DOM.chatMessages.scrollTo({
            top: DOM.chatMessages.scrollHeight,
            behavior: 'smooth'
        });

        // Add to state
        state.messages.push({
            role,
            content,
            metadata,
            timestamp: new Date().toISOString()
        });

        // Limit message history
        if (state.messages.length > CONFIG.MESSAGES.MAX_HISTORY) {
            state.messages.shift();
            DOM.chatMessages.firstChild?.remove();
        }
    }

    function renderSources(sources) {
        if (!sources || sources.length === 0) return '';
        
        const cleanSources = sources.slice(0, 3).map(source => {
            const fields = [];
            if (source.evaluationId) fields.push(`ID: ${source.evaluationId}`);
            if (source.partner) fields.push(`Partner: ${source.partner}`);
            if (source.agentName) fields.push(`Agent: ${source.agentName}`);
            if (source.weighted_score) fields.push(`Score: ${source.weighted_score}`);
            return fields.join(' • ');
        });

        return `
            <div class="message-sources">
                <div class="sources-header">
                    <span class="material-icons">source</span>
                    Data Sources
                </div>
                <div class="sources-list">
                    ${cleanSources.map(s => `<div class="source-item">${s}</div>`).join('')}
                </div>
                ${sources.length > 3 ? `<div class="sources-more">+${sources.length - 3} more</div>` : ''}
            </div>
        `;
    }

    function showTypingIndicator() {
        const indicator = document.createElement('div');
        indicator.id = 'typingIndicator';
        indicator.className = 'message message-assistant typing';
        indicator.innerHTML = `
            <div class="message-content">
                <div class="typing-dots">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;
        DOM.chatMessages.appendChild(indicator);
        DOM.chatMessages.scrollTop = DOM.chatMessages.scrollHeight;
    }

    function hideTypingIndicator() {
        document.getElementById('typingIndicator')?.remove();
    }

    // ==================== API Communication ====================
    async function sendMessage() {
        const message = DOM.chatInput.value.trim();
        if (!message || state.isLoading) return;

        state.isLoading = true;
        DOM.sendBtn.disabled = true;
        DOM.statusText.textContent = 'Processing...';

        // Add user message
        addMessage('user', message);
        
        // Clear input
        DOM.chatInput.value = '';
        DOM.chatInput.style.height = 'auto';
        
        // Show typing indicator
        showTypingIndicator();

        try {
            const response = await fetch(CONFIG.API_ENDPOINTS.CHAT, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    message,
                    filters: state.filters,
                    sessionId: state.currentSessionId,
                    history: state.messages.slice(-10)
                }),
                signal: AbortSignal.timeout(CONFIG.TIMEOUTS.API_CALL)
            });

            if (!response.ok) {
                throw new Error(`Server error: ${response.status}`);
            }

            const data = await response.json();
            
            // Hide typing indicator
            hideTypingIndicator();
            
            // Add assistant response
            if (data.reply) {
                addMessage('assistant', data.reply, {
                    sources: data.sources || [],
                    metadata: data.metadata || {}
                });
            }
            
        } catch (error) {
            console.error('Chat error:', error);
            hideTypingIndicator();
            addMessage('error', error.message || 'Failed to get response. Please try again.');
        } finally {
            state.isLoading = false;
            DOM.sendBtn.disabled = false;
            DOM.statusText.textContent = 'Ready';
        }
    }

    async function loadFilters() {
        try {
            const response = await fetch(CONFIG.API_ENDPOINTS.FILTERS);
            const data = await response.json();
            
            if (data.success) {
                state.filterOptions = data.filters;
                populateFilterDropdowns();
            }
        } catch (error) {
            console.error('Failed to load filters:', error);
        }
    }

    function populateFilterDropdowns() {
        // Helper function to populate a select element
        const populateSelect = (elementId, options) => {
            const select = document.getElementById(elementId);
            if (!select) return;
            
            // Keep first option (All)
            select.innerHTML = select.innerHTML.split('</option>')[0] + '</option>';
            
            // Add options
            options.forEach(option => {
                const optionEl = document.createElement('option');
                optionEl.value = option.value || option;
                optionEl.textContent = option.label || option;
                select.appendChild(optionEl);
            });
        };

        // Populate all dropdowns
        populateSelect('templateFilter', state.filterOptions.templates);
        populateSelect('programFilter', state.filterOptions.programs);
        populateSelect('partnerFilter', state.filterOptions.partners);
        populateSelect('siteFilter', state.filterOptions.sites);
        populateSelect('lobFilter', state.filterOptions.lobs);
        populateSelect('disposition', state.filterOptions.dispositions);
        populateSelect('subDisposition', state.filterOptions.subDispositions);
        populateSelect('language', state.filterOptions.languages);
        populateSelect('callType', state.filterOptions.callTypes);
    }

    // ==================== Filter Management ====================
    function applyFilters() {
        // Collect all filter values
        state.filters = {
            startDate: document.getElementById('startDate')?.value,
            endDate: document.getElementById('endDate')?.value,
            template: document.getElementById('templateFilter')?.value,
            program: document.getElementById('programFilter')?.value,
            partner: document.getElementById('partnerFilter')?.value,
            site: document.getElementById('siteFilter')?.value,
            lob: document.getElementById('lobFilter')?.value,
            agentName: document.getElementById('agentName')?.value,
            agentId: document.getElementById('agentId')?.value,
            evaluationId: document.getElementById('evaluationId')?.value,
            callType: document.getElementById('callType')?.value,
            disposition: document.getElementById('disposition')?.value,
            subDisposition: document.getElementById('subDisposition')?.value,
            language: document.getElementById('language')?.value
        };

        // Remove empty values
        Object.keys(state.filters).forEach(key => {
            if (!state.filters[key]) delete state.filters[key];
        });

        updateActiveFiltersDisplay();
        addMessage('system', `Filters applied: ${Object.keys(state.filters).length} active`);
    }

    function clearFilters() {
        // Clear all inputs
        document.querySelectorAll('.filter-input, .filter-select').forEach(input => {
            if (input.type === 'date' || input.type === 'text') {
                input.value = '';
            } else if (input.tagName === 'SELECT') {
                input.selectedIndex = 0;
            }
        });

        state.filters = {};
        updateActiveFiltersDisplay();
        addMessage('system', 'All filters cleared');
    }

    function updateActiveFiltersDisplay() {
        if (!DOM.activeFilters) return;

        const filterCount = Object.keys(state.filters).length;
        
        if (filterCount === 0) {
            DOM.activeFilters.innerHTML = '<span class="no-filters">No active filters</span>';
            return;
        }

        const filterTags = Object.entries(state.filters).map(([key, value]) => `
            <span class="filter-tag">
                ${key}: ${value}
                <button onclick="ChatUI.removeFilter('${key}')" class="filter-remove">×</button>
            </span>
        `).join('');

        DOM.activeFilters.innerHTML = filterTags;
    }

    function removeFilter(key) {
        delete state.filters[key];
        
        // Clear the corresponding input
        const input = document.getElementById(key) || 
                     document.getElementById(key + 'Filter');
        if (input) {
            if (input.type === 'date' || input.type === 'text') {
                input.value = '';
            } else if (input.tagName === 'SELECT') {
                input.selectedIndex = 0;
            }
        }
        
        updateActiveFiltersDisplay();
    }

    // ==================== UI Actions ====================
    function toggleSidebar() {
        DOM.sidebar?.classList.toggle('collapsed');
    }

    function clearChat() {
        if (!confirm('Clear all messages? This action cannot be undone.')) return;
        
        state.messages = [];
        DOM.chatMessages.innerHTML = '';
        DOM.welcomeScreen?.classList.remove('hidden');
        DOM.chatMessages?.classList.add('hidden');
    }

    async function exportChat() {
        if (state.messages.length === 0) {
            alert('No messages to export');
            return;
        }

        const exportData = {
            sessionId: state.currentSessionId,
            timestamp: new Date().toISOString(),
            filters: state.filters,
            messages: state.messages,
            version: CONFIG.VERSION
        };

        const blob = new Blob([JSON.stringify(exportData, null, 2)], {
            type: 'application/json'
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `chat-export-${new Date().toISOString().split('T')[0]}.json`;
        a.click();
        URL.revokeObjectURL(url);
    }

    function askQuestion(question) {
        DOM.chatInput.value = question;
        sendMessage();
    }

    function handleKeyPress(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            sendMessage();
        }
    }

    // ==================== Auto-resize textarea ====================
    function setupAutoResize() {
        DOM.chatInput?.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 120) + 'px';
        });
    }

    // ==================== Initialization ====================
    function init() {
        console.log('Initializing Metro AI Analytics v' + CONFIG.VERSION);
        
        // Set up event listeners
        setupAutoResize();
        
        // Load filters
        loadFilters();
        
        // Set default dates
        const today = new Date();
        const thirtyDaysAgo = new Date(today.getTime() - (30 * 24 * 60 * 60 * 1000));
        
        const startDate = document.getElementById('startDate');
        const endDate = document.getElementById('endDate');
        
        if (startDate) startDate.valueAsDate = thirtyDaysAgo;
        if (endDate) endDate.valueAsDate = today;
        
        // Initialize filter display
        updateActiveFiltersDisplay();
        
        console.log('Initialization complete');
    }

    // ==================== Public API ====================
    return {
        init,
        sendMessage,
        clearChat,
        exportChat,
        toggleSidebar,
        applyFilters,
        clearFilters,
        removeFilter,
        askQuestion,
        handleKeyPress
    };
})();

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', ChatUI.init);
} else {
    ChatUI.init();
}
