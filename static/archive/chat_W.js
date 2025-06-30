// Fixed chat.js - COMPLETE VERSION with working search functionality

let chatSessions = [];
let savedChats = [];
let currentChat = null;
let typingTimeout = null;
let programs = ['Metro'];
let currentTab = 'recent';
let expandedMessages = new Set();
let isSearchMode = false;

const SYSTEM_WELCOME_MSG =
    'How can I help you today? I can answer questions, walk you through processes, or help you find the information you need.';

// Load saved chats - using sessionStorage instead of localStorage
function loadSavedChats() {
    try {
        const saved = sessionStorage.getItem('metroAI_savedChats');
        if (saved) {
            savedChats = JSON.parse(saved);
        }
    } catch (error) {
        console.error('Failed to load saved chats:', error);
        savedChats = [];
    }
}

// Save starred chats - using sessionStorage instead of localStorage
function saveChatsTostorage() {
    try {
        const chatsToSave = savedChats.slice(0, 50);
        sessionStorage.setItem(
            'metroAI_savedChats',
            JSON.stringify(chatsToSave)
        );
    } catch (error) {
        console.error('Failed to save chats:', error);
    }
}

function switchTab(tab) {
    currentTab = tab;

    document
        .getElementById('recentTab')
        .classList.toggle('active', tab === 'recent');
    document
        .getElementById('savedTab')
        .classList.toggle('active', tab === 'saved');

    document.getElementById('recentChats').style.display =
        tab === 'recent' ? 'block' : 'none';
    document.getElementById('savedChats').style.display =
        tab === 'saved' ? 'block' : 'none';

    renderSidebar();
}

function truncate(str, n) {
    if (!str) return '';
    str = str.replace(/[\n\r]+/g, ' ').trim();
    return str.length > n ? str.slice(0, n) + '…' : str;
}

function renderSidebar() {
    const recentContainer = document.getElementById('recentChats');
    const savedContainer = document.getElementById('savedChats');

    if (!recentContainer || !savedContainer) return;

    recentContainer.innerHTML = '';
    savedContainer.innerHTML = '';

    // Render recent chats
    if (chatSessions.length === 0) {
        recentContainer.innerHTML =
            '<div class="empty-state">No recent chats</div>';
    } else {
        chatSessions.forEach((session, index) => {
            const active =
                currentChat && currentChat.id === session.id ? 'active' : '';
            const name = truncate(
                session.name || getLastUserMessage(session) || 'New Chat',
                30
            );
            const preview = getLastUserMessage(session);

            const el = document.createElement('div');
            el.className = `chat-link ${active}`;
            el.innerHTML = `
                <div>
                    <div>${name}</div>
                    ${
                        preview
                            ? `<div class="chat-preview">${truncate(
                                  preview,
                                  40
                              )}</div>`
                            : ''
                    }
                </div>
            `;
            el.onclick = () => {
                selectChat(session);
            };
            recentContainer.appendChild(el);
        });
    }

    // Render saved chats
    if (savedChats.length === 0) {
        savedContainer.innerHTML =
            '<div class="empty-state">No saved chats yet</div>';
    } else {
        savedChats.forEach((savedChat) => {
            const active =
                currentChat && currentChat.id === savedChat.id ? 'active' : '';
            const name = truncate(
                savedChat.name || savedChat.title || 'Saved Chat',
                30
            );
            const preview = savedChat.preview || getLastUserMessage(savedChat);

            const el = document.createElement('div');
            el.className = `chat-link ${active} starred`;
            el.innerHTML = `
                <div>
                    <div>${name}</div>
                    ${
                        preview
                            ? `<div class="chat-preview">${truncate(
                                  preview,
                                  40
                              )}</div>`
                            : ''
                    }
                </div>
            `;
            el.onclick = () => {
                loadSavedChat(savedChat);
            };
            savedContainer.appendChild(el);
        });
    }
}

function getLastUserMessage(session) {
    const userMessages = session.messages.filter((m) => m.role === 'user');
    return userMessages.length > 0
        ? userMessages[userMessages.length - 1].text
        : '';
}

function selectChat(session) {
    currentChat = session;
    isSearchMode = false;
    renderSidebar();
    renderMessages();
    updateStarButton();

    // Show chat messages, hide search results
    document.getElementById('messages').style.display = 'block';
    document.getElementById('search-results').style.display = 'none';
}

function loadSavedChat(savedChat) {
    let existingChat = chatSessions.find((s) => s.id === savedChat.id);

    if (!existingChat) {
        existingChat = {
            name: savedChat.name || savedChat.title,
            messages: [...savedChat.messages],
            id: savedChat.id,
            programs: savedChat.programs || [...programs],
            starred: true,
        };
        chatSessions.unshift(existingChat);
    }

    selectChat(existingChat);

    if (currentTab === 'saved') {
        switchTab('recent');
    }
}

function startNewChat() {
    const session = {
        name: '',
        messages: [],
        id: Date.now(),
        programs: [...programs],
        starred: false,
    };
    session.messages.push({ role: 'system', text: SYSTEM_WELCOME_MSG });
    chatSessions.unshift(session);
    selectChat(session);

    if (currentTab === 'saved') {
        switchTab('recent');
    }
}

function clearCurrentChat() {
    if (!currentChat) return;

    if (
        confirm(
            'Are you sure you want to clear this chat? This action cannot be undone.'
        )
    ) {
        chatSessions = chatSessions.filter((s) => s.id !== currentChat.id);
        startNewChat();
    }
}

function toggleStarChat() {
    if (!currentChat) {
        alert('Start a conversation first before saving!');
        return;
    }

    if (currentChat.messages.length <= 1) {
        alert('Have a conversation first before saving!');
        return;
    }

    if (currentChat.starred) {
        currentChat.starred = false;
        savedChats = savedChats.filter((chat) => chat.id !== currentChat.id);
    } else {
        currentChat.starred = true;
        const chatToSave = {
            id: currentChat.id,
            name: currentChat.name || generateChatTitle(),
            title: currentChat.name || generateChatTitle(),
            messages: [...currentChat.messages],
            programs: [...currentChat.programs],
            timestamp: new Date().toISOString(),
            preview: getLastUserMessage(currentChat),
        };

        savedChats = savedChats.filter((chat) => chat.id !== currentChat.id);
        savedChats.unshift(chatToSave);
    }

    saveChatsTostorage();
    updateStarButton();
    renderSidebar();
}

function generateChatTitle() {
    if (currentChat && currentChat.messages.length > 1) {
        const firstUserMessage = currentChat.messages.find(
            (m) => m.role === 'user'
        );
        if (firstUserMessage) {
            return firstUserMessage.text.length > 50
                ? firstUserMessage.text.substring(0, 50) + '...'
                : firstUserMessage.text;
        }
    }
    return 'New Chat';
}

function updateStarButton() {
    const starBtn = document.getElementById('starChatBtn');
    if (!starBtn) return;

    const icon = starBtn.querySelector('.material-icons');
    const text = starBtn.querySelector('span:last-child');

    if (currentChat && currentChat.starred) {
        starBtn.classList.add('starred');
        icon.textContent = 'star';
        text.textContent = 'Saved';
        starBtn.title = 'Remove from saved chats';
    } else {
        starBtn.classList.remove('starred');
        icon.textContent = 'star_border';
        text.textContent = 'Save';
        starBtn.title = 'Save this chat';
    }
}

function toggleProgram(checkbox) {
    const value = checkbox.value;
    const isChecked = checkbox.checked;

    console.log(`Program ${value} ${isChecked ? 'selected' : 'deselected'}`);

    if (isChecked) {
        if (!programs.includes(value)) {
            programs.push(value);
        }
    } else {
        programs = programs.filter((p) => p !== value);
    }

    console.log('Current programs:', programs);

    // Update current chat's programs if exists
    if (currentChat) {
        currentChat.programs = [...programs];

        if (!currentChat.programs.includes('All')) {
            currentChat.programs.push('All');
        }

        console.log('Updated currentChat.programs:', currentChat.programs);
    }

    // Show visual feedback
    const filterDiv = checkbox.closest('.program-filters');
    if (filterDiv) {
        const status = document.createElement('div');
        status.style.cssText =
            'color: #6E32A0; font-size: 0.8em; margin-top: 4px;';
        status.textContent = `Active programs: ${
            programs.join(', ') || 'None'
        }`;

        // Remove any existing status
        const existingStatus = filterDiv.querySelector('.program-status');
        if (existingStatus) existingStatus.remove();

        status.className = 'program-status';
        filterDiv.appendChild(status);

        // Remove after 3 seconds
        setTimeout(() => {
            if (status.parentNode) status.remove();
        }, 3000);
    }
}

function addMessage(role, text, rawData) {
    if (!currentChat) startNewChat();

    if (role === 'assistant') {
        const lastMsg = currentChat.messages[currentChat.messages.length - 1];
        const newText =
            typeof text === 'object' && text !== null
                ? text.summary || text.reply || JSON.stringify(text)
                : text;
        const lastText =
            lastMsg && lastMsg.role === 'assistant' ? lastMsg.text || '' : '';
        if (newText.trim() === lastText.trim()) {
            console.log('Duplicate assistant message suppressed.');
            return;
        }
    }

    if (role === 'assistant' && typeof text === 'object' && text !== null) {
        text = text.summary || text.reply || JSON.stringify(text);
    }

    currentChat.messages.push({ role, text, rawData });
    if (role === 'user') {
        currentChat.name = truncate(text, 34);
        renderSidebar();
    }
    renderMessages(role);
}

function showTyping() {
    const messagesDiv = document.getElementById('messages');
    const indicator = document.createElement('div');
    indicator.className = 'message-row';
    indicator.id = 'typing-indicator';
    indicator.innerHTML = `
        <div class="assistant-msg" style="background: #f3e3fa; color:#9646c3; font-style:italic; min-width: 60px;">
            <span style="display:inline-block;width:14px;height:14px;border-radius:50%;background:#E20074;opacity:.6;margin:2px 2px;animation:bounce 1s infinite alternate;"></span>
            <span style="display:inline-block;width:14px;height:14px;border-radius:50%;background:#E20074;opacity:.4;margin:2px 2px;animation:bounce 1s .2s infinite alternate;"></span>
            <span style="display:inline-block;width:14px;height:14px;border-radius:50%;background:#E20074;opacity:.2;margin:2px 2px;animation:bounce 1s .4s infinite alternate;"></span>
            <style>@keyframes bounce{0%{transform:translateY(0);}100%{transform:translateY(-6px);}}</style>
        </div>
    `;
    messagesDiv.appendChild(indicator);
    
    // For typing indicator, scroll to show it
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function hideTyping() {
    const el = document.getElementById('typing-indicator');
    if (el) el.remove();
}

function renderMessages() {
    if (isSearchMode) return; // Don't render messages in search mode

    const messagesDiv = document.getElementById('messages');
    messagesDiv.innerHTML = '';
    if (!currentChat) return;

    for (let i = 0; i < currentChat.messages.length; i++) {
        const msg = currentChat.messages[i];
        let row = document.createElement('div');

        if (msg.role === 'system') {
            row.className = 'message-row system';
            row.innerHTML = `<div class="system-msg">${msg.text}</div>`;
        } else if (msg.role === 'user') {
            row.className = 'message-row user';
            row.innerHTML = `<div class="user-msg">${msg.text}</div>`;
        } else {
            row.className = 'message-row';
            const messageId = `msg-${i}`;
            const isCollapsed = expandedMessages.has(messageId + '_collapsed');

            row.innerHTML = `
                <div class="assistant-msg">
                    <div class="message-header">
                        <strong>Assistant</strong>
                        <div class="message-actions">
                            <button class="action-btn" onclick="toggleMessageExpand('${messageId}')" title="${
                isCollapsed ? 'Expand' : 'Collapse'
            }">
                                <span class="material-icons">${
                                    isCollapsed ? 'expand_more' : 'expand_less'
                                }</span>
                            </button>
                        </div>
                    </div>
                    <div id="${messageId}" class="message-content ${
                isCollapsed ? 'collapsed' : 'expanded'
            }">
                        ${formatEnhancedAIResponse(
                            msg.rawData || msg.text,
                            msg.text,
                            messageId
                        )}
                    </div>
                </div>
            `;
        }
        messagesDiv.appendChild(row);
    }
    handleScrollAfterRender(lastMessageRole, wasScrolledToBottom, messagesDiv);
}

function handleScrollAfterRender(lastMessageRole, wasScrolledToBottom, messagesDiv) {
    /**
     * Smart scroll behavior:
     * - User message: Scroll to show the new user message
     * - Assistant message: Scroll to start of the new assistant response
     * - Other cases: Maintain previous scroll position if user was reading
     */
    
    if (lastMessageRole === 'user') {
        // User just sent a message - scroll to show it
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    } else if (lastMessageRole === 'assistant') {
        // New AI response - scroll to the START of the assistant message, not the bottom
        const assistantMessages = messagesDiv.querySelectorAll('.message-row:not(.user):not(.system)');
        if (assistantMessages.length > 0) {
            const lastAssistantMessage = assistantMessages[assistantMessages.length - 1];
            
            // Scroll to show the top of the assistant message
            lastAssistantMessage.scrollIntoView({ 
                behavior: 'smooth', 
                block: 'start'  // Show the TOP of the message
            });
        }
    } else if (wasScrolledToBottom) {
        // User was at bottom before, keep them there
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }
    // Otherwise, don't change scroll position (user is reading)
}

// Add this function to chat.js to make inline document names clickable

function makeDocumentNamesClickable(text, references) {
    /**
     * Convert document names mentioned in AI response text into clickable links
     * @param {string} text - The AI response text
     * @param {Array} references - Array of reference objects with title and url
     * @returns {string} - Text with clickable document links
     */
    
    if (!references || references.length === 0) {
        return text;
    }
    
    let processedText = text;
    
    // Create a map of document titles to URLs
    const documentMap = {};
    references.forEach(ref => {
        if (ref.title && ref.url) {
            documentMap[ref.title] = ref.url;
        }
    });
    
    // Process each document title
    Object.entries(documentMap).forEach(([title, url]) => {
        // Patterns to match document name mentions
        const patterns = [
            `"${title}"`,           // "Account PIN & SQA Support"  
            `'${title}'`,           // 'Account PIN & SQA Support'
            `"${title}"`,           // Smart quotes
            `'${title}'`,           // Smart quotes
            title                   // Plain title (be careful with this one)
        ];
        
        patterns.forEach(pattern => {
            // Create regex that matches the pattern but not if it's already in a link
            const regex = new RegExp(
                `(?<!<a[^>]*>)(?<!href="[^"]*")${escapeRegex(pattern)}(?![^<]*</a>)`, 
                'gi'
            );
            
            // Replace with clickable link
            processedText = processedText.replace(regex, (match) => {
                return `<a href="${url}" target="_blank" rel="noopener" class="inline-doc-link" title="View ${title}">${match}</a>`;
            });
        });
    });
    
    return processedText;
}

function escapeRegex(string) {
    return string.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// Updated formatEnhancedAIResponse function with clickable inline document names
function formatEnhancedAIResponse(raw, fallbackText, messageId) {
    let summary = '', steps = [], suggestions = [], references = [];

    if (typeof raw === 'object' && raw !== null) {
        summary = raw.summary || raw.reply || '';
        steps = raw.steps || [];
        suggestions = raw.suggestions || [];
        references = raw.references || [];
    } else if (typeof raw === 'string') {
        const lines = raw.split('\n').filter(Boolean);
        summary = lines[0] || '';
        steps = lines.slice(1, 10);
        suggestions = lines.slice(5);
    }

    let html = '';

    // Enhanced Summary Section with clickable document names
    if (summary) {
        html += `<div class="response-section">`;
        html += `<span class="section-heading"> MetroAI Found...</span>`;

        // Format summary with better line breaks and styling
        let formattedSummary = summary
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
        
        // MAKE DOCUMENT NAMES CLICKABLE
        formattedSummary = makeDocumentNamesClickable(formattedSummary, references);

        html += `<div class="summary-content"><p>${formattedSummary}</p></div>`;
        html += `</div>`;
    }

    // Enhanced Steps Section with clickable document names
    if (steps.length) {
        html += `<div class="response-section">`;
        html += `<span class="section-heading">Step-by-Step Guidance</span>`;
        html += `<ol class="enhanced-step-list">`;
        for (let step of steps) {
            let formattedStep = step.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
            
            // MAKE DOCUMENT NAMES CLICKABLE IN STEPS
            formattedStep = makeDocumentNamesClickable(formattedStep, references);
            
            html += `<li>${formattedStep}</li>`;
        }
        html += `</ol>`;
        html += `</div>`;
    }

    // Enhanced Suggestions Section with clickable document names
    if (suggestions.length) {
        html += `<div class="response-section">`;
        html += `<span class="section-heading">Additional Information & Related Topics</span>`;
        html += `<ul class="enhanced-suggest-list">`;
        for (let sug of suggestions) {
            // Check if suggestion contains URLs but don't display them
            if (sug.includes('http://') || sug.includes('https://')) {
                const urlMatch = sug.match(/(.*?):\s*(https?:\/\/[^\s]+)/);
                if (urlMatch) {
                    const title = urlMatch[1].replace('•', '').trim();
                    const url = urlMatch[2];
                    html += `<li><strong><a href="${url}" target="_blank" rel="noopener" class="doc-link">${title}</a></strong> - <span style="color: #666; font-style: italic;">Click to view document</span></li>`;
                } else {
                    let formattedSug = sug.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
                    formattedSug = makeDocumentNamesClickable(formattedSug, references);
                    html += `<li>${formattedSug}</li>`;
                }
            } else {
                let formattedSug = sug.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
                
                // MAKE DOCUMENT NAMES CLICKABLE IN SUGGESTIONS
                formattedSug = makeDocumentNamesClickable(formattedSug, references);
                
                html += `<li>${formattedSug}</li>`;
            }
        }
        html += `</ul>`;
        html += `</div>`;
    }

    // References section (already has clickable links)
    if (references && Array.isArray(references) && references.length > 0) {
        const unique = {};
        const uniqueRefs = references.filter((r) => {
            if (!unique[r.url]) {
                unique[r.url] = 1;
                return true;
            }
            return false;
        });

        html += `<div class="response-section references-section">`;
        html += `<span class="section-heading"> Source Documents</span>`;

        html += `<div class="reference-grid">`;
        for (let ref of uniqueRefs) {
            if (typeof ref === 'object' && ref !== null) {
                const title = ref.title || ref.name || 'Reference';
                const url = ref.url;
                const program = ref.program || 'All';
                const collection = ref.collection || 'Unknown';  // ADDED Collection support
                const lob = ref.lob || '';
                const score = ref.score || 0;

                html += `<div class="reference-card">`;
                html += `<div class="ref-header">`;

                if (url && url.trim()) {
                    html += `<a href="${url}" target="_blank" rel="noopener" class="reference-title-link">`;
                    html += ` ${title}`;
                    html += `<span class="external-icon">🔗</span>`;
                    html += `</a>`;
                } else {
                    html += `<span class="reference-title"> ${title}</span>`;
                }

                html += `</div>`;
                html += `<div class="ref-meta">`;
                html += `<span class="ref-collection"> ${collection}</span>`;   // Show collection first (primary grouping)
                logger.debug('Collection:', collection);
                html += `<span class="ref-program">${program}</span>`;  // I will remove later after testing program accuracy
                if (lob) html += `<span class="ref-lob">${lob}</span>`;  // will remove later after testing
                if (score > 0) html += `<span class="ref-score">Score: ${score.toFixed(2)}</span>`;
                html += `</div>`;

                if (url && url.trim()) {
                    html += `<div class="ref-action">`;
                    html += `<small><a href="${url}" target="_blank" rel="noopener" class="view-doc-link">Click to view document →</a></small>`;
                    html += `</div>`;
                }

                html += `</div>`;
            }
        }
        html += `</div>`;
        html += `</div>`;
    }

    // Helpful rating section (unchanged)
    const articleId = typeof raw === 'object' && raw !== null ? 
        raw.search_query || summary.slice(0, 100) : summary.slice(0, 100);

    html += `
    <div class="helpful-rating">
        <span class="rate-label">Was this response helpful?</span>
        <button class="thumb-btn" onclick="rateHelpful(this, 'up', '${articleId.replace(/'/g, '')}')">
            <span class="material-icons">thumb_up</span>
        </button>
        <button class="thumb-btn" onclick="rateHelpful(this, 'down', '${articleId.replace(/'/g, '')}')">
            <span class="material-icons">thumb_down</span>
        </button>
        <span class="rate-confirm">Thank you!</span>
    </div>
    `;

    return html;
}

function toggleMessageExpand(messageId) {
    console.log('inside toggleMessageExpand');
    const messageContent = document.getElementById(messageId);
    if (!messageContent) return;

    const button = messageContent.parentElement.querySelector('.action-btn');
    const icon = button.querySelector('.material-icons');
    const collapseKey = messageId + '_collapsed';

    // UPDATED logic: Default is expanded, track collapsed state

    if (expandedMessages.has(collapseKey)) {
        console.log('Expanding message:', messageId);
        expandedMessages.delete(collapseKey);
        messageContent.classList.remove('collapsed');
        messageContent.classList.add('expanded');
        icon.textContent = 'expand_les';
        button.title = 'collapse';
    } else {
        console.log('Collapsing message:', messageId);
        expandedMessages.add(collapseKey);
        messageContent.classList.remove('expanded');
        messageContent.classList.add('collapsed');
        icon.textContent = 'expand_more';
        button.title = 'Expand';
    }
}

// UPDATED Enhanced formatting function with clickable document links
function formatEnhancedAIResponse(raw, fallbackText, messageId) {
    let summary = '',
        steps = [],
        suggestions = [],
        references = [];

    if (typeof raw === 'object' && raw !== null) {
        summary = raw.summary || raw.reply || '';
        steps = raw.steps || [];
        suggestions = raw.suggestions || [];
        references = raw.references || [];
    } else if (typeof raw === 'string') {
        const lines = raw.split('\n').filter(Boolean);
        summary = lines[0] || '';
        steps = lines.slice(1, 10);
        suggestions = lines.slice(5);
    }

    let html = '';

    // Enhanced Summary Section
    if (summary) {
        html += `<div class="response-section">`;
        html += `<span class="section-heading"> MetroAi Found...</span>`;

        // Format summary with better line breaks and styling
        const formattedSummary = summary
            .replace(/\n\n/g, '</p><p>')
            .replace(/\n/g, '<br>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

        html += `<div class="summary-content"><p>${formattedSummary}</p></div>`;
        html += `</div>`;
    }

    // Enhanced Steps Section
    if (steps.length) {
        html += `<div class="response-section">`;
        html += `<span class="section-heading"> Step-by-Step Guidance</span>`;
        html += `<ol class="enhanced-step-list">`;
        for (let step of steps) {
            const formattedStep = step.replace(
                /\*\*(.*?)\*\*/g,
                '<strong>$1</strong>'
            );
            html += `<li>${formattedStep}</li>`;
        }
        html += `</ol>`;
        html += `</div>`;
    }

    // UPDATED: Enhanced Suggestions Section - NEVER show URLs directly
    if (suggestions.length) {
        console.log('>>>>> in suggestions.length');
        html += `<div class="response-section">`;
        html += `<span class="section-heading"> Additional Information & Related Topics</span>`;
        html += `<ul class="enhanced-suggest-list">`;
        for (let sug of suggestions) {
            // UPDATED: Check if suggestion contains URLs but don't display them
            if (sug.includes('http://') || sug.includes('https://')) {
                const urlMatch = sug.match(/(.*?):\s*(https?:\/\/[^\s]+)/);
                if (urlMatch) {
                    const title = urlMatch[1].replace('•', '').trim();
                    const url = urlMatch[2];
                    // UPDATED: Show title as clickable link, but don't show URL text
                    html += `<li><strong><a href="${url}" target="_blank" rel="noopener" class="doc-link">${title}</a></strong> - <span style="color: #666; font-style: italic;">Click to view document</span></li>`;
                } else {
                    // If URL format doesn't match expected pattern, just show as text
                    const formattedSug = sug.replace(
                        /\*\*(.*?)\*\*/g,
                        '<strong>$1</strong>'
                    );
                    html += `<li>${formattedSug}</li>`;
                }
            } else {
                const formattedSug = sug.replace(
                    /\*\*(.*?)\*\*/g,
                    '<strong>$1</strong>'
                );
                html += `<li>${formattedSug}</li>`;
            }
        }
        html += `</ul>`;
        html += `</div>`;
    }

    // Enhanced References Section with clickable document names (removes duplicates)
    if (references && Array.isArray(references) && references.length > 0) {
        console.log('>>>> references', references);
        const uni = {};
        const unique = references.filter((r) => {
            if (!uni[r.url]) {
                uni[r.url] = 1;
                return true;
            }

            return false;
        });
        console.log('>>>> unique', unique);

        html += `<div class="response-section references-section">`;
        html += `<span class="section-heading"> Source Documents</span>`;

        html += `<div class="reference-grid">`;
        for (let ref of unique) {
            if (typeof ref === 'object' && ref !== null) {
                const title = ref.title || ref.name || 'Reference';
                const url = ref.url;
                const program = ref.program || 'All';
                const lob = ref.lob || '';
                const score = ref.score || 0;

                html += `<div class="reference-card">`;
                html += `<div class="ref-header">`;

                if (url && url.trim()) {
                    html += `<a href="${url}" target="_blank" rel="noopener" class="reference-title-link">`;
                    html += ` ${title}`;
                    html += `<span class="external-icon"></span>`;
                    html += `</a>`;
                } else {
                    html += `<span class="reference-title"> ${title}</span>`;
                }

                html += `</div>`;
                html += `<div class="ref-meta">`;
                html += `<span class="ref-program">${program}</span>`;
                if (lob) html += `<span class="ref-lob">${lob}</span>`;
                if (score > 0)
                    html += `<span class="ref-score">Score: ${score.toFixed(
                        2
                    )}</span>`;
                html += `</div>`;

                if (url && url.trim()) {
                    html += `<div class="ref-action">`;
                    html += `<small><a href="${url}" target="_blank" rel="noopener" class="view-doc-link">Click to view document →</a></small>`;
                    html += `</div>`;
                }

                html += `</div>`;
            }
        }
        html += `</div>`;
        html += `</div>`;
    }

    // Helpful Rating Section
    const articleId =
        typeof raw === 'object' && raw !== null
            ? raw.search_query || summary.slice(0, 100)
            : summary.slice(0, 100);

    html += `
    <div class="helpful-rating">
        <span class="rate-label">Was this response helpful?</span>
        <button class="thumb-btn" onclick="rateHelpful(this, 'up', '${articleId.replace(
            /'/g,
            ''
        )}')">
            <span class="material-icons">thumb_up</span>
        </button>
        <button class="thumb-btn" onclick="rateHelpful(this, 'down', '${articleId.replace(
            /'/g,
            ''
        )}')">
            <span class="material-icons">thumb_down</span>
        </button>
        <span class="rate-confirm">Thank you!</span>
    </div>
    `;

    return html;
}

function rateHelpful(btn, value, articleId) {
    const storageKey = 'voted_' + articleId;
    if (sessionStorage.getItem(storageKey)) return;

    const parent = btn.parentElement;
    parent
        .querySelectorAll('.thumb-btn')
        .forEach((b) => b.classList.remove('selected'));
    btn.classList.add('selected');

    const confirm = parent.querySelector('.rate-confirm');
    if (confirm) {
        confirm.style.display = 'inline';
        setTimeout(() => {
            confirm.style.display = 'none';
        }, 2000);
    }

    fetch('/rate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            article_id: articleId,
            value: value,
        }),
    }).catch(() => {
        /* ignore errors */
    });

    sessionStorage.setItem(storageKey, value);
}

async function sendMessage() {
    const input = document.getElementById('chatInput');
    const text = input.value.trim();
    if (!text) return;

    // Exit search mode and show chat
    isSearchMode = false;
    document.getElementById('search-results').style.display = 'none';
    document.getElementById('messages').style.display = 'block';

    // Add user message and render
    addMessage('user', text);
    input.value = '';
    
    // Show typing indicator
    showTyping();

    const programsUsed = (currentChat && currentChat.programs) || programs;

    // Debug logging
    console.log('=== CHAT DEBUG ===');
    console.log('Current programs array:', programs);
    console.log('CurrentChat programs:', currentChat?.programs);
    console.log('Programs being sent:', programsUsed);
    console.log('Message:', text);

    let payload = {
        message: text,
        programs: programsUsed,
        history: currentChat
            ? currentChat.messages
                  .filter((m) => m.role && m.text)
                  .map((m) => ({
                      role: m.role === 'assistant' ? 'assistant' : 'user',
                      content: m.text,
                  }))
            : [],
    };

    console.log('Full payload:', JSON.stringify(payload, null, 2));

    let data;
    try {
        const res = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        console.log('Response status:', res.status);
        data = await res.json();
        console.log('Response data:', data);
    } catch (e) {
        console.error('Fetch error:', e);
        hideTyping();
        addMessage('assistant', 'Error: Could not connect to server.');
        return;
    }

    hideTyping();
    addMessage('assistant', data, data);  // This will trigger smart scroll to top of response
    updateStarButton();
}
// Fixed search function
function searchDocs() {
    const input = document.getElementById('searchInput');
    const query = input?.value.trim();
    if (!query) {
        alert('Please enter a search term');
        return;
    }

    console.log(' Searching for:', query);

    // Enter search mode
    isSearchMode = true;
    document.getElementById('messages').style.display = 'none';
    document.getElementById('search-results').style.display = 'block';

    // Show loading
    const resultsContainer = document.getElementById('search-results');
    resultsContainer.innerHTML = '<div class="loading">🔍 Searching...</div>';

    fetch(`/search?q=${encodeURIComponent(query)}`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
    })
        .then((res) => {
            console.log('Search response status:', res.status);
            return res.json();
        })
        .then((data) => {
            console.log('Search response data:', data);
            if (data && data.results) {
                renderSearchResults(data.results, query);
            } else {
                resultsContainer.innerHTML =
                    '<div class="empty-state">❌ No results found. Try different keywords.</div>';
            }
        })
        .catch((err) => {
            console.error('Search failed:', err);
            resultsContainer.innerHTML = `<div class="empty-state">❌ Search failed: ${err.message}</div>`;
        });
}

// Optional: Add a "scroll to top of message" button for long responses
function addScrollToTopButton(messageElement, messageId) {
    // Validate inputs
    if (!messageElement) {
        console.warn('addScrollToTopButton: messageElement is null');
        return;
    }

    const scrollButton = document.createElement('button');
    scrollButton.className = 'scroll-to-top-btn';
    scrollButton.innerHTML = '⬆ Back to Top';
    scrollButton.onclick = () => {
        messageElement.scrollIntoView({ 
            behavior: 'smooth', 
            block: 'start' 
        });
    };

    // FIXED: Find the correct element to append to
    const messageContent = messageElement.querySelector('.message-content');
    if (messageContent && messageContent.scrollHeight > 600) {
        // Check if button already exists to avoid duplicates
        const existingButton = messageContent.querySelector('.scroll-to-top-btn');
        if (!existingButton) {
            messageContent.appendChild(scrollButton);
        }
    }
}

function renderSearchResults(results, query) {
    const container = document.getElementById('search-results');
    container.innerHTML = '';

    if (results.length === 0) {
        container.innerHTML =
            '<div class="empty-state">No results found. Try a different search term.</div>';
        return;
    }

    // Add header
    const header = document.createElement('div');
    header.style.cssText =
        'background: #f8f9fa; padding: 16px; border-radius: 6px; margin-bottom: 20px; border-left: 4px solid #6E32A0;';
    header.innerHTML = `
        <h3 style="margin: 0 0 8px 0; color: #6E32A0;">🔍 Search Results for "${query}"</h3>
        <p style="margin: 0; color: #666;">Found ${results.length} relevant documents</p>
    `;
    container.appendChild(header);

    results.forEach((r) => {
        const docDiv = document.createElement('div');
        docDiv.className = 'search-result-item';

        // Create title (clickable if URL exists)
        const titleEl = document.createElement('div');
        if (r.url && r.url.trim() && r.url !== '#') {
            const link = document.createElement('a');
            link.href = r.url;
            link.target = '_blank';
            link.rel = 'noopener';
            link.className = 'search-title';
            link.innerHTML = ` ${
                r.title || 'Document'
            } <span style="font-size: 0.8em;"></span>`;
            titleEl.appendChild(link);
        } else {
            titleEl.className = 'search-title';
            titleEl.innerHTML = ` ${r.title || 'Document'}`;
            titleEl.style.color = '#333';
            titleEl.style.textDecoration = 'none';
        }
        docDiv.appendChild(titleEl);

        // Add metadata
        const metaDiv = document.createElement('div');
        metaDiv.style.cssText =
            'display: flex; gap: 10px; margin: 8px 0; font-size: 0.85em;';
        metaDiv.innerHTML = `
            <span style="background: #e3f2fd; padding: 2px 6px; border-radius: 3px;"> ${
                r.collection || 'Unknown'
            }</span>
            <span style="background: #f3e5f5; padding: 2px 6px; border-radius: 3px;"> ${
                r.id || 'N/A'
            }</span>
            ${
                r.program
                    ? `<span style="background: #e8f5e8; padding: 2px 6px; border-radius: 3px;"> ${r.program}</span>`
                    : ''
            }
            ${
                r.score
                    ? `<span style="background: #fff3e0; padding: 2px 6px; border-radius: 3px;"> ${
                          Math.round(r.score * 100) / 100
                      }</span>`
                    : ''
            }
        `;
        docDiv.appendChild(metaDiv);

        // Add preview text
        if (r.chunk && r.chunk.text) {
            const preview = document.createElement('div');
            preview.className = 'search-preview';
            preview.innerHTML = `<strong>Preview:</strong><br>${r.chunk.text.slice(
                0,
                200
            )}...`;
            docDiv.appendChild(preview);
        }

        // Add action buttons
        const actionsDiv = document.createElement('div');
        actionsDiv.style.cssText =
            'margin-top: 10px; display: flex; gap: 10px;';

        if (r.url && r.url.trim() && r.url !== '#') {
            const viewBtn = document.createElement('a');
            viewBtn.href = r.url;
            viewBtn.target = '_blank';
            viewBtn.rel = 'noopener';
            viewBtn.style.cssText =
                'background: #10b981; color: white; padding: 6px 12px; border-radius: 4px; text-decoration: none; font-size: 0.9em;';
            viewBtn.innerHTML = ' View Document';
            actionsDiv.appendChild(viewBtn);
        }

        const chatBtn = document.createElement('button');
        chatBtn.style.cssText =
            'background: #6E32A0; color: white; border: none; padding: 6px 12px; border-radius: 4px; cursor: pointer; font-size: 0.9em;';
        chatBtn.innerHTML = ' Ask About This';
        chatBtn.onclick = () => useInChat(query, r.title || 'Document');
        actionsDiv.appendChild(chatBtn);

        docDiv.appendChild(actionsDiv);
        container.appendChild(docDiv);
    });

    // Add back to chat button
    const backBtn = document.createElement('button');
    backBtn.textContent = '← Back to Chat';
    backBtn.style.cssText =
        'background: #6c757d; color: white; border: none; padding: 10px 20px; border-radius: 6px; cursor: pointer; margin-top: 20px; width: 100%;';
    backBtn.onclick = () => {
        isSearchMode = false;
        document.getElementById('search-results').style.display = 'none';
        document.getElementById('messages').style.display = 'block';
        document.getElementById('searchInput').value = '';
    };
    container.appendChild(backBtn);
}

// Helper function to use search result in chat
function useInChat(query, docName) {
    // Switch back to chat view
    isSearchMode = false;
    document.getElementById('search-results').style.display = 'none';
    document.getElementById('messages').style.display = 'block';

    // Pre-fill chat input with a question about the document
    const chatInput = document.getElementById('chatInput');
    if (chatInput) {
        chatInput.value = `Tell me more about "${docName}" related to ${query}`;
        chatInput.focus();
    }
}

function initializeApp() {
    console.log(' Initializing Metro AI Chat App');

    loadSavedChats();

    if (chatSessions.length === 0) {
        startNewChat();
    }

    renderSidebar();
    renderMessages();
    updateStarButton();
    document.getElementById('chatInput')?.focus();

    console.log(' App initialized successfully');
}

// Expose functions globally for onclick handlers
window.sendMessage = sendMessage;
window.searchDocs = searchDocs;
window.toggleStarChat = toggleStarChat;
window.startNewChat = startNewChat;
window.clearCurrentChat = clearCurrentChat;
window.switchTab = switchTab;
window.toggleProgram = toggleProgram;
window.toggleMessageExpand = toggleMessageExpand;
window.rateHelpful = rateHelpful;
window.useInChat = useInChat;

// Initialize when DOM is loaded
window.addEventListener('DOMContentLoaded', function () {
    console.log(' DOM Content Loaded');

    initializeApp();

    // Set up event listeners
    const sendBtn = document.getElementById('sendBtn');
    const chatInput = document.getElementById('chatInput');
    const searchBtn = document.getElementById('searchBtn');
    const searchInput = document.getElementById('searchInput');

    if (sendBtn) {
        sendBtn.addEventListener('click', sendMessage);
        console.log(' Send button listener attached');
    } else {
        console.error(' Send button not found');
    }

    if (chatInput) {
        chatInput.addEventListener('keydown', function (e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                sendMessage();
            }
        });
        console.log(' Chat input listener attached');
    } else {
        console.error(' Chat input not found');
    }

    if (searchBtn) {
        searchBtn.addEventListener('click', function (e) {
            e.preventDefault();
            console.log('🔍 Search button clicked');
            searchDocs();
        });
        console.log(' Search button listener attached');
    } else {
        console.error(' Search button not found');
    }

    if (searchInput) {
        searchInput.addEventListener('keydown', function (e) {
            if (e.key === 'Enter') {
                e.preventDefault();
                console.log(' Search input Enter pressed');
                searchDocs();
            }
        });
        console.log(' Search input listener attached');
    } else {
        console.error(' Search input not found');
    }

    console.log(' All event listeners set up successfully');
});
