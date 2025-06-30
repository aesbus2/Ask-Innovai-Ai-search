let chats = [];
let chatSessions = []; // [{name, messages, id, programs}]
let currentChat = null;
let typingTimeout = null;
let programs = ['Metro'];

const SYSTEM_WELCOME_MSG = "How can I help you today? I can answer questions, walk you through processes, or help you find the information you need.";

// Util for truncating
function truncate(str, n) {
    if (!str) return '';
    str = str.replace(/[\n\r]+/g, ' ').trim();
    return str.length > n ? str.slice(0, n) + "…" : str;
}

function renderSidebar() {
    const chatsDiv = document.getElementById('chats');
    chatsDiv.innerHTML = '';
    chatSessions.forEach((session, idx) => {
        const active = session === currentChat ? 'active' : '';
        const name = truncate(session.name || '[]', 30);
        const el = document.createElement('div');
        el.className = `chat-link ${active}`;
        el.textContent = name;
        el.onclick = () => { selectChat(session); };
        chatsDiv.appendChild(el);
    });
}

function selectChat(session) {
    currentChat = session;
    renderSidebar();
    renderMessages();
}

function startNewChat() {
    const session = { name: '', messages: [], id: Date.now(), programs: [...programs] };
    session.messages.push({ role: 'system', text: SYSTEM_WELCOME_MSG });
    chatSessions.unshift(session);
    selectChat(session);
}

function clearAllChats() {
    if (confirm('Clear all chat history?')) {
        chatSessions = [];
        startNewChat();
    }
}

function toggleProgram(checkbox) {
    if (checkbox.checked) {
        programs.push(checkbox.value);
    } else {
        programs = programs.filter(p => p !== checkbox.value);
    }
    if (currentChat) {
        currentChat.programs = [...programs];
    }
}

function addMessage(role, text, rawData) {
    if (!currentChat) startNewChat();
    if (role === 'assistant' && typeof text === 'object' && text !== null) {
        text = text.summary || text.reply || JSON.stringify(text);
    }
    currentChat.messages.push({ role, text, rawData });
    if (role === 'user') {
        currentChat.name = truncate(text, 34);
        renderSidebar();
    }
    renderMessages();
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
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}
function hideTyping() {
    const el = document.getElementById('typing-indicator');
    if (el) el.remove();
}

function renderMessages() {
    const messagesDiv = document.getElementById('messages');
    messagesDiv.innerHTML = '';
    if (!currentChat) return;
    for (const msg of currentChat.messages) {
        let row = document.createElement('div');
        if (msg.role === 'system') {
            row.className = 'message-row system';
            row.innerHTML = `<div class="system-msg">${msg.text}</div>`;
        } else if (msg.role === 'user') {
            row.className = 'message-row user';
            row.innerHTML = `<div class="user-msg">${msg.text}</div>`;
        } else {
            row.className = 'message-row';
            row.innerHTML = `<div class="assistant-msg">${formatAIResponse(msg.rawData || msg.text, msg.text)}</div>`;
        }
        messagesDiv.appendChild(row);
    }
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function formatAIResponse(raw, fallbackText) {
    let summary = '', steps = [], suggestions = [], articleId = '';
    if (typeof raw === 'object' && raw !== null && (raw.summary || raw.steps || raw.suggestions || raw.reply)) {
        summary = raw.summary || raw.reply || '';
        steps = raw.steps || [];
        suggestions = raw.suggestions || [];
        articleId = raw.id || raw.title || summary.slice(0, 1000);
    } else if (typeof raw === 'string') {
        const lines = raw.split('\n').filter(Boolean);
        summary = lines[0] || '';
        steps = lines.slice(1, 10);
        suggestions = lines.slice(5);
        articleId = summary.slice(0, 1000);
    }

    let html = '';
    html += `<span class="section-heading">Summary</span>`;
    html += `<div style="margin-bottom:14px;">${summary || fallbackText}</div>`;
    if (steps.length) {
        html += `<span class="section-heading">Step-by-Step Answer</span><ol class="step-list">`;
        for (let s of steps) html += `<li>${s}</li>`;
        html += `</ol>`;
    }
    if (suggestions.length) {
        html += `<span class="section-heading">Suggested Content</span><ul class="suggest-list">`;
        for (let sug of suggestions) html += `<li>${sug}</li>`;
        html += `</ul>`;
    }
    if (raw && typeof raw === 'object' && Array.isArray(raw.references) && raw.references.length > 0) {
        html += `<span class="section-heading">References</span><ul class="suggest-list">`;
        for (let ref of raw.references) {
            if (ref.url && ref.url !== "#") {
                html += `<li><a href="${ref.url}" target="_blank">${ref.title || 'Reference'}</a></li>`;
            } else if (ref.title) {
                html += `<li>${ref.title}</li>`;
            }
        }
        html += `</ul>`;
    }
    html += `
    <div class="helpful-rating">
        <span class="rate-label">Was this article helpful?</span>
        <button class="thumb-btn" onclick="rateHelpful(this, 'up', '${articleId.replace(/'/g,"")}')">
            <span class="material-icons">thumb_up</span>
        </button>
        <button class="thumb-btn" onclick="rateHelpful(this, 'down', '${articleId.replace(/'/g,"")}')">
            <span class="material-icons">thumb_down</span>
        </button>
        <span class="rate-confirm">Thank you!</span>
    </div>
    `;
    return html;
}

function rateHelpful(btn, value, articleId) {
    if (localStorage.getItem('voted_' + articleId)) return;
    const parent = btn.parentElement;
    parent.querySelectorAll('.thumb-btn').forEach(b => b.classList.remove('selected'));
    btn.classList.add('selected');
    const confirm = parent.querySelector('.rate-confirm');
    if (confirm) {
        confirm.style.display = '';
        setTimeout(() => { confirm.style.display = 'none'; }, 2000);
    }
    fetch('/rate', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({
            article_id: articleId,
            value: value
        })
    }).catch(() => { /* ignore errors for now */ });
    localStorage.setItem('voted_' + articleId, value);
}

async function sendMessage() {
    const input = document.getElementById('chatInput');
    const text = input.value.trim();
    if (!text) return;
    addMessage('user', text);
    input.value = '';
    renderMessages();
    showTyping();
    const programsUsed = (currentChat && currentChat.programs) || programs;
    let payload = {
        message: text,
        programs: programsUsed,
        history: (currentChat ? currentChat.messages.filter(m => m.role && m.text).map(m => ({
            role: m.role === 'assistant' ? 'assistant' : 'user',
            content: m.text
        })) : [])
    };
    let data;
    try {
        const res = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        data = await res.json();
    } catch (e) {
        hideTyping();
        addMessage('assistant', 'Error: Could not connect to server.');
        return;
    }
    hideTyping();
    addMessage('assistant', data, data);
}

// --- SEARCH RESULT UI ---
function renderSearchResults(results) {
  const container = document.getElementById('search-results');
  container.innerHTML = '';
  results.forEach(r => {
    const docDiv = document.createElement('div');
    docDiv.className = 'search-result-item';
    const link = document.createElement('a');
    link.href = r.url;
    link.target = '_blank';
    link.textContent = r.doc_name || 'View Document';
    docDiv.appendChild(link);
    if (r.full_doc && r.full_doc.chunk && r.full_doc.chunk.text) {
      const preview = document.createElement('div');
      preview.className = 'search-preview';
      preview.textContent = r.full_doc.chunk.text.slice(0, 180) + '...';
      docDiv.appendChild(preview);
    }
    if (r.collection) {
      const coll = document.createElement('span');
      coll.className = 'search-collection';
      coll.textContent = ` [${r.collection}]`;
      docDiv.appendChild(coll);
    }
    container.appendChild(docDiv);
  });
}
// To trigger, after getting search results from /search:
// renderSearchResults(data.results);

window.onload = function() {
    if (chatSessions.length === 0) startNewChat();
    renderSidebar();
    renderMessages();
    document.getElementById('chatInput').focus();
};
