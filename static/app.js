/* ══════════════════════════════
   RAG 知识库管理 v2 - 前端
   新增：SSE 流式聊天 + 设置
   ══════════════════════════════ */

const API = {
  async get(path) {
    const r = await fetch(path);
    const d = await r.json();
    if (!r.ok && d.error_type) { const e = new Error(d.detail || d.error); e.type = d.error_type; throw e; }
    return d;
  },
  async post(path, body) {
    const r = await fetch(path, { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body) });
    const d = await r.json();
    if (!r.ok && d.error_type) { const e = new Error(d.detail || d.error); e.type = d.error_type; throw e; }
    return d;
  },
  async del(path) {
    const r = await fetch(path, { method:'DELETE' });
    const d = await r.json();
    if (!r.ok && d.error_type) { const e = new Error(d.detail || d.error); e.type = d.error_type; throw e; }
    return d;
  },
};

function friendlyError(e) {
  if (e.type === 'rate_limit') return '操作过于频繁，请稍后再试';
  if (e.type === 'loading') return '系统加载中，请稍候';
  if (e.type === 'not_found') return '资源不存在';
  if (e.type === 'forbidden') return '权限不足';
  return e.message || '未知错误';
}

function escapeHtml(text) {
  const d = document.createElement('div');
  d.textContent = text;
  return d.innerHTML;
}

function safeJSONParse(text) {
  try { return JSON.parse(text); } catch (e) { return null; }
}

function toast(msg, type='') {
  let el = document.querySelector('.toast');
  if (!el) { el = document.createElement('div'); el.className='toast'; document.body.appendChild(el); }
  el.textContent = msg; el.className = `toast ${type} show`;
  clearTimeout(el._tid); el._tid = setTimeout(() => el.classList.remove('show'), 2500);
}

/* ══════════════════════════════
   Navigation
   ══════════════════════════════ */

document.querySelectorAll('.nav-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    const page = btn.dataset.page;
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.getElementById(`page-${page}`).classList.add('active');
    if (page === 'search') loadStatus();
    if (page === 'manage') loadDocs();
    if (page === 'settings') { loadStatus(); loadLLMSettings(); loadConfig(); }
    if (page === 'chat') { setChatSubtitle(); loadModels(); }
    if (page === 'crawler') loadSources();
    if (page === 'categories') loadCategories();
  });
});

/* ══════════════════════════════
   Chat (SSE Streaming + RAG)
   ══════════════════════════════ */

let chatHistory = [];
let currentSSE = null;

const chatInput = document.getElementById('chat-input');
const chatSendBtn = document.getElementById('chat-send-btn');
const chatStopBtn = document.getElementById('chat-stop-btn');
const chatMessages = document.getElementById('chat-messages');
const chatStatus = document.getElementById('chat-status');
const chatSubtitle = document.getElementById('chat-subtitle');

function setChatSubtitle() {
  API.get('/api/llm/status').then(d => {
    if (d.enabled) chatSubtitle.textContent = d.model + ' · 基于知识库的智能问答（RAG）';
    else chatSubtitle.textContent = 'LLM 未配置 · 请在「系统设置」中填写 API Key';
  }).catch(() => {});
}

setChatSubtitle();

chatInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChat(); }
});

chatInput.addEventListener('input', () => {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 160) + 'px';
});

chatSendBtn.addEventListener('click', sendChat);
chatStopBtn.addEventListener('click', stopChat);

function scrollChat() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function addMessage(role, content, isStreaming = false) {
  // Remove empty-state if exists
  const empty = chatMessages.querySelector('.chat-empty');
  if (empty) empty.remove();

  // Find or create streaming bubble
  let bubble;
  if (isStreaming && role === 'ai') {
    let existing = chatMessages.querySelector('.chat-streaming');
    if (existing) {
      bubble = existing;
      bubble.innerHTML = renderMarkdown(content);
    } else {
      const div = document.createElement('div');
      div.className = 'chat-message chat-msg-ai chat-streaming';
      div.innerHTML = `<div class="chat-bubble">${renderMarkdown(content)}</div>`;
      chatMessages.appendChild(div);
      bubble = div;
    }
  } else {
    // Remove streaming marker
    const streaming = chatMessages.querySelector('.chat-streaming');
    if (streaming) streaming.classList.remove('chat-streaming');

    const div = document.createElement('div');
    div.className = `chat-message chat-msg-${role === 'user' ? 'user' : 'ai'}`;
    div.innerHTML = `<div class="chat-bubble">${renderMarkdown(content)}</div>`;
    chatMessages.appendChild(div);
  }
  scrollChat();
}

function addContextTag(count) {
  const tag = document.createElement('div');
  tag.className = 'chat-context-tag';
  tag.textContent = `📚 参考 ${count} 条知识库内容`;
  chatMessages.appendChild(tag);
  scrollChat();
}

function addTyping() {
  const div = document.createElement('div');
  div.className = 'chat-typing';
  div.textContent = '正在检索知识库';
  div.id = 'chat-typing-indicator';
  chatMessages.appendChild(div);
  scrollChat();
}

function removeTyping() {
  const el = document.getElementById('chat-typing-indicator');
  if (el) el.remove();
}

function renderMarkdown(text) {
  // Simple Markdown → HTML (headers, bold, code, lists)
  let html = text
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/^### (.+)$/gm, '<h5>$1</h5>')
    .replace(/^## (.+)$/gm, '<h4>$1</h4>')
    .replace(/^# (.+)$/gm, '<h3>$1</h3>')
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/^- (.+)$/gm, '• $1')
    .replace(/\n/g, '<br>');
  return html;
}

async function sendChat() {
  const query = chatInput.value.trim();
  if (!query || currentSSE) return;

  const useRAG = document.getElementById('rag-toggle').checked;
  const topK = +document.getElementById('chat-topk').value || 5;

  // UI state
  chatInput.value = '';
  chatInput.style.height = 'auto';
  chatSendBtn.style.display = 'none';
  chatStopBtn.style.display = 'inline-flex';
  chatStatus.textContent = useRAG ? '正在检索知识库...' : '正在连接 AI...';

  addMessage('user', query);
  addTyping();

  try {
    const selectedModel = document.getElementById('chat-model-select').value;
    currentSSE = new EventSourcePolyfill('/api/chat', {
      query, use_rag: useRAG, top_k: topK,
      model: selectedModel || undefined,
      history: chatHistory.slice(-10),
    });

    let fullText = '';
    let contextCount = 0;
    let hasContextTag = false;

    currentSSE.onEvent = (event, data) => {
      if (event === 'context') {
        const ctx = safeJSONParse(data);
        if (!ctx) return;
        contextCount = ctx.context_count;
        removeTyping();
        if (contextCount > 0 && !hasContextTag) {
          addContextTag(contextCount);
          hasContextTag = true;
        }
        chatStatus.textContent = '';
        return;
      }

      if (event === 'token') {
        fullText += data;
        // Throttle re-render: every 8 tokens or 80ms
        if (fullText.length % 8 === 0) {
          addMessage('ai', fullText, true);
        }
        return;
      }

      if (event === 'result') {
        currentSSE = null;
        chatSendBtn.style.display = 'inline-flex';
        chatStopBtn.style.display = 'none';
        chatStatus.textContent = contextCount > 0
          ? `已用 ${contextCount} 条知识库文档回答`
          : '未使用知识库';
        // Finalize message
        const streaming = chatMessages.querySelector('.chat-streaming');
        if (streaming) streaming.classList.remove('chat-streaming');
        // Save to history
        chatHistory.push({ role: 'user', content: query });
        chatHistory.push({ role: 'assistant', content: fullText });
        if (chatHistory.length > 20) chatHistory = chatHistory.slice(-20);
        return;
      }

      if (event === 'error') {
        const err = safeJSONParse(data) || {};
        addMessage('ai', '❌ 错误: ' + (err.error || '未知错误'));
        cleanupChat();
        return;
      }

      if (event === 'done') {
        cleanupChat();
      }
    };

    currentSSE.start();

  } catch (e) {
    addMessage('ai', '❌ 连接失败: ' + e.message);
    cleanupChat();
  }
}

function cleanupChat() {
  currentSSE = null;
  chatSendBtn.style.display = 'inline-flex';
  chatStopBtn.style.display = 'none';
  removeTyping();
}

function stopChat() {
  if (currentSSE) { currentSSE.close(); currentSSE = null; }
  cleanupChat();
  chatStatus.textContent = '已停止';
}

/* ══════════════════════════════
   SSE Polyfill for POST
   ══════════════════════════════ */

class EventSourcePolyfill {
  constructor(url, body) {
    this.url = url;
    this.body = body;
    this.onEvent = null;
    this._xhr = null;
    this._lastIndex = 0;
  }

  start() {
    this._xhr = new XMLHttpRequest();
    this._xhr.open('POST', this.url, true);
    this._xhr.setRequestHeader('Content-Type', 'application/json');
    this._xhr.setRequestHeader('Accept', 'text/event-stream');

    let buffer = '';
    let dataBuffer = '';
    let eventType = '';
    this._xhr.onprogress = () => {
      const newData = this._xhr.responseText.substring(this._lastIndex);
      this._lastIndex = this._xhr.responseText.length;
      buffer += newData;

      // SSE parser: line-by-line to handle TCP fragmentation
      while (true) {
        const nl = buffer.indexOf('\n');
        if (nl === -1) break;
        let line = buffer.substring(0, nl);
        buffer = buffer.substring(nl + 1);
        if (line.endsWith('\r')) line = line.substring(0, line.length - 1);

        if (line.startsWith('event: ')) {
          eventType = line.slice(7).trim();
        } else if (line.startsWith('data: ')) {
          dataBuffer = line.slice(6);
        } else if (line === '' && dataBuffer) {
          if (this.onEvent) this.onEvent(eventType, dataBuffer);
          eventType = ''; dataBuffer = '';
        }
      }
    };

    this._xhr.onloadend = () => {
      // Final flush: parse remaining buffer
      if (buffer && dataBuffer && this.onEvent) {
        this.onEvent(eventType, dataBuffer);
      }
      if (this.onEvent) this.onEvent('done', '[DONE]');
      this._xhr = null;  // prevent double-fire from onerror after onloadend
    };

    this._xhr.onerror = () => {
      if (this.onEvent) this.onEvent('error', JSON.stringify({error: '网络连接失败'}));
    };

    this._xhr.send(JSON.stringify(this.body));
  }

  close() {
    if (this._xhr) { this._xhr.abort(); this._xhr = null; }
  }
}

/* ══════════════════════════════
   Search
   ══════════════════════════════ */

document.getElementById('search-btn').addEventListener('click', doSearch);
document.getElementById('search-input').addEventListener('keydown', e => { if (e.key === 'Enter') doSearch(); });

async function doSearch() {
  const query = document.getElementById('search-input').value.trim();
  if (!query) return;
  const loading = document.getElementById('search-loading');
  const results = document.getElementById('results');
  const count = document.getElementById('result-count');
  loading.style.display = 'block';
  results.innerHTML = ''; count.textContent = '';
  const data = await API.post('/api/search', {
    query,
    top_k: +document.getElementById('topk').value || 5,
    threshold: +document.getElementById('threshold').value || 0.4,
  });
  loading.style.display = 'none';
  count.textContent = `找到 ${data.count} 条`;
  if (!data.results || data.results.length === 0) {
    results.innerHTML = '<div class="empty-result">没有找到匹配的内容<br><span style="font-size:13px;color:#94a3b8">试试换个问法，或降低相似度阈值</span></div>';
    return;
  }
  data.results.forEach(r => {
    const pct = Math.round(r.score * 100);
    let level = 'high'; if (pct < 70) level = 'mid'; if (pct < 50) level = 'low';
    const docName = (r.metadata && r.metadata.title) || '';
    const card = document.createElement('div');
    card.className = 'result-card';
    card.innerHTML = `
      <div class="result-header">
        <span class="score-badge ${level}">${pct}%</span>
        <span class="result-doc-name">${escapeHtml(docName)}</span>
      </div>
      <div class="result-content">${escapeHtml(r.content)}</div>`;
    results.appendChild(card);
  });
}

/* ══════════════════════════════
   Ingest
   ══════════════════════════════ */

document.getElementById('ingest-text-btn').addEventListener('click', async () => {
  const content = document.getElementById('paste-content').value.trim();
  if (!content) return toast('请先输入文本内容', 'error');
  const title = document.getElementById('paste-title').value.trim() || '手动粘贴';
  const source = document.getElementById('paste-source').value.trim() || 'web_ui';
  const data = await API.post('/api/ingest/text', { content, title, source });
  if (data.success) {
    toast(`已摄入: ${title} (${data.chunk_count} 块)`, 'success');
    document.getElementById('paste-content').value = '';
    document.getElementById('paste-title').value = '';
    document.getElementById('paste-source').value = '';
  } else { toast(data.error || '摄入失败', 'error'); }
});

const fileQueue = [];
const dropZone = document.getElementById('drop-zone');
dropZone.addEventListener('click', () => document.getElementById('file-input').click());
document.getElementById('file-input').addEventListener('change', e => {
  [...e.target.files].forEach(f => addToQueue(f.name, false)); e.target.value = '';
});
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('dragover');
  [...e.dataTransfer.files].forEach(f => addToQueue(f.path || f.name, !!f.path));
});

function addToQueue(name, isPath) {
  if (fileQueue.find(f => f.name === name)) return;
  fileQueue.push({ name, path: isPath ? name : null }); renderFileQueue();
}

function renderFileQueue() {
  document.getElementById('file-queue').innerHTML = fileQueue.map((f, i) =>
    `<div class="file-item"><span>📄 ${f.name}</span><button onclick="removeFromQueue(${i})">✕</button></div>`
  ).join('');
  document.getElementById('batch-ingest-btn').disabled = fileQueue.length === 0;
}

function removeFromQueue(i) { fileQueue.splice(i, 1); renderFileQueue(); }

document.getElementById('import-dir-btn').addEventListener('click', async () => {
  const dir = document.getElementById('dir-path').value.trim();
  if (!dir) return toast('请输入文件夹路径', 'error');
  const btn = document.getElementById('import-dir-btn');
  btn.textContent = '导入中...'; btn.disabled = true;
  const data = await API.post('/api/ingest/directory', { directory: dir });
  btn.textContent = '导入文件夹'; btn.disabled = false;
  if (data.success) toast(`导入完成: ${data.imported}/${data.total}`, 'success');
  else toast(data.error || '导入失败', 'error');
});

document.getElementById('batch-ingest-btn').addEventListener('click', async () => {
  if (fileQueue.length === 0) return;
  const btn = document.getElementById('batch-ingest-btn');
  btn.textContent = '导入中...'; btn.disabled = true;
  const data = await API.post('/api/ingest/files', { files: fileQueue.map(f => f.path || f.name) });
  if (data.success) { toast(`导入完成: ${data.imported}/${data.total}`, 'success'); fileQueue.length = 0; renderFileQueue(); }
  else toast(data.error || '导入失败', 'error');
  btn.textContent = '🚀 批量导入'; btn.disabled = fileQueue.length === 0;
});

/* ══════════════════════════════
   Manage
   ══════════════════════════════ */

document.getElementById('refresh-docs-btn').addEventListener('click', loadDocs);
document.getElementById('select-all').addEventListener('change', e => {
  document.querySelectorAll('#docs-tbody input[type=checkbox]').forEach(cb => cb.checked = e.target.checked);
  updateDeleteBtn();
});

document.getElementById('delete-selected-btn').addEventListener('click', async () => {
  const checked = document.querySelectorAll('#docs-tbody input[type=checkbox]:checked');
  if (checked.length === 0) return;
  if (!confirm(`确定删除选中的 ${checked.length} 篇文档吗？此操作不可撤消。`)) return;
  const docIds = [...checked].map(cb => cb.dataset.docId);
  try {
    const data = await API.post('/api/documents/batch-delete', { doc_ids: docIds });
    toast(`删除 ${data.success_count}/${data.total} 篇`, 'success');
  } catch (e) {
    toast('批量删除失败: ' + e.message, 'error');
  }
  loadDocs();
});

async function loadDocs() {
  const data = await API.get('/api/documents');
  const tbody = document.getElementById('docs-tbody');
  const count = document.getElementById('doc-count');
  if (data.error) { tbody.innerHTML = `<tr><td colspan="7">${data.error}</td></tr>`; return; }
  const docs = data.documents || [];
  count.textContent = `共 ${docs.length} 篇文档`;
  tbody.innerHTML = docs.map(d => `
    <tr>
      <td><input type="checkbox" data-doc-id="${escapeHtml(d.doc_id)}" onchange="updateDeleteBtn()"></td>
      <td>${escapeHtml(d.title)}</td>
      <td style="color:var(--text-muted);font-size:13px">${escapeHtml(d.source)}</td>
      <td><span style="background:var(--bg-input);padding:2px 8px;border-radius:4px;font-size:12px;color:var(--text-secondary)">${escapeHtml(d.file_type)}</span></td>
      <td>${d.chunk_count}</td>
      <td style="color:var(--text-muted);font-size:13px">${(d.created_at||'').slice(0, 10)}</td>
      <td><button class="btn" style="padding:4px 10px;font-size:12px" onclick="openDocDetail('${escapeHtml(d.doc_id)}')">详情</button></td>
    </tr>`).join('');
  document.getElementById('select-all').checked = false;
  updateDeleteBtn();
}

function updateDeleteBtn() {
  const count = document.querySelectorAll('#docs-tbody input[type=checkbox]:checked').length;
  document.getElementById('delete-selected-btn').disabled = count === 0;
  document.getElementById('delete-selected-btn').textContent = count > 0 ? `🗑 删除选中文档 (${count})` : '🗑 删除选中文档';
}

/* ══════════════════════════════
   Settings
   ══════════════════════════════ */

async function loadLLMSettings() {
  const data = await API.get('/api/llm/status');
  document.getElementById('setting-api-base').value = data.api_base || '';
  document.getElementById('setting-model').value = data.model || '';
  document.getElementById('setting-api-key').value = '';
  document.getElementById('chat-subtitle').textContent = data.enabled
    ? `${data.model} · 基于知识库的智能问答（RAG）`
    : 'LLM 未配置 · 请在「系统设置」中填写 API Key';
}

/* --- Config Display & Model Selector --- */

async function loadConfig() {
  const configDiv = document.getElementById('config-display');
  if (!configDiv) return;

  try {
    const data = await API.get('/api/config');
    if (data.error) {
      configDiv.innerHTML = `<div class="config-error">${escapeHtml(data.error)}</div>`;
      return;
    }

    const llm = data.llm || {};
    configDiv.innerHTML = `
      <table class="config-table">
        <tr>
          <td class="config-label">状态</td>
          <td class="config-value"><span class="status-dot ${llm.enabled ? 'ok' : 'err'}"></span> ${llm.enabled ? '已配置' : '未配置'}</td>
          <td class="config-action"></td>
        </tr>
        <tr>
          <td class="config-label">API 地址</td>
          <td class="config-value config-mono">${escapeHtml(llm.api_base || '-')}</td>
          <td class="config-action"><button class="config-edit-btn" data-field="api_base">✏️</button></td>
        </tr>
        <tr>
          <td class="config-label">模型</td>
          <td class="config-value config-mono">${escapeHtml(llm.model || '-')}</td>
          <td class="config-action"><button class="config-edit-btn" data-field="model">✏️</button></td>
        </tr>
        <tr>
          <td class="config-label">API Key</td>
          <td class="config-value config-mono">${escapeHtml(llm.api_key || '(空)')}</td>
          <td class="config-action"><button class="config-edit-btn" data-field="api_key">✏️</button></td>
        </tr>
        <tr>
          <td class="config-label">Temperature</td>
          <td class="config-value">${llm.temperature}</td>
          <td class="config-action"><button class="config-edit-btn" data-field="temperature">✏️</button></td>
        </tr>
        <tr>
          <td class="config-label">Max Tokens</td>
          <td class="config-value">${llm.max_tokens}</td>
          <td class="config-action"><button class="config-edit-btn" data-field="max_tokens">✏️</button></td>
        </tr>
        <tr>
          <td class="config-label">超时 (s)</td>
          <td class="config-value">${llm.timeout || 120}</td>
          <td class="config-action"><button class="config-edit-btn" data-field="timeout">✏️</button></td>
        </tr>
      </table>
      <div class="config-hint">点击✏️编辑单个字段，保存后需重启服务生效</div>
    `;

    // Bind edit buttons: inline edit with prompt
    configDiv.querySelectorAll('.config-edit-btn').forEach(btn => {
      btn.addEventListener('click', async () => {
        const field = btn.dataset.field;
        const row = btn.closest('tr');
        const valEl = row.querySelector('.config-value');
        const oldVal = valEl ? valEl.textContent.trim() : '';
        const currentVal = oldVal !== '(空)' && oldVal !== '-' ? oldVal : '';
        const newVal = prompt(`请输入新值 ('${field}'):\n留空则取消, 输入 'RESET' 恢复默认\n当前: ${currentVal}`, '');

        if (newVal === null) return;

        if (newVal.toUpperCase() === 'RESET') {
          // Reset to default
          const res = await API.post('/api/config/reset', { field });
          if (res.success) {
            toast(`字段 ${field} 已恢复默认`, 'success');
            loadConfig();
          } else {
            toast(res.error || '恢复失败', 'error');
          }
          return;
        }

        if (!newVal) return; // cancelled

        // Update field
        const rawVal = field === 'temperature' || field === 'timeout' ? parseFloat(newVal) :
                       field === 'max_tokens' ? parseInt(newVal, 10) : newVal;
        const res = await API.post('/api/config/update', { field, value: rawVal });
        if (res.success) {
          toast(`字段 ${field} 已更新`, 'success');
          loadConfig();
        } else {
          toast(res.error || '更新失败', 'error');
        }
      });
    });
  } catch (e) {
    configDiv.innerHTML = `<div class="config-error">加载失败: ${escapeHtml(e.message)}</div>`;
  }
}

async function loadModels() {
  const sel = document.getElementById('chat-model-select');
  if (!sel) return;

  try {
    const data = await API.get('/api/config/models');
    sel.innerHTML = '';

    if (!data.models || data.models.length === 0) {
      sel.innerHTML = '<option value="">未配置 LLM</option>';
      return;
    }

    // Add configured models
    data.models.forEach(m => {
      const opt = document.createElement('option');
      opt.value = m.id;
      opt.textContent = m.name;
      if (m.default) opt.selected = true;
      sel.appendChild(opt);
    });

    // Add separator + custom input option
    const sep = document.createElement('option');
    sep.disabled = true;
    sep.textContent = '--- 自定义 ---';
    sel.appendChild(sep);

    const custom = document.createElement('option');
    custom.value = '__custom__';
    custom.textContent = '✏️ 输入自定义模型';
    sel.appendChild(custom);

    // Handle custom model selection
    sel.onchange = () => {
      if (sel.value === '__custom__') {
        const name = prompt('请输入模型名称:');
        if (name && name.trim()) {
          const opt = document.createElement('option');
          opt.value = name.trim();
          opt.textContent = name.trim() + ' (自定义)';
          opt.selected = true;
          sel.insertBefore(opt, sep);
          sel.value = name.trim();
        } else {
          // Restore to first model
          sel.value = data.default || (data.models[0] ? data.models[0].id : '');
        }
      }
    };
  } catch (e) {
    sel.innerHTML = '<option value="">加载失败</option>';
  }
}

document.getElementById('save-settings-btn').addEventListener('click', async () => {
  const config = {
    api_base: document.getElementById('setting-api-base').value.trim(),
    api_key: document.getElementById('setting-api-key').value.trim(),
    model: document.getElementById('setting-model').value.trim(),
    temperature: +document.getElementById('setting-temperature').value || 0.7,
    max_tokens: +document.getElementById('setting-max-tokens').value || 2048,
  };

  const data = await API.post('/api/settings/save', config);
  if (data.success) {
    toast('设置已保存！需要重启服务生效', 'success');
  } else {
    toast(data.error || '保存失败', 'error');
  }
});

/* ══════════════════════════════
   Status (shared)
   ══════════════════════════════ */

async function loadStatus() {
  const [sysData, llmData] = await Promise.all([API.get('/api/status'), API.get('/api/llm/status')]);
  updateStatusCard('pg', sysData.postgresql, 'PostgreSQL', sysData.postgresql ? '运行中' : '未连接');
  updateStatusCard('llm', llmData.enabled, 'LLM API', llmData.enabled ? `${llmData.model}` : '未配置');
  updateStatusCard('embed', sysData.embedding, 'BGE-M3', sysData.embedding ? `${sysData.model_dim}维, ${sysData.device}` : '未加载');
  updateStatusCard('gpu', sysData.cuda, 'CUDA', sysData.cuda ? `RTX 4060 (CUDA ${sysData.cuda_version})` : '不可用');

  const sidebarDot = document.getElementById('sidebar-status');
  const sidebarText = document.getElementById('sidebar-status-text');
  if (sysData.postgresql && sysData.embedding && sysData.cuda) {
    sidebarDot.className = 'status-dot ok'; sidebarText.textContent = '系统正常';
  } else if (sysData.embedding || sysData.postgresql) {
    sidebarDot.className = 'status-dot warn'; sidebarText.textContent = '部分可用';
  } else {
    sidebarDot.className = 'status-dot err'; sidebarText.textContent = '未就绪';
  }
}

function updateStatusCard(key, ok, name, text) {
  const card = document.getElementById(`card-${key}`);
  if (!card) return;
  const dot = card.querySelector('.status-dot');
  const textEl = card.querySelector('.status-text');
  dot.className = `status-dot ${ok ? 'ok' : 'err'}`;
  textEl.textContent = text;
}

/* Init — 等待引擎就绪后加载 */
(async function initOnReady() {
  const overlay = document.getElementById('loading-overlay');
  const elapsedEl = document.getElementById('loading-elapsed');
  let retries = 0;
  const maxRetries = 120; // 最多等 60 秒

  while (retries < maxRetries) {
    try {
      const r = await fetch('/api/ready');
      const data = await r.json();
      if (data.ready) {
        overlay.classList.add('hidden');
        loadStatus();
        loadDocs();
        setChatSubtitle();
        loadModels();
        return;
      }
      if (data.error) {
        overlay.querySelector('.loading-text').textContent = '引擎加载失败';
        overlay.querySelector('.loading-sub').innerHTML = `<span class="loading-error">${escapeHtml(data.error)}</span>`;
        overlay.querySelector('.loading-elapsed').textContent = '请检查控制台日志后重启';
        return;
      }
      elapsedEl.textContent = `已等待 ${Math.floor(retries * 0.5)} 秒...`;
    } catch (e) {
      // 服务器可能还没启动完毕，继续等待
    }
    await new Promise(r => setTimeout(r, 500));
    retries++;
  }
  // 超时
  overlay.querySelector('.loading-text').textContent = '启动超时';
  overlay.querySelector('.loading-sub').textContent = '请检查终端是否有错误输出';
})();


/* ══════════════════════════════
   Crawler Functions
   ══════════════════════════════ */
let _sourcesCache = [];
let _advancedResultArticles = [];

async function loadSources() {
  const container = document.getElementById('sources-list');
  if (!container) return;
  try {
    const data = await API.get('/api/sources');
    const sources = data.sources || [];
    _sourcesCache = sources;
    if (sources.length === 0) {
      container.innerHTML = '<div class="config-loading">暂无数据源，请在上方添加</div>';
      return;
    }
    container.innerHTML = `
      <table class="config-table">
        <thead><tr><th>名称</th><th>URL</th><th>分类</th><th>操作</th></tr></thead>
        <tbody>${sources.map(s => `
          <tr>
            <td class="config-label">${escapeHtml(s.name)}</td>
            <td class="config-mono" style="font-size:12px">${escapeHtml(s.url)}</td>
            <td><span style="font-size:13px">${escapeHtml(s.category)}</span></td>
            <td style="white-space:nowrap">
              <button class="btn btn-primary" style="padding:4px 12px;font-size:12px" onclick="crawlSource('${escapeHtml(s.name)}', this)">▶ 爬取</button>
              <button class="btn" style="padding:4px 8px;font-size:12px" onclick="toggleSourceSettings('${escapeHtml(s.name)}')" title="设置">⚙</button>
              <button class="btn btn-danger" style="padding:4px 8px;font-size:12px" onclick="removeSource('${escapeHtml(s.name)}')">删</button>
            </td>
          </tr>`).join('')}
        </tbody>
      </table>`;
  } catch (e) {
    container.innerHTML = '<div class="config-error">加载失败: ' + escapeHtml(e.message) + '</div>';
  }
}

function toggleSourceSettings(name) {
  const panel = document.getElementById('source-settings-panel-container');
  if (!panel) return;
  const source = _sourcesCache.find(s => s.name === name);
  if (!source) return;
  const settings = source.settings || {};

  if (panel.dataset.currentSource === name && panel.style.display !== 'none') {
    panel.style.display = 'none';
    return;
  }

  panel.dataset.currentSource = name;
  panel.style.display = 'block';
  panel.innerHTML = `
    <div class="card" style="padding:12px;margin-top:8px">
      <h4>⚙ ${escapeHtml(name)} 设置</h4>
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px">
        <div><label>超时(秒)</label><input id="ss-timeout" type="number" value="${settings.timeout || 20}" min="5" max="60" style="width:100%;padding:4px;border:1px solid var(--border);border-radius:4px"></div>
        <div><label>请求间隔(秒)</label><input id="ss-delay" type="number" value="${settings.delay || 1.5}" min="0.5" max="10" step="0.1" style="width:100%;padding:4px;border:1px solid var(--border);border-radius:4px"></div>
        <div><label>最低质量</label><select id="ss-mq" style="width:100%;padding:4px;border:1px solid var(--border);border-radius:4px">
          <option value="C" ${(settings.min_quality||'C')==='C'?'selected':''}>C-全部</option>
          <option value="B" ${settings.min_quality==='B'?'selected':''}>B-中等+</option>
          <option value="A" ${settings.min_quality==='A'?'selected':''}>A-仅高</option>
        </select></div>
        <div><label>列表页URL</label><input id="ss-list-url" value="${escapeHtml(source.list_url || source.url || '')}" style="width:100%;padding:4px;border:1px solid var(--border);border-radius:4px"></div>
        <div><label>文章链接正则</label><input id="ss-article-pattern" value="${escapeHtml(source.article_pattern || '')}" placeholder="留空自动匹配" style="width:100%;padding:4px;border:1px solid var(--border);border-radius:4px"></div>
        <div><label>正文CSS选择器</label><input id="ss-content-selector" value="${escapeHtml(source.content_selector || '')}" placeholder="留空自动提取" style="width:100%;padding:4px;border:1px solid var(--border);border-radius:4px"></div>
        <div><label>翻页深度</label><input id="ss-max-pages" type="number" value="${settings.max_pages || 3}" min="1" max="20" style="width:100%;padding:4px;border:1px solid var(--border);border-radius:4px"></div>
        <div><label>最多文章</label><input id="ss-max-articles" type="number" value="${settings.max_articles || 50}" min="1" max="200" style="width:100%;padding:4px;border:1px solid var(--border);border-radius:4px"></div>
      </div>
      <div style="margin-top:8px;display:flex;gap:8px">
        <button class="btn btn-primary" onclick="saveSourceSettings('${escapeHtml(name)}')">💾 保存</button>
        <button class="btn" onclick="document.getElementById('source-settings-panel-container').style.display='none'">关闭</button>
      </div>
    </div>`;
}

async function saveSourceSettings(name) {
  const settings = {
    timeout: parseInt(document.getElementById('ss-timeout')?.value) || 20,
    delay: parseFloat(document.getElementById('ss-delay')?.value) || 1.5,
    min_quality: document.getElementById('ss-mq')?.value || 'C',
    list_url: document.getElementById('ss-list-url')?.value || '',
    article_pattern: document.getElementById('ss-article-pattern')?.value || '',
    content_selector: document.getElementById('ss-content-selector')?.value || '',
    max_pages: parseInt(document.getElementById('ss-max-pages')?.value) || 3,
    max_articles: parseInt(document.getElementById('ss-max-articles')?.value) || 50,
  };
  const r = await API.post('/api/sources/settings', { name, settings });
  if (r.success) { toast(name + ' 设置已保存', 'success'); document.getElementById('source-settings-panel-container').style.display = 'none'; loadSources(); }
  else { toast(r.error || '保存失败', 'error'); }
}

async function removeSource(name) {
  if (!confirm('确定删除源「' + name + '」?')) return;
  const r = await API.post('/api/sources/remove', { name });
  if (r.success) { toast('已移除: ' + name, 'success'); loadSources(); }
  else { toast(r.error || '移除失败', 'error'); }
}

async function crawlSource(name, btn) {
  if (!btn) return;
  btn.disabled = true;
  btn.textContent = '...';
  try {
    await crawlSourceWithProgress(name);
  } catch (e) {
    toast('爬取失败: ' + e.message, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = '▶ 爬取';
  }
}

let currentCrawlXhr = null;
let crawlCancelFlag = false;
let currentCrawlSourceName = null;

// 取消爬取按钮 — 通知后端 + 前端 abort
document.getElementById('crawl-progress-cancel')?.addEventListener('click', async () => {
  const cancelBtn = document.getElementById('crawl-progress-cancel');
  if (cancelBtn) { cancelBtn.textContent = '取消中...'; cancelBtn.disabled = true; }
  // 通知后端取消
  if (currentCrawlSourceName) {
    try {
      await API.post('/api/crawl/cancel', { name: currentCrawlSourceName });
    } catch (e) {}
  }
  // 前端也 abort
  crawlCancelFlag = true;
  if (currentCrawlXhr) { currentCrawlXhr.abort(); currentCrawlXhr = null; }
  if (cancelBtn) { cancelBtn.style.display = 'none'; cancelBtn.disabled = false; cancelBtn.textContent = '取消'; }
  if (window._crawlTimer) { clearInterval(window._crawlTimer); window._crawlTimer = null; }
  renderCrawlProgress({type: 'cancelled', message: '用户取消爬取'});
  // 恢复爬取按钮状态
  document.querySelectorAll('button[onclick^="crawlSource"]').forEach(b => {
    b.disabled = false;
    b.textContent = '▶ 爬取';
  });
});

// 清空爬取结果
document.getElementById('clear-results-btn')?.addEventListener('click', () => {
  _advancedResultArticles = [];
  document.getElementById('crawl-results').style.display = 'none';
  toast('已清空爬取结果', 'success');
});

function renderCrawlProgress(data) {
  const container = document.getElementById('crawl-progress-container');
  const bar = document.getElementById('crawl-progress-bar');
  const text = document.getElementById('crawl-progress-text');
  const log = document.getElementById('crawl-progress-log');
  const stats = document.getElementById('crawl-progress-stats');
  const title = document.getElementById('crawl-progress-title');
  if (!container || !bar) return;
  container.style.display = 'block';
  switch (data.type) {
    case 'start':
      title.textContent = '🕷\uFE0F 正在爬取「' + data.source_name + '」...';
      text.textContent = '初始化...';
      bar.style.width = '0%';
      if (log) log.innerHTML = '';
      if (stats) stats.textContent = '最大 ' + data.max_pages + ' 页，最多 ' + data.max_articles + ' 篇';
      // 启动计时器
      if (window._crawlTimer) clearInterval(window._crawlTimer);
      window._crawlStartTime = Date.now();
      window._crawlTimer = setInterval(() => {
        const elapsed = Math.floor((Date.now() - window._crawlStartTime) / 1000);
        const mm = String(Math.floor(elapsed / 60)).padStart(2, '0');
        const ss = String(elapsed % 60).padStart(2, '0');
        const statsEl = document.getElementById('crawl-progress-stats');
        if (statsEl) statsEl.textContent = `已用时间 ${mm}:${ss}`;
      }, 1000);
      break;
    case 'phase':
      text.textContent = data.message || '';
      if (data.phase === 'crawling' && stats) stats.textContent = '共发现 ' + data.total_articles + ' 篇文章，开始抓取...';
      break;
    case 'page_discover':
      bar.style.width = Math.round((data.current_page / data.total_pages) * 50) + '%';
      text.textContent = '翻页 ' + data.current_page + '/' + data.total_pages + '，已发现 ' + data.found_so_far + ' 篇';
      break;
    case 'article_progress':
      bar.style.width = Math.min(50 + Math.round((data.current / data.total) * 45), 95) + '%';
      text.textContent = '抓取文章 ' + data.current + '/' + data.total + ' (' + data.percent + '%)';
      break;
    case 'article_success':
      if (log) {
        const el = document.createElement('div');
        el.style.color = 'var(--success)';
        el.textContent = '✅ ' + (data.title || '') + ' (' + data.quality + '级, ' + data.word_count + '字)';
        log.appendChild(el);
        log.scrollTop = log.scrollHeight;
      }
      break;
    case 'article_failed':
      if (log) {
        const el = document.createElement('div');
        el.style.color = 'var(--danger)';
        el.textContent = '❌ ' + (data.url || '') + ' - ' + (data.error || '');
        log.appendChild(el);
        log.scrollTop = log.scrollHeight;
      }
      break;
    case 'article_skipped':
      if (log) {
        const el = document.createElement('div');
        el.style.color = 'var(--text-muted)';
        el.textContent = '⏭\uFE0F 跳过重复: ' + (data.url || '');
        log.appendChild(el);
        log.scrollTop = log.scrollHeight;
      }
      break;
    case 'summary':
      if (stats) stats.textContent = '发现 ' + data.total_found + ' 篇，成功 ' + data.success_count + ' 篇，过滤后 ' + data.filtered_count + ' 篇';
      break;
    case 'cancelled':
      text.textContent = '⏹ 已取消: ' + (data.message || '用户取消');
      title.textContent = '⏹ 爬取已取消';
      if (window._crawlTimer) { clearInterval(window._crawlTimer); window._crawlTimer = null; }
      break;
    case 'complete':
      bar.style.width = '100%';
      text.textContent = '✅ 完成！共 ' + data.total + ' 篇，成功 ' + data.success_count + ' 篇';
      title.textContent = '✅ 爬取完成';
      if (window._crawlTimer) { clearInterval(window._crawlTimer); window._crawlTimer = null; }
      if (stats) stats.textContent = '成功 ' + data.success_count + '/' + data.total;
      if (data.articles) {
        _advancedResultArticles = data.articles;
        renderCrawlResults(data.articles, name);
      }
      break;
    case 'error':
      text.textContent = '❌ 错误: ' + (data.error || '未知');
      bar.style.width = '100%';
      bar.style.background = 'var(--danger)';
      if (window._crawlTimer) { clearInterval(window._crawlTimer); window._crawlTimer = null; }
      break;
  }
}

function crawlSourceWithProgress(name) {
  return new Promise((resolve) => {
    crawlCancelFlag = false;
    const container = document.getElementById('crawl-progress-container');
    if (container) {
      container.style.display = 'block';
      const log = document.getElementById('crawl-progress-log');
      if (log) log.innerHTML = '';
    }
    const cancelBtn = document.getElementById('crawl-progress-cancel');
    if (cancelBtn) {
      cancelBtn.style.display = 'inline-flex';
      cancelBtn.textContent = '取消';
    }

    const xhr = new XMLHttpRequest();
    currentCrawlXhr = xhr;
    currentCrawlSourceName = name;
    xhr.open('POST', '/api/crawl/source-progress', true);
    xhr.setRequestHeader('Content-Type', 'application/json');
    xhr.timeout = 300000;
    xhr.ontimeout = function() {
      renderCrawlProgress({type: 'error', error: '爬取超时（5分钟）'});
      toast('爬取超时', 'error');
      if (cancelBtn) cancelBtn.style.display = 'none';
      if (window._crawlTimer) { clearInterval(window._crawlTimer); window._crawlTimer = null; }
      resolve();
    };

    let buffer = '';
    let eventName = '';
    let successCount = 0;
    let totalCount = 0;

    xhr.onprogress = function() {
      buffer += xhr.responseText.substring(buffer.length);
      let idx;
      while ((idx = buffer.indexOf('\n')) >= 0) {
        let line = buffer.substring(0, idx).trim();
        buffer = buffer.substring(idx + 1);
        if (!line) { eventName = ''; continue; }
        if (line.startsWith('event: ')) {
          eventName = line.substring(7).trim();
        } else if (line.startsWith('data: ')) {
          const dataStr = line.substring(6);
          if (eventName === 'status' || eventName === 'progress' || eventName === 'result' || eventName === 'error') {
            try {
              const parsed = JSON.parse(dataStr);
              if (eventName === 'result') {
                renderCrawlProgress(Object.assign({type: 'complete'}, parsed));
                successCount = parsed.success_count || 0;
                totalCount = parsed.total || 0;
              } else if (eventName === 'error') {
                renderCrawlProgress({type: 'error', error: parsed.error || '未知'});
              } else {
                renderCrawlProgress(parsed);
              }
            } catch (e) {}
          } else if (eventName === 'done') {
            // finished
          }
          eventName = '';
        }
      }
    };

    xhr.onload = function() {
      if (cancelBtn) cancelBtn.style.display = 'none';
      if (window._crawlTimer) { clearInterval(window._crawlTimer); window._crawlTimer = null; }
      if (totalCount > 0) {
        toast('爬取完成: ' + successCount + '/' + totalCount + ' 篇', 'success');
      }
      resolve();
    };
    xhr.onerror = function() {
      if (cancelBtn) cancelBtn.style.display = 'none';
      if (window._crawlTimer) { clearInterval(window._crawlTimer); window._crawlTimer = null; }
      renderCrawlProgress({type: 'error', error: '网络错误'});
      toast('爬取失败: 网络错误', 'error');
      resolve();
    };
    xhr.send(JSON.stringify({name: name}));
  });
}

document.getElementById('add-source-btn')?.addEventListener('click', async () => {
  const name = document.getElementById('new-source-name').value.trim();
  const url = document.getElementById('new-source-url').value.trim();
  const category = document.getElementById('new-source-category').value.trim() || '通用';
  if (!name || !url) { toast('名称和URL不能为空', 'error'); return; }
  const r = await API.post('/api/sources/add', { name, url, category });
  if (r.success) { toast('已添加: ' + name, 'success'); document.getElementById('new-source-name').value = ''; document.getElementById('new-source-url').value = ''; document.getElementById('new-source-category').value = ''; loadSources(); }
  else { toast(r.error || '添加失败', 'error'); }
});

// ========== 临时抓取 ==========

document.getElementById('preview-url-btn')?.addEventListener('click', async () => {
  const url = document.getElementById('temp-url-input').value.trim();
  if (!url) { toast('请输入URL', 'error'); return; }
  const btn = document.getElementById('preview-url-btn'); btn.disabled = true; btn.textContent = '...';
  try {
    const r = await API.post('/api/crawl/url', { url });
    if (r.success) {
      document.getElementById('crawl-preview-title').textContent = r.title || '(无标题)';
      document.getElementById('crawl-preview-content').textContent = r.content || '(无内容)';
      document.getElementById('crawl-preview').style.display = 'block';
    } else { toast(r.error || '预览失败', 'error'); }
  } catch (e) { toast('失败: ' + e.message, 'error'); }
  btn.disabled = false; btn.textContent = '👁 预览';
});

document.getElementById('crawl-ingest-btn')?.addEventListener('click', async () => {
  const url = document.getElementById('temp-url-input').value.trim();
  if (!url) { toast('请输入URL', 'error'); return; }
  const btn = document.getElementById('crawl-ingest-btn'); btn.disabled = true; btn.textContent = '...';
  try {
    const r = await API.post('/api/crawl/ingest', { url });
    if (r.success !== false) { toast('摄入: ' + (r.title || ''), 'success'); }
    else { toast(r.error || '失败', 'error'); }
  } catch (e) { toast('失败: ' + e.message, 'error'); }
  btn.disabled = false; btn.textContent = '📥 抓取并摄入';
});

let _crawlResultsPage = 0;
const CRAWL_PAGE_SIZE = 20;

function renderCrawlResults(articles, sourceName) {
  const container = document.getElementById('crawl-results');
  const list = document.getElementById('crawl-articles-list');
  const stats = document.getElementById('crawl-results-stats');
  if (!container || !list) return;
  container.style.display = 'block';
  stats.textContent = '共 ' + articles.length + ' 篇';
  if (articles.length === 0) { list.innerHTML = '<p style="color:var(--text-muted);padding:12px">未抓到文章</p>'; return; }
  _crawlResultsPage = 0;
  renderCrawlPage(articles);
  // 分页控件
  const pager = document.getElementById('crawl-pagination') || document.createElement('div');
  pager.id = 'crawl-pagination';
  pager.style.cssText = 'display:flex;justify-content:center;gap:8px;padding:8px;font-size:13px';
  if (articles.length > CRAWL_PAGE_SIZE) {
    pager.innerHTML = `<button class="btn" id="crawl-prev-page" style="padding:2px 10px;font-size:12px">上一页</button>
      <span id="crawl-page-info" style="line-height:24px;color:var(--text-muted)">1/${Math.ceil(articles.length / CRAWL_PAGE_SIZE)}</span>
      <button class="btn" id="crawl-next-page" style="padding:2px 10px;font-size:12px">下一页</button>`;
    if (!container.contains(pager)) container.appendChild(pager);
    document.getElementById('crawl-prev-page').onclick = () => { if (_crawlResultsPage > 0) { _crawlResultsPage--; renderCrawlPage(articles); updatePageInfo(articles); } };
    document.getElementById('crawl-next-page').onclick = () => { if ((_crawlResultsPage+1)*CRAWL_PAGE_SIZE < articles.length) { _crawlResultsPage++; renderCrawlPage(articles); updatePageInfo(articles); } };
  } else if (container.contains(pager)) { pager.remove(); }
}

function updatePageInfo(articles) {
  const info = document.getElementById('crawl-page-info');
  if (info) info.textContent = (_crawlResultsPage+1) + '/' + Math.ceil(articles.length / CRAWL_PAGE_SIZE);
}

function renderCrawlPage(articles) {
  const list = document.getElementById('crawl-articles-list');
  if (!list) return;
  const start = _crawlResultsPage * CRAWL_PAGE_SIZE;
  const page = articles.slice(start, start + CRAWL_PAGE_SIZE);
  list.innerHTML = page.map((a) => {
    const realIdx = start + page.indexOf(a);
    const ok = a.success !== false && a.content && a.content.length >= 20;
    const title = escapeHtml(a.title || '(无标题)');
    const quality = a.quality ? `<span style="font-size:10px;padding:1px 4px;border-radius:3px;background:${a.quality==='A'?'#e8f5e9':a.quality==='B'?'#fff3e0':'#ffebee'};color:${a.quality==='A'?'#2e7d32':a.quality==='B'?'#e65100':'#c62828'}">${a.quality}</span>` : '';
    const wc = a.word_count ? `<span style="font-size:11px;color:var(--text-muted)">${a.word_count}字</span>` : '';
    const preview = escapeHtml((a.content || '').substring(0, 300));
    return `<div style="display:flex;align-items:flex-start;gap:10px;padding:8px 0;border-bottom:1px solid var(--border);font-size:13px">
      <input type="checkbox" class="crawl-checkbox" data-index="${realIdx}" style="margin-top:3px;flex-shrink:0" ${ok ? 'checked' : ''}>
      <div style="flex:1">
        <div style="display:flex;align-items:center;gap:6px">
          <strong>${title}</strong>
          ${quality} ${wc}
        </div>
        <span style="color:var(--text-muted);font-size:11px">${escapeHtml(a.url || '')}</span>
        <div style="color:var(--text-secondary);margin-top:2px;font-size:12px">${preview}...</div>
      </div>
    </div>`;
  }).join('');
}

document.getElementById('batch-select-all')?.addEventListener('click', () => {
  document.querySelectorAll('#crawl-articles-list .crawl-checkbox').forEach(cb => cb.checked = true);
});

document.getElementById('batch-ingest-selected')?.addEventListener('click', async () => {
  const checked = document.querySelectorAll('#crawl-articles-list .crawl-checkbox:checked');
  if (checked.length === 0) { toast('请先勾选文章', 'error'); return; }
  const articles = [];
  checked.forEach(cb => {
    const idx = parseInt(cb.dataset.index);
    if (!isNaN(idx) && _advancedResultArticles[idx]) articles.push(_advancedResultArticles[idx]);
  });
  const btn = document.getElementById('batch-ingest-selected'); btn.disabled = true; btn.textContent = '摄入中...';
  try {
    const minQ = document.getElementById('min-quality-select')?.value || 'C';
    const r = await API.post('/api/crawl/batch-ingest', { articles, source_name: '', min_quality: minQ });
    toast('摄入: ' + (r.success_count || 0) + '/' + articles.length + (r.skipped ? ' (跳过 ' + r.skipped + ')' : ''), 'success');
  } catch (e) { toast('失败: ' + friendlyError(e), 'error'); }
  btn.disabled = false; btn.textContent = '📥 摄入选中';
});


// ========== 分类管理 ==========

let _categoriesCache = [];
let _currentCatId = null;
let _catDocs = [];

async function loadCategories() {
  try {
    const data = await API.get('/api/categories');
    _categoriesCache = data.categories || [];
    const container = document.getElementById('categories-list');
    if (!container) return;
    if (_categoriesCache.length === 0) {
      container.innerHTML = '<p style="color:var(--text-muted);padding:12px;font-size:13px">暂无分类，请创建</p>';
    } else {
      container.innerHTML = _categoriesCache.map(c => `
        <div class="cat-item" style="display:flex;align-items:center;justify-content:space-between;padding:8px 10px;margin:2px 0;border-radius:4px;cursor:pointer;font-size:14px"
             onclick="selectCategory(${c.id}, '${escapeHtml(c.name)}')">
          <span>🗂️ ${escapeHtml(c.name)}</span>
          <button class="btn" style="padding:2px 6px;font-size:11px" onclick="event.stopPropagation();deleteCategory(${c.id})">✕</button>
        </div>`).join('');
    }
    // Populate move-to-category select
    const sel = document.getElementById('move-to-cat-select');
    if (sel) {
      sel.innerHTML = '<option value="">移动到分类...</option>' + _categoriesCache.map(c =>
        `<option value="${c.id}">${escapeHtml(c.name)}</option>`
      ).join('');
    }
  } catch (e) { toast('加载分类失败: ' + e.message, 'error'); }
}

async function selectCategory(catId, catName) {
  _currentCatId = catId;
  document.getElementById('cat-docs-title').textContent = '📁 ' + catName;
  document.getElementById('cat-docs-actions').style.display = 'flex';
  document.getElementById('cat-batch-remove-btn').style.display = 'inline-flex';
  await refreshCatDocs();
}

async function refreshCatDocs() {
  if (!_currentCatId) return;
  try {
    const data = await API.get('/api/categories/' + _currentCatId + '/docs');
    _catDocs = data.documents || [];
    renderCatDocsTable(_catDocs);
  } catch (e) { toast('加载失败: ' + e.message, 'error'); }
}

function renderCatDocsTable(docs) {
  const container = document.getElementById('cat-docs-table-container');
  if (!container) return;
  if (docs.length === 0) {
    container.innerHTML = '<p style="color:var(--text-muted);text-align:center;padding:30px">此分类下暂无文档</p>';
    return;
  }
  container.innerHTML = `
    <table>
      <thead><tr><th><input type="checkbox" id="cat-select-all"></th><th>标题</th><th>来源</th><th>时间</th><th>操作</th></tr></thead>
      <tbody>${docs.map(d => `
        <tr>
          <td><input type="checkbox" class="cat-doc-check" data-doc-id="${escapeHtml(d.doc_id)}"></td>
          <td>${escapeHtml(d.title || d.doc_id || '')}</td>
          <td style="color:var(--text-muted);font-size:13px">${escapeHtml(d.source || '-')}</td>
          <td style="color:var(--text-muted);font-size:13px">${(d.created_at || '').slice(0, 10)}</td>
          <td>
            <button class="btn" style="padding:4px 8px;font-size:12px" onclick="openDocDetail('${escapeHtml(d.doc_id)}')">编辑</button>
            <button class="btn btn-danger" style="padding:4px 8px;font-size:12px" onclick="removeDocFromCat('${escapeHtml(d.doc_id)}')">移除</button>
          </td>
        </tr>`).join('')}
      </tbody>
    </table>`;
  document.getElementById('cat-select-all')?.addEventListener('change', e => {
    document.querySelectorAll('.cat-doc-check').forEach(cb => cb.checked = e.target.checked);
  });
}

async function removeDocFromCat(docId) {
  if (!_currentCatId) return;
  const r = await API.post('/api/categories/remove-doc', { doc_id: docId, category_id: _currentCatId });
  if (r.success) { toast('已从分类移除', 'success'); refreshCatDocs(); }
  else { toast(r.error || '失败', 'error'); }
}

document.getElementById('create-cat-btn')?.addEventListener('click', async () => {
  const name = document.getElementById('cat-name-input').value.trim();
  if (!name) { toast('请输入分类名称', 'error'); return; }
  const r = await API.post('/api/categories/create', { name });
  if (r.success) { toast('已创建: ' + name, 'success'); document.getElementById('cat-name-input').value = ''; loadCategories(); }
  else { toast(r.error || '创建失败', 'error'); }
});

async function deleteCategory(catId) {
  if (!confirm('确定删除此分类？(文档不会被删除，仅取消关联)')) return;
  const r = await API.del('/api/categories/' + catId);
  if (r.success) { toast('已删除', 'success'); loadCategories(); if (_currentCatId === catId) { _currentCatId = null; document.getElementById('cat-docs-table-container').innerHTML = '<p style="color:var(--text-muted);text-align:center;padding:30px">已删除</p>'; } }
  else { toast(r.error || '删除失败', 'error'); }
}

document.getElementById('cat-batch-move-btn')?.addEventListener('click', async () => {
  const targetCatId = document.getElementById('move-to-cat-select').value;
  if (!targetCatId) { toast('请选择目标分类', 'error'); return; }
  const checks = document.querySelectorAll('.cat-doc-check:checked');
  if (checks.length === 0) { toast('请先勾选文档', 'error'); return; }
  for (const cb of checks) {
    await API.post('/api/categories/assign-doc', { doc_id: cb.dataset.docId, category_id: parseInt(targetCatId) });
  }
  toast('已移动 ' + checks.length + ' 篇', 'success');
  refreshCatDocs(); loadCategories();
});

document.getElementById('cat-batch-remove-btn')?.addEventListener('click', async () => {
  if (!_currentCatId) return;
  const checks = document.querySelectorAll('.cat-doc-check:checked');
  if (checks.length === 0) { toast('请先勾选文档', 'error'); return; }
  for (const cb of checks) {
    await API.post('/api/categories/remove-doc', { doc_id: cb.dataset.docId, category_id: _currentCatId });
  }
  toast('已移除 ' + checks.length + ' 篇', 'success');
  refreshCatDocs();
});

document.getElementById('show-uncategorized-btn')?.addEventListener('click', async () => {
  const data = await API.get('/api/sources/uncategorized-docs');
  const docs = data.documents || [];
  _currentCatId = null;
  document.getElementById('cat-docs-title').textContent = '📋 未分类文档 (' + docs.length + ' 篇)';
  document.getElementById('cat-docs-actions').style.display = 'flex';
  document.getElementById('cat-batch-remove-btn').style.display = 'none';
  renderCatDocsTable(docs);
});

document.getElementById('refresh-cat-docs-btn')?.addEventListener('click', refreshCatDocs);

// ========== 文档编辑器 ==========

async function openDocDetail(docId) {
  const modal = document.getElementById('doc-modal');
  const body = document.getElementById('doc-modal-body');
  const title = document.getElementById('doc-modal-title');
  if (!modal || !body) return;
  modal.style.display = 'flex';
  title.textContent = '文档编辑器';
  body.innerHTML = '<div class="doc-detail-loading">加载中...</div>';

  try {
    const doc = await API.get('/api/documents/' + docId);
    if (doc.error) { body.innerHTML = '<p style="color:red">' + escapeHtml(doc.error) + '</p>'; return; }
    // Get full content via keywords endpoint (has chunks)
    let content = '';
    try {
      const kwData = await API.get('/api/documents/' + docId + '/keywords');
      content = kwData.content || '';
    } catch (e) {}
    if (!content) {
      // Fallback: get from chunks in doc object
      const chunks = doc.chunks || [];
      content = chunks.map(c => c.content || '').join(String.fromCharCode(10));
    }
    body.innerHTML = `
      <div class="form-group">
        <label>标题</label>
        <input type="text" id="editor-title" value="${escapeHtml(doc.title || '')}" style="width:100%;padding:8px 12px;border:1px solid var(--border);border-radius:var(--radius-sm);font-size:14px">
      </div>
      <div class="form-group">
        <label>描述</label>
        <input type="text" id="editor-desc" value="${escapeHtml(doc.description || '')}" style="width:100%;padding:8px 12px;border:1px solid var(--border);border-radius:var(--radius-sm);font-size:14px">
      </div>
      <div class="form-group">
        <label>标签 (逗号分隔)</label>
        <input type="text" id="editor-tags" value="${escapeHtml(Array.isArray(doc.tags) ? doc.tags.join(',') : (doc.tags || ''))}" style="width:100%;padding:8px 12px;border:1px solid var(--border);border-radius:var(--radius-sm);font-size:14px">
      </div>
      <div class="form-group">
        <label>正文</label>
        <textarea id="editor-content" style="width:100%;min-height:400px;padding:12px;border:1px solid var(--border);border-radius:var(--radius-sm);font-size:13px;font-family:monospace;line-height:1.6;resize:vertical">${escapeHtml(content)}</textarea>
      </div>
      <div class="form-group">
        <label>关键词 & 词云</label>
        <div id="editor-keywords" style="display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px">${(doc.keywords||[]).map(k=>`<span style="background:var(--bg-input);padding:2px 10px;border-radius:12px;font-size:12px;color:var(--text-secondary)">${escapeHtml(k.word)} <span style="color:var(--text-muted)">${k.count}</span></span>`).join('')}</div>
        <div id="editor-wordcloud" style="text-align:center"></div>
      </div>
      <div style="display:flex;gap:10px;margin-top:12px">
        <button class="btn btn-primary" id="save-editor-btn">💾 保存并重新向量化</button>
        <button class="btn" id="ai-summarize-btn">🤖 AI 摘要</button>
        <button class="btn btn-danger" onclick="document.getElementById('doc-modal').style.display='none'">关闭</button>
        <span id="editor-status" style="padding:8px;font-size:13px;color:var(--text-secondary)"></span>
      </div>`;

    document.getElementById('save-editor-btn')?.addEventListener('click', async () => {
      const btn = document.getElementById('save-editor-btn'); btn.disabled = true; btn.textContent = '保存中...';
      const payload = {
        doc_id: docId,
        title: document.getElementById('editor-title').value.trim(),
        description: document.getElementById('editor-desc').value.trim(),
        tags: document.getElementById('editor-tags').value.split(',').map(s => s.trim()).filter(Boolean),
        content: document.getElementById('editor-content').value,
      };
      try {
        const r = await API.post('/api/documents/editor', payload);
        if (r.success) { toast('已保存', 'success'); document.getElementById('editor-status').textContent = '✅ 已保存'; }
        else { toast(r.error || '保存失败', 'error'); }
      } catch (e) { toast('失败: ' + e.message, 'error'); }
      btn.disabled = false; btn.textContent = '💾 保存并重新向量化';
    });

    document.getElementById('ai-summarize-btn')?.addEventListener('click', async () => {
      const btn = document.getElementById('ai-summarize-btn'); btn.disabled = true; btn.textContent = '生成中...';
      try {
        const r = await API.post('/api/documents/ai-summarize', { doc_id: docId });
        if (r.summary) { document.getElementById('editor-status').textContent = '摘要: ' + r.summary.substring(0, 200); }
      } catch (e) { toast('摘要失败: ' + e.message, 'error'); }
      btn.disabled = false; btn.textContent = '🤖 AI 摘要';
    });

    // 加载词云
    try {
      const wcData = await API.get('/api/documents/' + docId + '/wordcloud');
      if (wcData.wordcloud_url) {
        const wcDiv = document.getElementById('editor-wordcloud');
        if (wcDiv) wcDiv.innerHTML = `<img src="${wcData.wordcloud_url}?t=${Date.now()}" alt="词云" style="max-width:100%;border-radius:var(--radius-sm)">`;
      }
    } catch (e) {}

  } catch (e) {
    body.innerHTML = '<p style="color:red">加载失败: ' + escapeHtml(e.message) + '</p>';
  }
}

document.getElementById('doc-modal-close')?.addEventListener('click', () => {
  document.getElementById('doc-modal').style.display = 'none';
});

// Close modal on overlay click
document.getElementById('doc-modal')?.addEventListener('click', (e) => {
  if (e.target === document.getElementById('doc-modal')) document.getElementById('doc-modal').style.display = 'none';
});
