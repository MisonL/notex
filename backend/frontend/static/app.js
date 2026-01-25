// LocalCache - 本地缓存管理类
class LocalCache {
  constructor(ttlMinutes = 5) {
    this.ttl = ttlMinutes * 60 * 1000; // 转换为毫秒
    this.prefix = "notex_cache_";
  }

  // 生成缓存键
  _makeKey(key) {
    return `${this.prefix}${key}`;
  }

  // 获取缓存
  get(key) {
    try {
      const fullKey = this._makeKey(key);
      const item = localStorage.getItem(fullKey);
      if (!item) return null;

      const data = JSON.parse(item);

      // 检查是否过期
      if (Date.now() > data.expiresAt) {
        localStorage.removeItem(fullKey);
        return null;
      }

      return data.value;
    } catch (e) {
      console.warn("Cache get error:", e);
      return null;
    }
  }

  // 设置缓存
  set(key, value, customTTL = null) {
    try {
      const fullKey = this._makeKey(key);
      const ttl = customTTL || this.ttl;
      const data = {
        value,
        expiresAt: Date.now() + ttl,
      };
      localStorage.setItem(fullKey, JSON.stringify(data));
    } catch (e) {
      console.warn("Cache set error:", e);
    }
  }

  // 删除缓存
  delete(key) {
    try {
      const fullKey = this._makeKey(key);
      localStorage.removeItem(fullKey);
    } catch (e) {
      console.warn("Cache delete error:", e);
    }
  }

  // 按前缀删除缓存
  deletePattern(pattern) {
    try {
      const prefix = this._makeKey(pattern);
      const keys = [];
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key && key.startsWith(prefix)) {
          keys.push(key);
        }
      }
      keys.forEach((key) => localStorage.removeItem(key));
    } catch (e) {
      console.warn("Cache deletePattern error:", e);
    }
  }

  // 清空所有缓存
  clear() {
    try {
      const keys = [];
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key && key.startsWith(this.prefix)) {
          keys.push(key);
        }
      }
      keys.forEach((key) => localStorage.removeItem(key));
    } catch (e) {
      console.warn("Cache clear error:", e);
    }
  }

  // 清理过期缓存
  cleanup() {
    try {
      const now = Date.now();
      const keys = [];
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key && key.startsWith(this.prefix)) {
          keys.push(key);
        }
      }
      keys.forEach((key) => {
        try {
          const item = localStorage.getItem(key);
          if (item) {
            const data = JSON.parse(item);
            if (now > data.expiresAt) {
              localStorage.removeItem(key);
            }
          }
        } catch (e) {
          // 忽略解析错误，删除无效条目
          localStorage.removeItem(key);
        }
      });
    } catch (e) {
      console.warn("Cache cleanup error:", e);
    }
  }
}

class OpenNotebook {
  constructor() {
    this.notebooks = [];
    this.currentNotebook = null;
    this.apiBase = "/api";
    this.currentChatSession = null;
    this.currentPublicToken = null;
    this.config = {
      allowDelete: true,
    };

    // Auth state
    this.token = localStorage.getItem("token");
    this.currentUser = null;

    // Sync token from localStorage to cookie for image loading
    if (this.token) {
      document.cookie = `token=${this.token}; path=/; SameSite=Lax`;
    }

    // 初始化本地缓存 (5分钟TTL)
    this.cache = new LocalCache(5);

    // Note type name mapping
    this.noteTypeNameMap = {
      summary: "摘要",
      faq: "常见问题",
      study_guide: "学习指南",
      outline: "大纲",
      podcast: "播客",
      timeline: "时间线",
      glossary: "术语表",
      quiz: "测验",
      mindmap: "思维导图",
      infograph: "信息图",
      ppt: "幻灯片",
      insight: "洞察报告",
    };

    this.init();
  }

  async init() {
    await this.initAuth();
    await this.loadConfig();
    this.bindEvents();
    this.initResizers();
    this.initNotebookNameEditor();

    // 清理过期缓存
    this.cache.cleanup();

    // Check if URL contains /notes/:id or /public/:token for direct notebook access
    // Only load notebooks if not accessing a public notebook directly
    if (!this.checkURLForNotebook() && !this.checkURLForPublicNotebook()) {
      await this.loadNotebooks();
      this.applyConfig();
      this.switchView("landing");
    } else {
      this.applyConfig();
    }
  }

  async loadConfig() {
    try {
      const config = await this.api("/config");
      // Convert snake_case to camelCase
      this.config = {
        allowDelete: config.allow_delete !== undefined ? config.allow_delete : true,
        embeddingProvider: config.embedding_provider || 'google',
        imageModel: config.image_model || 'gemini-2.5-flash-image-preview',
        chatProvider: config.chat_provider || 'openai',
        chatModel: config.chat_model || 'qwen3-max'
      };
    } catch (error) {
      console.error("Failed to load config:", error);
    }
  }

  applyConfig() {
    // Show/hide delete buttons based on allowDelete config
    const deleteButtons = document.querySelectorAll(
      ".btn-delete, .btn-remove-source, .btn-delete-note, .btn-delete-card, .btn-delete-notebook, .btn-delete-compact-note"
    );
    deleteButtons.forEach((btn) => {
      if (this.config.allowDelete) {
        btn.style.display = "";
        btn.style.opacity = "1";
      } else {
        btn.style.display = "none";
      }
    });
  }

  initResizers() {
    const resizerLeft = document.getElementById("resizerLeft");
    const resizerRight = document.getElementById("resizerRight");
    const grid = document.querySelector(".main-grid");

    if (!resizerLeft || !resizerRight) return;

    let isDragging = false;
    let currentResizer = null;

    const startDragging = (e, resizer) => {
      isDragging = true;
      currentResizer = resizer;
      resizer.classList.add("dragging");
      document.body.style.cursor = "col-resize";
      e.preventDefault();
    };

    const stopDragging = () => {
      if (!isDragging) return;
      isDragging = false;
      currentResizer.classList.remove("dragging");
      document.body.style.cursor = "";
      currentResizer = null;
    };

    const drag = (e) => {
      if (!isDragging) return;

      const gridRect = grid.getBoundingClientRect();
      if (currentResizer === resizerLeft) {
        const width = e.clientX - gridRect.left;
        if (width > 150 && width < 600) {
          grid.style.setProperty("--left-width", `${width}px`);
        }
      } else if (currentResizer === resizerRight) {
        const width = gridRect.right - e.clientX;
        if (width > 200 && width < 600) {
          grid.style.setProperty("--right-width", `${width}px`);
        }
      }
    };

    resizerLeft.addEventListener("mousedown", (e) =>
      startDragging(e, resizerLeft)
    );
    resizerRight.addEventListener("mousedown", (e) =>
      startDragging(e, resizerRight)
    );
    document.addEventListener("mousemove", drag);
    document.addEventListener("mouseup", stopDragging);
  }

  bindEvents() {
    const safeAddEventListener = (id, event, handler) => {
      const el = document.getElementById(id);
      if (el) el.addEventListener(event, handler);
    };

    safeAddEventListener("btnNewNotebook", "click", () =>
      this.showNewNotebookModal()
    );
    safeAddEventListener("btnNewNotebookLanding", "click", () =>
      this.showNewNotebookModal()
    );
    safeAddEventListener("btnBackToList", "click", () =>
      this.handleBackToList()
    );
    safeAddEventListener("btnToggleRight", "click", () =>
      this.toggleRightPanel()
    );
    safeAddEventListener("btnToggleLeft", "click", () =>
      this.toggleLeftPanel()
    );
    safeAddEventListener("btnShowNotesDetails", "click", () =>
      this.showNotesListTab()
    );
    safeAddEventListener("btnCloseNotesList", "click", (e) => {
      e.stopPropagation();
      this.closeNotesListTab();
    });
    safeAddEventListener("btnCloseNote", "click", (e) => {
      e.stopPropagation();
      this.closeNoteTab();
    });

    // Share listeners
    safeAddEventListener('btnShareNotebook', 'click', () => {
        if (this.currentNotebook) {
            this.showShareDialog(this.currentNotebook);
        }
    });
    safeAddEventListener('btnCloseShareModal', 'click', () => this.closeShareModal());
    safeAddEventListener('btnCancelShare', 'click', () => this.closeShareModal());
    safeAddEventListener('btnCopyLink', 'click', () => this.copyShareLink());
    safeAddEventListener('btnToggleShare', 'click', () => this.toggleShareFromModal());

    // Auth events
    safeAddEventListener('btnLogin', 'click', () => this.handleLogin());
    safeAddEventListener('btnLogout', 'click', () => this.handleLogout());
    safeAddEventListener('btnLoginWorkspace', 'click', () => this.handleLogin());
    safeAddEventListener('btnLogoutWorkspace', 'click', () => this.handleLogout());

    // Panel tabs
    document.querySelectorAll(".tab-btn").forEach((tab) => {
      tab.addEventListener("click", () => {
        this.switchPanelTab(tab.dataset.tab);
      });
    });

    safeAddEventListener("newNotebookForm", "submit", (e) =>
      this.handleCreateNotebook(e)
    );
    safeAddEventListener("btnCloseNotebookModal", "click", () =>
      this.closeModals()
    );
    safeAddEventListener("btnCancelNotebook", "click", () =>
      this.closeModals()
    );

    safeAddEventListener("btnAddSource", "click", () =>
      this.showAddSourceModal()
    );
    safeAddEventListener("btnCloseSourceModal", "click", () =>
      this.closeModals()
    );
    const dropZone = document.getElementById("dropZone");
    if (dropZone) {
      dropZone.addEventListener("click", () =>
        document.getElementById("fileInput").click()
      );
      dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("drag-over");
      });
      dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("drag-over");
      });
      dropZone.addEventListener("drop", (e) => this.handleDrop(e));
    }

    safeAddEventListener("fileInput", "change", (e) =>
      this.handleFileUpload(e)
    );
    safeAddEventListener("textSourceForm", "submit", (e) =>
      this.handleTextSource(e)
    );
    safeAddEventListener("urlSourceForm", "submit", (e) =>
      this.handleURLSource(e)
    );
    safeAddEventListener("btnCancelText", "click", () => this.closeModals());
    safeAddEventListener("btnCancelURL", "click", () => this.closeModals());

    document.querySelectorAll(".source-tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        document
          .querySelectorAll(".source-tab")
          .forEach((t) => t.classList.remove("active"));
        document
          .querySelectorAll(".source-content")
          .forEach((c) => c.classList.remove("active"));
        tab.classList.add("active");
        const targetId = `source${
          tab.dataset.source.charAt(0).toUpperCase() +
          tab.dataset.source.slice(1)
        }`;
        const target = document.getElementById(targetId);
        if (target) target.classList.add("active");
      });
    });

    document.querySelectorAll(".transform-card").forEach((card) => {
      card.addEventListener("click", (e) => {
        e.preventDefault();
        this.handleTransform(card.dataset.type, card);
      });
    });

    safeAddEventListener("btnCustomTransform", "click", (e) => {
      this.handleTransform("custom", e.currentTarget);
    });

    safeAddEventListener("chatForm", "submit", (e) => this.handleChat(e));

    safeAddEventListener("modalOverlay", "click", (e) => {
      if (e.target.id === "modalOverlay") {
        this.closeModals();
      }
    });

    safeAddEventListener("btnSettings", "click", () =>
      this.showSettingsModal()
    );
    safeAddEventListener("btnCloseSettingsModal", "click", () =>
      this.closeModals()
    );
    safeAddEventListener("btnCancelSettings", "click", () =>
      this.closeModals()
    );
    safeAddEventListener("settingsForm", "submit", (e) =>
      this.handleSaveSettings(e)
    );

    const chatProviderSelect = document.getElementById('chatProviderSelect');
    if (chatProviderSelect) {
      chatProviderSelect.addEventListener('change', () => this.updateChatModelVisibility());
    }
    const embeddingProviderSelect = document.getElementById('embeddingProviderSelect');
    if (embeddingProviderSelect) {
      embeddingProviderSelect.addEventListener('change', () => this.updateEmbeddingModelVisibility());
    }

    // Popstate
    window.addEventListener('popstate', (event) => {
        const path = window.location.pathname;
        const match = path.match(/^\/notes\/([a-f0-9-]+)$/);
        if (match) {
            const notebookId = match[1];
            const notebook = this.notebooks.find(nb => nb.id === notebookId);
            if (notebook && (!this.currentNotebook || this.currentNotebook.id !== notebookId)) {
                this.selectNotebook(notebookId);
            }
        } else if (path === '/' && this.currentNotebook) {
            this.switchView('landing');
        }
    });
  }

  async updateChatModelVisibility() {
    const provider = document.getElementById('chatProviderSelect').value;
    const openaiSelect = document.getElementById('chatModelOpenAI');
    const ollamaSelect = document.getElementById('chatModelOllama');

    if (provider === 'openai') {
      openaiSelect.classList.remove('hidden');
      ollamaSelect.classList.add('hidden');
    } else {
      openaiSelect.classList.add('hidden');
      ollamaSelect.classList.remove('hidden');
    }
    
    // Refresh models for the selected provider
    await this.loadModels(provider, 'chat');
  }

  async updateEmbeddingModelVisibility() {
    const provider = document.getElementById('embeddingProviderSelect').value;
    // Embedding dropdown is single for now, but we'll fetch based on provider
    await this.loadModels(provider, 'embedding');
  }

  async loadModels(provider, type = 'chat') {
    let selectId;
    if (type === 'chat') {
        selectId = provider === 'openai' ? 'chatModelOpenAI' : 'chatModelOllama';
    } else {
        selectId = 'embeddingModelSelect';
    }
    const select = document.getElementById(selectId);
    if (!select) return;

    // Use specific config field based on type
    const configModel = type === 'chat' ? this.config.chatModel : (this.config.embeddingModel || '');
    const configProvider = type === 'chat' ? this.config.chatProvider : (this.config.embeddingProvider || '');

    // Preserve current selection if possible, or use config
    const currentVal = select.value || (configProvider === provider ? configModel : '');

    // Show loading if empty
    if (select.options.length === 0 || select.innerHTML.includes('Failed')) {
       select.innerHTML = '<option value="">Loading...</option>';
    }

    try {
        const res = await this.api(`/models?provider=${provider}&type=${type}`);
        const models = res.models || [];

        select.innerHTML = '';
        let foundCurrent = false;

        models.forEach(modelItem => {
            const opt = document.createElement('option');
            const id = typeof modelItem === 'string' ? modelItem : modelItem.id;
            const name = typeof modelItem === 'string' ? modelItem : modelItem.display_name;
            
            opt.value = id;
            opt.textContent = name;
            if (id === currentVal) {
                opt.selected = true;
                foundCurrent = true;
            }
            select.appendChild(opt);

            if (id === currentVal) foundCurrent = true;
        });

        // If configured model is in list, select it
        if (foundCurrent) {
            select.value = currentVal;
        } else if (configProvider === provider && configModel) {
             const exists = models.some(m => {
                  const id = typeof m === 'string' ? m : m.id;
                  return id === configModel;
             });
             if (!exists) {
                  const opt = document.createElement('option');
                  opt.value = configModel;
                  opt.textContent = configModel;
                  select.insertBefore(opt, select.firstChild);
                  select.value = configModel;
             }
        }
    } catch (e) {
        console.error("Failed to load models:", e);
        if (select.options.length === 0 || select.options[0].text === 'Loading...') {
             select.innerHTML = '<option value="">Failed to load models</option>';
        }
    }
  }
    // Check if URL contains /public/:token and load the public notebook
    checkURLForPublicNotebook() {
        const path = window.location.pathname;
        const match = path.match(/^\/public\/([a-f0-9-]+)$/);
        if (match) {
            this.loadPublicNotebook(match[1]);
            return true;
        }
        return false;
    }

    // Load public notebook by token
    async loadPublicNotebook(token) {
        try {
            this.setStatus('加载公开笔记本...');

            const [notebook, sources, notes] = await Promise.all([
                fetch(`/public/notebooks/${token}`).then(r => {
                    if (!r.ok) throw new Error('Failed to load notebook');
                    return r.json();
                }),
                fetch(`/public/notebooks/${token}/sources`).then(r => {
                    if (!r.ok) throw new Error('Failed to load sources');
                    return r.json();
                }),
                fetch(`/public/notebooks/${token}/notes`).then(r => {
                    if (!r.ok) throw new Error('Failed to load notes');
                    return r.json();
                })
            ]);

            this.currentNotebook = notebook;
            this.currentPublicToken = token;

            // 先显示笔记列表 tab（创建容器）
            this.showNotesListTab();

            // 渲染 sources
            await this.renderSourcesList(sources);

            // 渲染 notes 到紧凑网格视图（容器已创建）
            await this.renderNotesCompactGridPublic(notes);

            // 设置为只读模式
            this.setReadOnlyMode(true);

            this.switchView('workspace');
            this.setStatus('公开笔记本: ' + notebook.name);
        } catch (error) {
            console.error('Failed to load public notebook:', error);
            this.showError('加载公开笔记本失败');
            this.switchView('landing');
        }
    }

    // Handle back to list button click
    async handleBackToList() {
        // Clear public notebook state
        this.currentPublicToken = null;
        this.currentNotebook = null;

        // Reload user's notebooks
        await this.loadNotebooks();

        // Clear status
        this.setStatus('就绪');

        // Switch to landing view
        this.switchView('landing');
    }

    // 设置只读模式
    setReadOnlyMode(readOnly) {
        const workspace = document.getElementById('workspaceContainer');
        if (readOnly) {
            workspace.classList.add('readonly-mode');
            // 禁用编辑功能
            const addSourceBtn = document.getElementById('btnAddSource');
            if (addSourceBtn) addSourceBtn.style.display = 'none';

            // 隐藏编辑按钮
            document.querySelectorAll('.transform-card').forEach(btn => {
                btn.style.pointerEvents = 'none';
                btn.style.opacity = '0.5';
            });

            // 隐藏聊天功能
            const chatWrapper = document.querySelector('.chat-messages-wrapper');
            if (chatWrapper) chatWrapper.style.display = 'none';
            const chatInput = document.querySelector('.chat-input-wrapper');
            if (chatInput) chatInput.style.display = 'none';

            // 显示公开标识
            this.showPublicBadge();
        } else {
            workspace.classList.remove('readonly-mode');
            const addSourceBtn = document.getElementById('btnAddSource');
            if (addSourceBtn) addSourceBtn.style.display = '';

            document.querySelectorAll('.transform-card').forEach(btn => {
                btn.style.pointerEvents = '';
                btn.style.opacity = '';
            });

            const chatWrapper = document.querySelector('.chat-messages-wrapper');
            if (chatWrapper) chatWrapper.style.display = '';

            const chatInput = document.querySelector('.chat-input-wrapper');
            if (chatInput) chatInput.style.display = '';

            const badge = document.querySelector('.public-badge');
            if (badge) badge.remove();
        }
    }

    // 显示公开标识
    showPublicBadge() {
        // 移除已存在的 badge
        const existingBadge = document.querySelector('.public-badge');
        if (existingBadge) existingBadge.remove();

        const nameDisplay = document.getElementById('currentNotebookName');
        if (nameDisplay && !document.querySelector('.public-badge')) {
            const badge = document.createElement('div');
            badge.className = 'public-badge';
            badge.innerHTML = `
                <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M5 9l2-2 2 2m-4 0l2-2 2-2"/>
                </svg>
                <span>公开</span>
            `;
            nameDisplay.parentNode.appendChild(badge);
        }
    }

    // Check if URL contains /notes/:id and auto-load the notebook
    // Returns true if a notebook was found and loaded, false otherwise
    checkURLForNotebook() {
        const path = window.location.pathname;
        const match = path.match(/^\/notes\/([a-f0-9-]+)$/);
        if (match) {
            const notebookId = match[1];
            // Check if notebook exists in loaded notebooks
            const notebook = this.notebooks.find(nb => nb.id === notebookId);
            if (notebook) {
                this.selectNotebook(notebookId);
                return true;  // Notebook found and loaded
            } else {
                // Notebook not found or user doesn't have access
                this.setStatus('笔记本不存在或无权访问', true);
                return false;  // Notebook not found
            }
        }
        return false;  // No notebook ID in URL
    }

    // Update URL when notebook is selected
    updateURL(notebookId) {
        const newURL = `/notes/${notebookId}`;
        window.history.pushState({ notebookId }, '', newURL);
    }

    async loadConfig() {
        // Config loading - no longer needed, all features enabled
    }

    applyConfig() {
        // All features enabled by default, no config to apply
    }

    initResizers() {
        const resizerLeft = document.getElementById('resizerLeft');
        const resizerRight = document.getElementById('resizerRight');
        const grid = document.querySelector('.main-grid');

        if (!resizerLeft || !resizerRight) return;

        let isDragging = false;
        let currentResizer = null;

        const startDragging = (e, resizer) => {
            isDragging = true;
            currentResizer = resizer;
            resizer.classList.add('dragging');
            document.body.style.cursor = 'col-resize';
            e.preventDefault();
        };

        const stopDragging = () => {
            if (!isDragging) return;
            isDragging = false;
            currentResizer.classList.remove('dragging');
            document.body.style.cursor = '';
            currentResizer = null;
        };

        const drag = (e) => {
            if (!isDragging) return;

            const gridRect = grid.getBoundingClientRect();
            if (currentResizer === resizerLeft) {
                const width = e.clientX - gridRect.left;
                if (width > 150 && width < 600) {
                    grid.style.setProperty('--left-width', `${width}px`);
                }
            } else if (currentResizer === resizerRight) {
                const width = gridRect.right - e.clientX;
                if (width > 200 && width < 600) {
                    grid.style.setProperty('--right-width', `${width}px`);
                }
            }
        };

        resizerLeft.addEventListener('mousedown', (e) => startDragging(e, resizerLeft));
        resizerRight.addEventListener('mousedown', (e) => startDragging(e, resizerRight));
        document.addEventListener('mousemove', drag);
        document.addEventListener('mouseup', stopDragging);
    }

  bindEvents() {
    const safeAddEventListener = (id, event, handler) => {
      const el = document.getElementById(id);
      if (el) el.addEventListener(event, handler);
    };

    safeAddEventListener("btnNewNotebook", "click", () =>
      this.showNewNotebookModal()
    );
    safeAddEventListener("btnNewNotebookLanding", "click", () =>
      this.showNewNotebookModal()
    );
    safeAddEventListener("btnBackToList", "click", () =>
      this.handleBackToList()
    );
    safeAddEventListener("btnToggleRight", "click", () =>
      this.toggleRightPanel()
    );
    safeAddEventListener("btnToggleLeft", "click", () =>
      this.toggleLeftPanel()
    );
    safeAddEventListener("btnShowNotesDetails", "click", () =>
      this.showNotesListTab()
    );
    safeAddEventListener("btnCloseNotesList", "click", (e) => {
      e.stopPropagation();
      this.closeNotesListTab();
    });
    safeAddEventListener("btnCloseNote", "click", (e) => {
      e.stopPropagation();
      this.closeNoteTab();
    });
    safeAddEventListener('btnShareNotebook', 'click', () => {
        if (this.currentNotebook) {
            this.showShareDialog(this.currentNotebook);
        }
    });

    // Share modal events
    safeAddEventListener('btnCloseShareModal', 'click', () => this.closeShareModal());
    safeAddEventListener('btnCancelShare', 'click', () => this.closeShareModal());
    safeAddEventListener('btnCopyLink', 'click', () => this.copyShareLink());
    safeAddEventListener('btnToggleShare', 'click', () => this.toggleShareFromModal());

    // Auth events
    safeAddEventListener('btnLogin', 'click', () => this.handleLogin());
    safeAddEventListener('btnLogout', 'click', () => this.handleLogout());
    safeAddEventListener('btnLoginWorkspace', 'click', () => this.handleLogin());
    safeAddEventListener('btnLogoutWorkspace', 'click', () => this.handleLogout());

    // Panel tabs
    document.querySelectorAll(".tab-btn").forEach((tab) => {
      tab.addEventListener("click", () => {
        this.switchPanelTab(tab.dataset.tab);
      });
    });

    safeAddEventListener("newNotebookForm", "submit", (e) =>
      this.handleCreateNotebook(e)
    );
    safeAddEventListener("btnCloseNotebookModal", "click", () =>
      this.closeModals()
    );
    safeAddEventListener("btnCancelNotebook", "click", () =>
      this.closeModals()
    );

    safeAddEventListener("btnAddSource", "click", () =>
      this.showAddSourceModal()
    );
    safeAddEventListener("btnCloseSourceModal", "click", () =>
      this.closeModals()
    );
    const dropZone = document.getElementById("dropZone");
    if (dropZone) {
      dropZone.addEventListener("click", () =>
        document.getElementById("fileInput").click()
      );
      dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("drag-over");
      });
      dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("drag-over");
      });
      dropZone.addEventListener("drop", (e) => this.handleDrop(e));
    }

    safeAddEventListener("fileInput", "change", (e) =>
      this.handleFileUpload(e)
    );
    safeAddEventListener("textSourceForm", "submit", (e) =>
      this.handleTextSource(e)
    );
    safeAddEventListener("urlSourceForm", "submit", (e) =>
      this.handleURLSource(e)
    );
    safeAddEventListener("btnCancelText", "click", () => this.closeModals());
    safeAddEventListener("btnCancelURL", "click", () => this.closeModals());

    document.querySelectorAll(".source-tab").forEach((tab) => {
      tab.addEventListener("click", () => {
        document
          .querySelectorAll(".source-tab")
          .forEach((t) => t.classList.remove("active"));
        document
          .querySelectorAll(".source-content")
          .forEach((c) => c.classList.remove("active"));
        tab.classList.add("active");
        const targetId = `source${
          tab.dataset.source.charAt(0).toUpperCase() +
          tab.dataset.source.slice(1)
        }`;
        const target = document.getElementById(targetId);
        if (target) target.classList.add("active");
      });
    });

    document.querySelectorAll(".transform-card").forEach((card) => {
      card.addEventListener("click", (e) => {
        e.preventDefault();
        this.handleTransform(card.dataset.type, card);
      });
    });

    safeAddEventListener("btnCustomTransform", "click", (e) => {
      this.handleTransform("custom", e.currentTarget);
    });

    safeAddEventListener("chatForm", "submit", (e) => this.handleChat(e));

    safeAddEventListener("modalOverlay", "click", (e) => {
      if (e.target.id === "modalOverlay") {
        this.closeModals();
      }
    });

    safeAddEventListener("btnSettings", "click", () =>
      this.showSettingsModal()
    );
    safeAddEventListener("btnCloseSettingsModal", "click", () =>
      this.closeModals()
    );
    safeAddEventListener("btnCancelSettings", "click", () =>
      this.closeModals()
    );
    safeAddEventListener("settingsForm", "submit", (e) =>
      this.handleSaveSettings(e)
    );

    const chatProviderSelect = document.getElementById('chatProviderSelect');
    if (chatProviderSelect) {
      chatProviderSelect.addEventListener('change', () => this.updateChatModelVisibility());
    }
    const embeddingProviderSelect = document.getElementById('embeddingProviderSelect');
    if (embeddingProviderSelect) {
      embeddingProviderSelect.addEventListener('change', () => this.updateEmbeddingModelVisibility());
    }

    // Handle browser back/forward buttons
    window.addEventListener('popstate', (event) => {
        const path = window.location.pathname;
        const match = path.match(/^\/notes\/([a-f0-9-]+)$/);
        if (match) {
            const notebookId = match[1];
            const notebook = this.notebooks.find(nb => nb.id === notebookId);
            if (notebook && (!this.currentNotebook || this.currentNotebook.id !== notebookId)) {
                this.selectNotebook(notebookId);
            }
        } else if (path === '/' && this.currentNotebook) {
            this.switchView('landing');
        }
    });
  }

    async initAuth() {
        if (!this.token) {
            this.updateAuthUI();
            return;
        }

        try {
            const user = await this.api('/auth/me');
            this.currentUser = user;
            this.updateAuthUI();
        } catch (error) {
            console.warn('Auth check failed:', error);
            this.handleLogout();
        }
    }

    updateAuthUI() {
        // Landing page auth UI
        const authContainer = document.getElementById('authContainer');
        const btnLogin = document.getElementById('btnLogin');
        const userProfile = document.getElementById('userProfile');
        const userAvatar = document.getElementById('userAvatar');
        const userName = document.getElementById('userName');

        // Workspace auth UI
        const btnLoginWorkspace = document.getElementById('btnLoginWorkspace');
        const userProfileWorkspace = document.getElementById('userProfileWorkspace');
        const userAvatarWorkspace = document.getElementById('userAvatarWorkspace');
        const userNameWorkspace = document.getElementById('userNameWorkspace');

        if (this.currentUser) {
            // Get provider display name
            const providerNames = {
                'github': 'GitHub',
                'google': 'Google'
            };
            const providerName = providerNames[this.currentUser.provider] || this.currentUser.provider;
            const tooltipText = `登录方式: ${providerName}\n账号ID: ${this.currentUser.email}`;

            // Update landing page
            if (btnLogin) btnLogin.classList.add('hidden');
            if (userProfile) userProfile.classList.remove('hidden');
            if (userAvatar) {
                userAvatar.src = this.currentUser.avatar_url;
                userAvatar.title = tooltipText;
            }
            if (userName) userName.textContent = this.currentUser.name;

            // Update workspace
            if (btnLoginWorkspace) btnLoginWorkspace.classList.add('hidden');
            if (userProfileWorkspace) userProfileWorkspace.classList.remove('hidden');
            if (userAvatarWorkspace) {
                userAvatarWorkspace.src = this.currentUser.avatar_url;
                userAvatarWorkspace.title = tooltipText;
            }
            if (userNameWorkspace) userNameWorkspace.textContent = this.currentUser.name;
        } else {
            // Update landing page
            if (btnLogin) btnLogin.classList.remove('hidden');
            if (userProfile) userProfile.classList.add('hidden');

            // Update workspace
            if (btnLoginWorkspace) btnLoginWorkspace.classList.remove('hidden');
            if (userProfileWorkspace) userProfileWorkspace.classList.add('hidden');
        }
    }

    handleLogin() {
        // Show login modal
        this.showLoginModal();
    }

    showLoginModal() {
        // Create or get existing modal
        let modal = document.getElementById('loginModal');
        if (!modal) {
            // Create modal dynamically
            modal = document.createElement('div');
            modal.id = 'loginModal';
            modal.className = 'login-modal';
            modal.innerHTML = `
                <div class="login-modal-content">
                    <div class="login-modal-header">
                        <h3>选择登录方式</h3>
                        <button class="btn-close-login" id="btnCloseLoginModal">×</button>
                    </div>
                    <div class="login-modal-body">
                        <button class="btn-login-provider" id="btnLoginGithub">
                            <svg width="20" height="20" viewBox="0 0 16 16" fill="currentColor">
                                <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
                            </svg>
                            使用 GitHub 登录
                        </button>
                        <button class="btn-login-provider" id="btnLoginGoogle">
                            <svg width="20" height="20" viewBox="0 0 16 16">
                                <path fill="#4285F4" d="M14.9 8.16c0-.95-.08-1.65-.21-2.37H8v4.4h3.83c-.17.96-.69 2.05-1.55 2.68v2.19h2.48c1.46-1.34 2.3-3.31 2.3-5.64z"/>
                                <path fill="#34A853" d="M8 16c2.07 0 3.83-.69 5.11-1.87l-2.48-2.19c-.69.46-1.57.73-2.63.73-2.02 0-3.74-1.37-4.35-3.19H1.11v2.26C2.38 13.89 4.99 16 8 16z"/>
                                <path fill="#FBBC05" d="M3.65 9.52c-.16-.46-.25-.95-.25-1.47s.09-1.01.25-1.47V4.48H1.11C.4 5.87 0 7.39 0 8s.4 2.13 1.11 3.52l2.54-2z"/>
                                <path fill="#EA4335" d="M8 3.24c1.14 0 2.17.39 2.98 1.15l2.2-2.2C11.83.87 10.07 0 8 0 4.99 0 2.38 2.11 1.11 4.48l2.54 2.26c.61-1.82 2.33-3.5 4.35-3.5z"/>
                            </svg>
                            使用 Google 登录
                        </button>
                    </div>
                </div>
            `;
            document.body.appendChild(modal);

            // Add event listeners
            document.getElementById('btnCloseLoginModal').addEventListener('click', () => {
                this.closeLoginModal();
            });
            document.getElementById('btnLoginGithub').addEventListener('click', () => {
                this.loginWithProvider('github');
            });
            document.getElementById('btnLoginGoogle').addEventListener('click', () => {
                this.loginWithProvider('google');
            });
        }

        modal.classList.add('active');
    }

    closeLoginModal() {
        const modal = document.getElementById('loginModal');
        if (modal) {
            modal.classList.remove('active');
        }
    }

    loginWithProvider(provider) {
        this.closeLoginModal();

        // Open popup
        const width = 600;
        const height = 700;
        const left = (screen.width - width) / 2;
        const top = (screen.height - height) / 2;

        window.open(
            `/auth/login/${provider}`,
            'NotexLogin',
            `width=${width},height=${height},top=${top},left=${left}`
        );

        // Listen for message with origin validation
        const messageHandler = (event) => {
            // Validate origin for security
            if (event.origin !== window.location.origin) {
                console.warn('Received message from untrusted origin:', event.origin);
                return;
            }

            if (event.data.token && event.data.user) {
                this.token = event.data.token;
                this.currentUser = event.data.user;
                localStorage.setItem('token', this.token);

                // Also set token as cookie for image loading
                document.cookie = `token=${this.token}; path=/; SameSite=Lax`;

                this.updateAuthUI();

                // Reload data
                this.loadNotebooks();
            }
        };

        window.addEventListener('message', messageHandler, { once: true });
    }

    handleLogout() {
        this.token = null;
        this.currentUser = null;
        localStorage.removeItem('token');

        // Also remove token cookie
        document.cookie = 'token=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT';

        // Clear cache
        this.cache.delete('notebooks');

        this.updateAuthUI();

        // Clear data
        this.notebooks = [];
        this.renderNotebooks();
        this.switchView('landing');
    }

    // API 方法
    async api(endpoint, options = {}) {
        const timeout = options.timeout || 300000; // 默认 300 秒
        const controller = new AbortController();
        const id = setTimeout(() => controller.abort(), timeout);

        const defaults = {
            cache: 'no-store',
            signal: controller.signal
        };

        // Set Content-Type header (but not for FormData - let browser set it)
        if (!(options.body instanceof FormData)) {
            defaults.headers = {
                'Content-Type': 'application/json',
            };
        } else {
            defaults.headers = {};
        }

        if (this.token) {
            defaults.headers['Authorization'] = `Bearer ${this.token}`;
        }

        let url = `${this.apiBase}${endpoint}`;
        if (!options.method || options.method === 'GET') {
            const separator = url.includes('?') ? '&' : '?';
            url += `${separator}_t=${Date.now()}`;
        }

        const config = { ...defaults, ...options };
        if (config.body && typeof config.body === 'object' && !(config.body instanceof FormData)) {
            config.body = JSON.stringify(config.body);
        }

        try {
            const response = await fetch(url, config);
            clearTimeout(id);

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ message: response.statusText }));
                throw new Error(errorData.message || `API request failed with status ${response.status}`);
            }

            // Check if response is JSON
            const contentType = response.headers.get("content-type");
            if (contentType && contentType.indexOf("application/json") !== -1) {
                return await response.json();
            } else {
                return await response.text(); // Or handle as appropriate for non-JSON
            }
        } catch (error) {
            clearTimeout(id);
            if (error.name === 'AbortError') {
                throw new Error('API request timed out');
            }
            throw error;
        }
    }

    async loadModels(provider, select, currentVal) {
        try {
            const models = await this.api(`/models/${provider}`);
            select.innerHTML = '';
            let foundCurrent = false;

            models.forEach(modelItem => {
                const opt = document.createElement('option');
                // Check if modelItem is string (legacy) or object (new)
                const id = typeof modelItem === 'string' ? modelItem : modelItem.id;
                const name = typeof modelItem === 'string' ? modelItem : modelItem.display_name;
                
                opt.value = id;
                opt.textContent = name;
                if (id === currentVal) {
                    opt.selected = true;
                    foundCurrent = true;
                }
                select.appendChild(opt);

                if (id === currentVal) foundCurrent = true;
            });

            // If configured model is in list, select it
            if (foundCurrent) {
                select.value = currentVal;
            } else if (this.config.chatProvider === provider && this.config.chatModel) {
                 const exists = models.some(m => {
                      const id = typeof m === 'string' ? m : m.id;
                      return id === this.config.chatModel;
                 });
                 if (!exists) {
                      const opt = document.createElement('option');
                      opt.value = this.config.chatModel;
                      opt.textContent = this.config.chatModel;
                      select.insertBefore(opt, select.firstChild);
                      select.value = this.config.chatModel;
                 }
            }
        } catch (e) {
            console.error("Failed to load models:", e);
            // On error, keep existing or show error option
            if (select.options.length === 0 || select.options[0].text === 'Loading...') {
                 select.innerHTML = '<option value="">Failed to load models</option>';
            }
        }
    }

    // 笔记本方法
    async loadNotebooks() {
        // Always load public notebooks showcase (regardless of private notebooks)
        await this.loadPublicNotebooksShowcase();

        try {
            // 先尝试从缓存获取
            const cached = this.cache.get('notebooks');
            if (cached) {
                this.notebooks = cached;
                this.renderNotebooks();
                this.updateFooter();
            }

            // 从服务器获取最新数据（包含统计信息）
            const notebooks = await this.api("/notebooks/stats");
            this.notebooks = notebooks;

            // 更新缓存
            this.cache.set('notebooks', notebooks);

            this.renderNotebooks();
            this.updateFooter();
        } catch (error) {
            // 401 Unauthorized is expected for non-logged-in users, don't show error
            if (error.message && !error.message.includes('401')) {
                console.warn('用户未登录，跳过私有笔记本加载');
            } else {
                this.showError('加载笔记本失败');
            }
        }
    }

    renderNotebooks() {
        this.renderNotebookCards();
    }

    renderNotebookCards() {
        const container = document.getElementById("notebookGridLanding");
        const template = document.getElementById("notebookCardTemplate");

        container.innerHTML = "";

        if (this.notebooks.length === 0) {
            container.innerHTML = `
                <div class="empty-state">
                    <svg width="64" height="64" viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="1">
                        <rect x="12" y="12" width="40" height="40" rx="4"/>
                        <line x1="20" y1="24" x2="44" y2="24"/>
                        <line x1="20" y1="32" x2="40" y2="32"/>
                    </svg>
                    <p>开启你的知识之旅</p>
                    <button class="btn-primary" onclick="app.showNewNotebookModal()">创建第一个笔记本</button>
                </div>
            `;
            return;
        }

        this.notebooks.forEach(nb => {
            const clone = template.content.cloneNode(true);
            const card = clone.querySelector('.notebook-card');

            card.dataset.id = nb.id;
            card.querySelector('.notebook-card-name').textContent = nb.name;
            card.querySelector('.notebook-card-desc').textContent = nb.description || '暂无描述';

            // 直接使用从 API 获取的统计信息
            card.querySelector('.stat-sources').textContent = `${nb.source_count || 0} 来源`;
            card.querySelector('.stat-notes').textContent = `${nb.note_count || 0} 笔记`;
            // Check if stat-date exists
            const dateEl = card.querySelector('.stat-date');
            if (dateEl) {
                dateEl.textContent = this.formatDate(nb.created_at);
            }

            // 更新分享按钮状态
            const shareCardBtn = clone.querySelector('.btn-share-card');
            if (shareCardBtn) {
                if (nb.is_public) {
                    shareCardBtn.classList.add('active');
                    shareCardBtn.setAttribute('title', '已公开');
                } else {
                    shareCardBtn.classList.remove('active');
                    shareCardBtn.setAttribute('title', '分享');
                }

                // Share button event
                shareCardBtn.addEventListener('click', (e) => {
                    e.stopPropagation();
                    this.showShareDialog(nb);
                });
            }

            card.addEventListener('click', (e) => {
                if (!e.target.closest('.btn-delete-card') && !e.target.closest('.btn-share-card')) {
                    this.selectNotebook(nb.id);
                }
            });

            const deleteCardBtn = card.querySelector('.btn-delete-card');
            deleteCardBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                if (confirm('确定要删除此笔记本吗？')) {
                    this.deleteNotebook(nb.id);
                }
            });

            // Apply config to delete button
            if (this.config.allowDelete) {
              deleteCardBtn.style.display = "flex";
            } else {
              deleteCardBtn.style.display = "none";
            }

            container.appendChild(clone);
        });
    }

    // Load and render public notebooks showcase
    async loadPublicNotebooksShowcase() {
        // Prevent duplicate calls
        if (this._publicNotebooksLoaded) return;
        this._publicNotebooksLoaded = true;

        try {
            const response = await fetch('/public/notebooks');
            if (!response.ok) return;

            const notebooks = await response.json();
            this.renderPublicNotebooksShowcase(notebooks);
        } catch (error) {
            console.error('Failed to load public notebooks showcase:', error);
        }
    }

    // Cache to avoid duplicate calls
    _publicNotebooksLoaded = false;

    renderPublicNotebooksShowcase(notebooks) {
        const container = document.getElementById('publicShowcase');
        const grid = document.getElementById('publicShowcaseGrid');

        if (!container || !grid) return;

        grid.innerHTML = '';

        if (notebooks.length === 0) {
            container.style.display = 'none';
            return;
        }

        container.style.display = 'block';

        notebooks.forEach(nb => {
            const card = document.createElement('a');
            card.className = 'public-showcase-card';
            card.href = `/public/${nb.public_token}`;

            // Generate background style if cover image exists
            if (nb.cover_image_url) {
                card.style.backgroundImage = `url('${nb.cover_image_url}')`;
                card.style.backgroundSize = 'cover';
                card.style.backgroundPosition = 'center';
                card.classList.add('has-cover-image');
            }

            card.innerHTML = `
                <div class="public-showcase-card-content">
                    <h3 class="public-showcase-card-title">${this.escapeHtml(nb.name)}</h3>
                </div>
            `;

            grid.appendChild(card);
        });
    }

    switchView(view) {
        const landing = document.getElementById('landingPage');
        const workspace = document.getElementById('workspaceContainer');
        const header = document.querySelector('.app-header');

        if (view === 'workspace') {
            landing.classList.add('hidden');
            workspace.classList.remove('hidden');
            header.classList.add('hidden');
        } else {
            landing.classList.remove('hidden');
            workspace.classList.add('hidden');
            header.classList.remove('hidden');
        }
  }

  switchView(view) {
    const landing = document.getElementById("landingPage");
    const workspace = document.getElementById("workspaceContainer");
    const header = document.querySelector(".app-header");

    if (view === "workspace") {
      landing.classList.add("hidden");
      workspace.classList.remove("hidden");
      header.classList.add("hidden");
    } else {
      landing.classList.remove("hidden");
      workspace.classList.add("hidden");
      header.classList.remove("hidden");
      this.currentNotebook = null;
      this.renderNotebookCards();
    }
  }

  toggleRightPanel() {
    const grid = document.querySelector(".main-grid");
    grid.classList.toggle("right-collapsed");
  }

  toggleLeftPanel() {
    const grid = document.querySelector(".main-grid");
    grid.classList.toggle("left-collapsed");
  }

  switchPanelTab(tab) {
    // Update tab buttons
    document.querySelectorAll(".tab-btn").forEach((t) => {
      t.classList.toggle("active", t.dataset.tab === tab);
    });

    // Update content visibility
    const chatWrapper = document.querySelector(".chat-messages-wrapper");
    const noteViewContainer = document.querySelector(".note-view-container");
    const notesDetailsView = document.querySelector(".notes-details-view");

    if (tab === "note") {
      chatWrapper.style.display = "none";
      if (notesDetailsView) notesDetailsView.style.display = "none";
      if (noteViewContainer) {
        noteViewContainer.style.display = "flex";
      }
    } else if (tab === "chat") {
      chatWrapper.style.display = "flex";
      if (notesDetailsView) notesDetailsView.style.display = "none";
      if (noteViewContainer) {
        noteViewContainer.style.display = "none";
      }
    } else if (tab === "notes_list") {
      chatWrapper.style.display = "none";
      if (noteViewContainer) noteViewContainer.style.display = "none";
      if (notesDetailsView) {
        notesDetailsView.style.display = "flex";
        this.renderNotesCompactGrid();
      }
    }
  }

  async showNotesListTab() {
    const tabBtn = document.getElementById("tabBtnNotesList");
    tabBtn.classList.remove("hidden");

    // Ensure notesDetailsView container exists
    let notesDetailsView = document.querySelector(".notes-details-view");
    if (!notesDetailsView) {
      const chatWrapper = document.querySelector(".chat-messages-wrapper");
      notesDetailsView = document.createElement("div");
      notesDetailsView.className = "notes-details-view";
      notesDetailsView.innerHTML = '<div class="notes-compact-grid"></div>';
      chatWrapper.insertAdjacentElement("afterend", notesDetailsView);
    }

    this.switchPanelTab("notes_list");
  }

  closeNotesListTab() {
    const tabBtn = document.getElementById("tabBtnNotesList");
    tabBtn.classList.add("hidden");

    const notesDetailsView = document.querySelector(".notes-details-view");
    if (notesDetailsView) notesDetailsView.style.display = "none";

    if (tabBtn.classList.contains("active")) {
      this.switchPanelTab("chat");
    }
  }

  closeNoteTab() {
    const noteViewContainer = document.querySelector(".note-view-container");
    if (noteViewContainer) noteViewContainer.remove();

    const tabBtnNote = document.getElementById("tabBtnNote");
    if (tabBtnNote) tabBtnNote.style.display = "none";

    this.switchPanelTab("chat");
  }

  async renderNotesCompactGrid() {
    if (!this.currentNotebook) return;

    const container = document.querySelector(".notes-compact-grid");
    if (!container) return;

    try {
      const notes = await this.api(
        `/notebooks/${this.currentNotebook.id}/notes`
      );
      container.innerHTML = "";

      notes.forEach((note) => {
        const card = document.createElement("div");
        card.className = "compact-note-card";

        const plainText = note.content
          .replace(/^#+\s+/gm, "")
          .replace(/\*\*/g, "")
          .replace(/\*/g, "")
          .replace(/`/g, "")
          .replace(/\n+/g, " ")
          .trim();

        card.innerHTML = `
                    <button class="btn-delete-compact-note" title="删除笔记">
                        <svg width="14" height="14" viewBox="0 0 14 14" fill="none" stroke="currentColor" stroke-width="1.5">
                            <path d="M4.5 4.5L9.5 9.5M9.5 4.5L4.5 9.5"/>
                        </svg>
                    </button>
                    <div class="note-type">${note.type}</div>
                    <h4 class="note-title">${note.title}</h4>
                    <p class="note-preview">${plainText}</p>
                    <div class="note-footer">
                        <span>${this.formatDate(note.created_at)}</span>
                        <span>${note.source_ids?.length || 0} 来源</span>
                    </div>
                `;

        // Delete button event
        const deleteBtn = card.querySelector(".btn-delete-compact-note");
        deleteBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          if (confirm("确定要删除此笔记吗？")) {
            this.deleteNote(note.id);
          }
        });

        // Apply config to delete button
        if (this.config.allowDelete) {
          deleteBtn.style.display = "flex";
        } else {
          deleteBtn.style.display = "none";
        }

        card.addEventListener("click", () => this.viewNote(note));
        container.appendChild(card);
      });
    } catch (error) {
      console.error("Failed to load notes for grid:", error);
    }
  }

    // Render notes compact grid for public notebooks (without API call)
    async renderNotesCompactGridPublic(notes) {
        const container = document.querySelector('.notes-compact-grid');
        if (!container) return;

        container.innerHTML = '';

        notes.forEach(note => {
            const card = document.createElement('div');
            card.className = 'compact-note-card';

            const plainText = note.content
                .replace(/^#+\s+/gm, '')
                .replace(/\*\*/g, '')
                .replace(/\*/g, '')
                .replace(/`/g, '')
                .replace(/\n+/g, ' ')
                .trim();

            card.innerHTML = `
                <div class="note-type">${note.type}</div>
                <h4 class="note-title">${note.title}</h4>
                <p class="note-preview">${plainText}</p>
                <div class="note-footer">
                    <span>${this.formatDate(note.created_at)}</span>
                    <span>${note.source_ids?.length || 0} 来源</span>
                </div>
            `;

            card.addEventListener('click', () => this.viewNote(note));
            container.appendChild(card);
        });
    }

    async selectNotebook(id) {
        this.currentNotebook = this.notebooks.find(nb => nb.id === id);
        this.currentPublicToken = null;  // Clear public token when selecting regular notebook

        const nameDisplay = document.getElementById('currentNotebookName');
        nameDisplay.textContent = this.currentNotebook.name;
        nameDisplay.classList.add('editable');

        // Update URL to /notes/:id for shareable links
        this.updateURL(id);

        // 更新分享按钮状态
        this.updateShareButtonState();

        this.switchView('workspace');

        // Reset tab to notes list and remove any existing note view
        this.showNotesListTab();
        const noteView = document.querySelector('.note-view-container');
        if (noteView) noteView.remove();

        await Promise.all([
          this.loadSources(),
          this.loadNotes(),
          this.loadChatSessions(),
        ]);

        this.setStatus(`当前选择: ${this.currentNotebook.name}`);
    }

    showNewNotebookModal() {
        document.getElementById("newNotebookModal").classList.add("active");
        document.getElementById("modalOverlay").classList.add("active");
        document.querySelector('#newNotebookForm input[name="name"]').focus();
    }

    async showSettingsModal() {
        const modal = document.getElementById('settingsModal');
        const selectEmbedding = document.getElementById('embeddingProviderSelect');
        const selectImage = document.getElementById('imageModelSelect');
        
        if (this.config.embeddingProvider) {
            selectEmbedding.value = this.config.embeddingProvider;
        }
        if (this.config.imageModel) {
            selectImage.value = this.config.imageModel;
        }
        if (this.config.chatProvider) {
            document.getElementById('chatProviderSelect').value = this.config.chatProvider;
        }
        
        // Set appropriate chat model based on provider (and load models)
        await this.updateChatModelVisibility();
        // Load embedding models
        await this.updateEmbeddingModelVisibility();
        
        modal.classList.add('active');
        document.getElementById('modalOverlay').classList.add('active');
    }

    // 更新分享按钮状态
    updateShareButtonState() {
        const shareBtn = document.getElementById('btnShareNotebook');
        const shareText = document.getElementById('shareButtonText');
        if (!shareBtn || !this.currentNotebook) return;

        if (this.currentNotebook.is_public) {
            shareText.textContent = '已公开';
            shareBtn.classList.add('active');
        } else {
            shareText.textContent = '分享';
            shareBtn.classList.remove('active');
        }
    }

    // 显示分享对话框
    showShareDialog(notebook) {
        this.currentShareNotebook = notebook;
        const modal = document.getElementById('shareModal');
        const overlay = document.getElementById('modalOverlay');

        // 设置笔记本名称
        document.getElementById('shareNotebookName').textContent = notebook.name;

        // 更新状态显示
        this.updateShareModalState(notebook);

        // 显示模态框
        modal.classList.add('active');
        overlay.classList.add('active');
    }

    // 更新分享对话框状态
    updateShareModalState(notebook) {
        const statusIcon = document.getElementById('shareStatusIcon');
        const statusText = document.getElementById('shareStatusText');
        const linkSection = document.getElementById('shareLinkSection');
        const linkInput = document.getElementById('shareLinkInput');
        const toggleBtn = document.getElementById('btnToggleShare');

        if (notebook.is_public) {
            statusIcon.className = 'status-icon public';
            statusIcon.innerHTML = '<svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 9l2-2 2 2m-4 0l2-2 2-2"/></svg>';
            statusText.textContent = '笔记本已公开';
            linkSection.style.display = 'flex';
            linkInput.value = `${window.location.origin}/public/${notebook.public_token}`;
            toggleBtn.textContent = '取消公开';
            toggleBtn.className = 'btn-secondary';
        } else {
            statusIcon.className = 'status-icon private';
            statusIcon.innerHTML = '<svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="2"><rect x="2" y="6" width="8" height="4" rx="1"/></svg>';
            statusText.textContent = '笔记本未公开';
            linkSection.style.display = 'none';
            toggleBtn.textContent = '公开笔记本';
            toggleBtn.className = 'btn-primary';
        }
    }

    // 关闭分享对话框
    closeShareModal() {
        const modal = document.getElementById('shareModal');
        const overlay = document.getElementById('modalOverlay');
        modal.classList.remove('active');
        overlay.classList.remove('active');
        this.currentShareNotebook = null;
    }

    // 复制分享链接
    copyShareLink() {
        const linkInput = document.getElementById('shareLinkInput');
        linkInput.select();
        linkInput.setSelectionRange(0, 99999); // For mobile devices

        navigator.clipboard.writeText(linkInput.value).then(() => {
            this.showToast('链接已复制到剪贴板', 'success');
        }).catch(() => {
            // Fallback
            try {
                document.execCommand('copy');
                this.showToast('链接已复制到剪贴板', 'success');
            } catch (err) {
                this.showError('复制失败，请手动复制');
            }
        });
    }

    // 切换笔记本公开状态（从对话框调用）
    async toggleShareFromModal() {
        if (!this.currentShareNotebook) return;

        const newPublicState = !this.currentShareNotebook.is_public;
        try {
            const result = await this.api(`/notebooks/${this.currentShareNotebook.id}/public`, {
                method: 'PUT',
                body: JSON.stringify({ is_public: newPublicState })
            });

            // 更新当前笔记本
            if (this.currentNotebook && this.currentNotebook.id === this.currentShareNotebook.id) {
                this.currentNotebook = result;
                this.updateShareButtonState();
            }

            // 更新笔记本列表中的数据
            const nb = this.notebooks.find(n => n.id === this.currentShareNotebook.id);
            if (nb) {
                nb.is_public = result.is_public;
                nb.public_token = result.public_token;
            }

            // 更新对话框状态
            this.currentShareNotebook = result;
            this.updateShareModalState(result);

            // 刷新笔记本列表
            this.renderNotebooks();

            this.showToast(newPublicState ? '笔记本已公开' : '笔记本已取消公开', 'success');
        } catch (error) {
            this.showError(`操作失败: ${error.message}`);
        }
    }

    initNotebookNameEditor() {
        const nameDisplay = document.getElementById('currentNotebookName');
        const nameEditor = document.getElementById('notebookNameEditor');
        const nameInput = document.getElementById('notebookNameInput');
        const saveBtn = document.getElementById('btnSaveNotebookName');
        const cancelBtn = document.getElementById('btnCancelNotebookName');

        // 双击进入编辑模式
        nameDisplay.addEventListener('dblclick', () => {
            this.startEditingNotebookName();
        });

        // 点击保存按钮
        saveBtn.addEventListener('click', () => {
            this.saveNotebookName();
        });

        // 点击取消按钮
        cancelBtn.addEventListener('click', () => {
            this.cancelEditNotebookName();
        });

        // 输入框回车保存
        nameInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                this.saveNotebookName();
            } else if (e.key === 'Escape') {
                this.cancelEditNotebookName();
            }
        });
    }

    startEditingNotebookName() {
        const nameDisplay = document.getElementById('currentNotebookName');
        const nameEditor = document.getElementById('notebookNameEditor');
        const nameInput = document.getElementById('notebookNameInput');

        nameInput.value = this.currentNotebook.name;
        nameDisplay.classList.add('hidden');
        nameEditor.classList.remove('hidden');
        nameInput.focus();
        nameInput.select();
    }

    async saveNotebookName() {
        const nameInput = document.getElementById('notebookNameInput');
        const newName = nameInput.value.trim();

        if (!newName) {
            this.showError('笔记本名称不能为空');
            return;
        }

        if (newName === this.currentNotebook.name) {
            this.cancelEditNotebookName();
            return;
        }

        try {
            this.showLoading('保存中...');

            const updated = await this.api(`/notebooks/${this.currentNotebook.id}`, {
                method: 'PUT',
                body: JSON.stringify({
                    name: newName,
                    description: this.currentNotebook.description
                })
            });

            // 更新本地数据
            this.currentNotebook.name = newName;
            this.currentNotebook.updated_at = updated.updated_at;

            // 更新 notebooks 列表中的数据
            const nb = this.notebooks.find(n => n.id === this.currentNotebook.id);
            if (nb) {
                nb.name = newName;
                nb.updated_at = updated.updated_at;
            }

            // 使缓存失效
            this.cache.delete('notebooks');

            // 更新显示
            document.getElementById('currentNotebookName').textContent = newName;
            this.cancelEditNotebookName();
            this.hideLoading();
            this.setStatus('笔记本名称已更新');

        } catch (error) {
            this.hideLoading();
            this.showError(error.message);
        }
    }

    cancelEditNotebookName() {
        const nameDisplay = document.getElementById('currentNotebookName');
        const nameEditor = document.getElementById('notebookNameEditor');

        nameDisplay.classList.remove('hidden');
        nameEditor.classList.add('hidden');
    }

  async handleSaveSettings(e) {
    e.preventDefault();
    const form = e.target;
    const data = new FormData(form);
    const provider = data.get('embedding_provider');
    const embeddingModel = data.get('embedding_model');
    const imageModel = data.get('image_model');
    const chatProvider = data.get('chat_provider');
    const chatModel = chatProvider === 'openai' ? data.get('chat_model_openai') : data.get('chat_model_ollama');

    this.showLoading('保存设置中...');

    try {
      const res = await this.api('/config', {
        method: 'POST',
        body: JSON.stringify({
          embedding_provider: provider,
          embedding_model: embeddingModel,
          image_model: imageModel,
          chat_provider: chatProvider,
          chat_model: chatModel
        }),
      });

      this.config.embeddingProvider = res.embedding_provider;
      this.config.embeddingModel = res.embedding_model;
      this.config.imageModel = res.image_model;
      this.config.chatProvider = res.chat_provider;
      this.config.chatModel = res.chat_model;
      this.showError('设置已保存');
      this.closeModals();
      this.hideLoading();
    } catch (error) {
      this.hideLoading();
      this.showError("保存设置失败: " + error.message);
    }
  }

  async handleCreateNotebook(e) {
    e.preventDefault();
    const form = e.target;
    const data = new FormData(form);

    this.showLoading("处理中...");

    try {
      const notebook = await this.api("/notebooks", {
        method: "POST",
        body: JSON.stringify({
          name: data.get("name"),
          description: data.get("description") || undefined,
        }),
      });

      // 使缓存失效
      this.cache.delete("notebooks");

      this.notebooks.push(notebook);
      this.renderNotebooks();
      this.selectNotebook(notebook.id);
      this.closeModals();
      form.reset();
      this.hideLoading();
    } catch (error) {
      this.hideLoading();
      this.showError(error.message);
    }
  }

  async deleteNotebook(id) {
    try {
      await this.api(`/notebooks/${id}`, { method: "DELETE" });

      // 使缓存失效
      this.cache.delete("notebooks");
      this.cache.deletePattern(`sources_${id}`);
      this.cache.deletePattern(`notes_${id}`);
      this.cache.deletePattern(`chat_${id}`);

      this.notebooks = this.notebooks.filter((nb) => nb.id !== id);

      if (this.currentNotebook?.id === id) {
        this.currentNotebook = null;
        this.clearContentAreas();
        this.switchView("landing");
      }

      this.renderNotebooks();
      this.updateFooter();
    } catch (error) {
      this.showError("删除笔记本失败: " + error.message);
    }
  }

  clearContentAreas() {
    const sourcesContainer = document.getElementById("sourcesGrid");
    sourcesContainer.innerHTML = `
            <div class="empty-state">
                <svg width="64" height="64" viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="1">
                    <path d="M20 8 L44 8 L48 12 L48 56 L20 56 Z"/>
                    <polyline points="44,8 44,12 48,12"/>
                    <line x1="28" y1="24" x2="40" y2="24"/>
                    <line x1="28" y1="32" x2="40" y2="32"/>
                    <line x1="28" y1="40" x2="36" y2="40"/>
                </svg>
                <p>添加来源以开始</p>
                <p class="empty-hint">支持 PDF, TXT, MD, DOCX, HTML</p>
            </div>
        `;

    const notesContainer = document.getElementById("notesList");
    notesContainer.innerHTML = `
            <div class="empty-state">
                <svg width="48" height="48" viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="1.5">
                    <path d="M12 4 L36 4 L40 8 L40 44 L12 44 Z"/>
                    <polyline points="36,4 36,8 40,8"/>
                </svg>
                <p>暂无笔记</p>
                <p class="empty-hint">使用转换从来源生成笔记</p>
            </div>
        `;

    const chatContainer = document.getElementById("chatMessages");
    chatContainer.innerHTML = `
            <div class="chat-welcome">
                <svg width="40" height="40" viewBox="0 0 40 40" fill="none" stroke="currentColor" stroke-width="1.5">
                    <circle cx="20" cy="12" r="6"/>
                    <path d="M8 38 C8 28 14 22 20 22 C26 22 32 28 32 38"/>
                </svg>
                <h3>与来源对话</h3>
                <p>询问关于笔记本内容的问题</p>
            </div>
        `;

    this.currentChatSession = null;
  }

  // 来源方法
  async loadSources() {
    if (!this.currentNotebook) return;

    const container = document.getElementById("sourcesGrid");
    const template = document.getElementById("sourceTemplate");

    try {
      // 先尝试从缓存获取
      const cacheKey = `sources_${this.currentNotebook.id}`;
      const cached = this.cache.get(cacheKey);

      // 从服务器获取最新数据
      const sources = await this.api(
        `/notebooks/${this.currentNotebook.id}/sources`
      );

      // 更新缓存
      this.cache.set(cacheKey, sources);

      if (sources.length === 0) {
        this.clearContentAreas();
        return;
      }

      container.innerHTML = "";

      sources.forEach((source) => {
        const clone = template.content.cloneNode(true);
        const card = clone.querySelector(".source-card");

        card.dataset.id = source.id;
        card.querySelector(".source-type-badge").textContent = source.type;
        card.querySelector(".source-name").textContent = source.name;
        card.querySelector(".source-meta").textContent =
          this.formatFileSize(source.file_size) || "文本来源";
        card.querySelector(".chunk-count").textContent =
          source.chunk_count || 0;

        const icon = this.getSourceIcon(source.type);
        card.querySelector(".source-icon").innerHTML = icon;

        const removeBtn = card.querySelector(".btn-remove-source");
        removeBtn.addEventListener("click", () => {
          this.removeSource(source.id);
        });

        if (this.config.allowDelete) {
          removeBtn.style.display = "flex";
        } else {
          removeBtn.style.display = "none";
        }

        container.appendChild(clone);
      });

      this.updateFooter();
    } catch (error) {
      console.error("加载来源失败:", error);
    }
  }

  // Render sources from data (for public notebooks)
  async renderSourcesList(sources) {
    const container = document.getElementById('sourcesGrid');
    const template = document.getElementById('sourceTemplate');

    if (!container || !template) return;

    container.innerHTML = '';

    if (sources.length === 0) {
        this.clearContentAreas();
        return;
    }

    sources.forEach(source => {
        const clone = template.content.cloneNode(true);
        const card = clone.querySelector('.source-card');

        card.dataset.id = source.id;
        card.querySelector('.source-type-badge').textContent = source.type;
        card.querySelector('.source-name').textContent = source.name;
        card.querySelector('.source-meta').textContent = this.formatFileSize(source.file_size) || '文本来源';
        card.querySelector('.chunk-count').textContent = source.chunk_count || 0;

        const icon = this.getSourceIcon(source.type);
        card.querySelector('.source-icon').innerHTML = icon;

        const removeBtn = card.querySelector('.btn-remove-source');
        if (removeBtn) {
            removeBtn.style.display = 'none';
        }

        container.appendChild(clone);
    });

    this.updateFooter();
  }

  // Render notes from data (for public notebooks)
  async renderNotesList(notes) {
    const container = document.getElementById('notesList');
    const template = document.getElementById('noteTemplate');

    if (!container || !template) return;

    container.innerHTML = '';

    if (notes.length === 0) {
        return;
    }

    notes.forEach(note => {
        const clone = template.content.cloneNode(true);
        const item = clone.querySelector('.note-item');

        item.dataset.id = note.id;
        item.querySelector('.note-type-badge').textContent = this.noteTypeNameMap[note.type] || note.type.toUpperCase();
        item.querySelector('.note-title').textContent = note.title;

        const plainText = note.content
            .replace(/^#+\s+/gm, '')
            .replace(/\*\*/g, '')
            .replace(/\*/g, '')
            .replace(/`/g, '')
            .replace(/\ \[([^\]]+)\]\([^)]+\)/g, '$1')
            .replace(/\n+/g, ' ')
            .trim();

        item.querySelector('.note-preview').textContent = plainText;
        item.querySelector('.note-date').textContent = this.formatDate(note.created_at);
        item.querySelector('.note-sources').textContent = `${note.source_ids?.length || 0} 来源`;

        const deleteBtn = item.querySelector('.btn-delete-note');
        if (deleteBtn) {
            deleteBtn.style.display = 'none';
        }

        item.addEventListener('click', (e) => {
            if (!e.target.closest('.btn-delete-note')) {
                this.viewNote(note);
            }
        });

        container.appendChild(clone);
    });

    this.updateFooter();
  }
  async handleTextSource(e) {
    e.preventDefault();
    const form = e.target;
    const data = new FormData(form);

    this.showLoading("处理中...");

    try {
      await this.api(`/notebooks/${this.currentNotebook.id}/sources`, {
        method: "POST",
        body: JSON.stringify({
          name: data.get("name"),
          type: "text",
          content: data.get("content"),
        }),
      });

      this.hideLoading();
      this.closeModals();
      form.reset();
      await this.loadSources();
      await this.updateCurrentNotebookCounts();
    } catch (error) {
      this.hideLoading();
      this.showError(error.message);
    }
  }

  async handleURLSource(e) {
    e.preventDefault();
    const form = e.target;
    const data = new FormData(form);

    this.showLoading("获取网址内容中...");

    try {
      await this.api(`/notebooks/${this.currentNotebook.id}/sources`, {
        method: "POST",
        body: JSON.stringify({
          name: data.get("name") || data.get("url"),
          type: "url",
          url: data.get("url"),
        }),
      });

      this.hideLoading();
      this.closeModals();
      form.reset();
      await this.loadSources();
      await this.updateCurrentNotebookCounts();
    } catch (error) {
      this.hideLoading();
      this.showError(error.message);
    }
  }

  handleDrop(e) {
    e.preventDefault();
    document.getElementById("dropZone").classList.remove("drag-over");

    const files = e.dataTransfer.files;
    if (!files.length) return;

    document.getElementById("fileInput").files = files;
    this.handleFileUpload({ target: { files } });
  }

  async removeSource(id) {
    try {
      await this.api(`/notebooks/${this.currentNotebook.id}/sources/${id}`, {
        method: "DELETE",
      });
      await this.loadSources();
      await this.updateCurrentNotebookCounts();
    } catch (error) {
      this.showError("移除来源失败");
    }
  }

  // 笔记方法
  async loadNotes() {
    if (!this.currentNotebook) return;

    const container = document.getElementById("notesList");
    const template = document.getElementById("noteTemplate");
    const countHeader = document.querySelector(".section-notes .panel-title");

    try {
      // 先尝试从缓存获取
      const cacheKey = `notes_${this.currentNotebook.id}`;
      const cached = this.cache.get(cacheKey);

      // 从服务器获取最新数据
      const notes = await this.api(
        `/notebooks/${this.currentNotebook.id}/notes`
      );

      // 更新缓存
      this.cache.set(cacheKey, notes);

      if (countHeader) {
        countHeader.textContent = `笔记 (${notes.length})`;
      }

      if (notes.length === 0) {
        container.innerHTML = `
                    <div class="empty-state">
                        <svg width="48" height="48" viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="1.5">
                            <path d="M12 4 L36 4 L40 8 L40 44 L12 44 Z"/>
                            <polyline points="36,4 36,8 40,8"/>
                        </svg>
                        <p>暂无笔记</p>
                        <p class="empty-hint">使用转换从来源生成笔记</p>
                    </div>
                `;
        return;
      }

      container.innerHTML = "";

      notes.forEach((note) => {
        const clone = template.content.cloneNode(true);
        const item = clone.querySelector(".note-item");

        item.dataset.id = note.id;
        item.querySelector(".note-type-badge").textContent =
          this.noteTypeNameMap[note.type] || note.type.toUpperCase();
        item.querySelector(".note-title").textContent = note.title;

        const plainText = note.content
          .replace(/^#+\s+/gm, "")
          .replace(/\*\*/g, "")
          .replace(/\*/g, "")
          .replace(/`/g, "")
          .replace(/\ \[([^\]]+)\]\([^)]+\)/g, "$1")
          .replace(/\n+/g, " ")
          .trim();

        item.querySelector(".note-preview").textContent = plainText;
        item.querySelector(".note-date").textContent = this.formatDate(
          note.created_at
        );
        item.querySelector(".note-sources").textContent = `${
          note.source_ids?.length || 0
        } 来源`;

        const deleteBtn = item.querySelector(".btn-delete-note");
        deleteBtn.addEventListener("click", () => {
          this.deleteNote(note.id);
        });

        if (this.config.allowDelete) {
          deleteBtn.style.display = "flex";
        } else {
          deleteBtn.style.display = "none";
        }

        item.addEventListener("click", (e) => {
          if (!e.target.closest(".btn-delete-note")) {
            this.viewNote(note);
          }
        });

        container.appendChild(clone);
      });

      this.updateFooter();
    } catch (error) {
      console.error("加载笔记失败:", error);
    }
  }

  // Helper to rewrite image URLs for public notebooks
  rewriteImageUrlsForPublic(content) {
    if (!this.currentPublicToken) {
      return content;
    }
    // Regex to find image URLs in markdown: ![alt](url) or <img src="url">
    // And also direct URLs that might be in metadata
    return content.replace(
      /(src="|!\[.*?\]\()(\/api\/notebooks\/.*?\/files\/.*?)"/g,
      `$1/public/${this.currentPublicToken}$2"`
    );
  }

  async viewNote(note) {
    // Debug: log note metadata
    console.log('viewNote - metadata:', note.metadata);
    console.log('viewNote - image_url:', note.metadata?.image_url);
    console.log('viewNote - currentPublicToken:', this.currentPublicToken);

    // Rewrite image URLs for public notebooks
    const content = this.rewriteImageUrlsForPublic(note.content);
    const renderedContent = marked.parse(content);

    // 信息图错误提示 HTML
    let infographicErrorHTML = '';
    if (note.type === 'infograph' && note.metadata?.image_error) {
        const fullPrompt = note.content + '\n\n**注意：无论来源是什么语言，请务必使用中文**';
        const escapedPrompt = this.escapeHtml(fullPrompt);
        const escapedError = this.escapeHtml(note.metadata.image_error);

        infographicErrorHTML = `
            <div class="infographic-error-banner">
                <div class="error-banner-content">
                    <svg width="20" height="20" viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="2">
                        <circle cx="10" cy="10" r="8"/>
                        <line x1="10" y1="7" x2="10" y2="13"/>
                        <line x1="10" y1="16" x2="10" y2="16"/>
                    </svg>
                    <div>
                        <strong>图片生成失败</strong>
                        <p>${escapedError}</p>
                    </div>
                </div>
                <div class="error-banner-prompt">
                    <strong>生成的 Prompt（可用于手动生成）：</strong>
                    <pre>${escapedPrompt}</pre>
                </div>
            </div>
        `;
    }

    // Rewrite image URL for infographics if present
    const originalImageUrl = note.metadata?.image_url || null;
    const infographicImageUrl = originalImageUrl
        ? this.rewriteImageUrlsForPublic(originalImageUrl)
        : null;

    console.log('viewNote - originalImageUrl:', originalImageUrl);
    console.log('viewNote - infographicImageUrl:', infographicImageUrl);

    const infographicHTML = infographicImageUrl
        ? `<div class="infographic-container">
             <img src="${infographicImageUrl}" alt="Infographic" class="infographic-image" onerror="console.error('Failed to load image:', this.src)">
             <div class="infographic-actions">
                <a href="${infographicImageUrl}" target="_blank" class="btn-text">查看大图</a>
             </div>
           </div>`
        : "";

    // PPT Slider HTML
    let pptSliderHTML = "";
    if (note.metadata?.slides && note.metadata.slides.length > 0) {
      const slides = note.metadata.slides.map(src => {
        const rewritten = this.rewriteImageUrlsForPublic(src);
        console.log('viewNote - slide original:', src, 'rewritten:', rewritten);
        return rewritten;
      });
      pptSliderHTML = `
                <div class="ppt-viewer-container" id="pptViewer">
                    <div class="ppt-slides-wrapper">
                        ${slides
                          .map(
                            (src, index) => `
                            <div class="ppt-slide ${
                              index === 0 ? "active" : ""
                            }" data-index="${index}">
                                <img src="${src}" alt="Slide ${index + 1}" onerror="console.error('Failed to load slide:', this.src)">
                                <div class="ppt-slide-counter">${index + 1} / ${
                                slides.length
                              }</div>
                            </div>
                        `
                          )
                          .join("")}
                    </div>
                    <div class="ppt-controls">
                        <button class="btn-ppt-nav prev" id="btnPptPrev">
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="15 18 9 12 15 6"></polyline></svg>
                        </button>
                        <button class="btn-ppt-nav next" id="btnPptNext">
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"></polyline></svg>
                        </button>
                    </div>
                </div>
            `;
    }

    // Determine if we should show the text content
    const showMarkdownContent =
      (note.type !== "infograph" && note.type !== "ppt") ||
      (!note.metadata?.image_url && !note.metadata?.slides);

    // Show the Note tab button
    const tabBtnNote = document.getElementById("tabBtnNote");
    if (tabBtnNote) {
      tabBtnNote.style.display = "flex";
    }

    // Remove existing note view if any
    const existingNoteView = document.querySelector(".note-view-container");
    if (existingNoteView) {
      existingNoteView.remove();
    }

    // Create note view container and insert it after chat-messages-wrapper
    const noteViewHTML = `
            <div class="note-view-container">
                <div class="note-view-header">
                    <div class="note-view-info">
                        <span class="note-view-type">${note.type}</span>
                        <span class="note-view-title-text">${note.title}</span>
                    </div>
                    <div class="note-view-actions">
                        <button class="btn-copy-note" id="btnCopyNote" title="复制 Markdown">
                            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2">
                                <rect x="3" y="3" width="10" height="10" rx="1"/>
                                <path d="M7 3 L7 1 C7 1 13 1 13 1 L13 13 L11 13"/>
                            </svg>
                        </button>
                    </div>
                </div>
                <div class="note-view-content">
                    ${infographicErrorHTML}
                    ${infographicHTML}
                    ${pptSliderHTML}
                    <div class="markdown-content" style="${
                      showMarkdownContent ? "" : "display:none"
                    }">${renderedContent}</div>
                </div>
            </div>
        `;

    const chatWrapper = document.querySelector(".chat-messages-wrapper");
    chatWrapper.insertAdjacentHTML("afterend", noteViewHTML);

    // PPT Navigation Logic
    if (note.metadata?.slides) {
      let currentSlide = 0;
      const slidesCount = note.metadata.slides.length;
      const slideElements = document.querySelectorAll(".ppt-slide");

      const showSlide = (n) => {
        slideElements[currentSlide].classList.remove("active");
        currentSlide = (n + slidesCount) % slidesCount;
        slideElements[currentSlide].classList.add("active");
      };

      document
        .getElementById("btnPptPrev")
        .addEventListener("click", () => showSlide(currentSlide - 1));
      document
        .getElementById("btnPptNext")
        .addEventListener("click", () => showSlide(currentSlide + 1));

      // Key navigation
      const keyHandler = (e) => {
        if (e.key === "ArrowLeft") showSlide(currentSlide - 1);
        if (e.key === "ArrowRight") showSlide(currentSlide + 1);
      };
      document.addEventListener("keydown", keyHandler);
      // Cleanup handler on container remove? We'll leave it for now or add observer
    }

    // Render Mermaid diagrams if any
    if (window.mermaid) {
      try {
        mermaid.initialize({
          startOnLoad: false,
          theme: "base",
          securityLevel: "loose",
          fontFamily: "var(--font-sans)",
          themeVariables: {
            // Vibrant WeChat Green Theme
            primaryColor: "#ecfdf5", // Lighter, more vibrant green background
            primaryTextColor: "#065f46", // Deep emerald for text
            primaryBorderColor: "#10b981", // Bright emerald border
            lineColor: "#10b981", // Bright line color
            secondaryColor: "#f0fdf4",
            tertiaryColor: "#ffffff",
            fontSize: "14px",
            mainBkg: "#ecfdf5",
            nodeBorder: "#10b981",
            clusterBkg: "#f0fdf4",
            // Mindmap specific vibrancy
            nodeTextColor: "#065f46",
            edgeColor: "#34d399", // Slightly lighter green for edges
          },
          mindmap: {
            useMaxWidth: true,
            padding: 20,
          },
        });

        const contentArea = document.querySelector(".note-view-content");
        const mermaidBlocks = contentArea.querySelectorAll(
          "pre code.language-mermaid"
        );

        // Helper to fix common mermaid errors
        const sanitizeMermaid = (code) => {
          let sanitized = code.trim();

          // 1. If it's a graph and has unquoted brackets, try to wrap them
          if (sanitized.startsWith("graph")) {
            // Fix things like: A --> socket() --> B
            sanitized = sanitized.replace(
              /(\s+)-->(\s+)([^"\s][^-\n>]*\([^)]*\)[^-\n>]*)/g,
              '$1-->$2"$3"'
            );
            sanitized = sanitized.replace(
              /([^"\s][^-\n>]*\([^)]*\)[^-\n>]*)\s+-->/g,
              '"$1" -->'
            );
          }

          // 2. Fix mindmap - handle special characters in node labels
          if (sanitized.startsWith("mindmap")) {
            const lines = sanitized.split("\n");
            const processedLines = [];

            for (let i = 0; i < lines.length; i++) {
              let line = lines[i];
              const trimmed = line.trim();

              // Skip empty lines and the mindmap declaration
              if (!trimmed || trimmed === "mindmap") {
                processedLines.push(line);
                continue;
              }

              // Fix root if missing double parens
              if (trimmed.startsWith("root") && !line.includes("((")) {
                line = line.replace(/root\s+(.+)/, "root(($1))");
                processedLines.push(line);
                continue;
              }

              // For other nodes, check if they contain special characters that need quoting
              // Special chars: parentheses, brackets, braces, quotes, colons, semicolons
              const hasSpecialChars = /[\(\)\[\]\{\}"':;,\s]{2,}/.test(trimmed);
              const alreadyQuoted =
                /^["'].*["']$/.test(trimmed) ||
                /^\(.*\)$/.test(trimmed) ||
                /^\[.*\]$/.test(trimmed);

              if (hasSpecialChars && !alreadyQuoted && trimmed.length > 0) {
                // Extract indentation and node content
                const indentMatch = line.match(/^(\s*)/);
                const indent = indentMatch ? indentMatch[1] : "";
                const content = trimmed;

                // Wrap in quotes and preserve the original brackets for styling
                // Replace inner parentheses that are part of the content with quoted version
                processedLines.push(
                  indent + '"' + content.replace(/"/g, '\\"') + '"'
                );
              } else {
                processedLines.push(line);
              }
            }

            sanitized = processedLines.join("\n");
          }

          return sanitized;
        };

        for (let i = 0; i < mermaidBlocks.length; i++) {
          const block = mermaidBlocks[i];
          const pre = block.parentElement;
          const rawCode = block.textContent;
          const cleanCode = sanitizeMermaid(rawCode);

          const id = `mermaid-diag-${Date.now()}-${i}`;

          try {
            const { svg } = await mermaid.render(id, cleanCode);
            const container = document.createElement("div");
            container.className = "mermaid-diagram";
            container.innerHTML = svg;
            pre.parentNode.replaceChild(container, pre);
          } catch (renderErr) {
            console.error("Mermaid Render Error:", renderErr);
            // Final fallback: If rendering failed, try one more time by stripping ALL parentheses from labels
            try {
              const lastResort = cleanCode.replace(/\(|\)/g, "");
              const { svg } = await mermaid.render(`${id}-retry`, lastResort);
              const container = document.createElement("div");
              container.className = "mermaid-diagram";
              container.innerHTML = svg;
              pre.parentNode.replaceChild(container, pre);
            } catch (e) {
              pre.innerHTML = `<div style="color:red; font-size:12px; padding:10px;">渲染失败: ${renderErr.message}</div>`;
            }
          }
        }
      } catch (err) {
        console.error("Mermaid general error:", err);
      }
    }

    // Render MathJax if available
    if (window.MathJax && window.MathJax.typesetPromise) {
        try {
            await MathJax.typesetPromise([document.querySelector('.note-view-content')]);
        } catch (e) {
            console.warn('MathJax rendering error:', e);
        }
    }

    // Switch to note tab
    this.switchPanelTab("note");

    // Copy button
    const copyBtn = document.getElementById("btnCopyNote");
    copyBtn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(note.content);
        const originalHTML = copyBtn.innerHTML;
        copyBtn.innerHTML = `
                    <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="2">
                        <polyline points="4,8 6,10 12,4"/>
                    </svg>
                `;
        copyBtn.classList.add("copied");
        setTimeout(() => {
          copyBtn.innerHTML = originalHTML;
          copyBtn.classList.remove("copied");
        }, 2000);
        this.setStatus("已复制!");
      } catch (err) {
        this.showError("复制失败");
      }
    });

    // Highlight the selected note in the sidebar
    document.querySelectorAll(".note-item").forEach((el) => {
      el.classList.remove("selected");
    });
    const noteItem = document.querySelector(`.note-item[data-id="${note.id}"]`);
    if (noteItem) {
      noteItem.classList.add("selected");
    }
  }

  async deleteNote(id) {
    try {
      await this.api(`/notebooks/${this.currentNotebook.id}/notes/${id}`, {
        method: "DELETE",
      });
      await this.loadNotes();
      await this.updateCurrentNotebookCounts();

      // If notes_list tab is active or visible, refresh it
      const tabBtnNotesList = document.getElementById("tabBtnNotesList");
      if (tabBtnNotesList && !tabBtnNotesList.classList.contains("hidden")) {
        this.renderNotesCompactGrid();
      }
    } catch (error) {
      this.showError("删除笔记失败");
    }
  }

  // 转换方法
  async handleTransform(type, element) {
    if (!this.currentNotebook) {
      this.showError("请先选择一个笔记本");
      return;
    }

    const sources = await this.api(
      `/notebooks/${this.currentNotebook.id}/sources`
    );
    if (sources.length === 0) {
      this.showError("请先添加来源");
      return;
    }

    const customPrompt = document.getElementById("customPrompt").value;
    const typeName = this.noteTypeNameMap[type] || "内容";

    // 1. 开始动画
    if (element) {
      element.classList.add("loading");
    }

    // 2. 添加占位笔记
    const notesContainer = document.getElementById("notesList");
    const template = document.getElementById("noteTemplate");
    const placeholder = template.content
      .cloneNode(true)
      .querySelector(".note-item");

    placeholder.classList.add("placeholder");
    placeholder.querySelector(
      ".note-title"
    ).textContent = `正在生成${typeName}...`;
    placeholder.querySelector(".note-preview").textContent =
      "AI 正在分析您的来源并撰写笔记，请稍候...";
    placeholder.querySelector(".note-date").textContent = "刚刚";
    placeholder.querySelector(".note-type-badge").textContent =
      type.toUpperCase();

    // 占位符暂不显示删除按钮
    const delBtn = placeholder.querySelector(".btn-delete-note");
    if (delBtn) delBtn.style.display = "none";

    // 如果有“暂无笔记”状态，先清空
    const emptyState = notesContainer.querySelector(".empty-state");
    if (emptyState) emptyState.remove();

    notesContainer.prepend(placeholder);
    placeholder.scrollIntoView({ behavior: "smooth", block: "nearest" });

    try {
      const sourceIds = sources.map((s) => s.id);
      const note = await this.api(
        `/notebooks/${this.currentNotebook.id}/transform`,
        {
          method: "POST",
          body: JSON.stringify({
            type: type,
            prompt: customPrompt || undefined,
            source_ids: sourceIds,
            length: "medium",
            format: "markdown",
          }),
        }
      );

      // 3. 停止动画并更新占位符
      if (element) element.classList.remove("loading");

      // 替换占位符内容
      placeholder.classList.remove("placeholder");
      placeholder.dataset.id = note.id;
      placeholder.querySelector(".note-title").textContent = note.title;

      const plainText = note.content
        .replace(/^#+\s+/gm, "")
        .replace(/\*\*/g, "")
        .replace(/\*/g, "")
        .replace(/`/g, "")
        .replace(/\ \[([^\]]+)\]\([^)]+\)/g, "$1")
        .replace(/\n+/g, " ")
        .trim();

      placeholder.querySelector(".note-preview").textContent = plainText;
      placeholder.querySelector(".note-sources").textContent = `${
        note.source_ids?.length || 0
      } 来源`;

      // 恢复删除按钮并绑定事件
      if (delBtn) {
        // Apply config to delete button
        if (this.config.allowDelete) {
          delBtn.style.display = "flex";
        } else {
          delBtn.style.display = "none";
        }
        delBtn.addEventListener("click", (e) => {
          e.stopPropagation();
          this.deleteNote(note.id);
        });
      }

      // 绑定查看事件
      placeholder.addEventListener("click", (e) => {
        if (!e.target.closest(".btn-delete-note")) {
          this.viewNote(note);
        }
      });

      await this.updateCurrentNotebookCounts();
      this.updateFooter();
      document.getElementById("customPrompt").value = "";
      this.setStatus(`成功生成 ${typeName}`);

      // If type is insight, refresh sources list to show the injected insight report
      if (type === "insight") {
        await this.loadSources();
      }

      // If notes_list tab is active or visible, refresh it
      const tabBtnNotesList = document.getElementById("tabBtnNotesList");
      if (tabBtnNotesList && !tabBtnNotesList.classList.contains("hidden")) {
        this.renderNotesCompactGrid();
      }
    } catch (error) {
      if (element) element.classList.remove("loading");
      placeholder.remove(); // 失败则移除占位符
      this.showError(error.message);
    }
  }

  // 聊天方法
  async loadChatSessions() {
    if (!this.currentNotebook) return;

    try {
      await this.api(`/notebooks/${this.currentNotebook.id}/chat/sessions`);
      const container = document.getElementById("chatMessages");
      container.innerHTML = `
                <div class="chat-welcome">
                    <svg width="40" height="40" viewBox="0 0 40 40" fill="none" stroke="currentColor" stroke-width="1.5">
                        <circle cx="20" cy="12" r="6"/>
                        <path d="M8 38 C8 28 14 22 20 22 C26 22 32 28 32 38"/>
                    </svg>
                    <h3>与来源对话</h3>
                    <p>询问关于笔记本内容的问题</p>
                </div>
            `;
      this.currentChatSession = null;
    } catch (error) {
      console.error("加载对话失败:", error);
    }
  }

  async handleChat(e) {
    if (e) e.preventDefault();
    if (!this.currentNotebook) return;

    const input = document.getElementById("chatInput");
    const content = input.value.trim();
    if (!content) return;

    input.value = "";
    this.addMessage("user", content);
    this.setStatus("思考中...");

    const typingId = "typing-" + Date.now();
    this.addMessage("assistant", "...", [], typingId);

    try {
      const response = await this.api(
        `/notebooks/${this.currentNotebook.id}/chat`,
        {
          method: "POST",
          body: JSON.stringify({
            message: content,
            session_id: this.currentChatSession || undefined,
          }),
        }
      );

      const typingEl = document.getElementById(typingId);
      if (typingEl) typingEl.remove();

      this.addMessage("assistant", response.message, response.sources);
      this.currentChatSession = response.session_id;
      this.setStatus("就绪");
    } catch (error) {
      const typingEl = document.getElementById(typingId);
      if (typingEl) typingEl.remove();

      this.addMessage("assistant", `错误: ${error.message}`);
      this.setStatus("错误");
    }
  }

  addMessage(role, content, sources = []) {
    const container = document.getElementById("chatMessages");
    const template = document.getElementById("messageTemplate");

    const welcome = container.querySelector(".chat-welcome");
    if (welcome) welcome.remove();

    const clone = template.content.cloneNode(true);
    const message = clone.querySelector(".chat-message");

    message.dataset.role = role;

    const avatar = message.querySelector(".message-avatar");
    avatar.textContent = role === "assistant" ? "AI" : "你";

    const messageText = message.querySelector(".message-text");
    if (role === "assistant") {
      messageText.innerHTML = marked.parse(content);
    } else {
      messageText.textContent = content;
    }

    if (sources.length > 0) {
      const sourcesContainer = message.querySelector(".message-sources");
      sources.forEach((source) => {
        const tag = document.createElement("span");
        tag.className = "source-tag";
        tag.textContent = source.name || source.id;
        sourcesContainer.appendChild(tag);
      });
    }

    // Render MathJax for the new message if available
    if (window.MathJax && window.MathJax.typesetPromise && role === 'assistant') {
        MathJax.typesetPromise([messageText]).catch(err => {
            console.warn('MathJax rendering error:', err);
        });
    }

    container.appendChild(clone);
    container.scrollTop = container.scrollHeight;
  }

  // UI 方法
  closeModals() {
    document
      .querySelectorAll(".modal")
      .forEach((m) => m.classList.remove("active"));
    document.getElementById("modalOverlay").classList.remove("active");
    this.hideLoading();
  }

  showLoading(text) {
    document.getElementById("loadingText").textContent = text || "处理中...";
    document.getElementById("loadingOverlay").classList.add("active");
  }

  hideLoading() {
    document.getElementById("loadingOverlay").classList.remove("active");
  }

  setStatus(text) {
    document.getElementById("footerStatus").textContent = text;
  }

  showError(message) {
    this.setStatus(`错误: ${message}`);

    const toast = document.createElement("div");
    toast.className = "error-toast";
    toast.style.cssText = `
            position: fixed; bottom: 60px; right: 20px; padding: 12px 20px;
            background: var(--accent-red); color: white; font-family: var(--font-mono);
            font-size: 0.75rem; border-radius: 4px; box-shadow: var(--shadow-medium);
            animation: slideIn 0.3s ease; z-index: 3000; white-space: pre-wrap; max-width: 400px;
        `;
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = "0";
      setTimeout(() => toast.remove(), 300);
    }, 3000);
  }

  updateFooter() {
    const sourceCount = document.querySelectorAll(".source-card").length;
    const noteCount = document.querySelectorAll(".note-item").length;
    document.getElementById(
      "footerStats"
    ).textContent = `${sourceCount} 来源 · ${noteCount} 笔记`;
  }

  formatDate(dateString) {
    const date = new Date(dateString);
    const now = new Date();
    const diff = now - date;

    if (diff < 60000) return "刚刚";
    if (diff < 3600000) return `${Math.floor(diff / 60000)}分钟前`;
    if (diff < 86400000) return `${Math.floor(diff / 3600000)}小时前`;

    return date.toLocaleDateString("zh-CN", { month: "short", day: "numeric" });
  }

  rewriteImageUrlsForPublic(content) {
    if (!content || !this.currentPublicToken) return content;
    // Rewrite /api/notes/image/ to /public/notes/image/ and append public token
    return content.replace(/\/api\/notes\/image\/([a-f0-9-]+)/g, (match, id) => {
        return `/public/notes/image/${id}?token=${this.currentPublicToken}`;
    });
  }

  escapeHtml(unsafe) {
    return unsafe
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
  }

  async updateCurrentNotebookCounts() {
    if (!this.currentNotebook) return;

    const [sources, notes] = await Promise.all([
      this.api(`/notebooks/${this.currentNotebook.id}/sources`),
      this.api(`/notebooks/${this.currentNotebook.id}/notes`),
    ]);

    const notebookCard = document.querySelector(
      `.notebook-card[data-id="${this.currentNotebook.id}"]`
    );
    if (notebookCard) {
      notebookCard.querySelector(".stat-sources").textContent = `${sources.length} 来源`;
      notebookCard.querySelector(".stat-notes").textContent = `${notes.length} 笔记`;
    }
  }

  // 通用 toast 提示方法
  showToast(message, type = "error") {
    const colors = {
      error: "var(--accent-red)",
      warn: "var(--accent-orange)",
      success: "var(--accent-green)",
    };

    const toast = document.createElement("div");
    toast.className = `${type}-toast`;
    toast.style.cssText = `
          position: fixed; bottom: 60px; right: 20px; padding: 12px 20px;
          background: ${colors[type]}; color: white; font-family: var(--font-mono);
          font-size: 0.75rem; border-radius: 4px; box-shadow: var(--shadow-medium);
          animation: slideIn 0.3s ease; z-index: 3000; white-space: pre-wrap; max-width: 400px;
      `;
    toast.textContent = message;
    document.body.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = "0";
      setTimeout(() => toast.remove(), 300);
    }, 5000);
  }
}

// 初始化
document.addEventListener("DOMContentLoaded", () => {
  window.app = new OpenNotebook();
});
