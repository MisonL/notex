package backend

import (
	"context"
	"embed"
	"fmt"
	"io/fs"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
	"github.com/kataras/golog"
)

//go:embed frontend/index.html frontend/static
var frontendFS embed.FS

// Server handles HTTP requests
type Server struct {
	cfg         Config
	vectorStore *VectorStore
	store       *CachedStore
	agent       *Agent
	http        *gin.Engine
	auth        *AuthHandler
	// Track which notebooks have been loaded into vector store
	loadedNotebooks map[string]bool
	vectorMutex     sync.RWMutex
}

// NewServer creates a new server
func NewServer(cfg Config) (*Server, error) {
	// Initialize vector store
	vectorStore, err := NewVectorStore(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create vector store: %w", err)
	}

	// Initialize store
	baseStore, err := NewStore(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create store: %w", err)
	}

	// Wrap store with cache (5 minute TTL)
	store := NewCachedStore(baseStore, 5*time.Minute)

	// Initialize agent
	agent, err := NewAgent(cfg, vectorStore)
	if err != nil {
		return nil, fmt.Errorf("failed to create agent: %w", err)
	}
	
	// Initialize auth handler
	authHandler := NewAuthHandler(cfg, baseStore)

	// Create Gin router
	gin.SetMode(gin.ReleaseMode)
	router := gin.New()
	router.Use(gin.Recovery(), gin.Logger())

	s := &Server{
		cfg:             cfg,
		vectorStore:     vectorStore,
		store:           store,
		agent:           agent,
		http:            router,
		auth:            authHandler,
		loadedNotebooks: make(map[string]bool),
	}

	// 延迟加载向量索引，不在启动时加载
	golog.Infof("✅ server initialized (vector index will load on demand)")

	s.setupRoutes()

	return s, nil
}

// setupRoutes configures all routes
func (s *Server) setupRoutes() {
	// Serve static files from embedded filesystem (no audit)
	staticFS, _ := fs.Sub(frontendFS, "frontend/static")
	s.http.StaticFS("/static", http.FS(staticFS))

	// Serve uploaded files with auth protection
	// Remove public uploads route - files are now served via authenticated API
	// Old: uploads.Static("/", "./data/uploads")

	// Serve index.html at root (with audit)
	s.http.GET("/", AuditMiddlewareLite(), func(c *gin.Context) {
		c.Header("Cache-Control", "no-cache")
		content, _ := frontendFS.ReadFile("frontend/index.html")
		c.Data(http.StatusOK, "text/html; charset=utf-8", content)
	})

	// Serve index.html at /notes/:id (for shareable notebook links)
	// This route allows users to access a notebook directly via URL like /notes/xxxxxxxx
	// The frontend will parse the notebook ID from the URL and load it
	s.http.GET("/notes/:id", AuditMiddlewareLite(), func(c *gin.Context) {
		c.Header("Cache-Control", "no-cache")
		content, _ := frontendFS.ReadFile("frontend/index.html")
		c.Data(http.StatusOK, "text/html; charset=utf-8", content)
	})

	// Auth routes (OAuth - no auth required)
	auth := s.http.Group("/auth")
	{
		auth.GET("/login/:provider", s.auth.HandleLogin)
		auth.GET("/callback/:provider", s.auth.HandleCallback)
	}

	// File serving route - checks notebook public status internally
	golog.Info("Registering /api/files/:filename route")
	s.http.GET("/api/files/:filename", AuditMiddlewareLite(), OptionalAuthMiddleware(s.cfg.JWTSecret), s.handleServeFile)

	// API routes
	api := s.http.Group("/api")
	api.Use(AuditMiddlewareLite())
	api.Use(AuthMiddleware(s.cfg.JWTSecret)) // Apply JWT Auth
	{
		// Health check
		api.GET("/health", s.handleHealth)
		api.GET("/config", s.handleConfig)

		// Auth API (get current user)
		api.GET("/auth/me", s.auth.HandleMe)

		// Notebook routes
		notebooks := api.Group("/notebooks")
		{
			notebooks.GET("", s.handleListNotebooks)
			notebooks.GET("/stats", s.handleListNotebooksWithStats)
			notebooks.POST("", s.handleCreateNotebook)
			notebooks.GET("/:id", s.handleGetNotebook)
			notebooks.PUT("/:id", s.handleUpdateNotebook)
			notebooks.DELETE("/:id", s.handleDeleteNotebook)

			// Public sharing
			notebooks.PUT("/:id/public", s.handleSetNotebookPublic)

			// Sources within a notebook
			notebooks.GET("/:id/sources", s.handleListSources)
			notebooks.POST("/:id/sources", s.handleAddSource)
			notebooks.DELETE("/:id/sources/:sourceId", s.handleDeleteSource)

			// Notes within a notebook
			notebooks.GET("/:id/notes", s.handleListNotes)
			notebooks.POST("/:id/notes", s.handleCreateNote)
			notebooks.DELETE("/:id/notes/:noteId", s.handleDeleteNote)

			// Transformations
			notebooks.POST("/:id/transform", s.handleTransform)

			// Chat within a notebook
			notebooks.GET("/:id/chat/sessions", s.handleListChatSessions)
			notebooks.POST("/:id/chat/sessions", s.handleCreateChatSession)
			notebooks.DELETE("/:id/chat/sessions/:sessionId", s.handleDeleteChatSession)
			notebooks.POST("/:id/chat/sessions/:sessionId/messages", s.handleSendMessage)

			// Quick chat (auto-create session)
			notebooks.POST("/:id/chat", s.handleChat)
		}

		// Upload endpoint
		api.POST("/upload", s.handleUpload)
	}

	// Public notebook routes (no authentication required)
	public := s.http.Group("/public")
	public.Use(AuditMiddlewareLite())
	{
		// List all public notebooks with infograph or ppt notes
		public.GET("/notebooks", s.handleListPublicNotebooks)
		// Get public notebook by token
		public.GET("/notebooks/:token", s.handleGetPublicNotebook)
		// Get public notebook sources
		public.GET("/notebooks/:token/sources", s.handleListPublicSources)
		// Get public notebook notes
		public.GET("/notebooks/:token/notes", s.handleListPublicNotes)
	}

	// Serve public notebook page
	s.http.GET("/public/:token", AuditMiddlewareLite(), func(c *gin.Context) {
		c.Header("Cache-Control", "no-cache")
		content, _ := frontendFS.ReadFile("frontend/index.html")
		c.Data(http.StatusOK, "text/html; charset=utf-8", content)
	})
}

// loadNotebookVectorIndex loads a notebook's sources into the vector store on demand
func (s *Server) loadNotebookVectorIndex(ctx context.Context, notebookID string) error {
	s.vectorMutex.Lock()
	defer s.vectorMutex.Unlock()

	// Check if already loaded
	if s.loadedNotebooks[notebookID] {
		return nil
	}

	golog.Infof("🔄 loading vector index for notebook %s...", notebookID)

	sources, err := s.store.Store.ListSources(ctx, notebookID)
	if err != nil {
		return fmt.Errorf("failed to list sources: %w", err)
	}

	for _, src := range sources {
		if src.Content != "" {
			if _, err := s.vectorStore.IngestText(ctx, notebookID, src.Name, src.Content); err != nil {
				golog.Errorf("failed to load source %s: %v", src.Name, err)
			}
		}
	}

	s.loadedNotebooks[notebookID] = true
	stats, _ := s.vectorStore.GetStats(ctx)
	golog.Infof("✅ notebook %s loaded into vector store (%d total documents)", notebookID, stats.TotalDocuments)

	return nil
}

// Start starts the server
func (s *Server) Start() error {
	addr := fmt.Sprintf("%s:%s", s.cfg.ServerHost, s.cfg.ServerPort)
	golog.Infof("server starting on %s", addr)
	return s.http.Run(addr)
}

// Health check handler
func (s *Server) handleHealth(c *gin.Context) {
	c.JSON(http.StatusOK, HealthResponse{
		Status:    "ok",
		Version:   "1.0.0",
		Timestamp: time.Now().Unix(),
		Services: map[string]string{
			"vector_store": s.cfg.VectorStoreType,
			"llm":          s.cfg.OpenAIModel,
		},
	})
}

func (s *Server) handleConfig(c *gin.Context) {
	c.JSON(http.StatusOK, ConfigResponse{})
}

// Notebook handlers

func (s *Server) handleListNotebooks(c *gin.Context) {
	ctx := context.Background()
	userID := c.GetString("user_id")
	
	notebooks, err := s.store.ListNotebooks(ctx, userID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "Failed to list notebooks"})
		return
	}
	c.JSON(http.StatusOK, notebooks)
}

func (s *Server) handleListNotebooksWithStats(c *gin.Context) {
	ctx := context.Background()
	userID := c.GetString("user_id")

	// If no user ID (anonymous or invalid token), return empty list
	if userID == "" {
		c.JSON(http.StatusOK, []NotebookWithStats{})
		return
	}

	notebooks, err := s.store.ListNotebooksWithStats(ctx, userID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "Failed to list notebooks with stats"})
		return
	}
	c.JSON(http.StatusOK, notebooks)
}

func (s *Server) handleCreateNotebook(c *gin.Context) {
	ctx := context.Background()
	userID := c.GetString("user_id")

	var req struct {
		Name        string                 `json:"name" binding:"required"`
		Description string                 `json:"description"`
		Metadata    map[string]interface{} `json:"metadata"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: err.Error()})
		return
	}

	notebook, err := s.store.CreateNotebook(ctx, userID, req.Name, req.Description, req.Metadata)
	if err != nil {
		golog.Errorf("error creating notebook: %v", err)
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: fmt.Sprintf("Failed to create notebook: %v", err)})
		return
	}

	// Log notebook creation activity
	activityLog := &ActivityLog{
		UserID:       userID,
		Action:       "create_notebook",
		ResourceType: "notebook",
		ResourceID:   notebook.ID,
		ResourceName: notebook.Name,
		IPAddress:    c.ClientIP(),
		UserAgent:    c.GetHeader("User-Agent"),
	}
	if err := s.store.LogActivity(ctx, activityLog); err != nil {
		golog.Errorf("failed to log notebook creation activity: %v", err)
	}

	c.JSON(http.StatusCreated, notebook)
}

func (s *Server) handleGetNotebook(c *gin.Context) {
	ctx := context.Background()
	id := c.Param("id")
	userID := c.GetString("user_id")

	notebook, err := s.store.GetNotebook(ctx, id)
	if err != nil {
		c.JSON(http.StatusNotFound, ErrorResponse{Error: "Notebook not found"})
		return
	}
	
	// Check ownership
	if notebook.UserID != "" && notebook.UserID != userID {
		c.JSON(http.StatusForbidden, ErrorResponse{Error: "Access denied"})
		return
	}

	c.JSON(http.StatusOK, notebook)
}

func (s *Server) handleUpdateNotebook(c *gin.Context) {
	ctx := context.Background()
	id := c.Param("id")
	userID := c.GetString("user_id")

	// Check ownership first
	existing, err := s.store.GetNotebook(ctx, id)
	if err != nil {
		c.JSON(http.StatusNotFound, ErrorResponse{Error: "Notebook not found"})
		return
	}
	if existing.UserID != "" && existing.UserID != userID {
		c.JSON(http.StatusForbidden, ErrorResponse{Error: "Access denied"})
		return
	}

	var req struct {
		Name        string                 `json:"name"`
		Description string                 `json:"description"`
		Metadata    map[string]interface{} `json:"metadata"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: err.Error()})
		return
	}

	notebook, err := s.store.UpdateNotebook(ctx, id, req.Name, req.Description, req.Metadata)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "Failed to update notebook"})
		return
	}

	c.JSON(http.StatusOK, notebook)
}

func (s *Server) handleDeleteNotebook(c *gin.Context) {
	ctx := context.Background()
	id := c.Param("id")
	userID := c.GetString("user_id")

	// Check ownership first
	existing, err := s.store.GetNotebook(ctx, id)
	if err != nil {
		c.JSON(http.StatusNotFound, ErrorResponse{Error: "Notebook not found"})
		return
	}
	if existing.UserID != "" && existing.UserID != userID {
		c.JSON(http.StatusForbidden, ErrorResponse{Error: "Access denied"})
		return
	}

	if err := s.store.DeleteNotebook(ctx, id); err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "Failed to delete notebook"})
		return
	}

	c.Status(http.StatusNoContent)
}

// Source handlers

func (s *Server) handleListSources(c *gin.Context) {
	ctx := context.Background()
	notebookID := c.Param("id")
	userID := c.GetString("user_id")

	if err := s.checkNotebookAccess(ctx, notebookID, userID); err != nil {
		c.JSON(http.StatusForbidden, ErrorResponse{Error: err.Error()})
		return
	}

	sources, err := s.store.ListSources(ctx, notebookID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "Failed to list sources"})
		return
	}

	c.JSON(http.StatusOK, sources)
}

func (s *Server) handleAddSource(c *gin.Context) {
	ctx := context.Background()
	notebookID := c.Param("id")
	userID := c.GetString("user_id")

	if err := s.checkNotebookAccess(ctx, notebookID, userID); err != nil {
		c.JSON(http.StatusForbidden, ErrorResponse{Error: err.Error()})
		return
	}

	var req struct {
		Name     string                 `json:"name" binding:"required"`
		Type     string                 `json:"type" binding:"required"`
		URL      string                 `json:"url"`
		Content  string                 `json:"content"`
		Metadata map[string]interface{} `json:"metadata"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: err.Error()})
		return
	}

	source := &Source{
		NotebookID: notebookID,
		Name:       req.Name,
		Type:       req.Type,
		URL:        req.URL,
		Content:    req.Content,
		Metadata:   req.Metadata,
	}

	// If URL is provided and Content is empty, fetch content from URL
	if req.URL != "" {
		golog.Infof("fetching content from URL: %s", req.URL)
		content, err := s.vectorStore.ExtractFromURL(ctx, req.URL)
		if err != nil {
			golog.Errorf("failed to fetch URL content: %v", err)
			c.JSON(http.StatusInternalServerError, ErrorResponse{Error: fmt.Sprintf("Failed to fetch URL content: %v", err)})
			return
		}
		source.Content = content
		golog.Infof("URL content fetched successfully, size: %d bytes", len(content))
	}

	if err := s.store.CreateSource(ctx, source); err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "Failed to create source"})
		return
	}

	// Log source import activity
	activityLog := &ActivityLog{
		UserID:       userID,
		Action:       "add_source",
		ResourceType: "source",
		ResourceID:   source.ID,
		ResourceName: source.Name,
		Details:      fmt.Sprintf(`{"notebook_id": "%s", "source_type": "%s", "source_url": "%s"}`, notebookID, source.Type, source.URL),
		IPAddress:    c.ClientIP(),
		UserAgent:    c.GetHeader("User-Agent"),
	}
	if err := s.store.LogActivity(ctx, activityLog); err != nil {
		golog.Errorf("failed to log source import activity: %v", err)
	}

	// Ingest into vector store (synchronous for immediate availability)
	if source.Content != "" {
		if chunkCount, err := s.vectorStore.IngestText(ctx, notebookID, source.Name, source.Content); err != nil {
			golog.Errorf("failed to ingest text: %v", err)
		} else {
			s.store.UpdateSourceChunkCount(ctx, source.ID, chunkCount)
		}
	}

	c.JSON(http.StatusCreated, source)
}

func (s *Server) handleDeleteSource(c *gin.Context) {
	ctx := context.Background()
	sourceID := c.Param("sourceId")
	userID := c.GetString("user_id")

	// Need to check notebook ownership. First get source to get notebookID
	source, err := s.store.GetSource(ctx, sourceID)
	if err != nil {
		c.JSON(http.StatusNotFound, ErrorResponse{Error: "Source not found"})
		return
	}
	
	if err := s.checkNotebookAccess(ctx, source.NotebookID, userID); err != nil {
		c.JSON(http.StatusForbidden, ErrorResponse{Error: err.Error()})
		return
	}

	if err := s.store.DeleteSource(ctx, sourceID); err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "Failed to delete source"})
		return
	}

	c.Status(http.StatusNoContent)
}

func (s *Server) checkNotebookAccess(ctx context.Context, notebookID, userID string) error {
	notebook, err := s.store.GetNotebook(ctx, notebookID)
	if err != nil {
		return fmt.Errorf("notebook not found")
	}
	if notebook.UserID != "" && notebook.UserID != userID {
		return fmt.Errorf("access denied")
	}
	return nil
}

func (s *Server) handleUpload(c *gin.Context) {
	ctx := context.Background()
	userID := c.GetString("user_id")
	notebookID := c.PostForm("notebook_id")

	if notebookID == "" {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "notebook_id required"})
		return
	}

	if err := s.checkNotebookAccess(ctx, notebookID, userID); err != nil {
		c.JSON(http.StatusForbidden, ErrorResponse{Error: err.Error()})
		return
	}

	file, err := c.FormFile("file")
	if err != nil {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "file required"})
		return
	}

	// Generate unique filename to avoid conflicts
	ext := filepath.Ext(file.Filename)
	baseName := file.Filename[:len(file.Filename)-len(ext)]
	uniqueFileName := fmt.Sprintf("%s_%s%s", baseName, uuid.New().String()[:8], ext)

	// Store in user-specific directory for isolation
	userUploadDir := fmt.Sprintf("./data/uploads/%s", userID)
	tempPath := fmt.Sprintf("%s/%s", userUploadDir, uniqueFileName)

	// Ensure user uploads directory exists
	if err := os.MkdirAll(userUploadDir, 0755); err != nil {
		golog.Errorf("failed to create user uploads directory: %v", err)
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "Failed to create uploads directory"})
		return
	}

	// Save file
	if err := c.SaveUploadedFile(file, tempPath); err != nil {
		golog.Errorf("failed to save file: %v", err)
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: fmt.Sprintf("Failed to save file: %v", err)})
		return
	}

	// Create source
	source := &Source{
		NotebookID: notebookID,
		Name:       file.Filename, // Keep original filename for display
		Type:       "file",
		FileName:   uniqueFileName, // Store unique filename
		FileSize:   file.Size,
		Metadata:   map[string]interface{}{"path": tempPath, "user_id": userID},
	}

	// Extract content
	content, err := s.vectorStore.ExtractDocument(ctx, tempPath)
	if err != nil {
		golog.Errorf("failed to extract document content: %v", err)
		// Clean up uploaded file on error
		os.Remove(tempPath)
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: fmt.Sprintf("Failed to extract document content: %v", err)})
		return
	}
	source.Content = content

	if err := s.store.CreateSource(ctx, source); err != nil {
		golog.Errorf("failed to create source: %v", err)
		// Clean up uploaded file on error
		os.Remove(tempPath)
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "Failed to create source"})
		return
	}

	// Log file upload activity
	activityLog := &ActivityLog{
		UserID:       userID,
		Action:       "upload_file",
		ResourceType: "source",
		ResourceID:   source.ID,
		ResourceName: file.Filename,
		Details:      fmt.Sprintf(`{"notebook_id": "%s", "file_size": %d, "file_type": "%s"}`, notebookID, file.Size, filepath.Ext(file.Filename)),
		IPAddress:    c.ClientIP(),
		UserAgent:    c.GetHeader("User-Agent"),
	}
	if err := s.store.LogActivity(ctx, activityLog); err != nil {
		golog.Errorf("failed to log file upload activity: %v", err)
	}

	// Ingest into vector store (synchronous for immediate availability)
	// Get chunk count from vector store stats
	stats, _ := s.vectorStore.GetStats(ctx)
	totalDocsBefore := stats.TotalDocuments

	if source.Content != "" {
		if _, err := s.vectorStore.IngestText(ctx, notebookID, source.Name, source.Content); err != nil {
			golog.Errorf("failed to ingest document: %v", err)
		} else {
			// Get updated stats to calculate chunk count
			stats, _ = s.vectorStore.GetStats(ctx)
			chunkCount := stats.TotalDocuments - totalDocsBefore

			// Update source with chunk count
			source.ChunkCount = chunkCount

			// Update in database
			s.store.UpdateSourceChunkCount(ctx, source.ID, chunkCount)
		}
	}

	c.JSON(http.StatusCreated, source)
}

// Note handlers

func (s *Server) handleListNotes(c *gin.Context) {
	ctx := context.Background()
	notebookID := c.Param("id")

	notes, err := s.store.ListNotes(ctx, notebookID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "Failed to list notes"})
		return
	}

	c.JSON(http.StatusOK, notes)
}

func (s *Server) handleCreateNote(c *gin.Context) {
	ctx := context.Background()
	notebookID := c.Param("id")

	var req struct {
		Title     string   `json:"title" binding:"required"`
		Content   string   `json:"content" binding:"required"`
		Type      string   `json:"type" binding:"required"`
		SourceIDs []string `json:"source_ids"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: err.Error()})
		return
	}

	note := &Note{
		NotebookID: notebookID,
		Title:      req.Title,
		Content:    req.Content,
		Type:       req.Type,
		SourceIDs:  req.SourceIDs,
	}

	if err := s.store.CreateNote(ctx, note); err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "Failed to create note"})
		return
	}

	// Log note creation activity
	activityLog := &ActivityLog{
		UserID:       c.GetString("user_id"),
		Action:       "create_note",
		ResourceType: "note",
		ResourceID:   note.ID,
		ResourceName: note.Title,
		Details:      fmt.Sprintf(`{"notebook_id": "%s", "note_type": "%s", "source_count": %d}`, notebookID, note.Type, len(note.SourceIDs)),
		IPAddress:    c.ClientIP(),
		UserAgent:    c.GetHeader("User-Agent"),
	}
	if err := s.store.LogActivity(ctx, activityLog); err != nil {
		golog.Errorf("failed to log note creation activity: %v", err)
	}

	c.JSON(http.StatusCreated, note)
}

func (s *Server) handleDeleteNote(c *gin.Context) {
	ctx := context.Background()
	noteID := c.Param("noteId")

	if err := s.store.DeleteNote(ctx, noteID); err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "Failed to delete note"})
		return
	}

	c.Status(http.StatusNoContent)
}

// Transformation handlers

func (s *Server) handleTransform(c *gin.Context) {
	ctx := context.Background()
	notebookID := c.Param("id")
	userID := c.GetString("user_id")

	// 按需加载向量索引
	if err := s.loadNotebookVectorIndex(ctx, notebookID); err != nil {
		golog.Errorf("failed to load vector index: %v", err)
	}

	var req TransformationRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: err.Error()})
		return
	}

	// Check if multiple notes of same type are allowed
	if !s.cfg.AllowMultipleNotesOfSameType {
		existingNotes, err := s.store.ListNotes(ctx, notebookID)
		if err != nil {
			c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "Failed to check existing notes"})
			return
		}
		for _, note := range existingNotes {
			if note.Type == req.Type {
				c.JSON(http.StatusConflict, ErrorResponse{Error: "该笔记本已存在相同类型的笔记，不允许创建重复类型"})
				return
			}
		}
	}

	// Get sources
	sources, err := s.store.ListSources(ctx, notebookID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "Failed to get sources"})
		return
	}

	if len(req.SourceIDs) > 0 {
		// Filter by specified source IDs
		filtered := make([]Source, 0)
		sourceMap := make(map[string]bool)
		for _, id := range req.SourceIDs {
			sourceMap[id] = true
		}
		for _, src := range sources {
			if sourceMap[src.ID] {
				filtered = append(filtered, src)
			}
		}
		sources = filtered
	} else {
		// If no source IDs specified, use all and populate the list for the note
		req.SourceIDs = make([]string, len(sources))
		for i, src := range sources {
			req.SourceIDs[i] = src.ID
		}
	}

	if len(sources) == 0 {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "No sources available"})
		return
	}

	// Generate transformation
	response, err := s.agent.GenerateTransformation(ctx, &req, sources)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: fmt.Sprintf("Generation failed: %v", err)})
		return
	}

	metadata := map[string]interface{}{
		"length": req.Length,
		"format": req.Format,
	}

	// If type is infograph, generate the image as well
	if req.Type == "infograph" {
		extra := "**注意：无论来源是什么语言，请务必使用中文**"
		prompt := response.Content + "\n\n" + extra
		imageModel := s.getImageModelForProvider()
		imagePath, err := s.agent.provider.GenerateImage(ctx, imageModel, prompt, userID)
		if err != nil {
			golog.Errorf("failed to generate infographic image: %v", err)
			metadata["image_error"] = err.Error()
		} else {
			// Convert local path to web path (authenticated API)
			webPath := "/api/files/" + filepath.Base(imagePath)
			metadata["image_url"] = webPath
		}
	}

	// If type is ppt, generate images for each slide
	if req.Type == "ppt" {
		slides := s.agent.ParsePPTSlides(response.Content)
		if len(slides) > 10 {
			golog.Errorf("ppt contains too many slides (%d), maximum allowed is 20. skipping image generation.", len(slides))
			metadata["image_error"] = "PPT页数超过20页上限，已停止生成图片"
		} else {
			var slideURLs []string
			golog.Infof("generating %d slides for ppt...", len(slides))

			for i, slide := range slides {
				golog.Infof("generating image for slide %d/%d...", i+1, len(slides))
				// Combine style and slide content for the image generator
				prompt := fmt.Sprintf("Style: %s\n\nSlide Content: %s", slides[0].Style, slide.Content)
				prompt += "\n\n**注意：无论来源是什么语言，请务必使用中文**\n"
				imageModel := s.getImageModelForProvider()
				imagePath, err := s.agent.provider.GenerateImage(ctx, imageModel, prompt, userID)
				if err != nil {
					golog.Errorf("failed to generate slide %d: %v", i+1, err)
					continue
				}
				slideURLs = append(slideURLs, "/api/files/"+filepath.Base(imagePath))
			}
			metadata["slides"] = slideURLs
		}
	}

	// Save as note
	// For infograph type: clear content only when image generation succeeds
	// If image generation fails, keep the prompt as content so user can see/retry it
	noteContent := response.Content
	if req.Type == "infograph" {
		// Check if image generation succeeded
		if metadata["image_url"] != nil {
			noteContent = "" // Clear content when image was generated successfully
		}
		// If image generation failed, noteContent remains as response.Content (the prompt)
	}

	note := &Note{
		NotebookID: notebookID,
		Title:      getTitleForType(req.Type),
		Content:    noteContent,
		Type:       req.Type,
		SourceIDs:  req.SourceIDs,
		Metadata:   metadata,
	}

	if err := s.store.CreateNote(ctx, note); err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "Failed to save note"})
		return
	}

	// Log transformation activity
	activityLog := &ActivityLog{
		UserID:       userID,
		Action:       "transform",
		ResourceType: "note",
		ResourceID:   note.ID,
		ResourceName: note.Title,
		Details:      fmt.Sprintf(`{"notebook_id": "%s", "transform_type": "%s", "length": "%s", "format": "%s", "source_count": %d}`, notebookID, req.Type, req.Length, req.Format, len(req.SourceIDs)),
		IPAddress:    c.ClientIP(),
		UserAgent:    c.GetHeader("User-Agent"),
	}
	if err := s.store.LogActivity(ctx, activityLog); err != nil {
		golog.Errorf("failed to log transformation activity: %v", err)
	}

	// If type is insight, inject the insight report as a new source
	if req.Type == "insight" {
		insightSource := &Source{
			NotebookID: notebookID,
			Name:       "洞察报告",
			Type:       "insight",
			Content:    response.Content,
			Metadata: map[string]interface{}{
				"generated_at": time.Now(),
				"source_ids":   req.SourceIDs,
			},
		}

		if err := s.store.CreateSource(ctx, insightSource); err != nil {
			golog.Errorf("failed to create insight source: %v", err)
		} else {
			// Ingest into vector store for future reference
			if chunkCount, err := s.vectorStore.IngestText(ctx, notebookID, insightSource.Name, insightSource.Content); err != nil {
				golog.Errorf("failed to ingest insight text: %v", err)
			} else {
				s.store.UpdateSourceChunkCount(ctx, insightSource.ID, chunkCount)
			}
		}
	}

	c.JSON(http.StatusOK, note)
}

func getTitleForType(t string) string {
	titles := map[string]string{
		"summary":     "摘要",
		"faq":         "常见问题解答",
		"study_guide": "学习指南",
		"outline":     "大纲",
		"podcast":     "播客脚本",
		"timeline":    "时间线",
		"glossary":    "术语表",
		"quiz":        "测验",
		"infograph":   "信息图",
		"ppt":         "幻灯片",
		"mindmap":     "思维导图",
		"insight":     "洞察报告",
	}
	if title, ok := titles[t]; ok {
		return title
	}
	return "笔记"
}

// Chat handlers

func (s *Server) handleListChatSessions(c *gin.Context) {
	ctx := context.Background()
	notebookID := c.Param("id")

	sessions, err := s.store.ListChatSessions(ctx, notebookID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "Failed to list chat sessions"})
		return
	}

	c.JSON(http.StatusOK, sessions)
}

func (s *Server) handleCreateChatSession(c *gin.Context) {
	ctx := context.Background()
	notebookID := c.Param("id")

	var req struct {
		Title string `json:"title"`
	}

	c.ShouldBindJSON(&req)

	session, err := s.store.CreateChatSession(ctx, notebookID, req.Title)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "Failed to create chat session"})
		return
	}

	c.JSON(http.StatusCreated, session)
}

func (s *Server) handleDeleteChatSession(c *gin.Context) {
	ctx := context.Background()
	sessionID := c.Param("sessionId")

	if err := s.store.DeleteChatSession(ctx, sessionID); err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "Failed to delete chat session"})
		return
	}

	c.Status(http.StatusNoContent)
}

func (s *Server) handleSendMessage(c *gin.Context) {
	ctx := context.Background()
	notebookID := c.Param("id")
	sessionID := c.Param("sessionId")

	// 按需加载向量索引
	if err := s.loadNotebookVectorIndex(ctx, notebookID); err != nil {
		golog.Errorf("failed to load vector index: %v", err)
	}

	var req ChatRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: err.Error()})
		return
	}

	// Add user message
	_, err := s.store.AddChatMessage(ctx, sessionID, "user", req.Message, nil)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "Failed to add message"})
		return
	}

	// Get session history
	session, err := s.store.GetChatSession(ctx, sessionID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "Failed to get session"})
		return
	}

	// Generate response
	response, err := s.agent.Chat(ctx, notebookID, req.Message, session.Messages)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: fmt.Sprintf("Chat failed: %v", err)})
		return
	}

	// Add assistant message
	sourceIDs := make([]string, len(response.Sources))
	for i, src := range response.Sources {
		sourceIDs[i] = src.ID
	}
	_, err = s.store.AddChatMessage(ctx, sessionID, "assistant", response.Message, sourceIDs)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "Failed to save response"})
		return
	}

	c.JSON(http.StatusOK, response)
}

func (s *Server) handleChat(c *gin.Context) {
	ctx := context.Background()
	notebookID := c.Param("id")

	// 按需加载向量索引
	if err := s.loadNotebookVectorIndex(ctx, notebookID); err != nil {
		golog.Errorf("failed to load vector index: %v", err)
	}

	var req ChatRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: err.Error()})
		return
	}

	// Create or get session
	sessionID := req.SessionID
	if sessionID == "" {
		session, err := s.store.CreateChatSession(ctx, notebookID, "")
		if err != nil {
			c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "Failed to create session"})
			return
		}
		sessionID = session.ID
	}

	// Get session history
	session, err := s.store.GetChatSession(ctx, sessionID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "Failed to get session"})
		return
	}

	// Generate response
	response, err := s.agent.Chat(ctx, notebookID, req.Message, session.Messages)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: fmt.Sprintf("Chat failed: %v", err)})
		return
	}

	response.SessionID = sessionID

	// Add messages
	sourceIDs := make([]string, len(response.Sources))
	for i, src := range response.Sources {
		sourceIDs[i] = src.ID
	}
	s.store.AddChatMessage(ctx, sessionID, "user", req.Message, nil)
	s.store.AddChatMessage(ctx, sessionID, "assistant", response.Message, sourceIDs)

	c.JSON(http.StatusOK, response)
}

// Utility functions

// handleServeFile serves uploaded files with proper access control
// Rules:
// 1. If notebook is public -> allow access
// 2. If notebook is private -> require authentication and ownership
//
// Files can come from two sources:
// 1. Uploaded files (stored in sources table)
// 2. Generated files (infographics, PPT slides) - stored in note metadata
func (s *Server) handleServeFile(c *gin.Context) {
	golog.Info("===== handleServeFile called =====")
	ctx := context.Background()
	filename := c.Param("filename")
	userID := c.GetString("user_id")

	golog.Infof("Request for file: %s, userID: %s", filename, userID)

	if filename == "" {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "filename required"})
		return
	}

	var ownerUserID string
	var isPublic bool
	var notebookID string

	// Try to find the file in sources table first (uploaded files)
	golog.Infof("Trying to find file %s in sources table", filename)
	source, notebook, err := s.store.GetSourceByFileName(ctx, filename)
	if err == nil && source != nil && notebook != nil {
		// File is from a source upload
		golog.Infof("File found in sources table, source_id: %s, notebook_id: %s", source.ID, notebook.ID)
		ownerUserID = notebook.UserID
		isPublic = notebook.IsPublic
		notebookID = notebook.ID
	} else {
		golog.Infof("File not in sources table (err: %v), trying notes table", err)
		// File not in sources table - try notes table (generated files like infographics)
		note, nb, err := s.store.GetNoteByFileName(ctx, filename)
		if err == nil && note != nil && nb != nil {
			golog.Infof("File found in notes table, note_id: %s, notebook_id: %s, is_public: %v", note.ID, nb.ID, nb.IsPublic)
			ownerUserID = nb.UserID
			isPublic = nb.IsPublic
			notebookID = nb.ID
		} else {
			// File not found in either table
			golog.Errorf("File not found in either table (notes err: %v)", err)
			c.JSON(http.StatusNotFound, ErrorResponse{Error: "File not found"})
			return
		}
	}

	golog.Infof("File owner: %s, isPublic: %v, notebookID: %s", ownerUserID, isPublic, notebookID)

	// Access control logic
	if isPublic {
		// Public notebook - allow access
		golog.Debugf("Serving public file: %s from notebook: %s", filename, notebookID)
	} else {
		// Private notebook - require authentication and ownership
		if userID == "" {
			c.JSON(http.StatusUnauthorized, ErrorResponse{Error: "Authorization required"})
			return
		}
		if userID != ownerUserID {
			golog.Warnf("Unauthorized access attempt by user %s to file %s owned by %s", userID, filename, ownerUserID)
			c.JSON(http.StatusForbidden, ErrorResponse{Error: "Access denied"})
			return
		}
	}

	// Build file path using the owner's user ID
	filePath := filepath.Join("./data/uploads", ownerUserID, filename)

	golog.Infof("Trying to load file: %s (owner: %s, public: %v)", filePath, ownerUserID, isPublic)

	// Security check
	absPath, err := filepath.Abs(filePath)
	if err != nil {
		golog.Errorf("Failed to get absolute path for %s: %v", filePath, err)
		c.JSON(http.StatusNotFound, ErrorResponse{Error: "File not found"})
		return
	}

	golog.Infof("Absolute path: %s", absPath)

	// Verify the path is within the uploads directory
	absUploadDir, _ := filepath.Abs("./data/uploads")
	if !strings.HasPrefix(absPath, absUploadDir) {
		golog.Warnf("Attempted directory traversal for file: %s", filename)
		c.JSON(http.StatusForbidden, ErrorResponse{Error: "Access denied"})
		return
	}

	// Check if file exists
	if _, err := os.Stat(absPath); os.IsNotExist(err) {
		golog.Errorf("File not found: %s", absPath)
		c.JSON(http.StatusNotFound, ErrorResponse{Error: "File not found"})
		return
	}

	golog.Infof("File found and serving: %s", absPath)

	// Determine content type
	ext := filepath.Ext(filename)
	contentType := "application/octet-stream"
	switch ext {
	case ".jpg", ".jpeg":
		contentType = "image/jpeg"
	case ".png":
		contentType = "image/png"
	case ".gif":
		contentType = "image/gif"
	case ".webp":
		contentType = "image/webp"
	case ".svg":
		contentType = "image/svg+xml"
	case ".pdf":
		contentType = "application/pdf"
	}

	c.Header("Content-Type", contentType)
	// Cache public files for 1 hour, private files for no-cache
	if isPublic {
		c.Header("Cache-Control", "public, max-age=3600")
	} else {
		c.Header("Cache-Control", "no-cache")
	}
	c.File(absPath)

	golog.Infof("File served: %s (notebook: %s, public: %v, user: %s)",
		filename, notebookID, isPublic, userID)
}

func writeFile(path, content string) error {
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}
	return os.WriteFile(path, []byte(content), 0644)
}

func removeFile(path string) error {
	return os.Remove(path)
}

// getImageModelForProvider returns the image model based on configured provider
func (s *Server) getImageModelForProvider() string {
	switch s.cfg.ImageProvider {
	case "glm":
		return s.cfg.GLMImageModel
	case "zimage":
		return s.cfg.ZImageModel
	case "gemini":
		return s.cfg.GeminiImageModel
	default:
		// Default to Gemini if provider is unknown
		return s.cfg.GeminiImageModel
	}
}

// Public sharing handlers

// handleSetNotebookPublic sets the notebook's public status
func (s *Server) handleSetNotebookPublic(c *gin.Context) {
	ctx := context.Background()
	id := c.Param("id")
	userID := c.GetString("user_id")

	// Check ownership first
	existing, err := s.store.GetNotebook(ctx, id)
	if err != nil {
		c.JSON(http.StatusNotFound, ErrorResponse{Error: "Notebook not found"})
		return
	}
	if existing.UserID != "" && existing.UserID != userID {
		c.JSON(http.StatusForbidden, ErrorResponse{Error: "Access denied"})
		return
	}

	var req struct {
		IsPublic bool `json:"is_public"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: err.Error()})
		return
	}

	notebook, err := s.store.SetNotebookPublic(ctx, id, req.IsPublic)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "Failed to update notebook"})
		return
	}

	// Log activity
	action := "make_public"
	if !req.IsPublic {
		action = "make_private"
	}
	activityLog := &ActivityLog{
		UserID:       userID,
		Action:       action,
		ResourceType: "notebook",
		ResourceID:   notebook.ID,
		ResourceName: notebook.Name,
		IPAddress:    c.ClientIP(),
		UserAgent:    c.GetHeader("User-Agent"),
	}
	if err := s.store.LogActivity(ctx, activityLog); err != nil {
		golog.Errorf("failed to log activity: %v", err)
	}

	c.JSON(http.StatusOK, notebook)
}

// handleGetPublicNotebook retrieves a public notebook by its token
func (s *Server) handleGetPublicNotebook(c *gin.Context) {
	ctx := context.Background()
	token := c.Param("token")

	notebook, err := s.store.GetNotebookByPublicToken(ctx, token)
	if err != nil {
		c.JSON(http.StatusNotFound, ErrorResponse{Error: "Public notebook not found"})
		return
	}

	c.JSON(http.StatusOK, notebook)
}

// handleListPublicSources lists sources for a public notebook
func (s *Server) handleListPublicSources(c *gin.Context) {
	ctx := context.Background()
	token := c.Param("token")

	// First verify the notebook is public
	notebook, err := s.store.GetNotebookByPublicToken(ctx, token)
	if err != nil {
		c.JSON(http.StatusNotFound, ErrorResponse{Error: "Public notebook not found"})
		return
	}

	sources, err := s.store.ListSources(ctx, notebook.ID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "Failed to list sources"})
		return
	}

	c.JSON(http.StatusOK, sources)
}

// handleListPublicNotes lists notes for a public notebook
func (s *Server) handleListPublicNotes(c *gin.Context) {
	ctx := context.Background()
	token := c.Param("token")

	// First verify the notebook is public
	notebook, err := s.store.GetNotebookByPublicToken(ctx, token)
	if err != nil {
		c.JSON(http.StatusNotFound, ErrorResponse{Error: "Public notebook not found"})
		return
	}

	notes, err := s.store.ListNotes(ctx, notebook.ID)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "Failed to list notes"})
		return
	}

	c.JSON(http.StatusOK, notes)
}

// handleListPublicNotebooks lists all public notebooks with infograph or ppt notes
func (s *Server) handleListPublicNotebooks(c *gin.Context) {
	ctx := context.Background()

	notebooks, err := s.store.ListPublicNotebooks(ctx)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "Failed to list public notebooks"})
		return
	}

	c.JSON(http.StatusOK, notebooks)
}

// handleServePublicFile serves files for public notebooks
