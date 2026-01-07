package backend

import (
	"context"
	"embed"
	"encoding/json"
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
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/llms/googleai"
	"github.com/tmc/langchaingo/llms/ollama"
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
	// Track which notebooks have been loaded into vector store
	loadedNotebooks map[string]bool
	vectorMutex     sync.RWMutex
}

// NewServer creates a new server
func NewServer(cfg Config) (*Server, error) {
	// Initialize Embedder based on provider
	var vsOpts []VectorStoreOption
	var embedder embeddings.Embedder
	var errEmbed error

	if cfg.EmbeddingModel != "" {
		ctx := context.Background()
		switch strings.ToLower(cfg.EmbeddingProvider) {
		case "google":
			if cfg.GoogleAPIKey != "" {
				golog.Infof("Initializing Gemini Embedder with model: %s", cfg.EmbeddingModel)
				llm, err := googleai.New(ctx, googleai.WithAPIKey(cfg.GoogleAPIKey), googleai.WithDefaultModel(cfg.EmbeddingModel))
				if err != nil {
					golog.Errorf("Failed to create GoogleAI client for embeddings: %v", err)
				} else {
					embedder, errEmbed = embeddings.NewEmbedder(llm)
				}
			} else {
				golog.Warnf("Embedding provider is 'google' but GOOGLE_API_KEY is unset")
			}
		case "ollama":
			golog.Infof("Initializing Ollama Embedder with model: %s (BaseURL: %s)", cfg.EmbeddingModel, cfg.OllamaBaseURL)
			llm, err := ollama.New(ollama.WithModel(cfg.EmbeddingModel), ollama.WithServerURL(cfg.OllamaBaseURL))
			if err != nil {
				golog.Errorf("Failed to create Ollama client for embeddings: %v", err)
			} else {
				embedder, errEmbed = embeddings.NewEmbedder(llm)
			}
		default:
			golog.Warnf("Unknown embedding provider: %s", cfg.EmbeddingProvider)
		}

		if errEmbed != nil {
			golog.Errorf("Failed to create embedder: %v", errEmbed)
		} else if embedder != nil {
			vsOpts = append(vsOpts, WithEmbedder(embedder))
		}
	}

	// Initialize vector store
	vectorStore, err := NewVectorStore(cfg, vsOpts...)
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

	// Serve uploaded files (with audit)
	uploads := s.http.Group("/uploads")
	uploads.Use(AuditMiddlewareLite())
	uploads.Static("/", "./data/uploads")

	// Serve index.html at root (with audit)
	s.http.GET("/", AuditMiddlewareLite(), func(c *gin.Context) {
		c.Header("Cache-Control", "no-cache")
		content, _ := frontendFS.ReadFile("frontend/index.html")
		c.Data(http.StatusOK, "text/html; charset=utf-8", content)
	})

	// API routes
	api := s.http.Group("/api")
	api.Use(AuditMiddlewareLite()) // Only audit API routes, not static resources
	{
		// Health check
		api.GET("/health", s.handleHealth)
		api.GET("/config", s.handleConfig)
		api.POST("/config", s.handleUpdateConfig)
		api.GET("/models", s.handleListModels)

		// Notebook routes
		notebooks := api.Group("/notebooks")
		{
			notebooks.GET("", s.handleListNotebooks)
			notebooks.GET("/stats", s.handleListNotebooksWithStats)
			notebooks.POST("", s.handleCreateNotebook)
			notebooks.GET("/:id", s.handleGetNotebook)
			notebooks.PUT("/:id", s.handleUpdateNotebook)
			notebooks.DELETE("/:id", s.handleDeleteNotebook)

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
			if _, err := s.vectorStore.IngestText(ctx, src.Name, src.Content); err != nil {
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
	c.JSON(http.StatusOK, ConfigResponse{
		AllowDelete:       s.cfg.AllowDelete,
		EmbeddingProvider: s.cfg.EmbeddingProvider,
		ImageModel:        s.cfg.ImageModel,
		ChatProvider:      s.cfg.ChatProvider,
		ChatModel:         s.cfg.ChatModel,
	})
}

func (s *Server) handleUpdateConfig(c *gin.Context) {
	var req struct {
		EmbeddingProvider string `json:"embedding_provider"`
		ImageModel        string `json:"image_model"`
		ChatProvider      string `json:"chat_provider"`
		ChatModel         string `json:"chat_model"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: err.Error()})
		return
	}



	if req.ImageModel != "" {
		s.cfg.ImageModel = req.ImageModel
		golog.Infof("Image model updated to: %s", s.cfg.ImageModel)
	}

	if req.ChatProvider != "" {
		s.cfg.ChatProvider = req.ChatProvider
		golog.Infof("Chat provider updated to: %s", s.cfg.ChatProvider)
	}

	if req.ChatModel != "" {
		s.cfg.ChatModel = req.ChatModel
		golog.Infof("Chat model updated to: %s", s.cfg.ChatModel)
	}

	if req.EmbeddingProvider != "" {
		s.cfg.EmbeddingProvider = req.EmbeddingProvider
		
		// Re-initialize embedder and vector store
		var vsOpts []VectorStoreOption
		var embedder embeddings.Embedder
		var errEmbed error
		
		ctx := context.Background()
		switch strings.ToLower(s.cfg.EmbeddingProvider) {
		case "google":
			if s.cfg.GoogleAPIKey != "" {
				golog.Infof("Switching to Gemini Embedder with model: %s", s.cfg.EmbeddingModel)
				llm, err := googleai.New(ctx, googleai.WithAPIKey(s.cfg.GoogleAPIKey), googleai.WithDefaultModel(s.cfg.EmbeddingModel))
				if err != nil {
					golog.Errorf("Failed to create GoogleAI client for embeddings: %v", err)
				} else {
					embedder, errEmbed = embeddings.NewEmbedder(llm)
				}
			} else {
				golog.Warnf("Embedding provider is 'google' but GOOGLE_API_KEY is unset")
			}
		case "ollama":
			golog.Infof("Switching to Ollama Embedder with model: %s (BaseURL: %s)", s.cfg.EmbeddingModel, s.cfg.OllamaBaseURL)
			llm, err := ollama.New(ollama.WithModel(s.cfg.EmbeddingModel), ollama.WithServerURL(s.cfg.OllamaBaseURL))
			if err != nil {
				golog.Errorf("Failed to create Ollama client for embeddings: %v", err)
			} else {
				embedder, errEmbed = embeddings.NewEmbedder(llm)
			}
		default:
			golog.Warnf("Unknown embedding provider: %s", s.cfg.EmbeddingProvider)
		}

		if errEmbed != nil {
			golog.Errorf("Failed to create embedder: %v", errEmbed)
		} else if embedder != nil {
			vsOpts = append(vsOpts, WithEmbedder(embedder))
		}

		vectorStore, err := NewVectorStore(s.cfg, vsOpts...)
		if err != nil {
			golog.Errorf("Failed to recreate vector store: %v", err)
			// Don't fail the request, just log error, maybe old store is still usable or it is broken now
		} else {
			s.vectorMutex.Lock()
			s.vectorStore = vectorStore
			// Clear loaded notebooks check so they re-index/load if needed (though current implementation loads all at startup? no, on demand)
			s.loadedNotebooks = make(map[string]bool)
			s.vectorMutex.Unlock()
			golog.Infof("Vector store re-initialized with provider: %s", s.cfg.EmbeddingProvider)
		}
	}

	c.JSON(http.StatusOK, gin.H{
		"status":             "ok",
		"embedding_provider": s.cfg.EmbeddingProvider,
		"image_model":        s.cfg.ImageModel,
		"chat_provider":      s.cfg.ChatProvider,
		"chat_model":         s.cfg.ChatModel,
	})
}

// Notebook handlers


func (s *Server) handleListModels(c *gin.Context) {
	provider := c.Query("provider")
	if provider == "" {
		provider = "openai"
	}

	type ModelInfo struct {
		ID string `json:"id"`
	}

	var models []string
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	client := &http.Client{Timeout: 5 * time.Second}

	if provider == "openai" {
		url := fmt.Sprintf("%s/v1/models", strings.TrimSuffix(s.cfg.GetBaseURL(), "/"))
		// If BaseURL is empty, use default OpenAI
		if s.cfg.OpenAIBaseURL == "" {
			url = "https://api.openai.com/v1/models"
		}

		req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
		if err == nil {
			if s.cfg.OpenAIAPIKey != "" {
				req.Header.Set("Authorization", "Bearer "+s.cfg.OpenAIAPIKey)
			}
			resp, err := client.Do(req)
			if err == nil {
				defer resp.Body.Close()
				var result struct {
					Data []ModelInfo `json:"data"`
				}
				if json.NewDecoder(resp.Body).Decode(&result) == nil {
					for _, m := range result.Data {
						models = append(models, m.ID)
					}
				}
			}
		}
	} else if provider == "ollama" {
		url := fmt.Sprintf("%s/api/tags", strings.TrimSuffix(s.cfg.OllamaBaseURL, "/"))
		req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
		if err == nil {
			resp, err := client.Do(req)
			if err == nil {
				defer resp.Body.Close()
				var result struct {
					Models []struct {
						Name string `json:"name"`
					} `json:"models"`
				}
				if json.NewDecoder(resp.Body).Decode(&result) == nil {
					for _, m := range result.Models {
						models = append(models, m.Name)
					}
				}
			}
		}
	}

	// Prioritize and deduplicate
	currentModel := s.cfg.ChatModel
	isCurrentProvider := s.cfg.ChatProvider == provider

	finalList := []ModelItem{}
	seen := make(map[string]bool)

	// Define default models for each provider
	defaultModels := []string{}
	if provider == "openai" {
		defaultModels = []string{"qwen3-max", "gpt-4o", "gpt-4o-mini", "o1-preview"}
	} else if provider == "ollama" {
		defaultModels = []string{"llama3.2", "qwen2.5-coder:7b", "mistral"}
	}

	// 1. Add env configured models with (Env Config) label
	envModelStr := ""
	if provider == "openai" {
		envModelStr = s.cfg.OpenAIModel
	} else if provider == "ollama" {
		envModelStr = s.cfg.OllamaModel
	}

	if envModelStr != "" {
		// Support comma-separated list
		parts := strings.Split(envModelStr, ",")
		for _, m := range parts {
			m = strings.TrimSpace(m)
			if m != "" && !seen[m] {
				finalList = append(finalList, ModelItem{
					ID:          m,
					DisplayName: fmt.Sprintf("%s (Env Config)", m),
				})
				seen[m] = true
			}
		}
	}

	// 2. Add current selected model if it's not the env model (don't add label)
	if isCurrentProvider && currentModel != "" && !seen[currentModel] {
		finalList = append(finalList, ModelItem{
			ID:          currentModel,
			DisplayName: currentModel,
		})
		seen[currentModel] = true
	}

	// 3. Add default preset models
	for _, m := range defaultModels {
		if !seen[m] {
			finalList = append(finalList, ModelItem{
				ID:          m,
				DisplayName: m,
			})
			seen[m] = true
		}
	}

	// 4. Add dynamic models from provider API
	for _, m := range models {
		if !seen[m] {
			finalList = append(finalList, ModelItem{
				ID:          m,
				DisplayName: m,
			})
			seen[m] = true
		}
	}

	c.JSON(http.StatusOK, gin.H{"models": finalList})
}

func (s *Server) handleListNotebooks(c *gin.Context) {
	ctx := context.Background()
	notebooks, err := s.store.ListNotebooks(ctx)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "Failed to list notebooks"})
		return
	}
	c.JSON(http.StatusOK, notebooks)
}

func (s *Server) handleListNotebooksWithStats(c *gin.Context) {
	ctx := context.Background()
	notebooks, err := s.store.ListNotebooksWithStats(ctx)
	if err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "Failed to list notebooks with stats"})
		return
	}
	c.JSON(http.StatusOK, notebooks)
}

func (s *Server) handleCreateNotebook(c *gin.Context) {
	ctx := context.Background()

	var req struct {
		Name        string                 `json:"name" binding:"required"`
		Description string                 `json:"description"`
		Metadata    map[string]interface{} `json:"metadata"`
	}

	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: err.Error()})
		return
	}

	notebook, err := s.store.CreateNotebook(ctx, req.Name, req.Description, req.Metadata)
	if err != nil {
		golog.Errorf("error creating notebook: %v", err)
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: fmt.Sprintf("Failed to create notebook: %v", err)})
		return
	}

	c.JSON(http.StatusCreated, notebook)
}

func (s *Server) handleGetNotebook(c *gin.Context) {
	ctx := context.Background()
	id := c.Param("id")

	notebook, err := s.store.GetNotebook(ctx, id)
	if err != nil {
		c.JSON(http.StatusNotFound, ErrorResponse{Error: "Notebook not found"})
		return
	}

	c.JSON(http.StatusOK, notebook)
}

func (s *Server) handleUpdateNotebook(c *gin.Context) {
	ctx := context.Background()
	id := c.Param("id")

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

	// Ingest into vector store (synchronous for immediate availability)
	if source.Content != "" {
		if chunkCount, err := s.vectorStore.IngestText(ctx, source.Name, source.Content); err != nil {
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

	if err := s.store.DeleteSource(ctx, sourceID); err != nil {
		c.JSON(http.StatusInternalServerError, ErrorResponse{Error: "Failed to delete source"})
		return
	}

	c.Status(http.StatusNoContent)
}

func (s *Server) handleUpload(c *gin.Context) {
	ctx := context.Background()
	notebookID := c.PostForm("notebook_id")
	if notebookID == "" {
		c.JSON(http.StatusBadRequest, ErrorResponse{Error: "notebook_id required"})
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
	tempPath := fmt.Sprintf("./data/uploads/%s", uniqueFileName)

	// Ensure uploads directory exists
	if err := os.MkdirAll("./data/uploads", 0755); err != nil {
		golog.Errorf("failed to create uploads directory: %v", err)
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
		Metadata:   map[string]interface{}{"path": tempPath},
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

	// Ingest into vector store (synchronous for immediate availability)
	// Get chunk count from vector store stats
	stats, _ := s.vectorStore.GetStats(ctx)
	totalDocsBefore := stats.TotalDocuments

	if source.Content != "" {
		if _, err := s.vectorStore.IngestText(ctx, source.Name, source.Content); err != nil {
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
		imagePath, err := s.agent.provider.GenerateImage(ctx, s.cfg.ImageModel, prompt)
		if err != nil {
			golog.Errorf("failed to generate infographic image: %v", err)
			metadata["image_error"] = err.Error()
		} else {
			// Convert local path to web path
			webPath := "/uploads/" + filepath.Base(imagePath)
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
				imagePath, err := s.agent.provider.GenerateImage(ctx, s.cfg.ImageModel, prompt)
				if err != nil {
					golog.Errorf("failed to generate slide %d: %v", i+1, err)
					continue
				}
				slideURLs = append(slideURLs, "/uploads/"+filepath.Base(imagePath))
			}
			metadata["slides"] = slideURLs
		}
	}

	// Save as note
	// For infograph type, don't save text content (only show the image)
	noteContent := response.Content
	if req.Type == "infograph" {
		noteContent = "" // Clear content for infograph, only show image
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
			if chunkCount, err := s.vectorStore.IngestText(ctx, insightSource.Name, insightSource.Content); err != nil {
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
