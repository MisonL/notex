package backend

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/google/uuid"
	_ "modernc.org/sqlite"
)

// Store handles data persistence for notebooks, sources, notes, and chat sessions
type Store struct {
	db     *sql.DB
	dbPath string
}

// NewStore creates a new store
func NewStore(cfg Config) (*Store, error) {
	// Ensure data directory exists
	if err := os.MkdirAll(filepath.Dir(cfg.StorePath), 0755); err != nil {
		return nil, fmt.Errorf("failed to create data directory: %w", err)
	}

	absPath, _ := filepath.Abs(cfg.StorePath)
	fmt.Printf("📦 Initializing SQLite Store at: %s\n", absPath)

	db, err := sql.Open("sqlite", cfg.StorePath)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	// Enable foreign key constraints
	if _, err := db.Exec("PRAGMA foreign_keys = ON"); err != nil {
		return nil, fmt.Errorf("failed to enable foreign keys: %w", err)
	}

	store := &Store{db: db, dbPath: cfg.StorePath}

	// Initialize schema
	if err := store.initSchema(); err != nil {
		return nil, fmt.Errorf("failed to initialize schema: %w", err)
	}

	return store, nil
}

// initSchema creates the database schema
func (s *Store) initSchema() error {
	schema := `
	CREATE TABLE IF NOT EXISTS users (
		id TEXT PRIMARY KEY,
		email TEXT NOT NULL UNIQUE,
		name TEXT,
		avatar_url TEXT,
		provider TEXT,
		created_at INTEGER NOT NULL,
		updated_at INTEGER NOT NULL
	);

	CREATE TABLE IF NOT EXISTS notebooks (
		id TEXT PRIMARY KEY,
		user_id TEXT,
		name TEXT NOT NULL,
		description TEXT,
		created_at INTEGER NOT NULL,
		updated_at INTEGER NOT NULL,
		metadata TEXT,
		FOREIGN KEY (user_id) REFERENCES users(id)
	);
	`
	if _, err := s.db.Exec(schema); err != nil {
		return err
	}

	// Check if user_id column exists in notebooks table (migration)
	var count int
	err := s.db.QueryRow("SELECT COUNT(*) FROM pragma_table_info('notebooks') WHERE name='user_id'").Scan(&count)
	if err == nil && count == 0 {
		// Add user_id column
		if _, err := s.db.Exec("ALTER TABLE notebooks ADD COLUMN user_id TEXT REFERENCES users(id)"); err != nil {
			return fmt.Errorf("failed to add user_id column to notebooks: %w", err)
		}
	}

	restSchema := `
	CREATE TABLE IF NOT EXISTS sources (
		id TEXT PRIMARY KEY,
		notebook_id TEXT NOT NULL,
		name TEXT NOT NULL,
		type TEXT NOT NULL,
		url TEXT,
		content TEXT,
		file_name TEXT,
		file_size INTEGER,
		chunk_count INTEGER DEFAULT 0,
		created_at INTEGER NOT NULL,
		updated_at INTEGER NOT NULL,
		metadata TEXT,
		FOREIGN KEY (notebook_id) REFERENCES notebooks(id) ON DELETE CASCADE
	);

	CREATE TABLE IF NOT EXISTS notes (
		id TEXT PRIMARY KEY,
		notebook_id TEXT NOT NULL,
		title TEXT NOT NULL,
		content TEXT NOT NULL,
		type TEXT NOT NULL,
		source_ids TEXT,
		created_at INTEGER NOT NULL,
		updated_at INTEGER NOT NULL,
		metadata TEXT,
		FOREIGN KEY (notebook_id) REFERENCES notebooks(id) ON DELETE CASCADE
	);

	CREATE TABLE IF NOT EXISTS chat_sessions (
		id TEXT PRIMARY KEY,
		notebook_id TEXT NOT NULL,
		title TEXT NOT NULL,
		created_at INTEGER NOT NULL,
		updated_at INTEGER NOT NULL,
		metadata TEXT,
		FOREIGN KEY (notebook_id) REFERENCES notebooks(id) ON DELETE CASCADE
	);

	CREATE TABLE IF NOT EXISTS chat_messages (
		id TEXT PRIMARY KEY,
		session_id TEXT NOT NULL,
		role TEXT NOT NULL,
		content TEXT NOT NULL,
		sources TEXT,
		created_at INTEGER NOT NULL,
		metadata TEXT,
		FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
	);

	CREATE TABLE IF NOT EXISTS podcasts (
		id TEXT PRIMARY KEY,
		notebook_id TEXT NOT NULL,
		title TEXT NOT NULL,
		script TEXT,
		audio_url TEXT,
		duration INTEGER DEFAULT 0,
		voice TEXT NOT NULL,
		status TEXT NOT NULL,
		source_ids TEXT,
		created_at INTEGER NOT NULL,
		updated_at INTEGER NOT NULL,
		metadata TEXT,
		FOREIGN KEY (notebook_id) REFERENCES notebooks(id) ON DELETE CASCADE
	);

	CREATE INDEX IF NOT EXISTS idx_sources_notebook ON sources(notebook_id);
	CREATE INDEX IF NOT EXISTS idx_notes_notebook ON notes(notebook_id);
	CREATE INDEX IF NOT EXISTS idx_chat_sessions_notebook ON chat_sessions(notebook_id);
	CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(session_id);
	CREATE INDEX IF NOT EXISTS idx_podcasts_notebook ON podcasts(notebook_id);
	`

	_, err = s.db.Exec(restSchema)
	return err
}

// User operations

// CreateUser creates or updates a user
func (s *Store) CreateUser(ctx context.Context, user *User) error {
	now := time.Now()
	if user.CreatedAt.IsZero() {
		user.CreatedAt = now
	}
	user.UpdatedAt = now

	// Check if user exists
	existing, err := s.GetUserByEmail(ctx, user.Email)
	if err == nil && existing != nil {
		// Update existing user
		user.ID = existing.ID
		user.CreatedAt = existing.CreatedAt // Keep original created_at
		_, err := s.db.ExecContext(ctx, `
			UPDATE users 
			SET name = ?, avatar_url = ?, provider = ?, updated_at = ?
			WHERE id = ?
		`, user.Name, user.AvatarURL, user.Provider, now.Unix(), user.ID)
		return err
	}

	if user.ID == "" {
		user.ID = uuid.New().String()
	}

	_, err = s.db.ExecContext(ctx, `
		INSERT INTO users (id, email, name, avatar_url, provider, created_at, updated_at)
		VALUES (?, ?, ?, ?, ?, ?, ?)
	`, user.ID, user.Email, user.Name, user.AvatarURL, user.Provider, user.CreatedAt.Unix(), user.UpdatedAt.Unix())

	return err
}

// GetUser retrieves a user by ID
func (s *Store) GetUser(ctx context.Context, id string) (*User, error) {
	var user User
	var createdAt, updatedAt int64

	err := s.db.QueryRowContext(ctx, `
		SELECT id, email, name, avatar_url, provider, created_at, updated_at
		FROM users WHERE id = ?
	`, id).Scan(&user.ID, &user.Email, &user.Name, &user.AvatarURL, &user.Provider, &createdAt, &updatedAt)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("user not found")
	}
	if err != nil {
		return nil, err
	}

	user.CreatedAt = time.Unix(createdAt, 0)
	user.UpdatedAt = time.Unix(updatedAt, 0)

	return &user, nil
}

// GetUserByEmail retrieves a user by Email
func (s *Store) GetUserByEmail(ctx context.Context, email string) (*User, error) {
	var user User
	var createdAt, updatedAt int64

	err := s.db.QueryRowContext(ctx, `
		SELECT id, email, name, avatar_url, provider, created_at, updated_at
		FROM users WHERE email = ?
	`, email).Scan(&user.ID, &user.Email, &user.Name, &user.AvatarURL, &user.Provider, &createdAt, &updatedAt)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("user not found")
	}
	if err != nil {
		return nil, err
	}

	user.CreatedAt = time.Unix(createdAt, 0)
	user.UpdatedAt = time.Unix(updatedAt, 0)

	return &user, nil
}

// Notebook operations

// CreateNotebook creates a new notebook
func (s *Store) CreateNotebook(ctx context.Context, userID, name, description string, metadata map[string]interface{}) (*Notebook, error) {
	id := uuid.New().String()
	now := time.Now()

	metadataJSON, _ := json.Marshal(metadata)

	_, err := s.db.ExecContext(ctx, `
		INSERT INTO notebooks (id, user_id, name, description, created_at, updated_at, metadata)
		VALUES (?, ?, ?, ?, ?, ?, ?)
	`, id, userID, name, description, now.Unix(), now.Unix(), string(metadataJSON))
	if err != nil {
		return nil, err
	}

	return s.GetNotebook(ctx, id)
}

// GetNotebook retrieves a notebook by ID
func (s *Store) GetNotebook(ctx context.Context, id string) (*Notebook, error) {
	var nb Notebook
	var metadataJSON string
	var createdAt, updatedAt int64
	var userID sql.NullString

	err := s.db.QueryRowContext(ctx, `
		SELECT id, user_id, name, description, created_at, updated_at, metadata
		FROM notebooks WHERE id = ?
	`, id).Scan(&nb.ID, &userID, &nb.Name, &nb.Description, &createdAt, &updatedAt, &metadataJSON)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("notebook not found")
	}
	if err != nil {
		return nil, err
	}
	
	if userID.Valid {
		nb.UserID = userID.String
	}

	nb.CreatedAt = time.Unix(createdAt, 0)
	nb.UpdatedAt = time.Unix(updatedAt, 0)

	if metadataJSON != "" {
		json.Unmarshal([]byte(metadataJSON), &nb.Metadata)
	} else {
		nb.Metadata = make(map[string]interface{})
	}

	return &nb, nil
}

// ListNotebooks retrieves all notebooks for a user
func (s *Store) ListNotebooks(ctx context.Context, userID string) ([]Notebook, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, user_id, name, description, created_at, updated_at, metadata
		FROM notebooks 
		WHERE user_id = ?
		ORDER BY updated_at DESC
	`, userID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	notebooks := make([]Notebook, 0)
	for rows.Next() {
		var nb Notebook
		var metadataJSON string
		var createdAt, updatedAt int64
		var uid sql.NullString

		if err := rows.Scan(&nb.ID, &uid, &nb.Name, &nb.Description, &createdAt, &updatedAt, &metadataJSON); err != nil {
			return nil, err
		}
		
		if uid.Valid {
			nb.UserID = uid.String
		}

		nb.CreatedAt = time.Unix(createdAt, 0)
		nb.UpdatedAt = time.Unix(updatedAt, 0)

		if metadataJSON != "" {
			json.Unmarshal([]byte(metadataJSON), &nb.Metadata)
		} else {
			nb.Metadata = make(map[string]interface{})
		}

		notebooks = append(notebooks, nb)
	}

	return notebooks, nil
}

// UpdateNotebook updates a notebook
func (s *Store) UpdateNotebook(ctx context.Context, id string, name, description string, metadata map[string]interface{}) (*Notebook, error) {
	now := time.Now()

	metadataJSON, _ := json.Marshal(metadata)

	_, err := s.db.ExecContext(ctx, `
		UPDATE notebooks
		SET name = ?, description = ?, updated_at = ?, metadata = ?
		WHERE id = ?
	`, name, description, now.Unix(), string(metadataJSON), id)
	if err != nil {
		return nil, err
	}

	return s.GetNotebook(ctx, id)
}

// DeleteNotebook deletes a notebook and all its data
func (s *Store) DeleteNotebook(ctx context.Context, id string) error {
	_, err := s.db.ExecContext(ctx, `DELETE FROM notebooks WHERE id = ?`, id)
	return err
}

// ListNotebooksWithStats retrieves all notebooks with their source and note counts for a user
func (s *Store) ListNotebooksWithStats(ctx context.Context, userID string) ([]NotebookWithStats, error) {
	query := `
		SELECT
			n.id, n.user_id, n.name, n.description, n.created_at, n.updated_at, n.metadata,
			COALESCE((SELECT COUNT(*) FROM sources WHERE notebook_id = n.id), 0) as source_count,
			COALESCE((SELECT COUNT(*) FROM notes WHERE notebook_id = n.id), 0) as note_count
		FROM notebooks n
		WHERE n.user_id = ?
		ORDER BY n.updated_at DESC
	`

	rows, err := s.db.QueryContext(ctx, query, userID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	notebooks := make([]NotebookWithStats, 0)
	for rows.Next() {
		var nb NotebookWithStats
		var metadataJSON string
		var createdAt, updatedAt int64
		var uid sql.NullString

		if err := rows.Scan(&nb.ID, &uid, &nb.Name, &nb.Description, &createdAt, &updatedAt, &metadataJSON, &nb.SourceCount, &nb.NoteCount); err != nil {
			return nil, err
		}
		
		if uid.Valid {
			nb.UserID = uid.String
		}

		nb.CreatedAt = time.Unix(createdAt, 0)
		nb.UpdatedAt = time.Unix(updatedAt, 0)

		if metadataJSON != "" {
			json.Unmarshal([]byte(metadataJSON), &nb.Metadata)
		} else {
			nb.Metadata = make(map[string]interface{})
		}

		notebooks = append(notebooks, nb)
	}

	return notebooks, nil
}

// Source operations

// CreateSource creates a new source
func (s *Store) CreateSource(ctx context.Context, source *Source) error {
	source.ID = uuid.New().String()
	now := time.Now()
	source.CreatedAt = now
	source.UpdatedAt = now

	metadataJSON, _ := json.Marshal(source.Metadata)

	_, err := s.db.ExecContext(ctx, `
		INSERT INTO sources (id, notebook_id, name, type, url, content, file_name, file_size, chunk_count, created_at, updated_at, metadata)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
	`, source.ID, source.NotebookID, source.Name, source.Type, source.URL, source.Content,
		source.FileName, source.FileSize, source.ChunkCount, now.Unix(), now.Unix(), string(metadataJSON))

	return err
}

// GetSource retrieves a source by ID
func (s *Store) GetSource(ctx context.Context, id string) (*Source, error) {
	var src Source
	var metadataJSON string
	var createdAt, updatedAt int64

	err := s.db.QueryRowContext(ctx, `
		SELECT id, notebook_id, name, type, url, content, file_name, file_size, chunk_count, created_at, updated_at, metadata
		FROM sources WHERE id = ?
	`, id).Scan(&src.ID, &src.NotebookID, &src.Name, &src.Type, &src.URL, &src.Content,
		&src.FileName, &src.FileSize, &src.ChunkCount, &createdAt, &updatedAt, &metadataJSON)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("source not found")
	}
	if err != nil {
		return nil, err
	}

	src.CreatedAt = time.Unix(createdAt, 0)
	src.UpdatedAt = time.Unix(updatedAt, 0)

	if metadataJSON != "" {
		json.Unmarshal([]byte(metadataJSON), &src.Metadata)
	} else {
		src.Metadata = make(map[string]interface{})
	}

	return &src, nil
}

// ListSources retrieves all sources for a notebook
func (s *Store) ListSources(ctx context.Context, notebookID string) ([]Source, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, notebook_id, name, type, url, content, file_name, file_size, chunk_count, created_at, updated_at, metadata
		FROM sources WHERE notebook_id = ? ORDER BY created_at DESC
	`, notebookID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	sources := make([]Source, 0)
	for rows.Next() {
		var src Source
		var metadataJSON string
		var createdAt, updatedAt int64

		if err := rows.Scan(&src.ID, &src.NotebookID, &src.Name, &src.Type, &src.URL, &src.Content,
			&src.FileName, &src.FileSize, &src.ChunkCount, &createdAt, &updatedAt, &metadataJSON); err != nil {
			return nil, err
		}

		src.CreatedAt = time.Unix(createdAt, 0)
		src.UpdatedAt = time.Unix(updatedAt, 0)

		if metadataJSON != "" {
			json.Unmarshal([]byte(metadataJSON), &src.Metadata)
		} else {
			src.Metadata = make(map[string]interface{})
		}

		sources = append(sources, src)
	}

	return sources, nil
}

// DeleteSource deletes a source
func (s *Store) DeleteSource(ctx context.Context, id string) error {
	_, err := s.db.ExecContext(ctx, `DELETE FROM sources WHERE id = ?`, id)
	return err
}

// UpdateSourceChunkCount updates the chunk count for a source
func (s *Store) UpdateSourceChunkCount(ctx context.Context, id string, chunkCount int) error {
	_, err := s.db.ExecContext(ctx, `UPDATE sources SET chunk_count = ? WHERE id = ?`, chunkCount, id)
	return err
}

// Note operations

// CreateNote creates a new note
func (s *Store) CreateNote(ctx context.Context, note *Note) error {
	note.ID = uuid.New().String()
	now := time.Now()
	note.CreatedAt = now
	note.UpdatedAt = now

	metadataJSON, _ := json.Marshal(note.Metadata)
	sourceIDsJSON, _ := json.Marshal(note.SourceIDs)

	_, err := s.db.ExecContext(ctx, `
		INSERT INTO notes (id, notebook_id, title, content, type, source_ids, created_at, updated_at, metadata)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
	`, note.ID, note.NotebookID, note.Title, note.Content, note.Type, string(sourceIDsJSON),
		now.Unix(), now.Unix(), string(metadataJSON))

	return err
}

// GetNote retrieves a note by ID
func (s *Store) GetNote(ctx context.Context, id string) (*Note, error) {
	var note Note
	var metadataJSON, sourceIDsJSON string
	var createdAt, updatedAt int64

	err := s.db.QueryRowContext(ctx, `
		SELECT id, notebook_id, title, content, type, source_ids, created_at, updated_at, metadata
		FROM notes WHERE id = ?
	`, id).Scan(&note.ID, &note.NotebookID, &note.Title, &note.Content, &note.Type,
		&sourceIDsJSON, &createdAt, &updatedAt, &metadataJSON)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("note not found")
	}
	if err != nil {
		return nil, err
	}

	note.CreatedAt = time.Unix(createdAt, 0)
	note.UpdatedAt = time.Unix(updatedAt, 0)

	if metadataJSON != "" {
		json.Unmarshal([]byte(metadataJSON), &note.Metadata)
	} else {
		note.Metadata = make(map[string]interface{})
	}

	if sourceIDsJSON != "" {
		json.Unmarshal([]byte(sourceIDsJSON), &note.SourceIDs)
	}

	return &note, nil
}

// ListNotes retrieves all notes for a notebook
func (s *Store) ListNotes(ctx context.Context, notebookID string) ([]Note, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, notebook_id, title, content, type, source_ids, created_at, updated_at, metadata
		FROM notes WHERE notebook_id = ? ORDER BY created_at DESC
	`, notebookID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	notes := make([]Note, 0)
	for rows.Next() {
		var note Note
		var metadataJSON, sourceIDsJSON string
		var createdAt, updatedAt int64

		if err := rows.Scan(&note.ID, &note.NotebookID, &note.Title, &note.Content, &note.Type,
			&sourceIDsJSON, &createdAt, &updatedAt, &metadataJSON); err != nil {
			return nil, err
		}

		note.CreatedAt = time.Unix(createdAt, 0)
		note.UpdatedAt = time.Unix(updatedAt, 0)

		if metadataJSON != "" {
			json.Unmarshal([]byte(metadataJSON), &note.Metadata)
		} else {
			note.Metadata = make(map[string]interface{})
		}

		if sourceIDsJSON != "" {
			json.Unmarshal([]byte(sourceIDsJSON), &note.SourceIDs)
		}

		notes = append(notes, note)
	}

	return notes, nil
}

// DeleteNote deletes a note
func (s *Store) DeleteNote(ctx context.Context, id string) error {
	_, err := s.db.ExecContext(ctx, `DELETE FROM notes WHERE id = ?`, id)
	return err
}

// Chat operations

// CreateChatSession creates a new chat session
func (s *Store) CreateChatSession(ctx context.Context, notebookID, title string) (*ChatSession, error) {
	id := uuid.New().String()
	now := time.Now()

	if title == "" {
		title = "New Chat"
	}

	metadataJSON, _ := json.Marshal(map[string]interface{}{})

	_, err := s.db.ExecContext(ctx, `
		INSERT INTO chat_sessions (id, notebook_id, title, created_at, updated_at, metadata)
		VALUES (?, ?, ?, ?, ?, ?)
	`, id, notebookID, title, now.Unix(), now.Unix(), string(metadataJSON))
	if err != nil {
		return nil, err
	}

	return s.GetChatSession(ctx, id)
}

// GetChatSession retrieves a chat session by ID
func (s *Store) GetChatSession(ctx context.Context, id string) (*ChatSession, error) {
	var session ChatSession
	var metadataJSON string
	var createdAt, updatedAt int64

	err := s.db.QueryRowContext(ctx, `
		SELECT id, notebook_id, title, created_at, updated_at, metadata
		FROM chat_sessions WHERE id = ?
	`, id).Scan(&session.ID, &session.NotebookID, &session.Title, &createdAt, &updatedAt, &metadataJSON)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("chat session not found")
	}
	if err != nil {
		return nil, err
	}

	session.CreatedAt = time.Unix(createdAt, 0)
	session.UpdatedAt = time.Unix(updatedAt, 0)

	if metadataJSON != "" {
		json.Unmarshal([]byte(metadataJSON), &session.Metadata)
	} else {
		session.Metadata = make(map[string]interface{})
	}

	// Load messages
	session.Messages, err = s.listChatMessages(ctx, id)
	if err != nil {
		return nil, err
	}

	return &session, nil
}

// ListChatSessions retrieves all chat sessions for a notebook
func (s *Store) ListChatSessions(ctx context.Context, notebookID string) ([]ChatSession, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, notebook_id, title, created_at, updated_at, metadata
		FROM chat_sessions WHERE notebook_id = ? ORDER BY updated_at DESC
	`, notebookID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	sessions := make([]ChatSession, 0)
	for rows.Next() {
		var session ChatSession
		var metadataJSON string
		var createdAt, updatedAt int64

		if err := rows.Scan(&session.ID, &session.NotebookID, &session.Title, &createdAt, &updatedAt, &metadataJSON); err != nil {
			return nil, err
		}

		session.CreatedAt = time.Unix(createdAt, 0)
		session.UpdatedAt = time.Unix(updatedAt, 0)

		if metadataJSON != "" {
			json.Unmarshal([]byte(metadataJSON), &session.Metadata)
		} else {
			session.Metadata = make(map[string]interface{})
		}

		sessions = append(sessions, session)
	}

	return sessions, nil
}

// AddChatMessage adds a message to a chat session
func (s *Store) AddChatMessage(ctx context.Context, sessionID, role, content string, sources []string) (*ChatMessage, error) {
	id := uuid.New().String()
	now := time.Now()

	metadataJSON, _ := json.Marshal(map[string]interface{}{})
	sourcesJSON, _ := json.Marshal(sources)

	_, err := s.db.ExecContext(ctx, `
		INSERT INTO chat_messages (id, session_id, role, content, sources, created_at, metadata)
		VALUES (?, ?, ?, ?, ?, ?, ?)
	`, id, sessionID, role, content, string(sourcesJSON), now.Unix(), string(metadataJSON))
	if err != nil {
		return nil, err
	}

	// Update session timestamp
	_, err = s.db.ExecContext(ctx, `UPDATE chat_sessions SET updated_at = ? WHERE id = ?`, now.Unix(), sessionID)
	if err != nil {
		return nil, err
	}

	return s.getChatMessage(ctx, id)
}

// listChatMessages retrieves all messages for a session
func (s *Store) listChatMessages(ctx context.Context, sessionID string) ([]ChatMessage, error) {
	rows, err := s.db.QueryContext(ctx, `
		SELECT id, session_id, role, content, sources, created_at, metadata
		FROM chat_messages WHERE session_id = ? ORDER BY created_at ASC
	`, sessionID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	messages := make([]ChatMessage, 0)
	for rows.Next() {
		var msg ChatMessage
		var metadataJSON, sourcesJSON string
		var createdAt int64

		if err := rows.Scan(&msg.ID, &msg.SessionID, &msg.Role, &msg.Content, &sourcesJSON, &createdAt, &metadataJSON); err != nil {
			return nil, err
		}

		msg.CreatedAt = time.Unix(createdAt, 0)

		if metadataJSON != "" {
			json.Unmarshal([]byte(metadataJSON), &msg.Metadata)
		} else {
			msg.Metadata = make(map[string]interface{})
		}

		if sourcesJSON != "" {
			json.Unmarshal([]byte(sourcesJSON), &msg.Sources)
		}

		messages = append(messages, msg)
	}

	return messages, nil
}

// getChatMessage retrieves a single message by ID
func (s *Store) getChatMessage(ctx context.Context, id string) (*ChatMessage, error) {
	var msg ChatMessage
	var metadataJSON, sourcesJSON string
	var createdAt int64

	err := s.db.QueryRowContext(ctx, `
		SELECT id, session_id, role, content, sources, created_at, metadata
		FROM chat_messages WHERE id = ?
	`, id).Scan(&msg.ID, &msg.SessionID, &msg.Role, &msg.Content, &sourcesJSON, &createdAt, &metadataJSON)
	if err == sql.ErrNoRows {
		return nil, fmt.Errorf("chat message not found")
	}
	if err != nil {
		return nil, err
	}

	msg.CreatedAt = time.Unix(createdAt, 0)

	if metadataJSON != "" {
		json.Unmarshal([]byte(metadataJSON), &msg.Metadata)
	} else {
		msg.Metadata = make(map[string]interface{})
	}

	if sourcesJSON != "" {
		json.Unmarshal([]byte(sourcesJSON), &msg.Sources)
	}

	return &msg, nil
}

// DeleteChatSession deletes a chat session
func (s *Store) DeleteChatSession(ctx context.Context, id string) error {
	_, err := s.db.ExecContext(ctx, `DELETE FROM chat_sessions WHERE id = ?`, id)
	return err
}

// Close closes the database connection
func (s *Store) Close() error {
	return s.db.Close()
}
