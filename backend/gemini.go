package backend

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/kataras/golog"
	"github.com/tmc/langchaingo/llms"
	"google.golang.org/genai"
)

// LLMProvider defines the interface for LLM operations
type LLMProvider interface {
	// GenerateImage generates an image using the provider
	GenerateImage(ctx context.Context, model, prompt string, userID string) (string, error)

	// GenerateTextWithModel generates text using a specific model
	GenerateTextWithModel(ctx context.Context, prompt string, model string) (string, error)

	// GenerateFromSinglePrompt generates text from a single prompt using the default LLM
	GenerateFromSinglePrompt(ctx context.Context, llm llms.Model, prompt string, options ...llms.CallOption) (string, error)
}

// GeminiClient is the default implementation of LLMProvider using Google GenAI
type GeminiClient struct {
	googleAPIKey string
	llm          llms.Model // maybe other llm except gemini for chat/summary etc.
}

// NewGeminiClient creates a new GeminiClient
func NewGeminiClient(googleAPIKey string, llm llms.Model) *GeminiClient {
	return &GeminiClient{
		googleAPIKey: googleAPIKey,
		llm:          llm,
	}
}

// GenerateImage generates an image using the Google GenAI SDK
func (n *GeminiClient) GenerateImage(ctx context.Context, model, prompt string, userID string) (string, error) {
	if n.googleAPIKey == "" {
		golog.Errorf("google_api_key is not set")
		return "", fmt.Errorf("google_api_key is not set")
	}

	httpClient := &http.Client{
		Timeout: time.Hour, // Give the model enough time to "think"
		Transport: &http.Transport{
			DisableKeepAlives: false,
			MaxIdleConns:      100,
			IdleConnTimeout:   time.Hour,
		},
	}

	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:     n.googleAPIKey,
		Backend:    genai.BackendGeminiAPI,
		HTTPClient: httpClient,
	})
	if err != nil {
		return "", fmt.Errorf("failed to create genai client: %w", err)
	}

	var lastErr error
	for attempt := 1; attempt <= 3; attempt++ {
		if attempt > 1 {
			golog.Infof("retrying image generation (attempt %d/3)...", attempt)
			time.Sleep(2 * time.Second)
		} else {
			golog.Infof("generating images with model %s using GenerateContent...", model)
		}

		genCtx, cancel := context.WithTimeout(ctx, 300*time.Second)
		resp, err := client.Models.GenerateContent(genCtx, model, genai.Text(prompt), nil)
		if err != nil {
			cancel()
			golog.Errorf("failed to generate content (attempt %d): %v", attempt, err)
			lastErr = err
			continue
		}

		if len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
			cancel()
			golog.Errorf("no candidates returned by the model (attempt %d)", attempt)
			lastErr = fmt.Errorf("no candidates generated")
			continue
		}

		var imageData []byte
		for _, part := range resp.Candidates[0].Content.Parts {
			if part.InlineData != nil {
				imageData = part.InlineData.Data
				break
			}
		}

		if len(imageData) == 0 {
			cancel()
			golog.Errorf("no image data found in the response parts (attempt %d)", attempt)
			lastErr = fmt.Errorf("no image data in response")
			continue
		}

		cancel()
		golog.Infof("image data received successfully, saving...")

		// Save the image to user-specific directory
		fileName := fmt.Sprintf("infograph_%d.png", time.Now().UnixNano())
		var uploadDir string
		if userID != "" {
			uploadDir = filepath.Join("./data/uploads", userID)
		} else {
			uploadDir = "./data/uploads"
		}

		if err := os.MkdirAll(uploadDir, 0755); err != nil {
			return "", fmt.Errorf("failed to create upload directory: %w", err)
		}

		filePath := filepath.Join(uploadDir, fileName)
		if err := os.WriteFile(filePath, imageData, 0644); err != nil {
			golog.Errorf("failed to save image to %s: %v", filePath, err)
			return "", fmt.Errorf("failed to save image: %w", err)
		}

		golog.Infof("infographic saved to %s", filePath)
		return filePath, nil
	}

	return "", fmt.Errorf("failed to generate image after 3 attempts: %w", lastErr)
}

// GenerateTextWithModel generates text using the Google GenAI SDK with a specific model
func (n *GeminiClient) GenerateTextWithModel(ctx context.Context, prompt string, model string) (string, error) {
	if n.googleAPIKey == "" {
		golog.Errorf("google_api_key is not set")
		return "", fmt.Errorf("google_api_key is not set")
	}

	httpClient := &http.Client{
		Timeout: 5 * time.Minute, // Give the model enough time to "think"
		Transport: &http.Transport{
			DisableKeepAlives: false,
			MaxIdleConns:      100,
			IdleConnTimeout:   5 * time.Minute,
		},
	}

	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:     n.googleAPIKey,
		Backend:    genai.BackendGeminiAPI,
		HTTPClient: httpClient,
	})
	if err != nil {
		return "", fmt.Errorf("failed to create genai client: %w", err)
	}

	golog.Infof("generating text with model %s using GenerateContent...", model)

	// Set a timeout for the text generation
	ctx, cancel := context.WithTimeout(ctx, 300*time.Second)
	defer cancel()

	resp, err := client.Models.GenerateContent(ctx, model, genai.Text(prompt), nil)
	if err != nil {
		golog.Errorf("failed to generate gemini text: %v", err)
		return "", fmt.Errorf("failed to generate gemini text: %w", err)
	}

	if len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil || len(resp.Candidates[0].Content.Parts) == 0 {
		golog.Errorf("no text candidates returned by the model")
		return "", fmt.Errorf("no text generated")
	}

	var textContent strings.Builder
	for _, part := range resp.Candidates[0].Content.Parts {
		if part.Text != "" {
			textContent.WriteString(part.Text)
		}
	}

	result := textContent.String()
	if result == "" {
		golog.Errorf("empty text content in response")
		return "", fmt.Errorf("empty response from model")
	}

	return result, nil
}

// GenerateFromSinglePrompt generates text from a single prompt using the specified LLM
func (n *GeminiClient) GenerateFromSinglePrompt(ctx context.Context, llm llms.Model, prompt string, options ...llms.CallOption) (string, error) {
	return llms.GenerateFromSinglePrompt(ctx, n.llm, prompt, options...)
}
