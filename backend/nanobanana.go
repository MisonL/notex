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
	"google.golang.org/genai"
)

// GenerateInfographImage generates an infographic image using the Nano Banana Pro SDK
func (a *Agent) GenerateInfographImage(ctx context.Context, model, prompt string) (string, error) {
	if a.cfg.GoogleAPIKey == "" {
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
		APIKey:     a.cfg.GoogleAPIKey,
		Backend:    genai.BackendGeminiAPI,
		HTTPClient: httpClient,
	})
	if err != nil {
		return "", fmt.Errorf("failed to create genai client: %w", err)
	}

	// Using gemini-3-pro-image-preview as requested
	// model := "gemini-3-pro-image-preview"
	golog.Infof("generating infographic with model %s using GenerateContent...", model)

	ctx, cancel := context.WithTimeout(ctx, 300*time.Second)
	defer cancel()

	resp, err := client.Models.GenerateContent(ctx, model, genai.Text(prompt), nil)
	if err != nil {
		golog.Errorf("failed to generate content: %v", err)
		return "", fmt.Errorf("failed to generate content: %w", err)
	}

	if len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
		golog.Errorf("no candidates returned by the model")
		return "", fmt.Errorf("no candidates generated")
	}

	var imageData []byte
	for _, part := range resp.Candidates[0].Content.Parts {
		if part.InlineData != nil {
			imageData = part.InlineData.Data
			break
		}
	}

	if len(imageData) == 0 {
		golog.Errorf("no image data found in the response parts")
		return "", fmt.Errorf("no image data in response")
	}

	golog.Infof("image data received successfully, saving...")

	// Save the image
	fileName := fmt.Sprintf("infograph_%d.png", time.Now().UnixNano())
	uploadDir := "./data/uploads"
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

// GenerateGeminiText generates text using the Google GenAI SDK with a specific model
func (a *Agent) GenerateGeminiText(ctx context.Context, prompt string, model string) (string, error) {
	if a.cfg.GoogleAPIKey == "" {
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
		APIKey:     a.cfg.GoogleAPIKey,
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
