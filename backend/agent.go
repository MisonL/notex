package backend

import (
	"context"
	"fmt"
	"regexp"
	"strings"
	"time"

	"github.com/tmc/langchaingo/llms"
	ollamallm "github.com/tmc/langchaingo/llms/ollama"
	"github.com/tmc/langchaingo/llms/openai"
	"github.com/tmc/langchaingo/prompts"
)

// Agent handles AI operations for generating notes and chat responses
type Agent struct {
	vectorStore *VectorStore
	llm         llms.Model
	cfg         Config
}

// NewAgent creates a new agent
func NewAgent(cfg Config, vectorStore *VectorStore) (*Agent, error) {
	llm, err := createLLM(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create LLM: %w", err)
	}

	return &Agent{
		vectorStore: vectorStore,
		llm:         llm,
		cfg:         cfg,
	}, nil
}

// createLLM creates an LLM based on configuration
func createLLM(cfg Config) (llms.Model, error) {
	if cfg.IsOllama() {
		return ollamallm.New(
			ollamallm.WithModel(cfg.OllamaModel),
			ollamallm.WithServerURL(cfg.OllamaBaseURL),
		)
	}

	opts := []openai.Option{
		openai.WithToken(cfg.OpenAIAPIKey),
		openai.WithModel(cfg.OpenAIModel),
	}
	if cfg.OpenAIBaseURL != "" {
		opts = append(opts, openai.WithBaseURL(cfg.OpenAIBaseURL))
	}

	return openai.New(opts...)
}

// GenerateTransformation generates a note based on transformation type
func (a *Agent) GenerateTransformation(ctx context.Context, req *TransformationRequest, sources []Source) (*TransformationResponse, error) {
	// Build context from sources
	var sourceContext strings.Builder
	for i, src := range sources {
		sourceContext.WriteString(fmt.Sprintf("\n## Source %d: %s\n", i+1, src.Name))

		// Use MaxContextLength from config, or default to a safe large value if not set (or too small)
		limit := a.cfg.MaxContextLength
		if limit <= 0 {
			limit = 100000 // Default to 100k chars if config is invalid
		}

		if src.Content != "" {
			if len(src.Content) <= limit {
				sourceContext.WriteString(src.Content)
			} else {
				// Truncate content instead of replacing it entirely
				sourceContext.WriteString(src.Content[:limit])
				sourceContext.WriteString(fmt.Sprintf("\n... [Content truncated, total length: %d]", len(src.Content)))
			}
		} else {
			sourceContext.WriteString(fmt.Sprintf("[Source content: %s, type: %s]", src.Name, src.Type))
		}
		sourceContext.WriteString("\n")
	}

	// Build prompt using f-string format (no Go template reserved names issue)
	promptTemplate := a.getTransformationPrompt(req)

	prompt := prompts.NewPromptTemplate(
		promptTemplate,
		[]string{"sources", "type", "length", "format", "prompt"},
	)
	prompt.TemplateFormat = prompts.TemplateFormatFString

	promptValue, err := prompt.Format(map[string]any{
		"sources": sourceContext.String(),
		"type":    req.Type,
		"length":  req.Length,
		"format":  req.Format,
		"prompt":  req.Prompt,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to format prompt: %w", err)
	}

	// Generate response
	var response string
	var genErr error

	if req.Type == "ppt" {
		response, genErr = a.GenerateGeminiText(ctx, promptValue, "gemini-3-flash-preview")
	} else {
		ctx, cancel := context.WithTimeout(ctx, 300*time.Second)
		defer cancel()
		response, genErr = llms.GenerateFromSinglePrompt(ctx, a.llm, promptValue)
	}

	if genErr != nil {
		return nil, fmt.Errorf("failed to generate response: %w", genErr)
	}

	// Build source summaries
	sourceSummaries := make([]SourceSummary, len(sources))
	for i, src := range sources {
		sourceSummaries[i] = SourceSummary{
			ID:   src.ID,
			Name: src.Name,
			Type: src.Type,
		}
	}

	return &TransformationResponse{
		Type:      req.Type,
		Content:   response,
		Sources:   sourceSummaries,
		CreatedAt: time.Now(),
		Metadata: map[string]interface{}{
			"length": req.Length,
			"format": req.Format,
		},
	}, nil
}

// getTransformationPrompt returns the prompt template for each transformation type
func (a *Agent) getTransformationPrompt(req *TransformationRequest) string {
	switch req.Type {
	case "summary":
		return `你是一个擅长创建综合摘要的专家。请根据以下来源，以{format}格式创建一个{length}摘要。
**注意：无论来源是什么语言，请务必使用中文进行回复。不要使用 ` + "```markdown" + ` 标记包裹输出。**

来源：
{sources}

请提供一个结构良好的摘要，捕捉来源中的关键信息、主要主题和重要细节。`

	case "faq":
		return `你是一个擅长创建常见问题解答（FAQ）文档的专家。请根据以下来源，以{format}格式生成一个全面的FAQ。
**注意：无论来源是什么语言，请务必使用中文进行回复。不要使用 ` + "```markdown" + ` 标记包裹输出。**

来源：
{sources}

创建10-15个常见问题及其详细解答，涵盖来源中的主要主题和信息。`

	case "study_guide":
		return `你是一个教育专家。请根据以下来源，以{format}格式创建一个全面的学习指南。
**注意：无论来源是什么语言，请务必使用中文进行回复。不要使用 ` + "```markdown" + ` 标记包裹输出。**

来源：
{sources}

学习指南应包括：
1. 学习目标
2. 关键概念和定义
3. 重要主题和议题
4. 学习问题和练习
5. 要点总结

请针对{length}的学习课程进行格式化。`

	case "outline":
		return `你是一个擅长创建结构化大纲的专家。请根据以下来源，以{format}格式创建一个详细的层级大纲。
**注意：无论来源是什么语言，请务必使用中文进行回复。不要使用 ` + "```markdown" + ` 标记包裹输出。**

来源：
{sources}

大纲应：
- 使用适当的层级结构（I, A, 1, a）
- 涵盖所有主要主题和子主题
- 包含主要部分的简要说明
- 详细程度为{length}`

	case "podcast":
		return `你是一个播客脚本编剧。请根据以下来源创建一个引人入胜的播客脚本。
**注意：无论来源是什么语言，请务必使用中文进行回复。不要使用 ` + "```markdown" + ` 标记包裹输出。**

来源：
{sources}

脚本应：
- 具有对话性和吸引力
- 涵盖来源中的主要主题
- 包括两位主持人讨论材料
- 口语时长约为10-15分钟
- 包含自然的过渡和提问
- 有清晰的开场白和结束语

请将其格式化为带有演讲者标签（主持人1，主持人2）和[括号]中舞台指示的播客脚本。`

	case "timeline":
		return `你是一个擅长创建按时间顺序排列的时间线的专家。请根据以下来源，以{format}格式创建一个时间线。
**注意：无论来源是什么语言，请务必使用中文进行回复。不要使用 ` + "```markdown" + ` 标记包裹输出。**

来源：
{sources}

按时间顺序提取和组织事件，包括：
- 日期或时间段
- 事件描述
- 涉及的关键人物
- 每个事件的重要性`

	case "glossary":
		return `你是一个擅长创建术语表的专家。请根据以下来源，以{format}格式创建一个全面的术语表。
**注意：无论来源是什么语言，请务必使用中文进行回复。不要使用 ` + "```markdown" + ` 标记包裹输出。**

来源：
{sources}

包括：
- 重要术语和概念
- 清晰简洁的定义
- 来源中的上下文
- 相关术语之间的交叉引用`

	case "quiz":
		return `你是一个创建评估材料的教育家。请根据以下来源，以{format}格式创建一个测验。
**注意：无论来源是什么语言，请务必使用中文进行回复。不要使用 ` + "```markdown" + ` 标记包裹输出。**

来源：
{sources}

测验应包括：
- 混合题型（多项选择、判断正误、简答）
- 不同难度的问题
- 答案
- 测试理解力而非仅仅是记忆力的问题

创建一个包含10-20个问题的{length}测验。`

	case "mindmap":
		return `你是一位资深的信息架构师和知识管理专家。请将【文本内容】提炼并转换为 Mermaid.js 的 mindmap 格式。
**注意：无论来源是什么语言，请务必使用中文进行回复。**

# 样式规范：
1. **中心主题**：必须使用 root((内容)) 格式（圆圈）。
2. **主要分支**：使用 (内容) 格式（圆角矩形）。
3. **细节节点**：使用 [内容] 格式（普通矩形）或直接写文字。

# 严格逻辑规范：
1. **仅限 mindmap 语法**：严禁使用 graph, LR, --> 等字符。
2. **内容安全**：节点内容必须精炼（10字以内），严禁包含引号。
3. **严禁解释**：只输出以 ` + "```mermaid" + ` 开头和以 ` + "```" + ` 结尾的代码块。

来源：
{sources}

# 示例：
` + "```mermaid" + `
mindmap
  root((核心主题))
    (主要分支A)
      [细节1]
      [细节2]
    (主要分支B)
      [细节3]
` + "```" + `
`

	case "infograph":
		return `# Role
你是一位世界顶级的数据可视化设计师和信息图专家。你的任务是将复杂的文本信息转化为直观、吸引人且准确的视觉设计方案。

# Task
阅读所附文本，设计一张信息图（Infographic）。不要进行总结，而是描述这张图应该长什么样。你的输出将被直接用作 DALL-E 3 的绘画提示词。
**注意：无论来源是什么语言，请务必使用中文进行回复。确保信息图中的所有文本内容（如标题、标签、数据点）都使用中文。不要使用 ` + "```markdown" + ` 标记包裹输出。**

# Design Guidelines
1.  **核心信息提炼**：找出文本中最重要的 3-5 个数据点、流程步骤或对比项。
2.  **视觉隐喻**：使用形象的比喻。例如，讲网络安全用“盾牌和锁”，讲增长用“火箭或上升的箭头”。
3.  **布局结构**：明确定义图的结构（例如：“从左到右的流程图”、“分成两半的对比图”、“中心辐射图”）。
4.  **文本限制**：信息图中的文字必须极简。只保留标题、关键数据和极短的标签。
5.  **风格**：插画或手绘感，使用柔和的插画或轻松的手绘笔触，以增强亲和力和友好度。

# Output Format (DALL-E 3 Prompt style)
Start with "Infographic illustration created in a soft, hand-drawn digital art style with friendly and approachable vibes."
[描述整体布局和背景风格]
[详细描述主要视觉元素 1，包含其图标、颜色、位置和附带的文字标签]
[详细描述主要视觉元素 2，...]
[详细描述连接元素（如箭头、线条）]
[描述整体标题和配色方案]
End with "The background is a clean, light gradient suitable for a professional presentation."

# Input Text
{sources}
`

	case "ppt":
		return `# 演示文稿设计指令

## 角色定位

你是一位世界级的演示文稿设计师和故事讲述者。你创作视觉震撼且高度精致的幻灯片，能够有效传达复杂信息。你精通设计，并擅长讲故事。

你制作的幻灯片会根据源材料和目标受众进行调整。始终有一个故事主线，而你会找到最佳方式来讲述它。你融合了顶尖设计师的专业知识和创造力。

## 设计原则

这套幻灯片主要设计用于**阅读和分享**。结构应该是不言自明的，无需演讲者即可轻松理解。叙事和所有有用的数据都应包含在幻灯片的文本和视觉元素中。幻灯片应包含足够的上下文，使任何视觉元素都能独立被理解。如果有助于叙事，可以添加包含更密集信息（从源材料中提取）的特定幻灯片。

## 工作流程

你现在正在为下面描述的幻灯片编写一个**大纲**。
**重要：这套幻灯片严禁超过 20 页。如果内容过多，请务必进行精简或合并。**

我们将把这个大纲提供给专业设计师来制作最终成品。

幻灯片内容应使用  英文。占位符应保留为 英文。

---

## 第一步：生成风格指南

**首先**，在编写幻灯片大纲之前，你必须根据内容主题和用户要求生成一个全局的**风格指南**块。这应该包装在代码块内的XML标签中。

### 风格指南示例

` + "```xml" + `
<STYLE_INSTRUCTIONS>
设计美学：一种简洁、精致、极简的手绘水彩编辑风格，灵感来自建筑蓝图和高端技术期刊。整体感觉是精确、清晰和知识优雅。

背景颜色：带有肌理的柔和米白色，十六进制代码 #F8F7F5，让人联想到高质量的绘图纸。

主要字体：思源黑体，Neue Haas Grotesk Display Pro。用于所有幻灯片标题和主要标题。应使用粗体以产生冲击力和清晰度。

次要字体：思源宋体，Tiempos Text。用于所有正文、副标题和注释。其高可读性和经典感觉与简洁的无衬线标题形成专业对比。

色彩方案：
- 主要文本颜色：深石板灰，#2F3542
- 主要强调色（用于高亮、图表和关键元素）：生动、智能的蓝色，#007AFF

视觉元素：
始终使用细致、精确的线条、示意图和简洁的矢量图形。视觉效果是概念性和抽象的，旨在阐明想法而非描绘实际场景。布局宽敞且结构化，优先考虑信息层次和可读性。没有幻灯片编号、页脚、徽标或页眉。
</STYLE_INSTRUCTIONS>
` + "```" + `

### 风格指南模板结构

使用以下结构作为模板，但**动态调整**美学、字体和颜色以适应具体叙事：

` + "```markdown" + `
你是"建筑师"，一个复杂的AI，旨在将指令可视化为高端蓝图风格的数据展示。你的输出精确、分析性强且美学精致。

**核心指令：**

1. 分析用户提示的结构、意图和关键元素。
2. 将指令转化为简洁、结构化的视觉隐喻（蓝图、展示、示意图）。
3. 使用特定、克制的色彩方案和字体系列，以实现最大清晰度和专业影响力。
4. 为所有视觉输出保持严格的 16:9 纵横比。
5. 以三联画或网格布局呈现信息，平衡文本和视觉效果。

**风格指南：**

设计美学：[描述整体风格，例如：极简、俏皮、企业、建筑等]

背景颜色：[描述和十六进制代码]

主要字体：[标题字体名称]

次要字体：[正文字体名称]

色彩方案：
- 主要文本颜色：[十六进制代码]
- 主要强调色：[十六进制代码]

视觉元素：[描述线条、形状、图像风格的使用，摄影 vs 矢量等]

**要绘制的内容：**
` + "```" + `

---

## 第二步：定义内容焦点

对于这套特定的幻灯片，我们希望内容聚焦于：

{prompt}

我们还在下面附加了一些**制作说明**，这将有助于指导整套幻灯片的整体结构和叙事。

制作说明 (来源材料)：
{sources}

---

## 大纲编写规则

记住以下规则：

### 基本要求
- 专注于幻灯片的大纲以及每张幻灯片应涵盖的内容
- 每张幻灯片的描述应该全面且严格结构化
- **第1张幻灯片必须是封面幻灯片，最后一张幻灯片必须是封底幻灯片**
  - 这两张幻灯片的视觉风格和布局应与内部内容幻灯片明显不同（例如，使用"海报风格"布局、英雄式排版或全出血图像），以营造氛围并提供有力的结尾

### 每张幻灯片的结构

对于**每张幻灯片**，你必须使用以下 **4个部分** 准确输出内容：

#### // 叙事目标
解释这张幻灯片在整个叙事弧中的具体讲故事目的

#### // 关键内容
列出大标题、副标题和正文/要点。**每个具体数据点都必须可追溯到源材料。**

#### // 视觉元素
描述支持观点所需的图像、图表、图形或抽象视觉效果。

#### // 布局
描述构图、层次结构、空间排列或焦点。

### 内容要求
- 保留源材料中的关键元素
- **每个具体数据点都必须直接可追溯到源材料**
- 所有细节都需要提及，因为设计师稍后将无法访问源内容
- 始终假设受众比你想象的更有专业知识、兴趣和智慧

---

## 关键要点（必须遵守）

### ❌ 禁止事项

- **永远不要生成超过 20 张幻灯片**
- 避免使用"标题：副标题"格式的标题；它们看起来非常AI生成
- 明确避免陈词滥调的"AI套话"模式
  - 永远不要使用"这不仅仅是 [X]，更是 [Y]"之类的短语
- 永远不要包含任何带有作者姓名、日期等占位符的幻灯片
- 永远不要要求包含知名人物的照片级真实图像
- **永远不要以通用的"有问题吗？"或"谢谢"幻灯片结束**
  - 相反，封底应该是一个设计好的结束语、一个有意义的参考或一个强有力的视觉要点，以锚定整个叙事

### ✅ 必须做到

- 使用直接、自信、主动的**人性化语言**
- 倾向于使用叙事性的主题句来帮助将整套幻灯片联系在一起
- 确保所有数据点都有源材料支撑
- 为设计师提供足够详细的描述

---

## 总结

这份指令的目的是指导AI创建**高质量、专业、具有强烈叙事性**的演示文稿大纲。最终产品应该：

1. 有清晰的视觉风格指南
2. 讲述一个连贯的故事
3. 包含足够的细节供设计师执行
4. 避免常见的AI生成内容陷阱
5. 以有意义的方式开始和结束
`

	case "custom":
		return `你是一个有用的助手。根据以下来源和自定义请求，生成请求的内容。
**注意：无论来源是什么语言，请务必使用中文进行回复。不要使用 ` + "```markdown" + ` 标记包裹输出。**

来源：
{sources}

自定义请求：
{prompt}

请以{format}格式生成内容，保持{length}。`

	default:
		return `你是一个有用的助手。根据以下来源，以{format}格式提供一个{type}。
**注意：无论来源是什么语言，请务必使用中文进行回复。不要使用 ` + "```markdown" + ` 标记包裹输出。**

来源：
{sources}

生成{length}内容。`
	}
}

// Chat performs a chat query with RAG
func (a *Agent) Chat(ctx context.Context, notebookID, message string, history []ChatMessage) (*ChatResponse, error) {
	// Perform similarity search to find relevant sources
	docs, err := a.vectorStore.SimilaritySearch(ctx, message, a.cfg.MaxSources)
	if err != nil {
		return nil, fmt.Errorf("failed to search documents: %w", err)
	}

	// Build context from retrieved documents
	var contextBuilder strings.Builder
	if len(docs) > 0 {
		contextBuilder.WriteString("来源中的相关信息：\n\n")
		for i, doc := range docs {
			contextBuilder.WriteString(fmt.Sprintf("[来源 %d] %s\n", i+1, doc.PageContent))
			if source, ok := doc.Metadata["source"].(string); ok {
				contextBuilder.WriteString(fmt.Sprintf("来源: %s\n\n", source))
			}
		}
	}

	// Build chat history
	var historyBuilder strings.Builder
	for i, msg := range history {
		if i >= 10 { // Limit history
			break
		}
		role := "用户"
		if msg.Role == "assistant" {
			role = "助手"
		}
		historyBuilder.WriteString(fmt.Sprintf("%s: %s\n", role, msg.Content))
	}

	// Create RAG prompt using f-string format
	systemPrompt := `你是一个笔记本应用程序的有用人工智能助手。根据提供的上下文和聊天历史记录回答用户的问题。
**无论来源文件是什么语言，请务必使用中文回答用户的问题。不要使用 ` + "```markdown" + ` 标记包裹输出。**
如果上下文中没有足够的信息，请说明情况并提供一般性的回答。

聊天历史记录：
{history}

上下文：
{context}

用户问题：{question}

请提供有用的、准确的回答。当引用来源中的信息时，请提及信息来自哪个来源。`

	promptTemplate := prompts.NewPromptTemplate(
		systemPrompt,
		[]string{"history", "context", "question"},
	)
	promptTemplate.TemplateFormat = prompts.TemplateFormatFString

	promptValue, err := promptTemplate.Format(map[string]any{
		"history":  historyBuilder.String(),
		"context":  contextBuilder.String(),
		"question": message,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to format prompt: %w", err)
	}

	// Generate response
	ctx, cancel := context.WithTimeout(ctx, 300*time.Second)
	defer cancel()

	response, err := llms.GenerateFromSinglePrompt(ctx, a.llm, promptValue)
	if err != nil {
		return nil, fmt.Errorf("failed to generate response: %w", err)
	}

	// Build source summaries
	sourceSummaries := make([]SourceSummary, 0, len(docs))
	sourceMap := make(map[string]bool)
	for _, doc := range docs {
		if source, ok := doc.Metadata["source"].(string); ok {
			if !sourceMap[source] {
				sourceSummaries = append(sourceSummaries, SourceSummary{
					ID:   source,
					Name: source,
					Type: "file",
				})
				sourceMap[source] = true
			}
		}
	}

	return &ChatResponse{
		Message:   response,
		Sources:   sourceSummaries,
		SessionID: notebookID,
		Metadata: map[string]interface{}{
			"docs_retrieved": len(docs),
		},
	}, nil
}

// Slide represents a parsed PPT slide
type Slide struct {
	Style   string
	Content string
}

// ParsePPTSlides parses the LLM output into individual slides
func (a *Agent) ParsePPTSlides(content string) []Slide {
	var slides []Slide

	// 1. Extract style instructions
	style := ""
	styleStart := strings.Index(content, "<STYLE_INSTRUCTIONS>")
	styleEnd := strings.Index(content, "</STYLE_INSTRUCTIONS>")
	if styleStart != -1 && styleEnd > styleStart {
		style = content[styleStart+20 : styleEnd]
	}

	// 2. Split by Slide markers.
	// We look for "Slide X" or "幻灯片 X" with optional Markdown headers
	re := regexp.MustCompile(`(?m)^(?:\s*#{1,6}\s*)?(?:Slide|幻灯片|第\d+张幻灯片|##)\s*\d+[:\s]*.*$`)
	indices := re.FindAllStringIndex(content, -1)

	if len(indices) > 0 {
		for i := 0; i < len(indices); i++ {
			start := indices[i][0]
			end := len(content)
			if i+1 < len(indices) {
				end = indices[i+1][0]
			}

			slideContent := content[start:end]
			// Validation: Must contain at least one of the section markers
			lower := strings.ToLower(slideContent)
			if strings.Contains(lower, "叙事目标") ||
				strings.Contains(lower, "narrative goal") ||
				strings.Contains(lower, "关键内容") {
				slides = append(slides, Slide{
					Style:   style,
					Content: slideContent,
				})
			}
		}
	}

	// 3. If still nothing, try splitting by the required // NARRATIVE GOAL / // 叙事目标
	if len(slides) == 0 {
		// Use a more specific marker for splitting if Slide headers are missing
		marker := "// 叙事目标"
		if !strings.Contains(content, marker) {
			marker = "// NARRATIVE GOAL"
		}

		if strings.Contains(content, marker) {
			parts := strings.Split(content, marker)
			for i := 1; i < len(parts); i++ {
				slides = append(slides, Slide{
					Style:   style,
					Content: marker + parts[i],
				})
			}
		}
	}

	// Final fallback for completely unstructured content
	if len(slides) == 0 {
		slides = append(slides, Slide{Style: style, Content: content})
	}

	return slides
}

// GeneratePodcastScript generates a podcast script from sources
func (a *Agent) GeneratePodcastScript(ctx context.Context, sources []Source, voice string) (string, error) {
	req := &TransformationRequest{
		Type:   "podcast",
		Length: "medium",
		Format: "markdown",
	}

	resp, err := a.GenerateTransformation(ctx, req, sources)
	if err != nil {
		return "", err
	}

	return resp.Content, nil
}

// GenerateOutline generates an outline from sources
func (a *Agent) GenerateOutline(ctx context.Context, sources []Source) (string, error) {
	req := &TransformationRequest{
		Type:   "outline",
		Length: "detailed",
		Format: "markdown",
	}

	resp, err := a.GenerateTransformation(ctx, req, sources)
	if err != nil {
		return "", err
	}

	return resp.Content, nil
}

// GenerateFAQ generates an FAQ from sources
func (a *Agent) GenerateFAQ(ctx context.Context, sources []Source) (string, error) {
	req := &TransformationRequest{
		Type:   "faq",
		Length: "comprehensive",
		Format: "markdown",
	}

	resp, err := a.GenerateTransformation(ctx, req, sources)
	if err != nil {
		return "", err
	}

	return resp.Content, nil
}

// GenerateStudyGuide generates a study guide from sources
func (a *Agent) GenerateStudyGuide(ctx context.Context, sources []Source) (string, error) {
	req := &TransformationRequest{
		Type:   "study_guide",
		Length: "comprehensive",
		Format: "markdown",
	}

	resp, err := a.GenerateTransformation(ctx, req, sources)
	if err != nil {
		return "", err
	}

	return resp.Content, nil
}

// GenerateSummary generates a summary from sources
func (a *Agent) GenerateSummary(ctx context.Context, sources []Source, length string) (string, error) {
	req := &TransformationRequest{
		Type:   "summary",
		Length: length,
		Format: "markdown",
	}

	resp, err := a.GenerateTransformation(ctx, req, sources)
	if err != nil {
		return "", err
	}

	return resp.Content, nil
}
