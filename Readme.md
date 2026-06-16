# OptiChat

OptiChat is an advanced terminal-based chat application built with Python and Textual. It features a robust multi-tier memory system, personalized memory tracking, dynamic model connectivity, (including cloud and local Ollama models), web search support and a sophisticated prompt construction pipeline for high-quality, contextual AI responses.

## 🌟 Key Features

*   **Terminal-based UI**: A beautiful, responsive interface built with Textual, featuring tabs, chat session sidebars, and customizable themes.
*   **Multi-Tier Memory System**:
    *   **Short-Term Memory**: Token-budgeted rolling window for recent context.
    *   **LRU Memory**: Background-processed cache of frequently used messages.
    *   **Long-Term Memory**: Persistent vector store (ChromaDB) for semantic search across conversations.
    *   **Personalized Memory**: Automatically learns and updates user preferences, interests, and interaction styles with conflict resolution.
*   **Dynamic Model Connectivity**: Support for OpenAI, Anthropic, Gemini, and local models via Ollama.
*   **Web Search Support**: Utilizes duckduckgo-search to fetch real-time information from the internet.
*   **Prompt Construction Pipeline**: Utilizes LangGraph to dynamically classify queries, retrieve memory, apply personalization, and enforce structured output schemas.
*   **Thinking Logs**: Every assistant response includes a collapsible section showing the model's chain-of-thought ToDo plan – what the model thought before responding.
*   **Auto-Tool Calling**: The model can automatically call tools (web search, memory retrieval etc.) based on the query. To reduce latency it only calls the tools that are necessary and therefore short query it takes less time to respond, whereas for complex queries it will call the tools that are necessary and will take more time to respond.
*   **Adaptive Response**: Response length and depth dynamically adapt to question complexity (simple → concise, complex → thorough and comprehensive).
*   **Auto Chat Naming**: New chats are automatically renamed based on your first question (2-3 word title) via a background thread.
*   **Secure Local Storage**: All data, including settings, API keys (via `.env`), SQLite databases for chats, and ChromaDB vectors, are stored securely in your local `~/.optichat/` directory.

## 🏗️ Architecture

### Storage
OptiChat stores its data locally in `~/.optichat/`. This includes:
- `config.json` for global settings.
- `optichat.db` (SQLite) for storing chats, messages, and session metadata.
- `chroma/` for ChromaDB vector embeddings.
- Flat files for chat-specific short-term and LRU caches.

### Memory Pipeline
1.  **Short-term**: Retains the most recent 3-5 messages.
2.  **LRU Cache**: Frequently accessed context swapped in from long-term memory.
3.  **Long-term**: Chunks and embeds responses into ChromaDB for semantic retrieval.
4.  **Personalized**: Analyzes user behavior and explicitly stated preferences to tailor AI responses.

### Prompt Construction
Using LangChain and LangGraph, the pipeline:
1.  Classifies the user input (type, complexity).
2.  Retrieves relevant context (Short-term, LRU, or Long-term via semantic search).
3.  Scores and orders the context.
4.  Injects personalized memory (tone, length, interests).
5.  Selects an appropriate output schema (e.g., factual, procedural, coding).
6.  Checks if web search is needed.
7.  If web search is needed, it will search the web and add the results to the context.
8.  Instructs the model to produce a **chain-of-thought ToDo plan** (`<TRACE>…</TRACE>`) before answering and streams the thinking logs.
9.  Applies **adaptive response** instructions based on detected question complexity.
10. Streams the final response.

### Chat Trace Logs
Every assistant response includes a collapsible **Thinking Logs** widget before the response starts streaming. This displays the numbered ToDo plan (chain-of-thought) that the model produced before generating its answer. Click to expand and inspect the model's reasoning process.

### Adaptive Response
Response length automatically adapts to question complexity:
| Complexity | Behaviour |
| :--- | :--- |
| **Simple** | Concise, focused answer — a few sentences. |
| **Moderate** | Well-structured with paragraphs, lists, and examples. |
| **Complex** | Comprehensive and thorough — covers all aspects, edge cases, and examples. |

Complexity is auto-detected from signal words (e.g., *"briefly"* → simple, *"in detail"* → complex).

### Auto Chat Naming
New chats start with a generic "Chat N" name. After the first AI response, a background thread automatically renames the chat based on your first question, producing a short 2-3 word title.

## 🛠️ Setup & Installation

1. **Use pip to install the package**
   ```bash
   pip install optichat
   ```
   *Note: Since this package has large dependencies, it will take some time to install.*
   *Note: Installing this package will create the `~/.optichat/` directory and necessary files upon first launch.*

   ```bash
   optichat
   ```
   Use the above command to launch the application in the terminal.
   
5. **Configure AI Models:**
   - Launch the application and navigate to the **Settings** tab.
   - Enter your API keys for Cloud Providers (OpenAI, Anthropic, Gemini).
   - Alternatively, ensure [Ollama](https://ollama.com/) is running locally to auto-detect and use local models.
   - **DISCLAIMER: API models consume a lot of tokens for chats as multiple calls are used for a single response, use local models for longer conversations**
   

## ⌨️ Keyboard Shortcuts

| Shortcut | Action |
| :--- | :--- |
| `Ctrl+Q` | Quit OptiChat and close the layout |
| `Ctrl+R` | Toggle streaming on/off |
| `Ctrl+C` | Cancel current streaming response mid-output |
| `↑ / ↓` | Scroll through input history (previous commands/messages) |
| `Page Up / Page Down` | Scroll the main panel content |

## Disclaimer

- If you are using a local model make sure that ollama is running and models are downloaded.
- The first time you interact with any model it will take 3-5 minutes to load one of the libraries. This will not happen only the first time.
- When you use a local model for the first time in any session it will take 30 seconds to load the model inside the VRAM. This will not happen again in the same session.
- Make sure you have enough VRAM to run local models. (atleast 4-6GB VRAM is recommended for smaller models like gemma3:4B, tinyllama:1.1b)

**I have only tested on Windows so i am not sure how it will work on other OS**
**I am still learning and improving, so please help me with finding bugs and suggest improvements. I would really appreciate it!**  
---
*Developed using Textual, LangChain, and LangGraph.*
