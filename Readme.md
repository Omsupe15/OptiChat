# OptiChat

OptiChat is an advanced terminal-based chat application built with Python and Textual. It features a robust multi-tier memory system, personalized memory tracking, dynamic model connectivity (including cloud and local Ollama models), and a sophisticated prompt construction pipeline for high-quality, contextual AI responses.

## 🌟 Key Features

*   **Terminal-based UI**: A beautiful, responsive interface built with Textual, featuring tabs, chat session sidebars, and customizable themes.
*   **Multi-Tier Memory System**:
    *   **Short-Term Memory**: Token-budgeted rolling window for recent context.
    *   **LRU Memory**: Background-processed cache of frequently used messages.
    *   **Long-Term Memory**: Persistent vector store (ChromaDB) for semantic search across conversations.
    *   **Personalized Memory**: Automatically learns and updates user preferences, interests, and interaction styles with conflict resolution.
*   **Dynamic Model Connectivity**: Support for OpenAI, Anthropic, Gemini, and local models via Ollama.
*   **Prompt Construction Pipeline**: Utilizes LangGraph to dynamically classify queries, retrieve memory, apply personalization, and enforce structured output schemas.
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
6.  Streams the final response.

## 🛠️ Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd OptiChat
   ```

2. **Create a virtual environment (optional but recommended):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run OptiChat:**
   ```bash
   python main.py # runs in terminal 
   textual run --dev main.py # runs in textual UI (Slower startup)
   ```
   *Note: OptiChat will automatically create the `~/.optichat/` directory and necessary files upon first launch.*

5. **Configure AI Models:**
   - Launch the application and navigate to the **Settings** tab.
   - Enter your API keys for Cloud Providers (OpenAI, Anthropic, Gemini).
   - Alternatively, ensure [Ollama](https://ollama.com/) is running locally to auto-detect and use local models.
   **DISCLAIMER: API models consume a lot of tokens for chats as multiple calls are used for a single response, use local models for longer conversations**

## ⌨️ Keyboard Shortcuts

| Shortcut | Action |
| :--- | :--- |
| `Ctrl+Q` | Quit OptiChat and close the layout |
| `Ctrl+R` | Toggle streaming on/off |
| `Ctrl+C` | Cancel current streaming response mid-output |
| `↑ / ↓` | Scroll through input history (previous commands/messages) |
| `Page Up / Page Down` | Scroll the main panel content |

## 🚀 Development Roadmap

OptiChat is developed in structured phases:

*   **Phase 1: UI Design via Textual** - Building the responsive terminal interface, navigation, settings panels for API keys and themes, and chat windows.
*   **Phase 2: Core Backend & Model Connectivity** - Initializing the `~/.optichat/` environment, implementing SQLite for chat history, and connecting to Cloud/Local AI models using LangChain.
*   **Phase 3: Memory Storing Mechanism** - Implementing the background threads for Short-Term, LRU, and Long-Term (ChromaDB) memory handling, along with personalized memory updates.
*   **Phase 4: Prompt Construction Pipeline** - Orchestrating the advanced LangGraph pipeline for query classification, semantic retrieval, schema enforcement, and intelligent prompt assembly.

---
*Developed using Textual, LangChain, and LangGraph.*
