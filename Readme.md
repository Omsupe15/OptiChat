# OptiChat

<img src="https://github.com/Omsupe15/OptiChat/issues/1#issue-4692045509" alt="Alt text" width="500">


OptiChat is an advanced terminal-based AI Chat Optimisation Tool built with primarily using LangChain, LangGraph and Textual. It features a robust multi-tier memory system, personalized memory tracking, dynamic model connectivity, (including cloud and local Ollama models), web search support and a sophisticated prompt construction pipeline for high-quality, contextual AI responses.

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

## Pipeline for Cloud Models

The cloud pipeline is optimized for speed and API cost efficiency, running the entire pre-response phase in **exactly one LLM call**, followed by a second LLM call for response generation.

### Key Stages in Cloud Pipeline:
1. **Cloud Orchestrator Agent (LLM Call 1)**: In a single pass, the cloud model determines complexity, category schema, depth, personalization preferences, and generates the action plan.
2. **Programmatic Memory Search (No LLM)**: Short-term and LRU caches are searched lexically, and ChromaDB is searched semantically. Matching context chunks are sorted, filtered, and appended directly to the final prompt.
3. **Response Generation (LLM Call 2)**: Streams the response tokens to the UI or returns the completed text.
4. **Post-Processing (LLM Call 3)**: Logs token counts, writes message history to the local SQLite database, updates short-term memory, and triggers a periodic preference learning update every 3 message turns.


## Pipeline for Ollama(Local) Models

The local pipeline runs using LangGraph, executing specialized, single-purpose agents sequentially or concurrently to construct the prompt when API costs are not a factor.

### Key Stages in Local Pipeline:
1. **Classifier Agent (LLM Call 1)**: Analyzes the question and decides output formats, personalization, and whether memory/web search should be enabled.
2. **Memory Agent (LLM Call 2 - Optional)**: Dynamically retrieves relevant context from long-term memory (ChromaDB) and uses the local model to select and score the most relevant chunks.
3. **Websearch Agent (LLM Calls 3 & 4 - Optional)**: Uses a Query Planner to generate search phrases, fetches results from DuckDuckGo, uses a Source Ranker to extract key snippets, and queries additional missing facts if necessary.
4. **Prompt Assembly Agent (LLM Call 5)**: Produces the user-visible action plan and constructs the complete prompt block.
5. **Response Generation (LLM Call 6)**: The Ollama local model processes the assembled prompt and streams/returns the response.
6. **Post-Processing (LLM Call 7)**: Logs, updates database messages, short-term/LRU memories, and runs periodic personalized preference updates.

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

## Disclaimer:
* The application has large dependencies, and will take 1-1.5GB of space to install all the required libraries. Please check the requirements.txt file for the list of all the dependencies. Make sure you have enough space on your device before installing the application.
* Running a local model with this application requires atleast 4GB-6GB of VRAM. Also you will be required to download Ollama and a local model differently.
* If you are using a cloud model make sure the model you choose is actually available at a free tier and also make sure the model is not in deprecation phase. I tried using Gemini 2.0 and it was deprecated. You can try Gemini 3.1 flash or Gemini flash 2.5 and their lite versions. For other providers check their respective websites which models are available at a free tier.  
* 3 LLM calls are used for a single response in the cloud pipeline.
* The first launch for the application might take some time.
* This tool is still under development, so there may be bugs and issues.

*Developed using Textual, LangChain, and LangGraph.*
