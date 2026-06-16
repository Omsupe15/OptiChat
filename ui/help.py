from textual import on, work
from textual.app import ComposeResult
from textual.containers import Vertical, VerticalScroll
from textual.widgets import Button, Collapsible, Input, Label, Markdown, Select, Static, Switch, TabbedContent, TabPane

# ──────────────────────────────────────────────
#  Help Section
# ──────────────────────────────────────────────
class HelpSection(VerticalScroll):
    """Collapsible FAQ / documentation covering all OptiChat features."""

    def compose(self) -> ComposeResult:
        yield Label("📖  Help & Documentation", classes="settings-section-title")

        # ── Getting Started ──────────────────
        with Collapsible(title="Getting Started", collapsed=False):
            yield Markdown(
                "## Getting Started\n\n"
                "1. **Add an API key** – Go to *Settings → AI Models → Cloud Models* "
                "and add your provider API key (OpenAI, Anthropic, or Google Gemini).\n"
                "2. **Select a model** – Pick a default model from the dropdown.\n"
                "3. **Start chatting** – Click *＋ New Chat* in the sidebar.\n"
                "4. **Local models** – If Ollama is installed, click *Detect Models* "
                "under *Settings → AI Models → Local Models* to auto-detect local models.\n\n"
                "On first launch, OptiChat automatically creates the `~/.optichat/` directory "
                "with all required folders and configuration files."
            )

        # ── UI Features ──────────────────────
        with Collapsible(title="UI Features"):
            yield Markdown(
                "## User Interface\n\n"
                "- **Header bar** – displays the app name, current date/time, "
                "online status, and active chat count.\n"
                "- **Footer bar** – shows approximate token usage, active model name, "
                "and model status.\n"
                "- **Tabs** – navigate between *Chats*, *Settings*, and *Help*.\n"
                "- **Sidebar** – lists all chat sessions in a tree; click to switch.\n"
                "- **Chat window** – scrollable Markdown conversation with distinct "
                "user/assistant styling, a model selector, and a Delete Chat button.\n"
                "- **Splash screen** – ASCII art shown for a few seconds on startup.\n"
                "- **Themes** – switch between Dark, Light, and System Default "
                "in *Settings → Theme*.\n"
                "- **Auto chat naming** – new chats are automatically renamed "
                "based on your first question (2-3 word title) via a background thread."
            )

        # ── AI Model Connectivity ────────────
        with Collapsible(title="AI Model Connectivity"):
            yield Markdown(
                "## AI Model Connectivity\n\n"
                "- **Cloud providers** – OpenAI, Anthropic, and Google Gemini. "
                "Add your API key in *Settings → Cloud Models*; keys are validated "
                "before saving and stored securely in `~/.optichat/.env`.\n"
                "- **Local models** – Ollama models are auto-detected. Ensure Ollama "
                "is running (`ollama serve`) then click *Detect Models*.\n"
                "- **Per-chat model** – change the model for any chat via the "
                "dropdown in the chat header, even mid-conversation.\n"
                "- **Default model** – set a global default in "
                "*Settings → Cloud Models → Default Model*."
            )

        # ── Memory System ────────────────────
        with Collapsible(title="Memory System"):
            yield Markdown(
                "## Three-Tier Memory System\n\n"
                "OptiChat uses a multi-tier memory architecture that runs "
                "entirely in background threads:\n\n"
                "- **Short-term memory** – rolling window of the most recent 3 "
                "(large) or 5 (small) messages. Oldest messages are dropped "
                "when the limit is hit.\n"
                "- **LRU cache** – stores the most frequently referenced messages "
                "from previous conversations, updated automatically when "
                "short-term overflow occurs.\n"
                "- **Long-term memory** – every assistant response is chunked "
                "(400 tokens, 50 overlap) and embedded into ChromaDB via "
                "sentence-transformers. Retrieved via cosine similarity.\n\n"
                "### Personalized Memory\n\n"
                "- Stores your preferences (tone, response length, interests, "
                "dislikes) as JSON.\n"
                "- **Explicit updates** – detected from phrases like *\"I prefer…\"*, "
                "*\"don't use…\"*, *\"always…\"*.\n"
                "- **Implicit updates** – analysed at session close.\n"
                "- **Conflict resolution** – most-recent-wins for explicit, "
                "frequency-based for implicit; all changes logged.\n"
                "- Editable in *Settings → Memory*; can be toggled on/off."
            )

        # ── Prompt Construction Pipeline ─────
        with Collapsible(title="Prompt Construction Pipeline"):
            yield Markdown(
                "## Prompt Construction Pipeline\n\n"
                "Built with **LangChain** and **LangGraph**, the pipeline runs "
                "every time you send a message:\n\n"
                "1. **Question Classifier** – detects question type, complexity, "
                "and language; checks short-term/LRU for existing context.\n"
                "2. **Schema Classifier** – selects one of 10 output schemas "
                "(factual, how-to, comparison, code, etc.) with depth variants "
                "(quick, standard, detailed).\n"
                "3. **Memory Retrieval** – if local context is insufficient, "
                "retrieves top-5 chunks from ChromaDB via semantic search.\n"
                "4. **Relevance Scoring** – drops chunks below a 0.4 threshold "
                "and sorts the rest by score descending.\n"
                "5. **Personalization** – injects tone, length, interests.\n"
                "6. **WebSearch** - If you have enabled websearch, Optichat will "
                "search the web for relevant information and include it in the "
                "response.\n"
                "7. **Prompt Assembly** – combines all context into the final "
                "prompt template with Chain-of-Thought and Adaptive Response "
                "instructions.\n"
                "8. **LLM Invocation** – sends the prompt, parses the trace.\n"
                "9. **Post-processing** – stores in SQLite, updates short-term "
                "memory, triggers LRU and long-term embedding in background."
            )

        # ── Chat Trace Logs ──────────────────
        with Collapsible(title="Thinking Logs"):
            yield Markdown(
                "## Thinking Logs\n\n"
                "Every assistant response includes a collapsible *Thinking Logs* "
                "section shown when the response is being generated.\n\n"
                "- Click the collapsible to inspect what the model planned "
                "before generating its response.\n"
                "- Useful for **debugging**, **understanding reasoning**, and "
                "**evaluating response quality**."
            )

        # ── Adaptive Response ────────────────
        with Collapsible(title="Adaptive Response"):
            yield Markdown(
                "## Adaptive Response\n\n"
                "Response length and depth dynamically adapt to the detected "
                "complexity of your question:\n\n"
                "| Complexity | Behaviour |\n"
                "|---|---|\n"
                "| **Simple** | Concise, focused answer — a few sentences. |\n"
                "| **Moderate** | Well-structured with paragraphs and examples. |\n"
                "| **Complex** | Comprehensive and thorough — covers all aspects, "
                "edge cases, and examples even if the response is long. |\n\n"
                "Complexity is auto-detected from signal words (e.g. *\"briefly\"* "
                "→ simple, *\"in detail\"* → complex)."
            )

        # ── Keyboard Shortcuts ───────────────
        with Collapsible(title="Keyboard Shortcuts"):
            yield Markdown(
                "| Shortcut | Action |\n"
                "|---|---|\n"
                "| `Ctrl+Q` | Quit OptiChat |\n"
                "| `Ctrl+C` | Cancel current streaming response |\n"
                "| `↑ / ↓` | Scroll through input history |\n"
                "| `Page Up / Down` | Scroll the main panel content |"
            )

        # ── Troubleshooting ──────────────────
        with Collapsible(title="Troubleshooting"):
            yield Markdown(
                "- **API key invalid** – double-check in *Settings → Cloud Models*.\n"
                "- **Ollama not detected** – ensure Ollama is running (`ollama serve`).\n"
                "- **High token usage** – older messages are pruned automatically; "
                "check the short-term file in your chat folder.\n"
                "- **ChromaDB corrupted** – OptiChat will attempt an auto-rebuild "
                "from SQLite message history.\n"
                "- **Context window exceeded** – long-term results are auto-trimmed "
                "with an inline warning.\n"
                "- **Internet lost mid-session** – the API call is retried twice "
                "before surfacing the error.\n"
                "- **SQLite locked** – retried 3× with backoff before showing an error.\n"
                "- **App won't start** – delete `~/.optichat/config.json` and relaunch."
            )

        # ── About ────────────────────────────
        with Collapsible(title="About OptiChat"):
            yield Markdown(
                "**OptiChat** is an intelligent terminal-based AI chat client.\n\n"
                "It features a multi-tier memory system, adaptive response, "
                "chain-of-thought thinking logs, structured output schemas, "
                "multi-provider model support (OpenAI, Anthropic, Gemini, Ollama), "
                "and a beautiful Textual TUI.\n\n"
                "Built with **Textual**, **LangChain**, and **LangGraph**."
            )
