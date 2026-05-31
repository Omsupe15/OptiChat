from textual.app import App, ComposeResult
from textual.containers import (Container, Horizontal, Vertical, VerticalScroll)
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import (
    Button,Collapsible,
    Footer,Input,
    Label,LoadingIndicator,
    Markdown,Select,Static,Switch,TabbedContent,TabPane,TextArea,Tree)
import app.memory as mem
import json
from datetime import datetime

# ──────────────────────────────────────────────
#  ASCII splash shown on startup
# ──────────────────────────────────────────────
SPLASH_ART = r"""
            ____        __  _ ______ __          __   ______________
           / __ \____  / /_(_) ____// /_  ____ _/ /_ /              \
          / / / / __ \/ __/ / /    / __ \/ __ \/ __//   O    O    O  \
         / /_/ / /_/ / /_/ / /___ / / / / /_/ / /_  \                /
         \____/ .___/\__/_/\____//_/ /_/\__,_/\__/   \__________    /
             /_/                                                \  /
           ----SMART MEMORY OPTIMIZED PERSONALIZED CHATBOT ----  \/
"""

# ──────────────────────────────────────────────
#  Modal Dialogs
# ──────────────────────────────────────────────
class ConfirmDeleteScreen(ModalScreen[bool]):
    """Screen with a dialog to confirm chat deletion."""

    def compose(self) -> ComposeResult:
        with Vertical(id="confirm-dialog"):
            yield Label("Are you sure you want to delete this chat?", id="confirm-label")
            with Horizontal(id="confirm-buttons"):
                yield Button("Yes", id="btn-yes", variant="error")
                yield Button("No", id="btn-no", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "btn-yes")


# ──────────────────────────────────────────────
#  Custom Header Bar
# ──────────────────────────────────────────────
class HeaderBar(Horizontal):
    """Custom top bar: app name · date/time · status · active chats."""

    current_time: reactive[str] = reactive("")
    status: reactive[str] = reactive("● Online")
    active_chats: reactive[int] = reactive(0)

    def compose(self) -> ComposeResult:
        yield Static("⚡ OptiChat", id="header-app-name")
        yield Static("", id="header-datetime")
        yield Static("", id="header-status")
        yield Static("", id="header-chats-count")

    def on_mount(self) -> None:
        self._update_time()
        self.set_interval(1, self._update_time)

    def _update_time(self) -> None:
        now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        self.query_one("#header-datetime", Static).update(f"📅 {now}")
        self.query_one("#header-status", Static).update(self.status)
        self.query_one("#header-chats-count", Static).update(
            f"💬 Chats: {self.active_chats}"
        )


# ──────────────────────────────────────────────
#  Custom Footer Bar
# ──────────────────────────────────────────────
class FooterBar(Horizontal):
    """Bottom bar: token count · model name · model status."""

    tokens_used: reactive[int] = reactive(0)
    model_name: reactive[str] = reactive("No model selected")
    model_status: reactive[str] = reactive("Inactive")

    def compose(self) -> ComposeResult:
        yield Static("", id="footer-tokens")
        yield Static("", id="footer-model-name")
        yield Static("", id="footer-model-status")

    def on_mount(self) -> None:
        self._refresh_labels()

    def watch_tokens_used(self) -> None:
        self._refresh_labels()

    def watch_model_name(self) -> None:
        self._refresh_labels()

    def watch_model_status(self) -> None:
        self._refresh_labels()

    def _refresh_labels(self) -> None:
        try:
            self.query_one("#footer-tokens", Static).update(
                f"🔢 Tokens: ~{self.tokens_used}"
            )
            self.query_one("#footer-model-name", Static).update(
                f"🤖 {self.model_name}"
            )
            status_icon = "🟢" if self.model_status == "Active" else "🔴"
            self.query_one("#footer-model-status", Static).update(
                f"{status_icon} {self.model_status}"
            )
        except Exception:
            pass


# ──────────────────────────────────────────────
#  Chat Sidebar  (Tree widget for chat list)
# ──────────────────────────────────────────────
class ChatSidebar(Vertical):
    """Sidebar with a Tree listing chat sessions + new chat button."""

    def compose(self) -> ComposeResult:
        yield Button("＋ New Chat", id="btn-new-chat", variant="success")
        yield Label("Chat Sessions", id="sidebar-title")
        tree: Tree[str] = Tree("Chats", id="chat-tree")
        tree.root.expand()
        yield tree


# ──────────────────────────────────────────────
#  Single chat message bubble
# ──────────────────────────────────────────────
class ChatMessage(Static):
    """A single message rendered as Markdown inside the conversation.

    For assistant messages, an optional *trace_log* can be provided.
    When present, a ``Collapsible`` widget labelled **Chat Trace Logs**
    is rendered below the response body showing the model's chain-of-thought
    ToDo plan.
    """

    def __init__(self, role: str, content: str, trace_log: str = "", **kwargs) -> None:
        super().__init__(**kwargs)
        self.role = role
        self.content = content
        self.trace_log = trace_log
        self.add_class(f"msg-{role}")

    def compose(self) -> ComposeResult:
        label = "You" if self.role == "user" else "OptiChat"
        yield Static(f"[bold]{label}[/bold]", classes="msg-role-label")
        yield Markdown(self.content, classes="msg-body")

        # Show trace logs for assistant messages when available
        if self.role == "assistant" and self.trace_log:
            with Collapsible(title="Chat Trace Logs", collapsed=True, classes="trace-collapsible"):
                yield Markdown(
                    f"**Model's Chain-of-Thought Plan:**\n\n{self.trace_log}",
                    classes="trace-body",
                )


# ──────────────────────────────────────────────
#  Streaming Chat Message bubble
# ──────────────────────────────────────────────
class StreamingChatMessage(Static):
    """Assistant message bubble that supports live token streaming.

    Usage
    -----
    1. Mount this widget into the conversation container.
    2. Call ``append_token(text)`` for each token chunk received from the LLM.
    3. Call ``finish_streaming(trace_log)`` once the stream ends to finalise
       the Markdown content and mount the "Chat Trace Logs" Collapsible.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._accumulated: str = ""
        self.add_class("msg-assistant")

    def compose(self) -> ComposeResult:
        yield Static("[bold]OptiChat[/bold]", classes="msg-role-label")
        yield Markdown("", classes="msg-body", id=f"stream-md-{id(self)}")

    def append_token(self, text: str) -> None:
        """Append *text* to the live Markdown widget."""
        self._accumulated += text
        try:
            md = self.query_one(f"#stream-md-{id(self)}", Markdown)
            md.update(self._accumulated)
        except Exception:
            pass

    async def finish_streaming(self, trace_log: str = "") -> None:
        """Finalize the message.

        Ensures the Markdown shows the complete response text, then mounts
        the "Chat Trace Logs" Collapsible below it when *trace_log* is
        non-empty.  Must be awaited so Textual has time to complete the
        widget mount before the caller continues.
        """
        try:
            md = self.query_one(f"#stream-md-{id(self)}", Markdown)
            md.update(self._accumulated)
        except Exception:
            pass

        if trace_log:
            trace_md = Markdown(
                f"**Model's Chain-of-Thought Plan:**\n\n{trace_log}",
                classes="trace-body",
            )
            collapsible = Collapsible(
                trace_md,
                title="Chat Trace Logs",
                collapsed=True,
                classes="trace-collapsible",
            )
            await self.mount(collapsible)


# ──────────────────────────────────────────────
#  Phase 5: Web Search Toggle widget
# ──────────────────────────────────────────────
class WebSearchToggle(Horizontal):
    """Small inline widget that renders a labelled toggle for web search.

    Placed in the chat window header beside the model-select dropdown.
    The underlying ``Switch`` has id ``"websearch-toggle"`` so the main
    application can listen for ``Switch.Changed`` events from it.
    """

    def compose(self) -> ComposeResult:
        yield Label("🌐 Web Search", classes="websearch-label")
        yield Switch(value=False, id="websearch-toggle", classes="websearch-switch")


# ──────────────────────────────────────────────
#  Chat Window (conversation + input)
# ──────────────────────────────────────────────
class ChatWindow(Vertical):
    """Main chat area: scrollable conversation + input bar."""

    chat_name: reactive[str] = reactive("New Chat")

    def compose(self) -> ComposeResult:
        with Horizontal(id="chat-header"):
            yield Static("", id="chat-window-title")
            yield Select([], prompt="Select Model", id="chat-model-select", allow_blank=True)
            yield WebSearchToggle(id="websearch-toggle-widget")
            yield Button("Delete Chat", id="btn-delete-chat", variant="error")
        yield VerticalScroll(id="chat-messages")
        yield LoadingIndicator(id="chat-loading")
        with Horizontal(id="chat-input-bar"):
            yield Input(
                placeholder="Type your message…",
                id="chat-input",
            )
            yield Button("Send ➤", id="btn-send", variant="primary")

    def on_mount(self) -> None:
        self.query_one("#chat-loading", LoadingIndicator).display = False
        self._update_title()

    def watch_chat_name(self) -> None:
        self._update_title()

    def _update_title(self) -> None:
        try:
            self.query_one("#chat-window-title", Static).update(
                f"[bold]💬 {self.chat_name}[/bold]"
            )
        except Exception:
            pass

    def add_message(self, role: str, content: str, trace_log: str = "") -> None:
        container = self.query_one("#chat-messages", VerticalScroll)
        msg = ChatMessage(role, content, trace_log=trace_log)
        container.mount(msg)
        container.scroll_end(animate=False)

    def show_loading(self, show: bool = True) -> None:
        self.query_one("#chat-loading", LoadingIndicator).display = show


# ──────────────────────────────────────────────
#  Welcome / empty-state panel
# ──────────────────────────────────────────────
class WelcomePanel(Vertical):
    """Shown when no chat is active."""

    def compose(self) -> ComposeResult:
        yield Static(
            "[bold]Welcome to OptiChat![/bold]\n\n"
            "Start a new chat from the sidebar or select an existing one.\n\n"
            "Use the tabs above to explore Settings and Help.",
            id="welcome-text",
        )
        yield Static(SPLASH_ART,id="splash-screen")


# ──────────────────────────────────────────────
#  Settings — AI Models  (Cloud)
# ──────────────────────────────────────────────
class CloudModelsPane(Vertical):
    """Cloud provider API-key management."""

    def compose(self) -> ComposeResult:
        yield Label("☁️  Cloud Models", classes="settings-section-title")
        yield Label(
            "Add API keys for cloud providers to access their models.",
            classes="settings-desc",
        )

        with Horizontal(classes="form-row"):
            yield Select(
                [
                    ("OpenAI", "openai"),
                    ("Anthropic", "anthropic"),
                    ("Google Gemini", "gemini"),
                ],
                prompt="Select Provider",
                id="provider-select",
            )

        with Horizontal(classes="form-row"):
            yield Input(
                placeholder="Paste your API key here…",
                password=True,
                id="api-key-input",
            )
            yield Button("Save Key", id="btn-save-api-key", variant="success")

        yield Label("Saved Providers", classes="settings-subsection-title")
        yield VerticalScroll(
            Static("No API keys configured yet.", id="saved-providers-list"),
            id="saved-providers-scroll",
        )

        yield Label("Default Model", classes="settings-subsection-title")
        yield Select(
            [],
            prompt="Select default model",
            id="default-model-select",
            allow_blank=True,
        )


# ──────────────────────────────────────────────
#  Settings — AI Models  (Local / Ollama)
# ──────────────────────────────────────────────
class LocalModelsPane(Vertical):
    """Local Ollama model listing."""

    def compose(self) -> ComposeResult:
        yield Label("🖥️  Local Models (Ollama)", classes="settings-section-title")
        yield Label(
            "If Ollama is installed, available local models appear below.",
            classes="settings-desc",
        )
        yield Button("🔄 Detect Models", id="btn-detect-ollama", variant="default")
        yield VerticalScroll(
            Static(
                "No local models detected. Click Detect Models.",
                id="ollama-model-list",
            ),
            id="ollama-scroll",
        )


# ──────────────────────────────────────────────
#  Settings — Theme
# ──────────────────────────────────────────────
class ThemePane(Vertical):
    """Theme selection."""

    def compose(self) -> ComposeResult:
        yield Label("🎨  Theme", classes="settings-section-title")
        yield Label("Choose your preferred appearance.", classes="settings-desc")
        yield Select(
            [
                ("Dark", "dark"),
                ("Light", "light"),
                ("System Default", "system"),
            ],
            value="dark",
            id="theme-select",
        )


# ──────────────────────────────────────────────
#  Settings — Memory
# ──────────────────────────────────────────────
class MemoryPane(Vertical):
    """Personalized memory viewer / editor."""

    def compose(self) -> ComposeResult:
        yield Label("🧠  Personalized Memory", classes="settings-section-title")
        yield Label(
            "Edit the JSON structure that shapes your AI interactions.",
            classes="settings-desc",
        )

        with Horizontal(classes="switch-row"):
            yield Label("Enable personalized memory")
            yield Switch(value=True, id="memory-toggle")

        yield TextArea(
            "",
            language="json",
            id="memory-editor",
        )

        with Horizontal(classes="form-row"):
            yield Button("💾 Save Memory", id="btn-save-memory", variant="success")
            yield Static("", id="memory-validation-msg")

    def on_mount(self) -> None:
        """Load the current personalized memory from disk into the editor."""
        current = mem.load_personalized_memory()
        editor = self.query_one("#memory-editor", TextArea)
        editor.load_text(json.dumps(current, indent=2))