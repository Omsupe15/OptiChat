"""OptiChat – Phase 2 UI (Textual 8.x)

Complete TUI layout integrated with DB and Model Connection Layers.
"""


from __future__ import annotations

import asyncio
import json
from datetime import datetime

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import (
    Container,
    Horizontal,
    Vertical,
    VerticalScroll,
)
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import (
    Button,
    Collapsible,
    Footer,
    Input,
    Label,
    LoadingIndicator,
    Markdown,
    Select,
    Static,
    Switch,
    TabbedContent,
    TabPane,
    TextArea,
    Tree,
)

import app.connect_models as cm
from app.connect_models import send_message, send_message_via_pipeline, stream_message
import app.memory as mem
import db.database as db

# ──────────────────────────────────────────────
#  ASCII splash shown on startup
# ──────────────────────────────────────────────
SPLASH_ART = r"""
            ____        __  _ ______ __           __  ______________
           / __ \____  / /_(_) ____// /_  ____ _/ /_ /              \
          / / / / __ \/ __/ / /    / __ \/ __ `/ __//   O    O    O  \
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
    """A single message rendered as Markdown inside the conversation."""

    def __init__(self, role: str, content: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.role = role
        self.content = content
        self.add_class(f"msg-{role}")

    def compose(self) -> ComposeResult:
        label = "You" if self.role == "user" else "OptiChat"
        yield Static(f"[bold]{label}[/bold]", classes="msg-role-label")
        yield Markdown(self.content, classes="msg-body")


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

    def add_message(self, role: str, content: str) -> None:
        container = self.query_one("#chat-messages", VerticalScroll)
        msg = ChatMessage(role, content)
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


# ──────────────────────────────────────────────
#  Help Section
# ──────────────────────────────────────────────
class HelpSection(VerticalScroll):
    """Collapsible FAQ / documentation."""

    def compose(self) -> ComposeResult:
        yield Label("📖  Help & Documentation", classes="settings-section-title")

        with Collapsible(title="Getting Started", collapsed=False):
            yield Markdown(
                "## Getting Started\n\n"
                "1. **Add an API key** – Go to *Settings → AI Models → Cloud Models* "
                "and add your provider API key.\n"
                "2. **Select a model** – Pick a default model from the dropdown.\n"
                "3. **Start chatting** – Click *＋ New Chat* in the sidebar.\n\n"
                "You can also use **local models** via Ollama if you have it installed."
            )

        with Collapsible(title="Keyboard Shortcuts"):
            yield Markdown(
                "| Shortcut | Action |\n"
                "|---|---|\n"
                "| `Ctrl+Q` | Quit OptiChat |\n"
                "| `Ctrl+R` | Toggle streaming |\n"
                "| `Ctrl+C` | Cancel current response |\n"
                "| `↑ / ↓` | Scroll input history |\n"
                "| `Page Up / Down` | Scroll main panel |"
            )

        with Collapsible(title="Memory System"):
            yield Markdown(
                "OptiChat uses a **three-tier memory** system:\n\n"
                "- **Short-term** – rolling window of recent messages (≤ 2 500 tokens)\n"
                "- **Summarised** – compressed rolling summary\n"
                "- **Long-term** – vector-embedded past responses (ChromaDB)\n\n"
                "Your **personalized memory** (tone, interests, dislikes) is stored as JSON "
                "and can be edited in *Settings → Memory*."
            )

        with Collapsible(title="Troubleshooting"):
            yield Markdown(
                "- **API key invalid** – double-check in Settings.\n"
                "- **Ollama not detected** – ensure Ollama is running (`ollama serve`).\n"
                "- **High token usage** – older messages are pruned automatically; "
                "check the summary in your chat folder.\n"
                "- **App won't start** – delete `~/.optichat/config.json` and relaunch."
            )

        with Collapsible(title="About OptiChat"):
            yield Markdown(
                "**OptiChat** is an intelligent terminal-based AI chat client.\n\n"
                "It features adaptive memory, structured output schemas, "
                "multi-provider support, and a beautiful TUI."
            )


# ══════════════════════════════════════════════
#  Main Application
# ══════════════════════════════════════════════
class OptiChatApp(App):
    """OptiChat – the intelligent terminal AI chat client."""

    TITLE = "OptiChat"
    SUB_TITLE = "Intelligent AI Chat"
    CSS_PATH = "style.tcss"

    BINDINGS = [
        Binding("ctrl+q", "quit_app", "Quit", show=True, priority=True),
        Binding("ctrl+r", "toggle_streaming", "Toggle Streaming", show=True),
        Binding("ctrl+c", "cancel_response", "Cancel Response", show=True),
        Binding("pageup", "scroll_up", "Scroll Up", show=False),
        Binding("pagedown", "scroll_down", "Scroll Down", show=False),
    ]

    show_splash: reactive[bool] = reactive(True)
    active_chat_id: str | None = None

    # ── Compose ──────────────────────────────
    def compose(self) -> ComposeResult:
        # Splash overlay
        yield Static(SPLASH_ART, id="splash-screen")

        # Custom header
        yield HeaderBar(id="header-bar")

        # Main body: Tabs at top, content below
        with TabbedContent(id="main-tabs"):
            with TabPane("Chats", id="tab-chats"):
                with Horizontal(id="chats-body"):
                    yield ChatSidebar(id="chat-sidebar")
                    yield Container(
                        WelcomePanel(id="welcome-panel"),
                        ChatWindow(id="chat-window"),
                        id="chat-main-area",
                    )

            with TabPane("Settings", id="tab-settings"):
                with VerticalScroll(id="settings-scroll"):
                    yield CloudModelsPane(id="cloud-models-pane")
                    yield LocalModelsPane(id="local-models-pane")
                    yield ThemePane(id="theme-pane")
                    yield MemoryPane(id="memory-pane")

            with TabPane("Help", id="tab-help"):
                yield HelpSection(id="help-section")

        # Custom footer
        yield FooterBar(id="footer-bar")

        # Textual built-in footer for key bindings
        yield Footer()

    # ── Lifecycle ────────────────────────────
    def on_mount(self) -> None:
        # Phase 2: Bootstrap DB and Env
        db.bootstrap()
        db.load_env_into_process()

        # Hide chat window until a chat is opened
        self.query_one("#chat-window", ChatWindow).display = False

        # Phase 3: Update footer with active model info
        default_model = db.get_default_model()
        if default_model:
            footer = self.query_one("#footer-bar", FooterBar)
            footer.model_name = default_model
            footer.model_status = "Active"
        
        # Initialize UI state
        self._apply_config()
        self._refresh_providers_and_models()
        self._load_sidebar_chats()

        # Show splash, then hide after 3 seconds
        self._dismiss_splash()

    @work
    async def _dismiss_splash(self) -> None:
        await asyncio.sleep(3)
        splash = self.query_one("#splash-screen", Static)
        splash.display = False
        self.show_splash = False

    def _apply_config(self) -> None:
        cfg = db.load_config()
        theme_val = cfg.get("theme", "dark")
        self.theme = "textual-light" if theme_val == "light" else "textual-dark"
        self.query_one("#theme-select", Select).value = theme_val

    def _refresh_providers_and_models(self) -> None:
        """Fetch models from active providers and update Dropdowns."""
        providers = db.get_all_saved_providers()
        saved_str = "\n".join(f"• {p.title()}" for p in providers) if providers else "No API keys configured yet."
        self.query_one("#saved-providers-list", Static).update(saved_str)

        all_models = []
        for p in providers:
            api_key = db.get_api_key(p)
            if api_key:
                all_models.extend(cm.list_cloud_models(p, api_key))

        ollama_models = cm.detect_ollama_models()
        all_models.extend(ollama_models)

        # Update lists
        options = [(m["name"], m["id"]) for m in all_models]
        
        default_select = self.query_one("#default-model-select", Select)
        default_select.set_options(options)
        
        default_model = db.get_default_model()
        if default_model and default_model in [opt[1] for opt in options]:
            default_select.value = default_model

        chat_select = self.query_one("#chat-model-select", Select)
        chat_select.set_options(options)

    def _load_sidebar_chats(self) -> None:
        tree = self.query_one("#chat-tree", Tree)
        tree.root.collapse()
        tree.clear()
        
        chats = db.list_chats()
        for chat in chats:
            tree.root.add_leaf(f"💬 {chat['name']}", data=chat["id"])
        tree.root.expand()

        header = self.query_one("#header-bar", HeaderBar)
        header.active_chats = len(chats)

    # ── Actions ──────────────────────────────
    def action_toggle_streaming(self) -> None:
        self.notify("Streaming toggled", title="Streaming")

    def action_cancel_response(self) -> None:
        chat_window = self.query_one("#chat-window", ChatWindow)
        chat_window.show_loading(False)
        self.notify("Response cancelled", severity="warning")

    @work
    async def action_quit_app(self) -> None:
        """Update personalized memory on session close, then quit."""
        if self.active_chat_id:
            try:
                confirmations = await mem.on_session_close(self.active_chat_id)
                for c in confirmations:
                    self.notify(c, title="Memory")
            except Exception:
                pass
        self.exit()

    # ── Chat management ──────────────────────
    @on(Button.Pressed, "#btn-new-chat")
    def _on_new_chat(self) -> None:
        chats = db.list_chats()
        name = f"Chat {len(chats) + 1}"
        # Ensure unique name
        while db.get_chat_by_name(name):
            name += "_"

        default_model = db.get_default_model()
        chat_id = db.create_chat(name=name, model_id=default_model)

        self._load_sidebar_chats()
        self._open_chat(chat_id, name)
        self.notify(f"Created: {name}", title="New Chat")

    @on(Tree.NodeSelected, "#chat-tree")
    def _on_tree_select(self, event: Tree.NodeSelected) -> None:
        if event.node.data:
            chat_id = str(event.node.data)
            chat_data = db.get_chat_by_id(chat_id)
            if chat_data:
                self._open_chat(chat_id, chat_data["name"])

    def _open_chat(self, chat_id: str, name: str) -> None:
        self.active_chat_id = chat_id
        
        welcome = self.query_one("#welcome-panel", WelcomePanel)
        welcome.display = False
        chat_window = self.query_one("#chat-window", ChatWindow)
        chat_window.display = True
        chat_window.chat_name = name

        # Update specific model dropdown for this chat
        chat_data = db.get_chat_by_id(chat_id)
        if chat_data and chat_data.get("model_id"):
            model_select = self.query_one("#chat-model-select", Select)
            try:
                model_select.value = chat_data["model_id"]
            except Exception:
                pass # Dropdown might not have this model currently

        # Load messages
        messages = db.get_messages(chat_id)
        container = self.query_one("#chat-messages", VerticalScroll)
        # Clear existing messages safely
        for child in list(container.children):
            child.remove()
            
        for msg in messages:
            container.mount(ChatMessage(msg["role"], msg["content"]))
        container.scroll_end(animate=False)

    @on(Button.Pressed, "#btn-delete-chat")
    def _on_delete_chat_pressed(self) -> None:
        if not self.active_chat_id:
            return

        def check_delete(confirm: bool) -> None:
            if confirm and self.active_chat_id:
                db.delete_chat(self.active_chat_id)
                self.notify("Chat deleted.", title="Success")
                self.active_chat_id = None
                self.query_one("#chat-window", ChatWindow).display = False
                self.query_one("#welcome-panel", WelcomePanel).display = True
                self._load_sidebar_chats()

        self.push_screen(ConfirmDeleteScreen(), check_delete)

    @on(Select.Changed, "#chat-model-select")
    def _on_chat_model_change(self, event: Select.Changed) -> None:
        if self.active_chat_id and event.value and event.value != Select.BLANK:
            db.update_chat_model(self.active_chat_id, str(event.value))

    # ── Sending messages ─────────────────────
    @on(Button.Pressed, "#btn-send")
    def _on_send_click(self) -> None:
        self._send_message()

    @on(Input.Submitted, "#chat-input")
    def _on_input_submit(self) -> None:
        self._send_message()

    def _send_message(self) -> None:
        inp = self.query_one("#chat-input", Input)
        text = inp.value.strip()
        if not text or not self.active_chat_id:
            return

        chat_window = self.query_one("#chat-window", ChatWindow)
        chat_window.add_message("user", text)
        inp.value = ""

        chat_window.show_loading(True)
        self._get_ai_response(self.active_chat_id, text)

    @work
    async def _get_ai_response(self, chat_id: str, user_text: str) -> None:
        """Run the user message through the Phase 4 prompt construction pipeline.

        The pipeline handles:
        - Question classification and schema detection
        - Memory retrieval (short-term, LRU, long-term)
        - Relevance scoring and filtering
        - Personalization injection
        - Final prompt assembly
        - LLM invocation
        - Post-processing (DB storage + memory updates)
        """
        # Determine which model to use for this chat
        chat_data = db.get_chat_by_id(chat_id)
        model_id = (chat_data or {}).get("model_id") or db.get_default_model()
        chat_name = (chat_data or {}).get("name", "unknown")

        if not model_id:
            reply = "*No model selected.* Please choose a model from the dropdown or set a default in Settings."
            db.add_message(chat_id, "assistant", reply)
            if self.active_chat_id == chat_id:
                cw = self.query_one("#chat-window", ChatWindow)
                cw.show_loading(False)
                cw.add_message("assistant", reply)
            return

        try:
            reply = await send_message_via_pipeline(
                model_id=model_id,
                user_input=user_text,
                chat_name=chat_name,
                chat_id=chat_id,
            )
        except Exception as exc:
            reply = f"*Error communicating with model:* `{exc}`"
            # Pipeline failed; persist the error reply manually
            db.add_message(chat_id, "user", user_text)
            db.add_message(chat_id, "assistant", reply)

        # Update footer token count
        footer = self.query_one("#footer-bar", FooterBar)
        footer.tokens_used += mem.estimate_tokens(user_text) + mem.estimate_tokens(reply)

        # Update footer model info
        footer.model_name = model_id
        footer.model_status = "Active"

        # Render if user is still viewing this chat
        if self.active_chat_id == chat_id:
            chat_window = self.query_one("#chat-window", ChatWindow)
            chat_window.show_loading(False)
            chat_window.add_message("assistant", reply)

    # ── Settings handlers ────────────────────
    @on(Button.Pressed, "#btn-save-api-key")
    def _on_save_api_key(self) -> None:
        provider_select = self.query_one("#provider-select", Select)
        key_input = self.query_one("#api-key-input", Input)

        if provider_select.value == Select.BLANK:
            self.notify("Please select a provider first.", severity="error")
            return
        if not key_input.value.strip():
            self.notify("API key cannot be empty.", severity="error")
            return

        provider = str(provider_select.value)
        api_key = key_input.value.strip()
        
        self._validate_and_save_key(provider, api_key)

    @work
    async def _validate_and_save_key(self, provider: str, api_key: str) -> None:
        self.notify(f"Validating {provider.title()} API Key...", title="Validation")
        is_valid = await asyncio.to_thread(cm.validate_api_key, provider, api_key)
        
        def update_ui():
            if is_valid:
                db.save_api_key(provider, api_key)
                self.notify("API key valid and saved securely.", title="Success", severity="information")
                self._refresh_providers_and_models()
                self.query_one("#api-key-input", Input).value = ""
            else:
                self.notify("Invalid API key.", title="Error", severity="error")
                
        self.call_from_thread(update_ui)

    @on(Button.Pressed, "#btn-detect-ollama")
    def _on_detect_ollama(self) -> None:
        models = cm.detect_ollama_models()
        list_container = self.query_one("#ollama-model-list", Static)
        if models:
            lines = [f"• {m['name']} ({m.get('size', '?')})" for m in models]
            list_container.update("\n".join(lines))
            self.notify(f"Found {len(models)} local models.", title="Ollama")
            self._refresh_providers_and_models()
        else:
            list_container.update("No local models detected.")
            self.notify("Ensure Ollama is running.", severity="warning")

    @on(Select.Changed, "#default-model-select")
    def _on_default_model_change(self, event: Select.Changed) -> None:
        if event.value and event.value != Select.BLANK:
            db.set_default_model(str(event.value))
            self.notify(f"Default model set to {event.value}", title="Settings")

    @on(Select.Changed, "#theme-select")
    def _on_theme_change(self, event: Select.Changed) -> None:
        if event.value:
            value = str(event.value)
            self.theme = "textual-light" if value == "light" else "textual-dark"
            cfg = db.load_config()
            cfg["theme"] = value
            db.save_config(cfg)
            self.notify(f"Theme changed to {value}", title="Theme")

    @on(Button.Pressed, "#btn-save-memory")
    def _on_save_memory(self) -> None:
        editor = self.query_one("#memory-editor", TextArea)
        msg_label = self.query_one("#memory-validation-msg", Static)
        try:
            data = json.loads(editor.text)
            mem.save_personalized_memory(data)
            msg_label.update("[green]✓ Valid JSON – saved![/green]")
            self.notify("Personalized memory saved.", title="Memory")
        except json.JSONDecodeError as exc:
            msg_label.update(f"[red]✗ Invalid JSON: {exc}[/red]")
            self.notify("Invalid JSON – fix errors first.", severity="error")

    @on(Switch.Changed, "#memory-toggle")
    def _on_memory_toggle(self, event: Switch.Changed) -> None:
        state = "enabled" if event.value else "disabled"
        cfg = db.load_config()
        cfg["memory_enabled"] = event.value
        db.save_config(cfg)
        self.notify(f"Personalized memory {state}.", title="Memory")


# ──────────────────────────────────────────────
#  Entry-point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    app = OptiChatApp()
    app.run()
