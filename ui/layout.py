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
from app.pipeline import stream_pipeline
from app.pipeline_functions import StreamDone
import app.memory as mem
import db.database as db
import ui.help as help
from ui.layout_assets import (
    ChatMessage,
    ChatWindow,
    ChatSidebar,
    HeaderBar,
    FooterBar,
    ConfirmDeleteScreen,
    StreamingChatMessage,
    WebSearchToggle,
    SPLASH_ART
)
from ui.layout_assets import WelcomePanel
from ui.layout_assets import (
    CloudModelsPane,
    LocalModelsPane,
    ThemePane,
    MemoryPane,
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
    streaming_enabled: reactive[bool] = reactive(True)
    websearch_enabled: reactive[bool] = reactive(False)  # Phase 5: web search toggle
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
                yield help.HelpSection(id="help-section")

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
        self.streaming_enabled = not self.streaming_enabled
        state = "ON" if self.streaming_enabled else "OFF"
        self.notify(f"Streaming {state}", title="Streaming")

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

        When ``streaming_enabled`` is True the response is streamed token-by-
        token into a live :class:`StreamingChatMessage` widget.  The
        ``<TRACE>`` block is silently consumed during streaming and appended
        as a ``Collapsible`` once the stream finishes.

        When ``streaming_enabled`` is False the full pipeline runs without
        streaming and the complete response is rendered at once.
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

        trace_log = ""
        reply = ""

        if self.streaming_enabled:
            # ── Streaming path ────────────────────────────────────
            chat_window = self.query_one("#chat-window", ChatWindow)
            chat_window.show_loading(False)  # hide spinner; streaming is live

            # Mount a live streaming bubble
            container = self.query_one("#chat-messages", VerticalScroll)
            stream_bubble = StreamingChatMessage()
            await container.mount(stream_bubble)
            container.scroll_end(animate=False)

            try:
                async for item in stream_pipeline(
                    user_input=user_text,
                    chat_name=chat_name,
                    chat_id=chat_id,
                    model_id=model_id,
                    websearch_enabled=self.websearch_enabled,
                ):
                    if isinstance(item, StreamDone):
                        trace_log = item.trace_log
                        reply = item.response
                        if item.error:
                            reply = f"*Error communicating with model:* `{item.error}`"
                        # Finalise the bubble: set full content + add trace
                        await stream_bubble.finish_streaming(trace_log)
                        container.scroll_end(animate=False)
                    else:
                        # Plain token string – append to live bubble
                        stream_bubble.append_token(item)
                        container.scroll_end(animate=False)
            except Exception as exc:
                reply = f"*Error communicating with model:* `{exc}`"
                stream_bubble.append_token(reply)
                await stream_bubble.finish_streaming("")
                db.add_message(chat_id, "user", user_text)
                db.add_message(chat_id, "assistant", reply)
        else:
            # ── Non-streaming (pipeline) path ─────────────────────────
            try:
                result = await send_message_via_pipeline(
                    model_id=model_id,
                    user_input=user_text,
                    chat_name=chat_name,
                    chat_id=chat_id,
                    websearch_enabled=self.websearch_enabled,
                )
                reply = result["response"]
                trace_log = result.get("trace_log", "")
            except Exception as exc:
                reply = f"*Error communicating with model:* `{exc}`"
                db.add_message(chat_id, "user", user_text)
                db.add_message(chat_id, "assistant", reply)

            if self.active_chat_id == chat_id:
                chat_window = self.query_one("#chat-window", ChatWindow)
                chat_window.show_loading(False)
                chat_window.add_message("assistant", reply, trace_log=trace_log)

        # ── Update footer ──────────────────────────────────────────
        footer = self.query_one("#footer-bar", FooterBar)
        footer.tokens_used += mem.estimate_tokens(user_text) + mem.estimate_tokens(reply)
        footer.model_name = model_id
        footer.model_status = "Active"

        # ── Auto-rename chat ───────────────────────────────────────
        current_name = (db.get_chat_by_id(chat_id) or {}).get("name", "")
        if current_name.startswith("Chat "):
            self._auto_rename_chat(chat_id, user_text)

    @work
    async def _auto_rename_chat(self, chat_id: str, first_message: str) -> None:
        """Rename a chat based on the first user message in a background thread."""
        new_name = await asyncio.to_thread(db.auto_rename_chat, chat_id, first_message)
        if new_name:
            # Refresh sidebar and chat title on the main thread
            def _update_ui() -> None:
                self._load_sidebar_chats()
                if self.active_chat_id == chat_id:
                    cw = self.query_one("#chat-window", ChatWindow)
                    cw.chat_name = new_name
            _update_ui()

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
                
        update_ui()

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

    @on(Switch.Changed, "#websearch-toggle")
    def _on_websearch_toggle(self, event: Switch.Changed) -> None:
        """Phase 5: toggle web search on/off for subsequent messages."""
        self.websearch_enabled = event.value
        state = "ON 🌐" if event.value else "OFF"
        self.notify(f"Web Search {state}", title="Web Search")


# ──────────────────────────────────────────────
#  Entry-point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    app = OptiChatApp()
    app.run()
