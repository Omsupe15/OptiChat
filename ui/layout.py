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
from app.pipeline import stream_pipeline, TraceLogChunk
from app.pipeline_functions import StreamDone, preload_ollama_model, unload_ollama_model
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

    # Changes 3: pause / retry / ollama preloading state
    _stop_streaming: bool = False
    _last_user_query: str | None = None
    _loaded_ollama_model: str | None = None

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

        # Changes 3: Preload default Ollama model into VRAM
        if default_model and default_model.startswith("ollama/"):
            self._preload_ollama(default_model)

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
        self._stop_streaming = True
        chat_window = self.query_one("#chat-window", ChatWindow)
        chat_window.show_loading(False)
        chat_window.set_pause_enabled(False)
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
        # Unload any preloaded Ollama model
        if self._loaded_ollama_model:
            try:
                await unload_ollama_model(self._loaded_ollama_model)
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
        model_id = None
        if chat_data and chat_data.get("model_id"):
            model_id = chat_data["model_id"]
            model_select = self.query_one("#chat-model-select", Select)
            try:
                model_select.value = model_id
            except Exception:
                pass # Dropdown might not have this model currently

        # Disable websearch if it is a cloud model
        is_cloud = model_id and model_id.split("/", 1)[0] in ("openai", "anthropic", "gemini")
        chat_window.set_websearch_disabled(bool(is_cloud))
        if is_cloud:
            self.websearch_enabled = False

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
            new_model = str(event.value)
            db.update_chat_model(self.active_chat_id, new_model)
            
            # Disable websearch if it is a cloud model
            is_cloud = new_model.split("/", 1)[0] in ("openai", "anthropic", "gemini")
            chat_window = self.query_one("#chat-window", ChatWindow)
            chat_window.set_websearch_disabled(is_cloud)
            if is_cloud:
                self.websearch_enabled = False

            # Changes 3: preload/unload ollama model on switch
            if new_model.startswith("ollama/"):
                self._preload_ollama(new_model)
            elif self._loaded_ollama_model:
                self._unload_current_ollama()

    # ── Pause / Retry handlers (Changes 3) ────
    @on(Button.Pressed, "#btn-pause")
    def _on_pause_click(self) -> None:
        """Stop the current response generation."""
        self._stop_streaming = True
        chat_window = self.query_one("#chat-window", ChatWindow)
        chat_window.set_pause_enabled(False)
        self.notify("Stopping response...", severity="warning")

    @on(Button.Pressed, "#btn-retry")
    def _on_retry_click(self) -> None:
        """Delete last assistant response and re-send the same user query."""
        if not self.active_chat_id or not self._last_user_query:
            self.notify("Nothing to retry.", severity="warning")
            return

        chat_id = self.active_chat_id
        chat_data = db.get_chat_by_id(chat_id)
        chat_name = (chat_data or {}).get("name", "unknown")

        # Delete last assistant response from SQLite DB
        db.delete_last_message(chat_id, "assistant")

        # Delete last assistant response from short-term memory
        mem.remove_last_from_short_term(chat_name, "assistant")

        # Remove the last message widget from the chat UI
        container = self.query_one("#chat-messages", VerticalScroll)
        children = list(container.children)
        if children:
            children[-1].remove()

        # Disable retry during regeneration and re-send
        chat_window = self.query_one("#chat-window", ChatWindow)
        chat_window.set_retry_enabled(False)
        self._stop_streaming = False
        chat_window.show_loading(True)
        self._get_ai_response(chat_id, self._last_user_query)

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

        # Changes 3: track last query for retry, reset stop flag
        self._last_user_query = text
        self._stop_streaming = False
        chat_window.set_retry_enabled(False)

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

        chat_window = self.query_one("#chat-window", ChatWindow)

        if not model_id:
            reply = "*No model selected.* Please choose a model from the dropdown or set a default in Settings."
            db.add_message(chat_id, "assistant", reply)
            if self.active_chat_id == chat_id:
                chat_window.show_loading(False)
                chat_window.add_message("assistant", reply)
            return

        trace_log = ""
        reply = ""
        stopped_by_user = False

        if self.streaming_enabled:
            # ── Streaming path ────────────────────────────────────
            chat_window.show_loading(False)  # hide spinner; streaming is live
            chat_window.set_pause_enabled(True)  # Changes 3: activate pause

            # Mount a live streaming bubble
            container = self.query_one("#chat-messages", VerticalScroll)
            stream_bubble = StreamingChatMessage()
            await container.mount(stream_bubble)
            container.scroll_end(animate=False)

            response_started = False
            try:
                async for item in stream_pipeline(
                    user_input=user_text,
                    chat_name=chat_name,
                    chat_id=chat_id,
                    model_id=model_id,
                    websearch_enabled=self.websearch_enabled,
                ):
                    # Changes 3: check pause flag
                    if self._stop_streaming:
                        stopped_by_user = True
                        reply = stream_bubble._accumulated or ""
                        if not response_started:
                            stream_bubble.start_response()
                        stream_bubble.append_token("\n\n*Response stopped by user.*")
                        await stream_bubble.finish_streaming()
                        container.scroll_end(animate=False)
                        break

                    if isinstance(item, TraceLogChunk):
                        # Stream trace-log lines ABOVE the response
                        await stream_bubble.append_trace(item.text)
                        container.scroll_end(animate=False)
                    elif isinstance(item, StreamDone):
                        trace_log = item.trace_log
                        reply = item.response
                        if item.error:
                            reply = f"*Error communicating with model:* `{item.error}`"
                            stream_bubble.append_token(reply)
                        elif not stream_bubble._accumulated and reply:
                            stream_bubble.append_token(reply)
                        # Finalise the bubble (trace already shown above)
                        await stream_bubble.finish_streaming()
                        container.scroll_end(animate=False)
                    else:
                        # Plain token string – append to live bubble
                        if not response_started:
                            stream_bubble.start_response()
                            response_started = True
                        stream_bubble.append_token(item)
                        container.scroll_end(animate=False)
            except Exception as exc:
                reply = f"*Error communicating with model:* `{exc}`"
                stream_bubble.append_token(reply)
                await stream_bubble.finish_streaming()
                db.add_message(chat_id, "user", user_text)
                db.add_message(chat_id, "assistant", reply)

            # Changes 3: disable pause after streaming ends
            chat_window.set_pause_enabled(False)

            # If stopped by user, save partial response to DB
            if stopped_by_user and reply:
                partial = reply + "\n\n*Response stopped by user.*"
                db.add_message(chat_id, "user", user_text)
                db.add_message(chat_id, "assistant", partial)
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
                chat_window.show_loading(False)
                chat_window.add_message("assistant", reply, trace_log=trace_log)

        # ── Update footer ──────────────────────────────────────────
        footer = self.query_one("#footer-bar", FooterBar)
        footer.tokens_used += mem.estimate_tokens(user_text) + mem.estimate_tokens(reply)
        footer.model_name = model_id
        footer.model_status = "Active"

        # Changes 3: enable retry after response completes
        chat_window.set_retry_enabled(True)

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

    # ── Ollama preloading (Changes 3) ────────
    @work
    async def _preload_ollama(self, model_id: str) -> None:
        """Preload an Ollama model into VRAM with keep_alive=-1."""
        model_name = model_id.split("/", 1)[1] if "/" in model_id else model_id

        # Unload previous model if different
        if self._loaded_ollama_model and self._loaded_ollama_model != model_name:
            await unload_ollama_model(self._loaded_ollama_model)

        success = await preload_ollama_model(model_name)
        if success:
            self._loaded_ollama_model = model_name
            self.notify(f"Model {model_name} loaded into VRAM", title="Ollama")
        else:
            self.notify(f"Failed to preload {model_name}", severity="warning")

    @work
    async def _unload_current_ollama(self) -> None:
        """Unload the currently loaded Ollama model from VRAM."""
        if self._loaded_ollama_model:
            model_name = self._loaded_ollama_model
            await unload_ollama_model(model_name)
            self._loaded_ollama_model = None
            self.notify(f"Model {model_name} unloaded from VRAM", title="Ollama")

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
            new_model = str(event.value)
            db.set_default_model(new_model)
            self.notify(f"Default model set to {event.value}", title="Settings")
            # Changes 3: preload/unload ollama model on default change
            if new_model.startswith("ollama/"):
                self._preload_ollama(new_model)
            elif self._loaded_ollama_model:
                self._unload_current_ollama()

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
