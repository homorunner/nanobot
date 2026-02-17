"""Browser control tools (Playwright).

Optional extra: pip install nanobot-ai[browser] && playwright install chromium
"""

import asyncio
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from loguru import logger

from nanobot.agent.tools.base import Tool

# Chromium launch flags (suppress automation detection banner)
_LAUNCH_ARGS = ["--disable-blink-features=AutomationControlled"]

# WSL2 with homebrew: Chromium needs libasound.so.2 from linuxbrew if not in LD_LIBRARY_PATH
_LINUXBREW_LIB = "/home/linuxbrew/.linuxbrew/lib"


def _ensure_chromium_libs() -> None:
    """Add linuxbrew lib path to LD_LIBRARY_PATH if Chromium would otherwise fail in WSL2."""
    if not Path(_LINUXBREW_LIB).is_dir():
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if _LINUXBREW_LIB not in current:
        os.environ["LD_LIBRARY_PATH"] = f"{_LINUXBREW_LIB}:{current}" if current else _LINUXBREW_LIB


# Truncate browser_content output to keep token usage reasonable
_CONTENT_MAX_CHARS = 6000

# Mimic a regular browser to avoid bot-detection on common sites
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_2) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_EXTRA_HEADERS = {
    "Sec-CH-UA": '"Not A(Brand";v="24", "Chromium";v="120", "Google Chrome";v="120"',
    "Accept-Language": "en-US,en;q=0.9",
}

try:
    from playwright.async_api import async_playwright, Browser, BrowserContext, Page
    _PLAYWRIGHT_AVAILABLE = True
except ImportError:
    _PLAYWRIGHT_AVAILABLE = False
    Browser = None  # type: ignore[misc, assignment]
    BrowserContext = None  # type: ignore[misc, assignment]
    Page = None  # type: ignore[misc, assignment]


def _validate_url(url: str) -> tuple[bool, str]:
    """Return (True, '') if url is a valid http/https URL, else (False, reason)."""
    try:
        p = urlparse(url)
        if p.scheme not in ("http", "https"):
            return False, f"Only http/https allowed, got '{p.scheme or 'none'}'"
        if not p.netloc:
            return False, "Missing domain"
        return True, ""
    except Exception as e:
        return False, str(e)


def _resolve_proxy(proxy_server: str) -> str:
    """Return the effective proxy URL: explicit config, then HTTPS_PROXY/HTTP_PROXY env."""
    return proxy_server.strip() or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY") or ""


def _storage_state_path(config: Any, workspace: Path | None) -> str:
    """Return the resolved storage state file path."""
    explicit = (getattr(config, "storage_state_path", None) or "").strip()
    if explicit:
        return str(Path(explicit).expanduser())
    if workspace is None:
        return ""
    return str(workspace / "browser" / "cookie.json")


class BrowserSession:
    """Shared Playwright browser/page session for browser tools.

    Launches headless Chromium locally. Cookies are saved to storage_state_path
    so logins persist across restarts.
    """

    def __init__(
        self,
        headless: bool = True,
        timeout_ms: int = 30000,
        proxy_server: str = "",
        storage_state_path: str = "",
    ) -> None:
        self._headless = headless
        self._timeout_ms = timeout_ms
        self._proxy_server = proxy_server.strip()
        self._storage_state_path = storage_state_path.strip()
        self._playwright = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._page: Page | None = None

    async def get_page(self) -> Page:
        """Return the current page, launching Chromium if needed."""
        if self._page is not None:
            return self._page
        if not _PLAYWRIGHT_AVAILABLE:
            raise RuntimeError(
                "Playwright is not installed. Run: pip install nanobot-ai[browser] && playwright install"
            )
        # Must be set before async_playwright().start() so the Node.js driver inherits it
        _ensure_chromium_libs()
        self._playwright = await async_playwright().start()

        proxy = _resolve_proxy(self._proxy_server)
        logger.info("Browser: launching (headless={}, proxy={})", self._headless, proxy or "none")
        try:
            self._browser = await self._playwright.chromium.launch(
                headless=self._headless, args=_LAUNCH_ARGS,
            )
        except Exception as e:
            if "Executable doesn't exist" in str(e):
                raise RuntimeError(
                    "Chromium browser not found. Run: playwright install chromium"
                ) from e
            raise

        ctx_opts: dict[str, Any] = {
            "user_agent": _USER_AGENT,
            "viewport": {"width": 1280, "height": 720},
            "extra_http_headers": _EXTRA_HEADERS,
        }
        if proxy:
            ctx_opts["proxy"] = {"server": proxy}
        if self._storage_state_path and Path(self._storage_state_path).exists():
            ctx_opts["storage_state"] = self._storage_state_path
            logger.debug("Browser: loading storage state from {}", self._storage_state_path)
        try:
            self._context = await self._browser.new_context(**ctx_opts)
        except Exception as e:
            if ctx_opts.pop("storage_state", None):
                logger.warning("Browser: storage state load failed, starting clean: {}", e)
                self._context = await self._browser.new_context(**ctx_opts)
            else:
                raise

        self._page = await self._context.new_page()
        self._page.set_default_timeout(self._timeout_ms)
        return self._page

    async def reconfigure(self, headless: bool) -> None:
        """Close the current session and update headless mode for the next get_page() call."""
        if self._page is not None:
            await self.close()
        self._headless = headless

    async def save_storage_state(self) -> tuple[bool, str]:
        """Save cookies and storage to disk."""
        if not self._storage_state_path:
            return False, "No storage state path configured."
        if self._context is None:
            return False, "Browser not started yet."
        try:
            path = Path(self._storage_state_path).resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            await self._context.storage_state(path=str(path))
            path.chmod(0o600)
            logger.info("Browser: storage state saved to {}", path)
            return True, f"Saved to {path}"
        except Exception as e:
            return False, f"Save failed: {e}"

    async def close(self) -> None:
        """Close Chromium, saving storage state first."""
        if self._context and self._storage_state_path:
            try:
                path = Path(self._storage_state_path).resolve()
                path.parent.mkdir(parents=True, exist_ok=True)
                await self._context.storage_state(path=str(path))
                path.chmod(0o600)
            except Exception as e:
                logger.warning("Browser: storage state save on close failed: {}", e)

        if self._browser:
            await self._browser.close()
            self._browser = None
        self._context = None
        self._page = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None


# ── Tools ────────────────────────────────────────────────────────────────────


class BrowserNavigateTool(Tool):
    """Navigate the browser to a URL."""

    def __init__(self, session: BrowserSession) -> None:
        self._session = session

    @property
    def name(self) -> str:
        return "browser_navigate"

    @property
    def description(self) -> str:
        return (
            "Navigate the browser to a URL. Prefer web_fetch for simple read-only pages. "
            "Use browser for JS-rendered pages or when login is required. "
            "Set headless=false to watch the browser window live. "
            "After navigating, call browser_snapshot or browser_content to read the page."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL to open (http or https only)",
                },
                "headless": {
                    "type": "boolean",
                    "description": (
                        "Run without visible UI (true, default) or show the browser window (false). "
                        "Use false to watch the agent browse live. Persists for the session."
                    ),
                },
            },
            "required": ["url"],
        }

    async def execute(self, url: str, headless: bool | None = None, **kwargs: Any) -> str:
        ok, err = _validate_url(url)
        if not ok:
            return f"Error: {err}"

        if headless is not None and headless != self._session._headless:
            await self._session.reconfigure(headless=headless)

        try:
            page = await self._session.get_page()
            await page.goto(url, wait_until="load")
            logger.info("browser_navigate: {}", url)
            if self._session._storage_state_path:
                asyncio.create_task(self._session.save_storage_state())
            return f"Navigated to {url}"
        except Exception as e:
            logger.error("browser_navigate {}: {}: {}", url, type(e).__name__, e)
            return f"Error: {type(e).__name__}: {e}"


class BrowserSnapshotTool(Tool):
    """List interactive elements on the current page with numbered refs."""

    def __init__(self, session: BrowserSession) -> None:
        self._session = session

    @property
    def name(self) -> str:
        return "browser_snapshot"

    @property
    def description(self) -> str:
        return (
            "List interactive elements (buttons, links, inputs) on the current page with "
            "numbered refs. Use refs with browser_click and browser_type."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "max_elements": {
                    "type": "integer",
                    "description": "Max elements to list (default 50, max 200)",
                    "minimum": 1,
                    "maximum": 200,
                },
            },
            "required": [],
        }

    async def execute(self, max_elements: int = 50, **kwargs: Any) -> str:
        try:
            page = await self._session.get_page()
            result = await page.evaluate(
                """(maxEls) => {
                    const sel = 'button, a[href], input:not([type=hidden]), textarea, select, '
                        + '[role=button], [role=link], [role=textbox], [role=searchbox], [contenteditable="true"]';
                    const els = Array.from(document.querySelectorAll(sel)).slice(0, maxEls);
                    els.forEach((el, i) => el.setAttribute('data-nanobot-ref', String(i + 1)));
                    return els.map((el, i) => {
                        const ref = i + 1;
                        const role = el.getAttribute('role') || el.tagName.toLowerCase();
                        const name = el.getAttribute('aria-label') || el.getAttribute('placeholder') ||
                            (el.tagName === 'A' ? (el.textContent || '').trim().slice(0, 50) : '') || null;
                        const type = (el.getAttribute('type') || '').toLowerCase();
                        let label = (type && role === 'input') ? type + ' (input)' : role;
                        if (name) label += " '" + name.replace(/'/g, "\\'").slice(0, 40) + "'";
                        return ref + '. ' + label;
                    }).join('\\n');
                }""",
                min(max(max_elements, 1), 200),
            )
            return result or "No interactive elements found."
        except Exception as e:
            logger.error("browser_snapshot: {}: {}", type(e).__name__, e)
            return f"Error: {type(e).__name__}: {e}"


class BrowserContentTool(Tool):
    """Read the visible text content of the current page."""

    def __init__(self, session: BrowserSession) -> None:
        self._session = session

    @property
    def name(self) -> str:
        return "browser_content"

    @property
    def description(self) -> str:
        return (
            "Read the visible text content of the current page (headings, paragraphs, tables, "
            "error messages). Use after browser_navigate or browser_click to understand the page."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "selector": {
                    "type": "string",
                    "description": "CSS selector to scope content (default: body). E.g. 'main', '#content'",
                },
            },
            "required": [],
        }

    async def execute(self, selector: str = "body", **kwargs: Any) -> str:
        try:
            page = await self._session.get_page()
            sel = selector.strip() or "body"
            text = re.sub(r"\n{3,}", "\n\n", (await page.inner_text(sel)).strip())
            if len(text) > _CONTENT_MAX_CHARS:
                text = text[:_CONTENT_MAX_CHARS] + f"\n\n[...truncated — {len(text)} chars total]"
            return text or "(no text content)"
        except Exception as e:
            logger.error("browser_content: {}: {}", type(e).__name__, e)
            return f"Error: {type(e).__name__}: {e}"


class BrowserClickTool(Tool):
    """Click an element by ref from the last browser_snapshot."""

    def __init__(self, session: BrowserSession) -> None:
        self._session = session

    @property
    def name(self) -> str:
        return "browser_click"

    @property
    def description(self) -> str:
        return "Click an element by ref number from browser_snapshot."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ref": {"type": "integer", "description": "Ref number from browser_snapshot", "minimum": 1},
            },
            "required": ["ref"],
        }

    async def execute(self, ref: int, **kwargs: Any) -> str:
        try:
            page = await self._session.get_page()
            await page.locator(f'[data-nanobot-ref="{ref}"]').first.click()
            return f"Clicked ref {ref}"
        except Exception as e:
            logger.error("browser_click ref={}: {}: {}", ref, type(e).__name__, e)
            return f"Error: {type(e).__name__}: {e}"


class BrowserTypeTool(Tool):
    """Type text into an input by ref. Optionally submit with Enter."""

    def __init__(self, session: BrowserSession) -> None:
        self._session = session

    @property
    def name(self) -> str:
        return "browser_type"

    @property
    def description(self) -> str:
        return (
            "Type text into an input/textarea by ref from browser_snapshot. "
            "Set submit=true to press Enter after typing."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ref": {"type": "integer", "description": "Ref number from browser_snapshot", "minimum": 1},
                "text": {"type": "string", "description": "Text to type"},
                "submit": {"type": "boolean", "description": "Press Enter after typing (default false)"},
            },
            "required": ["ref", "text"],
        }

    async def execute(self, ref: int, text: str, submit: bool = False, **kwargs: Any) -> str:
        try:
            page = await self._session.get_page()
            el = page.locator(f'[data-nanobot-ref="{ref}"]').first
            await el.fill("")
            await el.press_sequentially(text)
            if submit:
                await el.press("Enter")
            return f"Typed into ref {ref}" + (" and pressed Enter." if submit else ".")
        except Exception as e:
            logger.error("browser_type ref={}: {}: {}", ref, type(e).__name__, e)
            return f"Error: {type(e).__name__}: {e}"


class BrowserPressTool(Tool):
    """Press a keyboard key (Enter, Tab, Escape, ArrowDown, etc.)."""

    def __init__(self, session: BrowserSession) -> None:
        self._session = session

    @property
    def name(self) -> str:
        return "browser_press"

    @property
    def description(self) -> str:
        return "Press a key in the browser (Enter, Tab, Escape, ArrowDown, etc.)."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": "Key name (Enter, Tab, Escape, ArrowDown, etc.)"},
            },
            "required": ["key"],
        }

    async def execute(self, key: str, **kwargs: Any) -> str:
        try:
            page = await self._session.get_page()
            k = key.strip() or "Enter"
            await page.keyboard.press(k)
            return f"Pressed {k!r}"
        except Exception as e:
            logger.error("browser_press key={!r}: {}: {}", key, type(e).__name__, e)
            return f"Error: {type(e).__name__}: {e}"


class BrowserSaveSessionTool(Tool):
    """Save browser session cookies to disk."""

    def __init__(self, session: BrowserSession) -> None:
        self._session = session

    @property
    def name(self) -> str:
        return "browser_save_session"

    @property
    def description(self) -> str:
        return (
            "Save the current browser session (cookies, storage) to disk so it persists across restarts. "
            "Call after logging in to a site."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        ok, msg = await self._session.save_storage_state()
        return msg if ok else f"Error: {msg}"


def create_browser_tools(
    config: Any, workspace: Path | None = None
) -> tuple[list[Tool], "BrowserSession | None"]:
    """Create browser tools that share one BrowserSession.

    Returns (tools, session). Returns ([], None) if Playwright is not installed.
    """
    if not _PLAYWRIGHT_AVAILABLE:
        logger.debug("Browser tools skipped: Playwright not installed")
        return [], None

    storage_path = _storage_state_path(config, workspace)
    logger.info("Browser: local mode (headless={}, storage={})",
                getattr(config, "headless", True), storage_path or "none")
    if storage_path:
        Path(storage_path).parent.mkdir(parents=True, exist_ok=True)

    session = BrowserSession(
        headless=getattr(config, "headless", True),
        timeout_ms=getattr(config, "timeout_ms", 30000),
        proxy_server=getattr(config, "proxy_server", "") or "",
        storage_state_path=storage_path,
    )
    tools: list[Tool] = [
        BrowserNavigateTool(session),
        BrowserSnapshotTool(session),
        BrowserContentTool(session),
        BrowserClickTool(session),
        BrowserTypeTool(session),
        BrowserPressTool(session),
        BrowserSaveSessionTool(session),
    ]
    return tools, session


def is_browser_available() -> bool:
    """Return True if Playwright is installed."""
    return _PLAYWRIGHT_AVAILABLE
