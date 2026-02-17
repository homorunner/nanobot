"""Tests for browser tools (no live browser required)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from nanobot.agent.tools.browser import (
    BrowserSession,
    BrowserNavigateTool,
    BrowserSnapshotTool,
    BrowserContentTool,
    BrowserClickTool,
    BrowserTypeTool,
    BrowserPressTool,
    BrowserSaveSessionTool,
    _validate_url,
    create_browser_tools,
    is_browser_available,
)
from nanobot.config.schema import BrowserToolConfig


# ── _validate_url ─────────────────────────────────────────────────────────────


def test_validate_url_valid_http():
    ok, err = _validate_url("http://example.com")
    assert ok is True
    assert err == ""


def test_validate_url_valid_https():
    ok, err = _validate_url("https://example.com/path?q=1")
    assert ok is True
    assert err == ""


def test_validate_url_rejects_ftp():
    ok, err = _validate_url("ftp://example.com")
    assert ok is False
    assert "ftp" in err


def test_validate_url_rejects_no_scheme():
    ok, err = _validate_url("example.com")
    assert ok is False


def test_validate_url_rejects_missing_domain():
    ok, err = _validate_url("https://")
    assert ok is False
    assert "Missing domain" in err


# ── BrowserToolConfig ─────────────────────────────────────────────────────────


def test_browser_tool_config_defaults():
    cfg = BrowserToolConfig()
    assert cfg.enabled is False
    assert cfg.headless is True
    assert cfg.timeout_ms == 30000
    assert cfg.proxy_server == ""
    assert cfg.storage_state_path == ""


def test_browser_tool_config_custom():
    cfg = BrowserToolConfig(enabled=True, headless=False, timeout_ms=10000)
    assert cfg.enabled is True
    assert cfg.headless is False
    assert cfg.timeout_ms == 10000


# ── Tool schema structure ─────────────────────────────────────────────────────


def _make_session() -> BrowserSession:
    return BrowserSession()


def test_tool_names():
    s = _make_session()
    assert BrowserNavigateTool(s).name == "browser_navigate"
    assert BrowserSnapshotTool(s).name == "browser_snapshot"
    assert BrowserContentTool(s).name == "browser_content"
    assert BrowserClickTool(s).name == "browser_click"
    assert BrowserTypeTool(s).name == "browser_type"
    assert BrowserPressTool(s).name == "browser_press"
    assert BrowserSaveSessionTool(s).name == "browser_save_session"


def test_navigate_schema_has_required_url():
    s = _make_session()
    params = BrowserNavigateTool(s).parameters
    assert "url" in params["properties"]
    assert "url" in params["required"]


def test_navigate_schema_has_headless_not_required():
    s = _make_session()
    params = BrowserNavigateTool(s).parameters
    assert "headless" in params["properties"]
    assert "headless" not in params["required"]


def test_click_schema_requires_ref():
    s = _make_session()
    params = BrowserClickTool(s).parameters
    assert "ref" in params["required"]
    assert params["properties"]["ref"]["type"] == "integer"


def test_type_schema_requires_ref_and_text():
    s = _make_session()
    params = BrowserTypeTool(s).parameters
    assert "ref" in params["required"]
    assert "text" in params["required"]
    assert "submit" not in params["required"]


def test_save_session_schema_no_required():
    s = _make_session()
    params = BrowserSaveSessionTool(s).parameters
    assert params["required"] == []


# ── create_browser_tools when Playwright unavailable ──────────────────────────


def test_create_browser_tools_no_playwright():
    with patch("nanobot.agent.tools.browser._PLAYWRIGHT_AVAILABLE", False):
        tools, session = create_browser_tools(BrowserToolConfig(enabled=True))
    assert tools == []
    assert session is None


def test_is_browser_available_type():
    result = is_browser_available()
    assert isinstance(result, bool)


# ── BrowserSession.reconfigure ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reconfigure_updates_headless():
    s = BrowserSession(headless=True)
    # reconfigure without an open page just updates the flag
    await s.reconfigure(headless=False)
    assert s._headless is False


@pytest.mark.asyncio
async def test_reconfigure_closes_existing_page():
    s = BrowserSession()
    # Simulate an open page
    mock_page = AsyncMock()
    mock_context = AsyncMock()
    mock_browser = AsyncMock()
    mock_pw = AsyncMock()
    s._page = mock_page
    s._context = mock_context
    s._browser = mock_browser
    s._playwright = mock_pw

    await s.reconfigure(headless=False)

    # close() should have been called (page cleared)
    assert s._page is None
    assert s._headless is False


# ── BrowserNavigateTool.execute ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_navigate_rejects_invalid_url():
    s = _make_session()
    tool = BrowserNavigateTool(s)
    result = await tool.execute(url="ftp://bad.com")
    assert result.startswith("Error:")


@pytest.mark.asyncio
async def test_navigate_uses_get_page():
    s = _make_session()
    mock_page = AsyncMock()
    mock_page.goto = AsyncMock(return_value=None)
    s._page = mock_page  # pre-seed so get_page() returns without launching

    # Patch get_page to avoid launching real browser
    with patch.object(s, "get_page", return_value=mock_page):
        tool = BrowserNavigateTool(s)
        result = await tool.execute(url="https://example.com")

    assert "Navigated to https://example.com" in result
    mock_page.goto.assert_called_once_with("https://example.com", wait_until="load")


# ── BrowserSaveSessionTool.execute ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_save_session_no_path_configured():
    s = BrowserSession(storage_state_path="")
    tool = BrowserSaveSessionTool(s)
    result = await tool.execute()
    assert "Error:" in result


@pytest.mark.asyncio
async def test_save_session_not_started():
    s = BrowserSession(storage_state_path="/tmp/test_cookies.json")
    tool = BrowserSaveSessionTool(s)
    result = await tool.execute()
    assert "Error:" in result
