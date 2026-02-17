"""Tests for browser tools (no live browser required)."""

import pytest
from unittest.mock import AsyncMock, patch

from nanobot.agent.tools.browser import (
    BrowserSession,
    BrowserNavigateTool,
    BrowserClickTool,
    BrowserTypeTool,
    BrowserSaveSessionTool,
    _validate_url,
    create_browser_tools,
)
from nanobot.config.schema import BrowserToolConfig


def _make_session() -> BrowserSession:
    return BrowserSession()


# ── _validate_url ─────────────────────────────────────────────────────────────


def test_validate_url_valid():
    ok, err = _validate_url("http://example.com")
    assert ok is True and err == ""


def test_validate_url_rejects_ftp():
    ok, err = _validate_url("ftp://example.com")
    assert ok is False and "ftp" in err


def test_validate_url_rejects_no_scheme():
    ok, _ = _validate_url("example.com")
    assert ok is False


def test_validate_url_rejects_missing_domain():
    ok, err = _validate_url("https://")
    assert ok is False and "Missing domain" in err


# ── create_browser_tools when Playwright unavailable ──────────────────────────


def test_create_browser_tools_no_playwright():
    with patch("nanobot.agent.tools.browser._PLAYWRIGHT_AVAILABLE", False):
        tools, session = create_browser_tools(BrowserToolConfig(enabled=True))
    assert tools == [] and session is None


# ── Tool schema ───────────────────────────────────────────────────────────────


def test_navigate_schema_has_required_url():
    params = BrowserNavigateTool(_make_session()).parameters
    assert "url" in params["properties"] and "url" in params["required"]


def test_click_schema_requires_ref():
    params = BrowserClickTool(_make_session()).parameters
    assert "ref" in params["required"] and params["properties"]["ref"]["type"] == "integer"


def test_type_schema_requires_ref_and_text():
    params = BrowserTypeTool(_make_session()).parameters
    assert "ref" in params["required"] and "text" in params["required"]
    assert "submit" not in params["required"]


def test_save_session_schema_no_required():
    assert BrowserSaveSessionTool(_make_session()).parameters["required"] == []


# ── BrowserSession.reconfigure ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reconfigure_updates_headless():
    s = BrowserSession(headless=True)
    await s.reconfigure(headless=False)
    assert s._headless is False


@pytest.mark.asyncio
async def test_reconfigure_closes_existing_page():
    s = BrowserSession()
    s._page = AsyncMock()
    s._context = AsyncMock()
    s._browser = AsyncMock()
    s._playwright = AsyncMock()

    await s.reconfigure(headless=False)

    assert s._page is None and s._headless is False


# ── BrowserNavigateTool.execute ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_navigate_rejects_invalid_url():
    result = await BrowserNavigateTool(_make_session()).execute(url="ftp://bad.com")
    assert result.startswith("Error:")


@pytest.mark.asyncio
async def test_navigate_uses_get_page():
    s = _make_session()
    mock_page = AsyncMock()
    mock_page.goto = AsyncMock(return_value=None)

    with patch.object(s, "get_page", return_value=mock_page):
        result = await BrowserNavigateTool(s).execute(url="https://example.com")

    assert "Navigated to https://example.com" in result
    mock_page.goto.assert_called_once_with("https://example.com", wait_until="load")


# ── BrowserSaveSessionTool.execute ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_save_session_no_path_configured():
    result = await BrowserSaveSessionTool(BrowserSession(storage_state_path="")).execute()
    assert "Error:" in result


@pytest.mark.asyncio
async def test_save_session_not_started():
    result = await BrowserSaveSessionTool(BrowserSession(storage_state_path="/tmp/x.json")).execute()
    assert "Error:" in result
