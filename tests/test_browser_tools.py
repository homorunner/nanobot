"""Tests for browser tools (no live browser required)."""

import pytest
from unittest.mock import AsyncMock, patch

from nanobot.agent.tools.browser import BrowserSession, BrowserNavigateTool, BrowserSaveSessionTool


@pytest.mark.asyncio
async def test_reconfigure_closes_existing_page():
    s = BrowserSession()
    s._page = AsyncMock()
    s._context = AsyncMock()
    s._browser = AsyncMock()
    s._playwright = AsyncMock()
    await s.reconfigure(headless=False)
    assert s._page is None and s._headless is False


@pytest.mark.asyncio
async def test_navigate_uses_get_page():
    s = BrowserSession()
    mock_page = AsyncMock()
    mock_page.goto = AsyncMock(return_value=None)
    with patch.object(s, "get_page", return_value=mock_page):
        result = await BrowserNavigateTool(s).execute(url="https://example.com")
    assert "Navigated to https://example.com" in result
    mock_page.goto.assert_called_once_with("https://example.com", wait_until="load")


@pytest.mark.asyncio
async def test_save_session_no_path_configured():
    result = await BrowserSaveSessionTool(BrowserSession(storage_state_path="")).execute()
    assert "Error:" in result


@pytest.mark.asyncio
async def test_save_session_not_started():
    result = await BrowserSaveSessionTool(BrowserSession(storage_state_path="/tmp/x.json")).execute()
    assert "Error:" in result
