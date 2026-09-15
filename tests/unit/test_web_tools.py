"""
Unit tests for Tavily search and BeautifulSoup scraper tools.
"""

from unittest.mock import MagicMock, patch
import pytest

from assistant.tools.web_information import WebInformationService
from assistant.agent_core.graph.tools import build_tools
from assistant.execution.outbound_registry import OutboundRegistry
from assistant.execution.task_manager import TaskManager
from assistant.execution.tool_router import ToolRouter
from assistant.agent_core.policy_engine import PolicyEngine
from assistant.agent_core.task_planner import TaskPlanner
from assistant.agent_core.approval_manager import ApprovalManager
from assistant.intelligence.priority_inbox import PriorityInbox


def test_web_information_no_api_key():
    service = WebInformationService(tavily_api_key="")
    assert "unavailable" in service.search("test query")
    assert "unavailable" in service.search_qna("test query")


@patch("tavily.TavilyClient")
def test_web_information_tavily_search(mock_tavily_cls):
    mock_client = MagicMock()
    mock_client.search.return_value = {
        "answer": "Python is a programming language.",
        "results": [{"title": "Python Overview", "content": "Python is dynamic.", "url": "https://python.org"}],
    }
    mock_tavily_cls.return_value = mock_client

    service = WebInformationService(tavily_api_key="test_key")
    res = service.search("python language", max_results=3, search_depth="advanced")

    assert "Python is a programming language." in res
    assert "Python Overview" in res
    mock_client.search.assert_called_once_with(
        query="python language", search_depth="advanced", max_results=3, include_answer=True
    )


@patch("tavily.TavilyClient")
def test_web_information_tavily_qna(mock_tavily_cls):
    mock_client = MagicMock()
    mock_client.qna_search.return_value = "Paris is the capital of France."
    mock_tavily_cls.return_value = mock_client

    service = WebInformationService(tavily_api_key="test_key")
    ans = service.search_qna("capital of France")

    assert ans == "Paris is the capital of France."


@patch("requests.get")
def test_beautifulsoup_extract_page_text(mock_get):
    mock_response = MagicMock()
    mock_response.headers = {"content-type": "text/html"}
    mock_response.raw.read.return_value = (
        b"<html><head><script>var x=1;</script></head>"
        b"<body><h1>Hello World</h1><p>This is test content.</p></body></html>"
    )
    mock_get.return_value = mock_response

    service = WebInformationService()
    text = service.extract_page_text("https://example.com")

    assert "Hello World" in text
    assert "This is test content." in text
    assert "var x=1" not in text


@patch("requests.get")
def test_beautifulsoup_extract_page_structure(mock_get):
    mock_response = MagicMock()
    mock_response.headers = {"content-type": "text/html"}
    mock_response.text = (
        "<html><head><title>Test Page Title</title>"
        "<meta name='description' content='Meta description text'>"
        "</head><body>"
        "<h1>Main Title</h1>"
        "<h2>Subtitle</h2>"
        "<a href='https://example.com/link1'>Link 1</a>"
        "<div class='content'>Target content</div>"
        "</body></html>"
    )
    mock_get.return_value = mock_response

    service = WebInformationService()
    summary = service.extract_page_structure("https://example.com")

    assert "Test Page Title" in summary
    assert "Meta description text" in summary
    assert "Main Title" in summary
    assert "Link 1" in summary

    # Test CSS selector parsing
    selected = service.extract_page_structure("https://example.com", selector=".content")
    assert "Target content" in selected


def test_build_tools_registers_web_tools():
    outbound = OutboundRegistry()
    tm = TaskManager(outbound)
    tr = ToolRouter(tm)
    pe = PolicyEngine({})
    tp = TaskPlanner(tm, pe)
    am = ApprovalManager(tm)
    pi = PriorityInbox()
    web_service = WebInformationService()

    tools = build_tools(
        task_planner=tp, approval_manager=am, priority_inbox=pi, tool_router=tr, web_information=web_service
    )
    tool_names = [t.name for t in tools]

    assert "search_latest_information" in tool_names
    assert "search_qna_information" in tool_names
    assert "extract_web_page_text" in tool_names
    assert "extract_web_page_structure" in tool_names
