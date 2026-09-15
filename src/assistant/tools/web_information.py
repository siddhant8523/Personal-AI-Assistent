"""Latest-information search and bounded HTML text extraction."""

from __future__ import annotations

import os
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


class WebInformationService:
    def __init__(self, tavily_api_key: str | None = None):
        self._tavily_api_key = tavily_api_key if tavily_api_key is not None else os.getenv("TAVILY_API_KEY", "")

    def search(self, query: str, max_results: int = 5, search_depth: str = "basic") -> str:
        if not self._tavily_api_key:
            return "Web search is unavailable: set TAVILY_API_KEY in .env."
        from tavily import TavilyClient

        depth = "advanced" if search_depth == "advanced" else "basic"
        response = TavilyClient(api_key=self._tavily_api_key).search(
            query=query, search_depth=depth, max_results=max(1, min(max_results, 10)), include_answer=True,
        )
        lines = []
        if response.get("answer"):
            lines.append(f"Summary: {response['answer']}")
        for item in response.get("results", []):
            lines.append(f"- {item.get('title', 'Untitled')}: {item.get('content', '')[:500]}\n  Source: {item.get('url', '')}")
        return "\n".join(lines) or "No results found."

    def search_qna(self, query: str) -> str:
        if not self._tavily_api_key:
            return "Web search Q&A is unavailable: set TAVILY_API_KEY in .env."
        from tavily import TavilyClient

        client = TavilyClient(api_key=self._tavily_api_key)
        if hasattr(client, "qna_search"):
            return client.qna_search(query=query) or "No direct Q&A answer found."
        
        # Fallback using search include_answer=True
        res = client.search(query=query, search_depth="basic", max_results=3, include_answer=True)
        return res.get("answer") or "No direct Q&A answer found for query."

    def extract_page_text(self, url: str, max_characters: int = 12000) -> str:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return "Only complete http:// or https:// URLs can be extracted."
        response = requests.get(url, timeout=15, headers={"User-Agent": "PersonalAIAssistant/1.0"}, stream=True)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "html" not in content_type.lower():
            return f"This URL is not an HTML page ({content_type or 'unknown content type'})."
        raw = response.raw.read(max_characters * 3, decode_content=True)
        soup = BeautifulSoup(raw, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()
        text = " ".join(soup.stripped_strings)
        return text[:max_characters] or "No readable text found on this page."

    def extract_page_structure(self, url: str, selector: str | None = None) -> str:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return "Only complete http:// or https:// URLs can be extracted."
        response = requests.get(url, timeout=15, headers={"User-Agent": "PersonalAIAssistant/1.0"})
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "html" not in content_type.lower():
            return f"This URL is not an HTML page ({content_type or 'unknown content type'})."

        soup = BeautifulSoup(response.text, "html.parser")
        lines = []

        if selector:
            elements = soup.select(selector)
            if not elements:
                return f"No elements matching selector '{selector}' found on {url}."
            lines.append(f"Elements matching '{selector}' ({len(elements)} found):")
            for idx, el in enumerate(elements[:15], 1):
                txt = " ".join(el.stripped_strings)[:300]
                lines.append(f"{idx}. [{el.name}] {txt}")
            return "\n".join(lines)

        # General structural summary
        title = soup.title.string.strip() if soup.title and soup.title.string else "No title"
        lines.append(f"Page Title: {title}")

        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            lines.append(f"Meta Description: {meta_desc['content'].strip()}")

        headings = []
        for h in soup.find_all(["h1", "h2", "h3"]):
            headings.append(f"- [{h.name.upper()}] {' '.join(h.stripped_strings)}")
        if headings:
            lines.append("\nHeadings:")
            lines.extend(headings[:15])

        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            label = " ".join(a.stripped_strings) or "Link"
            if href.startswith("http://") or href.startswith("https://"):
                links.append(f"- {label}: {href}")
        if links:
            lines.append("\nSample Outbound Links:")
            lines.extend(links[:10])

        return "\n".join(lines)

