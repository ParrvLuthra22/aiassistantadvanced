"""WebSearchAgent - web search + summarization pipeline using Tavily and Gemini."""

from __future__ import annotations

import asyncio
import os
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

try:
    import google.generativeai as genai

    GEMINI_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency import
    genai = None  # type: ignore
    GEMINI_AVAILABLE = False

try:
    from tavily import TavilyClient

    TAVILY_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency import
    TavilyClient = None  # type: ignore
    TAVILY_AVAILABLE = False

from agents.base_agent import AgentCapability, BaseAgent
from schemas.events import HUDSearchResultsEvent, IntentRecognizedEvent, VoiceOutputEvent
from utils.applescript import run_applescript
from utils.api_keys import get_gemini_api_key


TRIGGER_PHRASES = [
    "search for",
    "look up",
    "what is",
    "tell me about",
    "latest news on",
    "who is",
    "how does",
]
VISION_EXCLUSION_PHRASES = [
    "on my screen",
    "read my screen",
    "describe my screen",
    "describe screen",
    "read that",
    "what does it say",
]


class WebSearchAgent(BaseAgent):
    """Agent that searches the web and returns concise factual responses."""

    def __init__(
        self,
        name: Optional[str] = None,
        event_bus=None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name=name or "WebSearchAgent", event_bus=event_bus, config=config)
        self._tavily_client: Optional[TavilyClient] = None
        self._gemini_model = None
        self._last_tavily_call: float = 0.0
        self._cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
        self._cache_limit = 20

    @property
    def capabilities(self) -> List[AgentCapability]:
        return [
            AgentCapability(
                name="web_search",
                description="Searches web with Tavily and summarizes with Gemini",
                input_events=["IntentRecognizedEvent"],
                output_events=["VoiceOutputEvent", "HUDSearchResultsEvent"],
            )
        ]

    async def _setup(self) -> None:
        self._subscribe(IntentRecognizedEvent, self._handle_intent)
        self._initialize_clients()

    async def _teardown(self) -> None:
        self._tavily_client = None
        self._gemini_model = None
        self._cache.clear()

    def _initialize_clients(self) -> None:
        if not TAVILY_AVAILABLE:
            self._logger.warning("tavily-python SDK unavailable; WebSearchAgent disabled")
            return
        if not GEMINI_AVAILABLE or genai is None:
            self._logger.warning("google-generativeai SDK unavailable; WebSearchAgent disabled")
            return

        tavily_key = (
            os.getenv("TAVILY_API_KEY")
            or self._get_config("web_search.tavily_api_key")
            or self._get_config("system.apis.tavily.api_key")
        )

        gemini_key = get_gemini_api_key(self._get_config)

        if tavily_key:
            self._tavily_client = TavilyClient(api_key=tavily_key)
        else:
            self._logger.warning("TAVILY_API_KEY not configured; WebSearchAgent disabled")

        if gemini_key:
            genai.configure(api_key=gemini_key)
            self._gemini_model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        else:
            self._logger.warning("Gemini API key not configured; WebSearchAgent disabled")

    async def _handle_intent(self, event: IntentRecognizedEvent) -> None:
        text = self._event_text(event)
        text_lower = text.lower() if text else ""
        if not text or self._is_vision_query(text_lower) or not self._is_search_trigger(text_lower):
            return

        if self._tavily_client is None or self._gemini_model is None:
            await self._emit_voice("Web search is not configured yet.", event)
            return

        query = await self._extract_query(text)
        if not query:
            await self._emit_voice("I couldn't extract a search query.", event)
            return

        results = await self._search_tavily(query)
        if not results.get("results"):
            await self._emit_voice(f"I couldn't find results for {query}.", event)
            return

        summary = await self._summarize(query, results)
        await self._emit_voice(summary, event)

        sources = [
            {"title": str(item.get("title", "Untitled")), "url": str(item.get("url", ""))}
            for item in results.get("results", [])[:3]
        ]

        await self._emit(
            HUDSearchResultsEvent(
                query=query,
                summary=summary,
                sources=sources,
                source=self._name,
                correlation_id=event.correlation_id or event.event_id,
            )
        )

        lower = text.lower()
        if ("open" in lower or "show me" in lower) and sources and sources[0].get("url"):
            await self._open_in_safari(sources[0]["url"], event)

    async def _extract_query(self, text: str) -> str:
        prompt = f"Extract the search query from: '{text}'. Return only the query string."

        def _run() -> str:
            response = self._gemini_model.generate_content(prompt)
            return (getattr(response, "text", "") or "").strip()

        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, _run)
        except Exception:
            return ""

    async def _search_tavily(self, query: str) -> Dict[str, Any]:
        cached = self._cache.get(query)
        if cached is not None:
            self._cache.move_to_end(query)
            return cached

        now = time.monotonic()
        elapsed = now - self._last_tavily_call
        if elapsed < 1.0:
            await asyncio.sleep(1.0 - elapsed)

        def _run() -> Dict[str, Any]:
            return self._tavily_client.search(query, max_results=5, search_depth="advanced")

        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, _run)
        self._last_tavily_call = time.monotonic()

        self._cache[query] = results
        if len(self._cache) > self._cache_limit:
            self._cache.popitem(last=False)

        return results

    async def _summarize(self, query: str, results: Dict[str, Any]) -> str:
        context = "\n\n".join(
            [
                f"Source: {r.get('url', '')}\n{r.get('content', '')}"
                for r in results.get("results", [])
            ]
        )
        summary_prompt = (
            f"Based on these search results, answer '{query}' in 3 clear sentences. "
            "Be direct and factual.\n\n"
            f"{context}"
        )

        def _run() -> str:
            response = self._gemini_model.generate_content(summary_prompt)
            return (getattr(response, "text", "") or "").strip()

        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, _run)
        except Exception as exc:
            return f"I found sources for {query}, but summarization failed: {exc}"

    async def _open_in_safari(self, url: str, event: IntentRecognizedEvent) -> None:
        script = (
            "tell application \"Safari\"\n"
            "  activate\n"
            f"  open location \"{self._escape_applescript(url)}\"\n"
            "end tell"
        )

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, lambda: run_applescript(script))
        except Exception as exc:
            await self._emit_voice(f"I found results but couldn't open Safari: {exc}", event)

    async def _emit_voice(self, text: str, event: IntentRecognizedEvent) -> None:
        await self._emit(
            VoiceOutputEvent(
                text=text,
                source=self._name,
                correlation_id=event.correlation_id or event.event_id,
            )
        )

    @staticmethod
    def _event_text(event: IntentRecognizedEvent) -> str:
        intent = getattr(event, "intent", "") or ""
        text = getattr(event, "text", "") or ""
        raw = getattr(event, "raw_text", "") or ""
        if text.strip():
            return text.strip()
        if raw.strip():
            return raw.strip()
        return intent.strip()

    @staticmethod
    def _is_search_trigger(text_lower: str) -> bool:
        return any(phrase in text_lower for phrase in TRIGGER_PHRASES)

    @staticmethod
    def _is_vision_query(text_lower: str) -> bool:
        return any(phrase in text_lower for phrase in VISION_EXCLUSION_PHRASES)

    @staticmethod
    def _escape_applescript(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')
