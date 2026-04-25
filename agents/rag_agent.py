"""
RAGAgent - Retrieval-augmented context helper for the LangGraph orchestrator.

This agent provides lightweight retrieval from the existing MemoryAgent so the
orchestrator can enrich responses for question-like intents.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agents.memory_agent import MemoryAgent
from utils.logger import get_logger


logger = get_logger(__name__)


class RAGAgent:
    """Small retrieval adapter used by the LangGraph RAG node."""

    def __init__(self, memory_agent: Optional[MemoryAgent] = None):
        self._memory_agent = memory_agent

    async def retrieve(self, query: str, intent: str) -> Dict[str, Any]:
        """
        Retrieve context from memory and build a concise answer scaffold.

        Returns a dict with retrieved context and an optional synthesized answer.
        """
        logger.info(f"[RAGAgent] retrieve intent={intent} query='{query[:80]}'")

        if not self._memory_agent:
            return {
                "context": [],
                "answer": "I can help, but memory context is currently unavailable.",
            }

        semantic_matches = []
        try:
            semantic_matches = await self._memory_agent.semantic_retrieve(
                query=query,
                intent=intent,
                top_k=8,
            )
        except Exception as exc:
            logger.warning(f"[RAGAgent] semantic retrieval failed, using fallback context only: {exc}")

        recent_conversation: List[Dict[str, Any]] = self._memory_agent.get_recent_conversation(max_turns=5)
        frequent_apps = self._memory_agent.get_frequent_apps(limit=3)
        last_command = self._memory_agent.get_last_command()

        context = {
            "recent_conversation": recent_conversation,
            "frequent_apps": frequent_apps,
            "last_command": last_command,
            "semantic_matches": semantic_matches,
        }

        if intent == "RECALL_MEMORY":
            answer = (
                f"Your last command was: '{last_command}'."
                if last_command
                else "I don't have a recent command saved yet."
            )
        elif intent in {"HELP", "GENERAL_QUESTION"}:
            if semantic_matches:
                snippets = "; ".join(m.get("text", "")[:120] for m in semantic_matches[:2])
                answer = f"I found relevant memory context: {snippets}"
            else:
                answer = "I can help with app control, system info, memory recall, and vision commands."
        else:
            answer = "I found relevant context and passed it to the workflow."

        return {
            "context": context,
            "answer": answer,
        }
