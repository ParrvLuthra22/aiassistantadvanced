"""
ToolAgent - Execution adapter for LangGraph orchestrator tool node.

This class bridges intent-driven actions to existing concrete agent
implementations (SystemAgent, MemoryAgent) while keeping async orchestration.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

from agents.memory_agent import MemoryAgent
from agents.system_agent import SystemAgent
from bus.event_bus import EventBus
from schemas.events import VoiceOutputEvent
from utils.logger import get_logger


logger = get_logger(__name__)


class ToolAgent:
    """Executes tools/actions for a parsed intent."""

    def __init__(
        self,
        system_agent: Optional[SystemAgent] = None,
        memory_agent: Optional[MemoryAgent] = None,
        event_bus: Optional[EventBus] = None,
    ):
        self._system_agent = system_agent
        self._memory_agent = memory_agent
        self._event_bus = event_bus

    async def execute(
        self,
        intent: str,
        entities: Dict[str, Any],
        raw_text: str,
    ) -> Dict[str, Any]:
        """Execute an intent and return a structured result."""
        logger.info(f"[ToolAgent] execute intent={intent} entities={entities}")

        if intent == "SAVE_MEMORY":
            return await self._save_memory(entities)

        if intent == "RECALL_MEMORY":
            return self._recall_memory()

        if self._system_agent and intent in self._system_agent.INTENT_HANDLERS:
            handler_name = self._system_agent.INTENT_HANDLERS[intent]
            handler = getattr(self._system_agent, handler_name, None)
            if not handler:
                return {
                    "success": False,
                    "result": None,
                    "error": f"Handler not found for intent '{intent}'",
                }

            try:
                result = await asyncio.to_thread(handler, entities, raw_text)
                return {
                    "success": True,
                    "result": result,
                    "error": "",
                }
            except Exception as exc:
                logger.error(f"[ToolAgent] system handler failed: {exc}", exc_info=True)
                return {
                    "success": False,
                    "result": None,
                    "error": str(exc),
                }

        return {
            "success": False,
            "result": None,
            "error": f"No tool mapping available for intent '{intent}'",
        }

    async def emit_voice_feedback(self, text: str) -> None:
        """Optionally emit immediate voice feedback via event bus."""
        if not self._event_bus or not text:
            return

        await self._event_bus.emit(
            VoiceOutputEvent(
                text=text,
                source="ToolAgent",
            )
        )

    async def _save_memory(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        if not self._memory_agent or not self._memory_agent.store:
            return {
                "success": False,
                "result": None,
                "error": "Memory store is not available.",
            }

        key = str(entities.get("key", "note"))
        value = entities.get("value", entities.get("text", ""))
        if not value:
            return {
                "success": False,
                "result": None,
                "error": "Missing memory value.",
            }

        self._memory_agent.store.store(
            memory_type="long_term",
            category="context",
            key=key,
            value=value,
        )
        return {
            "success": True,
            "result": f"Saved memory '{key}'.",
            "error": "",
        }

    def _recall_memory(self) -> Dict[str, Any]:
        if not self._memory_agent:
            return {
                "success": False,
                "result": None,
                "error": "Memory agent is not available.",
            }

        last_command = self._memory_agent.get_last_command()
        result = (
            f"Your last command was '{last_command}'."
            if last_command
            else "I don't have a recent command saved yet."
        )

        return {
            "success": True,
            "result": result,
            "error": "",
        }
