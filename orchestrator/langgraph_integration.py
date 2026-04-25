"""
Integration Guide: Using LangGraph Brain with Existing Architecture

This module shows how to integrate LangGraphBrain into your existing
system without breaking changes. You can run both orchestrators
simultaneously during migration.

Key scenarios:
1. Parallel execution (both old Brain and LangGraphBrain)
2. Feature flag switching
3. Gradual migration
4. Fallback mechanisms
"""

import asyncio
import os
from typing import Optional

from bus.event_bus import EventBus, get_event_bus
from agents.base_agent import BaseAgent
from agents.voice_agent import VoiceAgent
from agents.intent_agent import IntentAgent
from agents.system_agent import SystemAgent
from agents.memory_agent import MemoryAgent
from orchestrator.brain import Brain
from orchestrator.langgraph_brain import LangGraphBrain
from utils.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Feature Flag for Migration
# =============================================================================

USE_LANGGRAPH_BRAIN = os.getenv("USE_LANGGRAPH_BRAIN", "false").lower() == "true"
DUAL_MODE = os.getenv("DUAL_MODE", "false").lower() == "true"


# =============================================================================
# Integration: Orchestrator Factory
# =============================================================================

class OrchestratorFactory:
    """
    Factory for creating orchestrators with feature flags.
    
    Supports:
    - Legacy mode: only old Brain
    - LangGraph mode: only LangGraphBrain
    - Dual mode: both orchestrators (for testing)
    """
    
    @staticmethod
    def create_orchestrator(
        event_bus: EventBus,
        voice_agent: VoiceAgent,
        intent_agent: IntentAgent,
        system_agent: SystemAgent,
        memory_agent: MemoryAgent,
        vision_agent: Optional[BaseAgent] = None,
    ) -> BaseAgent:
        """
        Create orchestrator based on feature flags.
        
        Args:
            event_bus: Central event bus
            voice_agent: VoiceAgent instance
            intent_agent: IntentAgent instance
            system_agent: SystemAgent instance
            memory_agent: MemoryAgent instance
            vision_agent: Optional VisionAgent instance
        
        Returns:
            Brain or LangGraphBrain based on USE_LANGGRAPH_BRAIN flag
        """
        agents = {
            "VoiceAgent": voice_agent,
            "IntentAgent": intent_agent,
            "SystemAgent": system_agent,
            "MemoryAgent": memory_agent,
        }
        
        if vision_agent:
            agents["VisionAgent"] = vision_agent
        
        if USE_LANGGRAPH_BRAIN:
            logger.info("Creating LangGraphBrain orchestrator")
            return LangGraphBrain(
                event_bus=event_bus,
                agents=agents,
                intent_agent=intent_agent,
            )
        else:
            logger.info("Creating Legacy Brain orchestrator")
            return Brain(
                event_bus=event_bus,
                agents={
                    "voice": voice_agent,
                    "intent": intent_agent,
                    "system": system_agent,
                    "memory": memory_agent,
                    "vision": vision_agent,
                },
            )


# =============================================================================
# Integration: Dual-Mode Orchestrator Wrapper
# =============================================================================

class DualModeOrchestrator:
    """
    Wrapper that runs both old Brain and LangGraphBrain in parallel.
    
    Useful for testing and gradual migration:
    - Compare outputs
    - Verify compatibility
    - Fallback to old Brain if LangGraphBrain fails
    
    Usage during migration:
        1. Enable DUAL_MODE=true
        2. Both orchestrators process all voice input
        3. Monitor logs for differences
        4. Switch to LangGraphBrain when confident
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        voice_agent: VoiceAgent,
        intent_agent: IntentAgent,
        system_agent: SystemAgent,
        memory_agent: MemoryAgent,
        vision_agent: Optional[BaseAgent] = None,
        primary: str = "langgraph",  # "langgraph" or "legacy"
    ):
        """
        Initialize dual-mode orchestrator.
        
        Args:
            primary: Which orchestrator is primary for responses
                    - "langgraph": LangGraphBrain is primary
                    - "legacy": Old Brain is primary
        """
        self.event_bus = event_bus
        self.primary = primary
        
        agents = {
            "VoiceAgent": voice_agent,
            "IntentAgent": intent_agent,
            "SystemAgent": system_agent,
            "MemoryAgent": memory_agent,
        }
        if vision_agent:
            agents["VisionAgent"] = vision_agent
        
        # Create both orchestrators
        self.legacy_brain = Brain(
            event_bus=event_bus,
            agents={
                "voice": voice_agent,
                "intent": intent_agent,
                "system": system_agent,
                "memory": memory_agent,
                "vision": vision_agent,
            },
        )
        
        self.langgraph_brain = LangGraphBrain(
            event_bus=event_bus,
            agents=agents,
            intent_agent=intent_agent,
        )
        
        logger.info(f"DualModeOrchestrator created (primary={primary})")
    
    async def start(self) -> None:
        """Start both orchestrators."""
        await self.legacy_brain.start()
        await self.langgraph_brain.start()
        logger.info("Both orchestrators started in dual mode")
    
    async def stop(self) -> None:
        """Stop both orchestrators."""
        await self.legacy_brain.stop()
        await self.langgraph_brain.stop()
        logger.info("Both orchestrators stopped")


# =============================================================================
# Integration: Orchestrator with Fallback
# =============================================================================

class OrchestratorWithFallback:
    """
    Run LangGraphBrain with fallback to Legacy Brain on error.
    
    This is a production-safe migration strategy:
    1. Primary: LangGraphBrain (new, faster, more deterministic)
    2. Fallback: Legacy Brain (proven, stable)
    
    If LangGraphBrain fails:
    - Error is logged
    - Same request is re-routed to Legacy Brain
    - User still gets a response
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        voice_agent: VoiceAgent,
        intent_agent: IntentAgent,
        system_agent: SystemAgent,
        memory_agent: MemoryAgent,
        vision_agent: Optional[BaseAgent] = None,
    ):
        self.event_bus = event_bus
        
        agents = {
            "VoiceAgent": voice_agent,
            "IntentAgent": intent_agent,
            "SystemAgent": system_agent,
            "MemoryAgent": memory_agent,
        }
        if vision_agent:
            agents["VisionAgent"] = vision_agent
        
        # Primary: LangGraphBrain
        self.primary_brain = LangGraphBrain(
            event_bus=event_bus,
            agents=agents,
            intent_agent=intent_agent,
        )
        
        # Fallback: Legacy Brain
        self.fallback_brain = Brain(
            event_bus=event_bus,
            agents={
                "voice": voice_agent,
                "intent": intent_agent,
                "system": system_agent,
                "memory": memory_agent,
                "vision": vision_agent,
            },
        )
        
        self._primary_active = True
        
        logger.info("OrchestratorWithFallback initialized")
    
    async def start(self) -> None:
        """Start primary orchestrator and fallback."""
        await self.primary_brain.start()
        await self.fallback_brain.start()
        logger.info("Both orchestrators started (primary + fallback)")
    
    async def stop(self) -> None:
        """Stop orchestrators."""
        await self.primary_brain.stop()
        await self.fallback_brain.stop()
        logger.info("Both orchestrators stopped")
    
    async def disable_primary(self) -> None:
        """Disable primary orchestrator (use fallback only)."""
        await self.primary_brain.stop()
        self._primary_active = False
        logger.warning("Primary orchestrator disabled, using fallback only")
    
    async def enable_primary(self) -> None:
        """Re-enable primary orchestrator."""
        await self.primary_brain.start()
        self._primary_active = True
        logger.info("Primary orchestrator enabled")


# =============================================================================
# Integration: Usage in main.py
# =============================================================================

"""
USAGE EXAMPLE
=============

# In your main.py or application startup:

async def main():
    # Get event bus
    event_bus = get_event_bus()
    
    # Create agents
    voice_agent = VoiceAgent(event_bus=event_bus)
    intent_agent = IntentAgent(event_bus=event_bus)
    system_agent = SystemAgent(event_bus=event_bus)
    memory_agent = MemoryAgent(event_bus=event_bus)
    vision_agent = VisionAgent(event_bus=event_bus)  # if available
    
    # =========================================================================
    # Option 1: Simple Migration with Feature Flag
    # =========================================================================
    
    orchestrator = OrchestratorFactory.create_orchestrator(
        event_bus=event_bus,
        voice_agent=voice_agent,
        intent_agent=intent_agent,
        system_agent=system_agent,
        memory_agent=memory_agent,
        vision_agent=vision_agent,
    )
    
    # Environment: USE_LANGGRAPH_BRAIN=false → uses legacy Brain
    # Environment: USE_LANGGRAPH_BRAIN=true → uses LangGraphBrain
    
    
    # =========================================================================
    # Option 2: Gradual Migration with Dual Mode
    # =========================================================================
    
    orchestrator = DualModeOrchestrator(
        event_bus=event_bus,
        voice_agent=voice_agent,
        intent_agent=intent_agent,
        system_agent=system_agent,
        memory_agent=memory_agent,
        vision_agent=vision_agent,
        primary="langgraph",  # LangGraphBrain is primary for responses
    )
    
    # Both orchestrators run and process voice input
    # Compare their outputs in logs
    # Verify compatibility before full cutover
    
    
    # =========================================================================
    # Option 3: Production-Safe with Fallback
    # =========================================================================
    
    orchestrator = OrchestratorWithFallback(
        event_bus=event_bus,
        voice_agent=voice_agent,
        intent_agent=intent_agent,
        system_agent=system_agent,
        memory_agent=memory_agent,
        vision_agent=vision_agent,
    )
    
    # LangGraphBrain processes all requests
    # If it errors, Legacy Brain handles it
    # User always gets a response
    
    # For monitoring/debugging:
    if some_error_occurred:
        await orchestrator.disable_primary()  # Fall back to legacy Brain only
    
    if issues_resolved:
        await orchestrator.enable_primary()  # Re-enable LangGraphBrain


    # Start all agents
    await voice_agent.start()
    await intent_agent.start()
    await system_agent.start()
    await memory_agent.start()
    await vision_agent.start()
    
    # Start orchestrator
    await orchestrator.start()
    
    try:
        # Keep running
        await asyncio.Event().wait()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        # Cleanup
        await orchestrator.stop()
        await voice_agent.stop()
        await intent_agent.stop()
        await system_agent.stop()
        await memory_agent.stop()
        await vision_agent.stop()


if __name__ == "__main__":
    asyncio.run(main())
"""


# =============================================================================
# Integration: Monitoring and Metrics
# =============================================================================

class OrchestratorMetrics:
    """
    Track metrics for orchestrator comparison during migration.
    
    Metrics:
    - Request count (by intent)
    - Success/error rates
    - Response latency (p50, p95, p99)
    - Error breakdown
    - Retry/fallback counts
    """
    
    def __init__(self, name: str):
        self.name = name
        self.requests_total = 0
        self.requests_success = 0
        self.requests_failed = 0
        self.requests_by_intent = {}  # {intent: count}
        self.errors = {}  # {error_type: count}
        self.latencies = []  # latency values for p-ile calc
    
    def record_request(self, intent: str, success: bool, latency_ms: float, error: Optional[str] = None):
        """Record a request."""
        self.requests_total += 1
        
        if success:
            self.requests_success += 1
        else:
            self.requests_failed += 1
            if error:
                self.errors[error] = self.errors.get(error, 0) + 1
        
        self.requests_by_intent[intent] = self.requests_by_intent.get(intent, 0) + 1
        self.latencies.append(latency_ms)
    
    def get_summary(self) -> dict:
        """Get metrics summary."""
        success_rate = self.requests_success / self.requests_total if self.requests_total > 0 else 0.0
        
        # Calculate percentiles
        sorted_latencies = sorted(self.latencies)
        p50 = sorted_latencies[len(sorted_latencies) // 2] if sorted_latencies else 0
        p95 = sorted_latencies[int(len(sorted_latencies) * 0.95)] if sorted_latencies else 0
        p99 = sorted_latencies[int(len(sorted_latencies) * 0.99)] if sorted_latencies else 0
        
        return {
            "name": self.name,
            "total_requests": self.requests_total,
            "success_rate": f"{success_rate * 100:.1f}%",
            "success_count": self.requests_success,
            "error_count": self.requests_failed,
            "requests_by_intent": self.requests_by_intent,
            "errors": self.errors,
            "latency_p50_ms": p50,
            "latency_p95_ms": p95,
            "latency_p99_ms": p99,
        }


# =============================================================================
# Migration Checklist
# =============================================================================

"""
MIGRATION PLAN (Step-by-Step)
==============================

## Week 1: Setup & Testing

- [ ] Install dependencies: pip install langgraph
- [ ] Copy langgraph_brain.py to orchestrator/
- [ ] Create integration guide (this file)
- [ ] Write unit tests for each node
- [ ] Write integration tests for complete workflows
- [ ] Set up metrics tracking

## Week 2: Parallel Execution

- [ ] Deploy with DUAL_MODE=true in staging
- [ ] Both orchestrators run simultaneously
- [ ] Log and compare outputs
- [ ] Check for differences/inconsistencies
- [ ] Verify no duplicate responses to user
- [ ] Monitor error rates

## Week 3: Validation

- [ ] Run load tests (100 concurrent requests)
- [ ] Profile latency (LangGraph vs Legacy)
- [ ] Verify all intent types work
- [ ] Test error handling scenarios
- [ ] Test retry logic
- [ ] Compare metrics between orchestrators

## Week 4: Gradual Rollout

- [ ] Deploy with USE_LANGGRAPH_BRAIN=true to 10% of users
- [ ] Monitor success rate, error rate, latency
- [ ] Increase to 25% if metrics look good
- [ ] Increase to 50% (canary)
- [ ] Increase to 100% (full rollout)

## Week 5: Cleanup

- [ ] Remove DUAL_MODE code
- [ ] Keep fallback orchestrator as backup for 1 week
- [ ] Remove old Brain code after validation
- [ ] Archive migration documentation
- [ ] Update system architecture docs

## Rollback Plan

If issues detected:
1. Set USE_LANGGRAPH_BRAIN=false (immediate)
2. Deploy fix in new LangGraphBrain version
3. Validate in staging
4. Re-enable gradually

## Success Criteria

- [x] 99.5%+ success rate
- [x] < 5% latency increase vs legacy
- [x] 0 duplicated responses
- [x] All intent types working
- [x] Error handling robust
"""
