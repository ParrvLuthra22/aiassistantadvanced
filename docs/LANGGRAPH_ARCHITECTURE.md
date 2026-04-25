# LangGraph-Based Orchestrator Architecture

## Executive Summary

This document outlines a complete migration from the event-driven Brain orchestrator to a **LangGraph-based workflow** that maintains full compatibility with your existing event bus while gaining:

- **Deterministic routing** with explicit state transitions
- **Centralized error handling** via LangGraph's retry/fallback mechanisms
- **Observable workflow execution** with built-in tracing
- **Simplified state management** through TypedDict schemas
- **Type-safe conditional routing** with @branch decorators

---

## 1. Architecture Overview

### Current Architecture (Event-Driven Brain)
```
┌─────────────────────────────────────────────────────────────┐
│                      EVENT BUS (pub/sub)                    │
└─────────────────────────────────────────────────────────────┘
     ↑              ↑              ↑              ↑
     │              │              │              │
  VOICE         INTENT          SYSTEM         MEMORY
  AGENT         AGENT           AGENT          AGENT
     ↑              ↑              ↑              ↑
     └──────────────┬──────────────┴──────────────┘
                    │
                 BRAIN (Orchestrator)
              (routes, plans, coordinates)
```

### New LangGraph Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Workflow                       │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│  │ CAPTURE  │───→│ CLASSIFY │───→│ ROUTE    │             │
│  │ INPUT    │    │ INTENT   │    │ INTENT   │             │
│  └──────────┘    └──────────┘    └──────────┘             │
│                                          ↓                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│  │ VOICE    │ ← │ EXECUTE  │ ← │ PLAN     │             │
│  │ AGENT    │    │ ACTION   │    │ MULTI    │             │
│  └──────────┘    └──────────┘    │ STEP     │             │
│                                    └──────────┘             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│  │ SYSTEM   │    │ MEMORY   │    │ INTENT   │             │
│  │ AGENT    │    │ AGENT    │    │ AGENT    │             │
│  └──────────┘    └──────────┘    └──────────┘             │
└─────────────────────────────────────────────────────────────┘
                          ↓
        ┌──────────────────────────────────────┐
        │  EVENT BUS (emit events unchanged)   │
        └──────────────────────────────────────┘
```

---

## 2. State Schema (TypedDict)

### WorkflowState
```python
from typing import TypedDict, Any, List, Dict, Optional
from uuid import UUID
from enum import Enum

class WorkflowState(TypedDict):
    """Unified state passed through LangGraph workflow."""
    
    # Input
    user_input: str                    # Raw user text
    correlation_id: UUID               # For tracing
    
    # Intent classification
    intent: Optional[str]              # Parsed intent (e.g., "OPEN_APP")
    intent_confidence: float           # Confidence score (0.0 - 1.0)
    entities: Dict[str, Any]           # Extracted parameters (e.g., {"app": "Safari"})
    raw_text: str                      # Original utterance for fallback parsing
    
    # Routing decision
    target_agent: Optional[str]        # Which agent handles this ("SystemAgent", "VisionAgent", etc.)
    action: Optional[str]              # Specific action for the agent
    is_multi_step: bool                # Whether this needs a plan
    
    # Execution state
    action_results: List[Dict]         # Results from each agent invocation
    current_step: int                  # For multi-step plans
    max_retries: int                   # Retry budget
    retry_count: int                   # Retries used so far
    
    # Error handling & fallback
    last_error: Optional[str]          # Last error message
    error_count: int                   # Total errors in workflow
    should_clarify: bool               # Should ask user for clarification?
    clarification_options: List[str]   # What to ask about
    
    # Response & context
    response_text: str                 # Final response to user
    context_snapshot: Dict             # Conversation context snapshot
    events_to_emit: List[Any]          # Events to emit to event bus
```

---

## 3. LangGraph Workflow Topology

### Node Definitions

#### Node 1: `capture_input`
**Responsibility:** Initialize workflow state from voice input  
**Input:** VoiceInputEvent  
**Output:** WorkflowState with user_input, correlation_id set

```python
def capture_input(state: WorkflowState) -> WorkflowState:
    """
    Capture user input and initialize workflow state.
    
    Called when VoiceInputEvent arrives from VoiceAgent.
    """
    logger.info(f"[NODE] capture_input: {state['user_input'][:50]}...")
    
    # Ensure required fields
    state = {
        **state,
        "intent": None,
        "intent_confidence": 0.0,
        "entities": {},
        "raw_text": state.get("user_input", ""),
        "target_agent": None,
        "action": None,
        "is_multi_step": False,
        "action_results": [],
        "current_step": 0,
        "max_retries": 3,
        "retry_count": 0,
        "last_error": None,
        "error_count": 0,
        "should_clarify": False,
        "clarification_options": [],
        "response_text": "",
        "context_snapshot": {},
        "events_to_emit": [],
    }
    return state
```

#### Node 2: `classify_intent`
**Responsibility:** Call IntentAgent to classify user input  
**Input:** WorkflowState with user_input  
**Output:** WorkflowState with intent, entities filled

```python
async def classify_intent(state: WorkflowState, intent_agent: IntentAgent, event_bus: EventBus) -> WorkflowState:
    """
    Classify user input into an intent via IntentAgent.
    
    Sends IntentRecognizedEvent to event bus for compatibility.
    """
    logger.info("[NODE] classify_intent")
    
    user_input = state["user_input"]
    
    try:
        # Call IntentAgent directly (or emit event and wait)
        # For now, simulate intent classification
        intent_result = await intent_agent.classify(user_input)
        
        state = {
            **state,
            "intent": intent_result.get("intent"),
            "intent_confidence": intent_result.get("confidence", 0.0),
            "entities": intent_result.get("entities", {}),
            "raw_text": user_input,
        }
        
        # Emit IntentRecognizedEvent for agents listening to event bus
        await event_bus.emit(IntentRecognizedEvent(
            intent=state["intent"],
            slots=state["entities"],
            raw_text=user_input,
            correlation_id=state["correlation_id"],
            source="LangGraphBrain",
        ))
        
        return state
        
    except Exception as e:
        logger.error(f"Intent classification failed: {e}")
        state = {
            **state,
            "last_error": str(e),
            "error_count": state["error_count"] + 1,
            "should_clarify": True,
            "clarification_options": ["I didn't understand that. Can you rephrase?"],
        }
        return state
```

#### Node 3: `route_intent`
**Responsibility:** Decide which agent handles the intent  
**Input:** WorkflowState with intent  
**Output:** WorkflowState with target_agent, action, is_multi_step

```python
def route_intent(state: WorkflowState) -> WorkflowState:
    """
    Route the classified intent to an appropriate agent.
    
    Uses INTENT_ROUTING table; handles multi-step detection.
    """
    logger.info(f"[NODE] route_intent: {state.get('intent')}")
    
    if not state.get("intent"):
        state = {
            **state,
            "should_clarify": True,
            "clarification_options": ["I didn't catch that. What would you like me to do?"],
        }
        return state
    
    # Look up routing
    routing = INTENT_ROUTING.get(state["intent"])
    
    if not routing:
        logger.warning(f"No routing for intent: {state['intent']}")
        state = {
            **state,
            "should_clarify": True,
            "clarification_options": [f"I'm not sure how to handle '{state['intent']}'."],
        }
        return state
    
    # Check if multi-step
    is_multi_step = _is_multi_step_command(state["intent"], state["entities"])
    
    state = {
        **state,
        "target_agent": routing["agent"],
        "action": routing["action"],
        "is_multi_step": is_multi_step,
    }
    
    return state
```

#### Node 4: `plan_multi_step`
**Responsibility:** Break complex commands into steps  
**Input:** WorkflowState with intent, entities, is_multi_step=True  
**Output:** WorkflowState with multi-step plan encoded as action_results

```python
def plan_multi_step(state: WorkflowState) -> WorkflowState:
    """
    Create a multi-step execution plan for complex commands.
    
    Example: "Open Safari and go to YouTube"
      → Step 1: open_app(app="Safari")
      → Step 2: open_url(url="https://youtube.com")
    """
    logger.info(f"[NODE] plan_multi_step: {state['intent']}")
    
    intent = state["intent"]
    entities = state["entities"]
    
    # Build steps based on intent
    steps = []
    
    if intent == "OPEN_APP_AND_NAVIGATE":
        steps = [
            {
                "agent": "SystemAgent",
                "action": "open_app",
                "params": {"app": entities.get("app", "")},
            },
            {
                "agent": "SystemAgent",
                "action": "open_url",
                "params": {"url": entities.get("url", "")},
            },
        ]
    elif intent == "RECORD_AND_SAVE":
        steps = [
            {
                "agent": "SystemAgent",
                "action": "start_recording",
                "params": {"duration": entities.get("duration", 10)},
            },
            {
                "agent": "SystemAgent",
                "action": "save_file",
                "params": {"filename": entities.get("filename", "recording.wav")},
            },
        ]
    
    state = {
        **state,
        "action_results": [{"step": i, "status": "pending"} for i in range(len(steps))],
        "current_step": 0,
    }
    
    return state
```

#### Node 5: `execute_action`
**Responsibility:** Invoke agent via event bus (maintains compatibility)  
**Input:** WorkflowState with target_agent, action  
**Output:** WorkflowState with action_results updated

```python
async def execute_action(
    state: WorkflowState,
    event_bus: EventBus,
    agents: Dict[str, Any],
) -> WorkflowState:
    """
    Execute an action by emitting ActionRequestEvent to event bus.
    
    Waits for ActionResultEvent response (with timeout).
    """
    logger.info(f"[NODE] execute_action: {state['target_agent']}.{state['action']}")
    
    target_agent = state["target_agent"]
    action = state["action"]
    parameters = state["entities"]
    
    # Set up response handler
    result_received = asyncio.Event()
    action_result = {}
    
    async def on_action_result(event: ActionResultEvent):
        nonlocal action_result
        if event.correlation_id == state["correlation_id"]:
            action_result = {
                "action": event.action,
                "success": event.success,
                "result": event.result,
                "error": event.error,
                "agent": event.source,
            }
            result_received.set()
    
    # Subscribe to result
    token = await event_bus.subscribe(ActionResultEvent, on_action_result)
    
    try:
        # Emit action request
        await event_bus.emit(ActionRequestEvent(
            action=action,
            target_agent=target_agent,
            parameters=parameters,
            correlation_id=state["correlation_id"],
            source="LangGraphBrain",
        ))
        
        # Wait for result (with timeout)
        await asyncio.wait_for(result_received.wait(), timeout=10.0)
        
        # Append result
        new_results = state["action_results"] + [action_result]
        
        success = action_result.get("success", False)
        state = {
            **state,
            "action_results": new_results,
            "response_text": str(action_result.get("result", "")),
            "last_error": action_result.get("error") if not success else None,
            "error_count": state["error_count"] + (0 if success else 1),
        }
        
        return state
        
    except asyncio.TimeoutError:
        logger.error(f"Action timeout: {target_agent}.{action}")
        state = {
            **state,
            "last_error": f"Agent {target_agent} did not respond in time",
            "error_count": state["error_count"] + 1,
        }
        return state
        
    finally:
        token.unsubscribe()
```

#### Node 6: `handle_error`
**Responsibility:** Decide whether to retry, fallback, or clarify  
**Input:** WorkflowState with last_error, error_count, retry_count  
**Output:** Decides next node (retry, fallback, clarify, or respond)

```python
def handle_error(state: WorkflowState) -> WorkflowState:
    """
    Error handling with retry and fallback logic.
    """
    logger.info(f"[NODE] handle_error: {state.get('last_error')}")
    
    if state["retry_count"] < state["max_retries"]:
        # Try again
        state = {
            **state,
            "retry_count": state["retry_count"] + 1,
            "last_error": None,
        }
        logger.info(f"Retrying (attempt {state['retry_count']})")
        return state
    
    # Max retries exceeded - ask for clarification
    state = {
        **state,
        "should_clarify": True,
        "clarification_options": [
            f"I'm having trouble with that ({state['last_error']}). Can you try again?",
            "What would you like me to do instead?",
        ],
    }
    return state
```

#### Node 7: `generate_response`
**Responsibility:** Create user-facing response  
**Input:** WorkflowState with response_text, action_results  
**Output:** WorkflowState with final response_text

```python
def generate_response(state: WorkflowState) -> WorkflowState:
    """
    Generate a natural language response for the user.
    """
    logger.info("[NODE] generate_response")
    
    if state["should_clarify"]:
        state = {
            **state,
            "response_text": state["clarification_options"][0],
        }
    elif state["action_results"]:
        last_result = state["action_results"][-1]
        if last_result.get("success"):
            state = {
                **state,
                "response_text": f"Done: {last_result.get('result', 'Success')}",
            }
        else:
            state = {
                **state,
                "response_text": f"Error: {last_result.get('error', 'Unknown error')}",
            }
    else:
        state = {
            **state,
            "response_text": "I'm ready to help.",
        }
    
    return state
```

#### Node 8: `emit_response`
**Responsibility:** Send response back via VoiceAgent  
**Input:** WorkflowState with response_text  
**Output:** Emits VoiceOutputEvent to event bus

```python
async def emit_response(state: WorkflowState, event_bus: EventBus) -> WorkflowState:
    """
    Emit VoiceOutputEvent to send response to user.
    """
    logger.info("[NODE] emit_response")
    
    await event_bus.emit(VoiceOutputEvent(
        text=state["response_text"],
        source="LangGraphBrain",
        correlation_id=state["correlation_id"],
    ))
    
    return state
```

---

## 4. Conditional Routing (@branch)

### Routing Logic

```python
def route_after_classify(state: WorkflowState) -> str:
    """Route after intent classification."""
    if state["error_count"] > 0:
        return "handle_error"
    if state["should_clarify"]:
        return "generate_response"
    return "route_intent"


def route_after_routing(state: WorkflowState) -> str:
    """Route after intent routing."""
    if state["should_clarify"]:
        return "generate_response"
    if state["is_multi_step"]:
        return "plan_multi_step"
    return "execute_action"


def route_after_plan(state: WorkflowState) -> str:
    """Route after multi-step planning."""
    return "execute_action"  # Execute first step


def route_after_action(state: WorkflowState) -> str:
    """Route after action execution."""
    if state["error_count"] > 0:
        return "handle_error"
    
    # Check if more steps in multi-step plan
    if state["is_multi_step"] and state["current_step"] < len(state["action_results"]):
        state["current_step"] += 1
        return "execute_action"
    
    return "generate_response"


def route_after_error(state: WorkflowState) -> str:
    """Route after error handling."""
    if state["retry_count"] < state["max_retries"]:
        return "execute_action"  # Retry
    return "generate_response"  # Give up, respond to user
```

---

## 5. LangGraph Workflow Construction

```python
from langgraph.graph import StateGraph, END
from typing_extensions import TypedDict

def build_workflow() -> StateGraph:
    """
    Build the complete LangGraph workflow.
    """
    workflow = StateGraph(WorkflowState)
    
    # =========================================================================
    # Add nodes
    # =========================================================================
    workflow.add_node("capture_input", capture_input)
    workflow.add_node("classify_intent", classify_intent)
    workflow.add_node("route_intent", route_intent)
    workflow.add_node("plan_multi_step", plan_multi_step)
    workflow.add_node("execute_action", execute_action)
    workflow.add_node("handle_error", handle_error)
    workflow.add_node("generate_response", generate_response)
    workflow.add_node("emit_response", emit_response)
    
    # =========================================================================
    # Add edges (transitions)
    # =========================================================================
    
    # Start
    workflow.set_entry_point("capture_input")
    
    # capture_input → classify_intent
    workflow.add_edge("capture_input", "classify_intent")
    
    # classify_intent → [route_intent | handle_error | generate_response]
    workflow.add_conditional_edges(
        "classify_intent",
        route_after_classify,
        {
            "route_intent": "route_intent",
            "handle_error": "handle_error",
            "generate_response": "generate_response",
        },
    )
    
    # route_intent → [execute_action | plan_multi_step | generate_response]
    workflow.add_conditional_edges(
        "route_intent",
        route_after_routing,
        {
            "execute_action": "execute_action",
            "plan_multi_step": "plan_multi_step",
            "generate_response": "generate_response",
        },
    )
    
    # plan_multi_step → execute_action
    workflow.add_edge("plan_multi_step", "execute_action")
    
    # execute_action → [handle_error | execute_action | generate_response]
    workflow.add_conditional_edges(
        "execute_action",
        route_after_action,
        {
            "handle_error": "handle_error",
            "execute_action": "execute_action",
            "generate_response": "generate_response",
        },
    )
    
    # handle_error → [execute_action | generate_response]
    workflow.add_conditional_edges(
        "handle_error",
        route_after_error,
        {
            "execute_action": "execute_action",
            "generate_response": "generate_response",
        },
    )
    
    # generate_response → emit_response
    workflow.add_edge("generate_response", "emit_response")
    
    # emit_response → END
    workflow.add_edge("emit_response", END)
    
    return workflow.compile()
```

---

## 6. Integration with Event Bus

### LangGraphBrain Class

```python
class LangGraphBrain:
    """
    LangGraph-based replacement for the orchestrator Brain.
    
    Maintains event bus compatibility while using LangGraph for routing.
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        agents: Dict[str, BaseAgent],
        voice_agent: VoiceAgent,
        intent_agent: IntentAgent,
        system_agent: SystemAgent,
        memory_agent: MemoryAgent,
    ):
        self.event_bus = event_bus
        self.agents = agents
        self.voice_agent = voice_agent
        self.intent_agent = intent_agent
        self.system_agent = system_agent
        self.memory_agent = memory_agent
        
        # Build LangGraph workflow
        self.workflow = build_workflow()
        
        # Subscribe to voice input
        self.event_bus.subscribe(VoiceInputEvent, self._on_voice_input)
    
    async def _on_voice_input(self, event: VoiceInputEvent) -> None:
        """
        Entry point: user speaks.
        
        Initializes workflow state and runs graph.
        """
        logger.info(f"[BRAIN] Voice input: {event.text[:50]}...")
        
        # Initialize state
        initial_state = WorkflowState(
            user_input=event.text,
            correlation_id=event.event_id,
            intent=None,
            intent_confidence=0.0,
            entities={},
            raw_text=event.text,
            target_agent=None,
            action=None,
            is_multi_step=False,
            action_results=[],
            current_step=0,
            max_retries=3,
            retry_count=0,
            last_error=None,
            error_count=0,
            should_clarify=False,
            clarification_options=[],
            response_text="",
            context_snapshot={},
            events_to_emit=[],
        )
        
        # Run workflow
        try:
            final_state = await self.workflow.ainvoke(
                initial_state,
                config={
                    "event_bus": self.event_bus,
                    "agents": self.agents,
                    "intent_agent": self.intent_agent,
                },
            )
            
            logger.info(f"[BRAIN] Workflow completed: {final_state['response_text']}")
            
        except Exception as e:
            logger.error(f"[BRAIN] Workflow error: {e}", exc_info=True)
            
            # Fallback response
            await self.event_bus.emit(VoiceOutputEvent(
                text="I encountered an error. Please try again.",
                source="LangGraphBrain",
                correlation_id=event.event_id,
            ))
```

---

## 7. Retry and Fallback Strategy

### Retry Pattern (Built into nodes)

```python
async def execute_action_with_retry(
    state: WorkflowState,
    event_bus: EventBus,
    agents: Dict[str, Any],
    max_retries: int = 3,
) -> WorkflowState:
    """
    Execute action with exponential backoff retry.
    """
    for attempt in range(max_retries):
        try:
            logger.info(f"[RETRY] Attempt {attempt + 1}/{max_retries}")
            
            result = await execute_action(state, event_bus, agents)
            
            if result["action_results"][-1].get("success"):
                return result
            
            # Exponential backoff
            await asyncio.sleep(2 ** attempt)
            
        except Exception as e:
            logger.error(f"[RETRY] Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
    
    return state
```

### Fallback Pattern

```python
def fallback_to_clarification(state: WorkflowState) -> WorkflowState:
    """
    Fallback when primary path fails.
    
    → Ask user to clarify or rephrase
    """
    state = {
        **state,
        "should_clarify": True,
        "clarification_options": [
            "I'm having trouble processing that request.",
            "Could you rephrase or try again?",
        ],
        "response_text": "I'm having trouble processing that request. Could you rephrase or try again?",
    }
    return state


def fallback_to_fallback_agent(state: WorkflowState) -> WorkflowState:
    """
    Fallback: try MemoryAgent or another agent.
    
    If SystemAgent fails, try asking MemoryAgent to recall context.
    """
    if state["target_agent"] != "MemoryAgent":
        state = {
            **state,
            "target_agent": "MemoryAgent",
            "action": "recall_context",
            "retry_count": 0,
        }
    return state
```

---

## 8. State Transitions Diagram

```
START
  ↓
[capture_input]
  ↓
[classify_intent] ──ERROR──→ [handle_error]
  ↓ (success)                   ↓
[route_intent] ──ERROR──→ [handle_error]
  ↓ (success)                   ↓
  ├─→ [execute_action]    [retry?]
  │     (single-step)        ↓
  │       ↓                  YES ──→ [execute_action]
  │     (success/error)       ↓
  │       ↓                  NO ──→ [generate_response]
  │   (more steps?) YES ──→ [execute_action] (loop)
  │       ↓
  │      NO
  │       ↓
  ├─→ [plan_multi_step]
  │     (multi-step)
  │       ↓
  │   [execute_action] (Step 1)
  │       ↓
  │   (more steps?) YES ──→ [execute_action] (Step 2, N...)
  │       ↓
  │      NO
  │       ↓
  └─→ [generate_response]
        ↓
    [emit_response]
        ↓
      END
```

---

## 9. Complete Example: "Open Safari and go to YouTube"

### Flow Trace

```
1. User: "Open Safari and go to YouTube"

2. [capture_input]
   user_input: "Open Safari and go to YouTube"
   correlation_id: <uuid>

3. [classify_intent]
   intent: OPEN_APP_AND_NAVIGATE
   entities: {
     "app": "Safari",
     "url": "https://youtube.com"
   }
   confidence: 0.92

4. [route_intent]
   target_agent: SystemAgent
   action: open_app_and_navigate
   is_multi_step: true

5. [plan_multi_step]
   action_results: [
     {step: 0, status: "pending", action: "open_app"},
     {step: 1, status: "pending", action: "open_url"}
   ]
   current_step: 0

6. [execute_action] ← Step 1
   Emit: ActionRequestEvent(
     action="open_app",
     target_agent="SystemAgent",
     parameters={"app": "Safari"}
   )
   Receive: ActionResultEvent(success=true, result="Safari opened")
   action_results[0].status = "completed"
   action_results[0].success = true

7. [execute_action] ← Step 2
   Emit: ActionRequestEvent(
     action="open_url",
     target_agent="SystemAgent",
     parameters={"url": "https://youtube.com"}
   )
   Receive: ActionResultEvent(success=true, result="YouTube loaded")
   action_results[1].status = "completed"
   action_results[1].success = true

8. [generate_response]
   response_text: "Done: Safari opened and YouTube loaded"

9. [emit_response]
   Emit: VoiceOutputEvent(
     text="Done: Safari opened and YouTube loaded",
     source="LangGraphBrain"
   )

10. END
```

---

## 10. Error Handling Example: "Open NonExistentApp"

### Flow Trace (with Retry & Fallback)

```
1. User: "Open NonExistentApp"

2. [capture_input]
   user_input: "Open NonExistentApp"

3. [classify_intent]
   intent: OPEN_APP
   entities: {"app": "NonExistentApp"}

4. [route_intent]
   target_agent: SystemAgent
   action: open_app
   is_multi_step: false

5. [execute_action] ← Attempt 1
   error_count: 0
   retry_count: 0
   Emit: ActionRequestEvent(action="open_app", parameters={"app": "NonExistentApp"})
   Receive: ActionResultEvent(success=false, error="App not found: NonExistentApp")
   error_count: 1

6. [handle_error]
   retry_count < max_retries? → YES (0 < 3)
   → retry_count: 1

7. [execute_action] ← Attempt 2 (same request, maybe with typo correction)
   Receive: ActionResultEvent(success=false, error="App not found")
   error_count: 2
   retry_count: 2

8. [handle_error]
   retry_count < max_retries? → YES (2 < 3)
   → retry_count: 3

9. [execute_action] ← Attempt 3 (final retry)
   Receive: ActionResultEvent(success=false, error="App not found")
   error_count: 3
   retry_count: 3

10. [handle_error]
    retry_count < max_retries? → NO (3 >= 3)
    → Fallback: should_clarify = true
    clarification_options: ["I couldn't find that app. What app would you like to open?"]

11. [generate_response]
    response_text: "I couldn't find that app. What app would you like to open?"

12. [emit_response]
    Emit VoiceOutputEvent

13. END
```

---

## 11. Migration Checklist

### Phase 1: Setup (Week 1)
- [ ] Install LangGraph: `pip install langgraph`
- [ ] Create `langgraph_brain.py` with state schema
- [ ] Copy node implementations
- [ ] Build workflow graph

### Phase 2: Integration (Week 2)
- [ ] Update `Brain.__init__` to instantiate `LangGraphBrain` alongside old Brain
- [ ] Keep event bus subscriptions intact
- [ ] Dual-write: both old and new orchestrator handle events
- [ ] Add feature flag: `USE_LANGGRAPH_BRAIN=True`

### Phase 3: Testing (Week 3)
- [ ] Unit test each node independently
- [ ] Integration tests for complete workflows
- [ ] Compare outputs (old Brain vs LangGraphBrain)
- [ ] Load test with concurrent requests

### Phase 4: Cutover (Week 4)
- [ ] Switch feature flag to `USE_LANGGRAPH_BRAIN=True` in production
- [ ] Keep old Brain as fallback
- [ ] Monitor error rates, latency, success rates
- [ ] Rollback plan if needed

### Phase 5: Cleanup (Week 5+)
- [ ] Deprecate old Brain
- [ ] Remove old orchestrator code
- [ ] Archive Brain tests that are now in LangGraph tests

---

## 12. Benefits & Trade-offs

### ✅ Benefits
1. **Deterministic routing**: Edges are explicit, not implicit subscriptions
2. **Easier debugging**: Clear state flow visible via LangGraph Studio
3. **Better error handling**: Retry/fallback baked into graph
4. **Type safety**: TypedDict + LangGraph validation
5. **Centralized execution**: Single place to trace, monitor, optimize
6. **Scalability**: LangGraph supports persistence, checkpointing

### ⚠️ Trade-offs
1. **Event bus still needed**: Can't fully eliminate it (other agents depend on events)
2. **Added dependency**: LangGraph version management
3. **State explosion**: WorkflowState may grow large with complex commands
4. **Async complexity**: Still need careful async/await handling

---

## 13. Future Enhancements

1. **Human-in-the-loop**: Add nodes for user approval/clarification
2. **Parallel steps**: Execute independent steps concurrently
3. **Tool calling**: Integrate LLM tool calls for dynamic agent invocation
4. **Persistence**: Save workflow state between sessions
5. **Analytics**: Track node execution times, error rates by intent
6. **Custom routing**: Add ML model to predict best agent for intent

---

## 14. Conclusion

By migrating to LangGraph, your orchestrator becomes:
- **Observable**: Clear node-by-node execution flow
- **Maintainable**: Centralized routing logic, explicit state transitions
- **Reliable**: Built-in retry, fallback, error handling patterns
- **Scalable**: Ready for distributed execution, human-in-the-loop workflows
- **Compatible**: Event bus remains for inter-agent communication

The event-driven architecture is preserved; LangGraph provides the **orchestration layer** on top.
