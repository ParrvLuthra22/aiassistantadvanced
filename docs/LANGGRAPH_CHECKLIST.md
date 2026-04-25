# LangGraph Migration - Quick Reference Checklist

## 📋 What You Received

### Documentation Files (4)
- [ ] `LANGGRAPH_ARCHITECTURE.md` - Complete design (14 sections, 3000+ words)
- [ ] `LANGGRAPH_MIGRATION_GUIDE.md` - Practical guide (2500+ words)
- [ ] `LANGGRAPH_DIAGRAMS.md` - Visual diagrams (8 diagrams)
- [ ] `LANGGRAPH_SUMMARY.md` - Overview & FAQ

### Implementation Files (2)
- [ ] `orchestrator/langgraph_brain.py` - Full LangGraph implementation (500 lines)
- [ ] `orchestrator/langgraph_integration.py` - Integration patterns (400 lines)

### Deliverables Checklist
- [x] Architecture diagram ✅
- [x] Node definitions ✅
- [x] State schema ✅
- [x] Conditional routing ✅
- [x] Retry logic ✅
- [x] Fallback patterns ✅
- [x] Event bus integration ✅
- [x] Pseudocode examples ✅
- [x] Migration plan ✅
- [x] Integration patterns ✅

---

## 🚀 Getting Started (30 minutes)

### Step 1: Install (5 min)
```bash
pip install langgraph langchain
```

### Step 2: Review Architecture (10 min)
- Open `LANGGRAPH_ARCHITECTURE.md`
- Read sections 1-3 (overview + state schema)
- Skim section 4 (node definitions)

### Step 3: Understand Implementation (10 min)
- Open `orchestrator/langgraph_brain.py`
- Read the class docstrings
- Scan through node implementations
- Note the LangGraphBrain.\_on_voice_input() entry point

### Step 4: Choose Integration Strategy (5 min)
- Simple: OrchestratorFactory with feature flag
- Safe: OrchestratorWithFallback for production
- Testing: DualModeOrchestrator for validation

---

## 📈 Phase-by-Phase Implementation

### Phase 1: Setup (Week 1)
- [ ] Read LANGGRAPH_ARCHITECTURE.md completely
- [ ] Copy langgraph_brain.py to orchestrator/
- [ ] Copy langgraph_integration.py to orchestrator/
- [ ] `pip install langgraph langchain`
- [ ] Create unit tests for each node
- [ ] Test LangGraphBrain locally

**Success criteria:** Can instantiate LangGraphBrain, no import errors

### Phase 2: Integration (Week 2)
- [ ] Use OrchestratorFactory in main.py
- [ ] Set feature flag: `USE_LANGGRAPH_BRAIN=false` (default to old Brain)
- [ ] Deploy to staging
- [ ] Verify old Brain still works
- [ ] Test basic voice input → response flow

**Success criteria:** Staging env works with old Brain

### Phase 3: Testing (Week 3)
- [ ] Set feature flag: `USE_LANGGRAPH_BRAIN=true`
- [ ] Deploy to staging
- [ ] Run manual tests:
  - [ ] Simple intent: "What's the time?"
  - [ ] Multi-step: "Open Safari and go to YouTube"
  - [ ] Error handling: "Open NonexistentApp"
  - [ ] Unclear input: Gibberish → clarification
- [ ] Load test: 50+ concurrent requests
- [ ] Monitor metrics
- [ ] Compare vs old Brain output

**Success criteria:** 99%+ success rate, latency ≤ 105% of baseline

### Phase 4: Validation (Week 3-4)
- [ ] Set feature flag: `USE_LANGGRAPH_BRAIN=true` + `DUAL_MODE=true`
- [ ] Deploy to staging
- [ ] Run for 24 hours
- [ ] Verify no duplicate responses
- [ ] Compare outputs between orchestrators
- [ ] Log any differences
- [ ] Adjust if needed

**Success criteria:** Zero duplicate responses, outputs match

### Phase 5: Production Rollout (Week 4)
- [ ] Set `DUAL_MODE=false`, `USE_LANGGRAPH_BRAIN=false` initially
- [ ] Deploy to production
- [ ] Verify working with old Brain
- [ ] Set feature flag to serve 10% of traffic: `USE_LANGGRAPH_BRAIN=true`
- [ ] Monitor for 6 hours
  - [ ] Success rate (target: 99.5%+)
  - [ ] Error rate (target: < 1%)
  - [ ] Latency p95 (target: ≤ 630ms)
  - [ ] No uncaught exceptions

**If all good:** Increase to 25%  
**If issues:** Rollback to 0% immediately

- [ ] 25% traffic (monitor 6 hours)
- [ ] 50% traffic (canary, monitor 12 hours)
- [ ] 100% traffic (full rollout)

**Success criteria:** All metrics passing at 100%

### Phase 6: Cleanup (Week 5)
- [ ] Keep both Brain and LangGraphBrain for 1 week
- [ ] Set OrchestratorWithFallback as permanent setup
- [ ] Monitor production metrics
- [ ] After 1 week stable: can remove fallback
- [ ] Archive old Brain code (don't delete, keep in git history)
- [ ] Update documentation

**Success criteria:** LangGraphBrain in production, metrics stable

---

## 🔍 Quick Debugging

### Check if LangGraphBrain is active
```python
if isinstance(orchestrator, LangGraphBrain):
    print("LangGraphBrain is active")
else:
    print("Old Brain is active")
```

### Enable detailed logging
```python
import logging
logging.getLogger("orchestrator.langgraph_brain").setLevel(logging.DEBUG)
```

### View node execution flow
```
[NODE] capture_input: User input...
[NODE] classify_intent: Intent=OPEN_APP confidence=0.95
[NODE] route_intent: target=SystemAgent action=open_app
[NODE] execute_action: Waiting for SystemAgent...
[NODE] generate_response: response_text='Done: ...'
[NODE] emit_response: Sent to user
```

### Check WorkflowState at each node
```python
# Add to any node:
logger.debug(f"State: {json.dumps(state, default=str, indent=2)}")
```

### Monitor metrics
```python
from orchestrator.langgraph_integration import OrchestratorMetrics

metrics = OrchestratorMetrics("LangGraphBrain")
# ... (track requests)
print(metrics.get_summary())
```

---

## 📊 Key Metrics to Monitor

### During Migration
| Metric | Target | Alarm |
|--------|--------|-------|
| Success Rate | ≥ 99.5% | < 99.0% |
| Latency P50 | ~300ms | > 320ms |
| Latency P95 | ≤ 630ms | > 660ms |
| Error Rate | < 1% | ≥ 2% |
| Retry Success | ≥ 90% | < 80% |
| Duplicate Responses | 0 | > 0 |

### Per-Intent Breakdown
Track success rate for each intent type:
- OPEN_APP
- CLOSE_APP
- GET_TIME
- CONTROL_VOLUME
- (etc.)

Look for patterns in failures.

---

## ⚡ Common Issues & Solutions

### Issue: Import Error
```
ImportError: No module named 'langgraph'
```
**Solution:**
```bash
pip install langgraph langchain
```

### Issue: Event Bus Not Working
```
Events not reaching agents
```
**Solution:**
- Verify event_bus.subscribe() called before orchestrator.start()
- Check event type matches (IntentRecognizedEvent, ActionRequestEvent)
- Enable debug logging on event bus

### Issue: Timeout Waiting for Agent
```
[NODE] execute_action: Waiting for SystemAgent...
(10 second timeout)
```
**Solution:**
- Check SystemAgent is running
- Check ActionResultEvent being emitted
- Verify correlation_id matches
- Increase timeout in execute_action() if needed

### Issue: Multi-Step Command Fails on Step 2
```
Step 1 succeeds, Step 2 fails repeatedly
```
**Solution:**
- Check if Step 2 depends on Step 1 result
- Verify parameters for Step 2 are correct
- Check if Step 2's agent is available
- Add explicit handling in _get_steps_for_intent()

### Issue: Duplicate Responses in Dual Mode
```
User hears response twice (both orchestrators)
```
**Solution:**
- Disable dual mode
- Only one orchestrator should emit VoiceOutputEvent
- Use OrchestratorFactory instead

### Issue: Rollback Doesn't Work
```
Still using LangGraphBrain after setting flag to false
```
**Solution:**
- Verify environment variable: `echo $USE_LANGGRAPH_BRAIN`
- Restart application (feature flag checked at startup)
- Check code: `USE_LANGGRAPH_BRAIN = os.getenv("USE_LANGGRAPH_BRAIN", "false").lower() == "true"`

---

## 🛑 Rollback Procedure

### Immediate Rollback (takes <1 minute)
```bash
# Terminal 1: Stop the app
^C

# Terminal 2: Set feature flag
export USE_LANGGRAPH_BRAIN=false

# Terminal 1: Restart
python main.py
```

### Verification After Rollback
```python
# Should see:
# - Old Brain starting up
# - All events routing through old Brain
# - No LangGraph-related log messages
```

### Gradual Rollback (if specific intent fails)
```python
# In route_after_routing():
if state["intent"] == "PROBLEMATIC_INTENT":
    # Force use of old Brain for this intent
    return "generate_response"  # Skip execution
```

### Full Rollback (keep both running indefinitely)
```python
# Use OrchestratorWithFallback indefinitely
# LangGraphBrain is primary, legacy is fallback
# If you need to disable LangGraphBrain:
await orchestrator.disable_primary()
```

---

## ✅ Success Checklist

### Week 1: Setup
- [ ] Code compiles without errors
- [ ] All imports successful
- [ ] Unit tests for nodes pass
- [ ] LangGraphBrain instantiates
- [ ] Graph compiles

### Week 2: Staging Integration
- [ ] Deploy with old Brain (USE_LANGGRAPH_BRAIN=false)
- [ ] All existing functionality works
- [ ] No new bugs introduced
- [ ] Metrics collection in place

### Week 3: Testing
- [ ] Deploy with LangGraphBrain (USE_LANGGRAPH_BRAIN=true)
- [ ] Simple intents work (GET_TIME, OPEN_APP)
- [ ] Multi-step intents work
- [ ] Error handling works (retries)
- [ ] Latency acceptable
- [ ] Load test passes (50+ concurrent)

### Week 3-4: Validation
- [ ] Dual mode (DUAL_MODE=true)
- [ ] Run for 24 hours
- [ ] Zero duplicate responses
- [ ] Output matches between orchestrators
- [ ] No conflicts or races

### Week 4: Production Rollout
- [ ] 10% traffic: Success rate ≥ 99.5%
- [ ] 10% traffic: Latency p95 ≤ 630ms
- [ ] 10% traffic: Error rate < 1%
- [ ] Increase to 25%: Same metrics ✓
- [ ] Increase to 50%: Same metrics ✓
- [ ] Increase to 100%: Same metrics ✓

### Week 5: Cleanup
- [ ] 1 week stable at 100%
- [ ] All metrics passing
- [ ] No production incidents
- [ ] Old Brain code archived
- [ ] Documentation updated

---

## 📖 Documentation Quick Links

| Need | File | Sections |
|------|------|----------|
| Understand architecture | LANGGRAPH_ARCHITECTURE.md | 1-10 |
| See examples | LANGGRAPH_ARCHITECTURE.md | 11-12 |
| Step-by-step guide | LANGGRAPH_MIGRATION_GUIDE.md | All |
| Visual diagrams | LANGGRAPH_DIAGRAMS.md | All |
| See code | orchestrator/langgraph_brain.py | All |
| Integration patterns | orchestrator/langgraph_integration.py | All |
| Q&A | LANGGRAPH_SUMMARY.md | FAQ section |

---

## 🎯 Decision Tree

```
Do you want to migrate to LangGraph?
├─ YES
│  ├─ Are you familiar with LangGraph?
│  │  ├─ NO → Read LANGGRAPH_ARCHITECTURE.md (30-45 min)
│  │  └─ YES → Continue
│  ├─ Ready to test in dev?
│  │  ├─ YES → Copy files, set USE_LANGGRAPH_BRAIN=true
│  │  └─ NO → Read LANGGRAPH_MIGRATION_GUIDE.md first
│  ├─ Want to test both simultaneously?
│  │  ├─ YES → Use DualModeOrchestrator
│  │  └─ NO → Use simple feature flag
│  └─ Ready for production?
│     ├─ YES → Use OrchestratorWithFallback
│     └─ NO → Stay in testing mode longer
└─ NO
   └─ Keep using old Brain (no changes needed)
```

---

## 🚀 TL;DR

1. **Install:** `pip install langgraph langchain`
2. **Copy:** langgraph_brain.py + langgraph_integration.py
3. **Test:** `USE_LANGGRAPH_BRAIN=true python main.py`
4. **Monitor:** Check metrics in LANGGRAPH_MIGRATION_GUIDE.md
5. **Rollout:** Follow 5-week plan in langgraph_integration.py
6. **Done:** LangGraph orchestrator in production ✅

---

**Everything is ready. Pick one document above and start.** 🎯
