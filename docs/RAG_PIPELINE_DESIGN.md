# Production-Grade RAG Pipeline Design for Personal AI Assistant

## Goals

Design a robust Retrieval-Augmented Generation (RAG) system that:

- Integrates tightly with existing `MemoryAgent`
- Uses a vector database (`FAISS` or `Chroma`)
- Supports reliable chunking, embedding, retrieval optimization, and context-window packing
- Separates and governs short-term vs long-term memory
- Balances privacy, latency, cost, and answer quality

---

## 1) High-Level Architecture

```text
User Input
   |
   v
Intent + Query Understanding
(IntentAgent / query normalizer)
   |
   v
MemoryRouter
(Short-term, Long-term, Hybrid)
   |
   +------------------------+
   |                        |
   v                        v
Short-Term Store        Long-Term Store
(MemoryAgent SQLite)    (Vector DB: Chroma/FAISS)
   |                        |
   +-----------+------------+
               |
               v
Retriever
- dense retrieval
- metadata filters
- recency weighting
- MMR diversification
               |
               v
Reranker (optional but recommended)
(cross-encoder or LLM scoring)
               |
               v
Context Assembler
- token budgeting
- relevance + novelty + recency packing
- conversation compression
               |
               v
LLM Response Generation
               |
               v
Memory Write-Back
- episodic summary (short-term)
- salient facts/tasks/preferences (long-term)
```

---

## 2) Integration with `MemoryAgent`

Use `MemoryAgent` as the canonical memory orchestrator, not just a passive store.

### Proposed responsibilities

### `MemoryAgent` keeps owning:

- Session turn tracking (`VoiceInputEvent`, `IntentRecognizedEvent`)
- Fast key-value recall (`last_command`, recent turns, preferences)
- Short-term TTL governance

### New RAG layer around `MemoryAgent` adds:

- Vector indexing for long-term semantic recall
- Metadata enrichment (`intent`, `timestamp`, `source`, `salience`)
- Retrieval scoring + reranking
- Context packing for LLM prompt construction

### Minimal interface contract

```python
class RAGMemoryService:
    async def ingest_turn(self, text: str, role: str, intent: str, entities: dict, timestamp: float) -> None: ...
    async def retrieve(self, query: str, intent: str, top_k: int = 20) -> list[dict]: ...
    async def assemble_context(self, query: str, retrieved: list[dict], token_budget: int) -> str: ...
    async def commit_long_term(self, memory_items: list[dict]) -> None: ...
```

---

## 3) Vector DB Choice: FAISS vs Chroma

## Recommendation

- **Default for this assistant:** `Chroma` (simpler persistence, metadata filtering, quick local ops)
- **Use FAISS when:** very large memory scale and custom ANN tuning is required

| Dimension | Chroma | FAISS |
|---|---|---|
| Setup simplicity | Very high | Medium |
| Persistence | Built-in easy local persistence | Manual index + metadata handling |
| Metadata filters | Strong/easy | Needs sidecar metadata DB |
| ANN tuning flexibility | Moderate | High |
| Scale potential | Good | Excellent |
| Operational complexity | Low | Medium/High |

### Practical hybrid

If you outgrow Chroma:

- Keep metadata in SQLite/Postgres
- Move vectors to FAISS ANN index
- Maintain ID synchronization via `doc_id`

---

## 4) Chunking Strategy (Critical)

A personal assistant memory corpus is heterogeneous (chat turns, reminders, notes, system events). Use **type-aware chunking**.

### Chunk types

1. **Conversation chunks**
   - Group by turn windows (2–6 turns)
   - Keep speaker role markers (`user`, `assistant`)
2. **Task/reminder chunks**
   - Atomic chunk per task/reminder entry
3. **Preference/fact chunks**
   - Atomic and canonicalized (`"User prefers dark mode"`)
4. **Document/note chunks**
   - Semantic paragraph chunks with overlap

### Baseline parameters

- Target size: **300–500 tokens** per chunk
- Overlap: **50–80 tokens** (for long note/document chunks)
- Do **not** overlap atomic chunks (preferences, reminders)
- Include metadata with every chunk:
  - `memory_type`: `short_term|long_term`
  - `intent`
  - `timestamp`
  - `source` (voice/manual/system)
  - `salience` (0-1)
  - `entities` (flattened)

### Why this works

- Smaller chunks improve pinpoint retrieval precision
- Slight overlap improves recall continuity
- Atomic facts avoid duplicate/conflicting retrieval

---

## 5) Embedding Model Selection + Tradeoffs

Use embedding model based on **privacy mode** and **latency budget**.

## Recommended model tiers

### Tier A (local/private-first)

- `bge-small-en-v1.5` (fast, good quality)
- `e5-small-v2` (good semantic retrieval)

### Tier B (higher quality local)

- `bge-base-en-v1.5`
- `e5-base-v2`

### Tier C (hosted API quality)

- `text-embedding-3-small` (cost-effective, strong baseline)
- `text-embedding-3-large` (best quality, higher cost/latency)

| Model class | Quality | Latency | Cost | Privacy | Best for |
|---|---|---|---|---|---|
| Small local | Medium/High | Low | Low | High | Real-time memory recall |
| Base local | High | Medium | Low | High | Better semantic precision |
| Hosted small | High | Medium | Medium | Medium | Fast rollout, good quality |
| Hosted large | Very high | Medium/High | High | Medium | Complex recall/reasoning |

## Production recommendation

- Start with **`bge-small` local** for privacy + speed
- Add optional API model behind feature flag for A/B quality checks
- Version embeddings (`embedding_model_version`) so you can reindex safely

---

## 6) Retrieval Optimization

Do not rely on plain top-k cosine search only.

## Retrieval pipeline

1. **Query rewriting** (optional)
   - Expand pronouns and references using short-term memory
   - Example: `"open it" -> "open last referenced app: Safari"`

2. **Candidate retrieval (top_k=30-50)**
   - Dense vector similarity
   - Metadata filtering (`intent`, time range, memory_type)

3. **Score fusion**

$$
score = 0.60 \cdot sim + 0.20 \cdot recency + 0.15 \cdot salience + 0.05 \cdot intent\_match
$$

4. **MMR diversification**
   - Reduce near-duplicate chunks
   - Keep coverage across subtopics

5. **Rerank top 12–20**
   - Cross-encoder reranker (if latency allows)
   - Or lightweight LLM scoring prompt

6. **Final select top 6–10 for context assembly**

## Additional optimizations

- Per-intent retrieval profiles
  - `RECALL_MEMORY`: increase recency weight
  - `GENERAL_QUESTION`: increase semantic similarity + reranking
- Cache frequent query embeddings and top retrieval IDs
- Deduplicate by canonical fact key (`fact_id`)

---

## 7) Context Window Management

Use strict token budgeting with ranked packing.

## Token budget policy

For an LLM context limit `C`:

- System + policy prompt: `0.15C`
- Conversation short-term summary: `0.15C`
- Retrieved evidence chunks: `0.45C`
- User query + tool state: `0.10C`
- Safety margin: `0.15C`

## Packing algorithm

1. Always include:
   - latest user turn
   - compressed short-term dialogue summary
2. Add retrieved chunks by descending `final_score`
3. Enforce diversity constraints:
   - max 2 chunks per source/type
4. If over budget:
   - compress lower-ranked chunks
   - drop lowest marginal utility chunks
5. Attach citations (`[memory:doc_id]`) for traceability

### Compression policy

- Compress stale conversation chunks to bullet summaries
- Keep atomic facts uncompressed
- Keep timestamps for temporal grounding

---

## 8) Short-Term vs Long-Term Memory Strategy

Treat these as separate stores + policies, not just one DB with tags.

## Short-term memory (STM)

- Store in `MemoryAgent` SQLite
- TTL-based (`30 min` to `24 hr` depending on type)
- High recency, low permanence
- Includes:
  - recent turns
  - active tasks
  - disambiguation context

## Long-term memory (LTM)

- Store in vector DB + metadata store
- No short TTL; retention policy by salience and confidence
- Includes:
  - durable preferences
  - repeated behavior patterns
  - important facts/reminders

## Promotion policy (STM -> LTM)

Promote only if one or more:

- Explicit user intent (`"remember that"`)
- Repeated mention count >= N
- High salience score (named entities, preferences, commitments)
- Task lifecycle event (created/completed/rescheduled)

## Decay policy

- LTM gets **soft decay**, not hard delete:
  - lower ranking weight over time unless reinforced
- Hard delete only for:
  - user deletion request
  - retention policy expiration
  - contradiction superseded by newer canonical fact

---

## 9) Data Model

```json
{
  "doc_id": "uuid",
  "text": "User prefers concise answers in morning meetings.",
  "embedding": "vector",
  "memory_type": "long_term",
  "intent": "PREFERENCE",
  "timestamp": 1713744000,
  "salience": 0.91,
  "confidence": 0.88,
  "entity_keys": ["communication_style", "meeting_context"],
  "source": "voice",
  "version": 3,
  "is_active": true
}
```

---

## 10) End-to-End Runtime Flow

1. User asks query
2. Build `query_context` from STM (`MemoryAgent`)
3. Retrieve LTM candidates from vector DB
4. Fuse + rerank
5. Assemble context under token budget
6. Generate response with citations
7. Write interaction to STM
8. Run salience classifier for promotion to LTM
9. Upsert promoted chunks to vector DB

---

## 11) Observability + Quality Controls

Track at least:

- Retrieval hit-rate@k
- Context utilization ratio (used_tokens / budget_tokens)
- Grounded response rate (citation-backed answers)
- Hallucination feedback rate
- Promotion precision (LTM pollution rate)
- Median and p95 retrieval latency

Add guardrails:

- If retrieval confidence below threshold -> ask clarifying question
- If no relevant evidence -> explicitly say uncertainty
- If conflicting memories found -> prefer newer high-confidence item, mention conflict if needed

---

## 12) Suggested Implementation Plan (Incremental)

### Phase 1 (MVP)

- Chroma integration
- local embeddings (`bge-small`)
- basic dense top-k + metadata filter
- simple token budget packing

### Phase 2

- MMR + score fusion
- STM->LTM promotion policy
- citation output

### Phase 3

- reranker
- query rewriting
- adaptive retrieval profiles by intent

### Phase 4

- memory contradiction resolver
- periodic memory consolidation jobs
- offline eval suite for retrieval quality

---

## 13) Concrete Defaults (Recommended)

- Vector DB: `Chroma`
- Embedding model: `bge-small-en-v1.5`
- Chunk size: `400 tokens`
- Overlap: `64 tokens` (docs only)
- Retrieval: `top_k=40 -> rerank -> top_8`
- MMR lambda: `0.65`
- Context budget allocation:
  - 15% system
  - 15% STM summary
  - 45% retrieved evidence
  - 10% query/tool state
  - 15% safety margin

---

## 14) Example: Integration Sketch with `MemoryAgent`

```python
async def handle_user_query(query: str, intent: str, memory_agent, rag_service, llm):
    # 1) Gather short-term memory
    stm = {
        "last_command": memory_agent.get_last_command(),
        "recent": memory_agent.get_recent_conversation(max_turns=5),
    }

    # 2) Retrieve long-term context
    candidates = await rag_service.retrieve(query=query, intent=intent, top_k=40)

    # 3) Assemble prompt context within token budget
    context = await rag_service.assemble_context(
        query=query,
        retrieved=candidates,
        token_budget=3500,
    )

    # 4) Generate answer
    answer = await llm.generate(query=query, stm=stm, context=context)

    # 5) Write-back to short-term memory
    # (emit events or use MemoryAgent APIs)

    # 6) Promote salient memory to long-term (async background)
    # await rag_service.commit_long_term(promoted_items)

    return answer
```

---

## 15) Final Recommendation

For your assistant, start with:

1. **`MemoryAgent` as STM authority**
2. **`Chroma + bge-small` for LTM semantic recall**
3. **Type-aware chunking + metadata-rich indexing**
4. **MMR + recency/salience score fusion**
5. **Strict context budget packing with citations**

This gives strong quality with low operational complexity, and a clear path to FAISS/reranker upgrades as usage grows.
