"""
Memory Agent Module - Context and conversation history management.

This agent handles persistent and ephemeral memory for the assistant:
    - Short-term memory (current session context)
    - Long-term memory (persistent storage)
    - Conversation history
    - User preferences and facts
    - Semantic search over memories

Architecture:
    - In-memory cache for fast access
    - SQLite backend for persistence
    - Future: Vector database for semantic search

Memory Types:
    - short_term: Session-scoped, expires after TTL
    - long_term: Persistent, survives restarts
    - episodic: Conversation fragments with timestamps
    - semantic: Embeddings for similarity search (future)

TODO: Add vector database integration (ChromaDB, Pinecone)
TODO: Add memory summarization for long conversations
TODO: Add memory importance scoring
TODO: Add memory consolidation (short -> long term)
TODO: Add user profile learning
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID, uuid4

from agents.base_agent import AgentCapability, BaseAgent
from bus.event_bus import EventBus
from schemas.events import (
    ConversationContextEvent,
    MemoryQueryEvent,
    MemoryQueryResultEvent,
    MemoryStoreEvent,
    VoiceInputEvent,
    IntentRecognizedEvent,
    ResponseGeneratedEvent,
)
from utils.logger import get_logger


logger = get_logger(__name__)


# =============================================================================
# Memory Item
# =============================================================================

@dataclass
class MemoryItem:
    """
    A single memory item.
    
    Attributes:
        id: Unique identifier
        key: Memory key for retrieval
        value: The stored value (any JSON-serializable data)
        memory_type: short_term, long_term, or episodic
        created_at: When the memory was created
        expires_at: When the memory expires (None = never)
        metadata: Additional metadata for search/filtering
    """
    
    id: UUID = field(default_factory=uuid4)
    key: str = ""
    value: Any = None
    memory_type: str = "short_term"
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if the memory has expired."""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": str(self.id),
            "key": self.key,
            "value": self.value,
            "memory_type": self.memory_type,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryItem":
        """Create from dictionary."""
        return cls(
            id=UUID(data["id"]),
            key=data["key"],
            value=data["value"],
            memory_type=data["memory_type"],
            created_at=datetime.fromisoformat(data["created_at"]),
            expires_at=datetime.fromisoformat(data["expires_at"]) if data["expires_at"] else None,
            metadata=data.get("metadata", {}),
        )


@dataclass
class ConversationTurn:
    """
    A single turn in a conversation.
    
    Attributes:
        role: 'user' or 'assistant'
        content: The message content
        timestamp: When the turn occurred
        intent: Detected intent (for user turns)
    """
    
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    intent: Optional[str] = None
    
    def to_dict(self) -> Dict[str, str]:
        """Convert to dict for LLM context."""
        return {
            "role": self.role,
            "content": self.content,
        }


# =============================================================================
# Memory Storage Backend
# =============================================================================

class MemoryStore:
    """
    SQLite-based memory storage.
    
    Provides persistent storage for long-term memory
    and session-based caching for short-term memory.
    
    TODO: Add connection pooling
    TODO: Add async operations
    TODO: Add migration support
    """
    
    def __init__(self, db_path: str = "data/memory.db"):
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        
        # In-memory cache for fast access
        self._cache: Dict[str, MemoryItem] = {}
    
    async def initialize(self) -> None:
        """Initialize the database."""
        # Create directory if needed
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Connect to database
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        
        # Create tables
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT,
                metadata TEXT,
                UNIQUE(key, memory_type)
            )
        """)
        
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_key 
            ON memories(key)
        """)
        
        self._conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_memories_type 
            ON memories(memory_type)
        """)
        
        self._conn.commit()
        
        logger.info(f"Memory store initialized at {self._db_path}")
    
    async def shutdown(self) -> None:
        """Close database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
    
    async def store(self, item: MemoryItem) -> None:
        """
        Store a memory item.
        
        Args:
            item: The memory item to store
        """
        # Add to cache
        cache_key = f"{item.memory_type}:{item.key}"
        self._cache[cache_key] = item
        
        # Persist long-term memories
        if item.memory_type == "long_term" and self._conn:
            self._conn.execute("""
                INSERT OR REPLACE INTO memories 
                (id, key, value, memory_type, created_at, expires_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                str(item.id),
                item.key,
                json.dumps(item.value),
                item.memory_type,
                item.created_at.isoformat(),
                item.expires_at.isoformat() if item.expires_at else None,
                json.dumps(item.metadata),
            ))
            self._conn.commit()
    
    async def get(self, key: str, memory_type: str = "short_term") -> Optional[MemoryItem]:
        """
        Retrieve a memory by key.
        
        Args:
            key: Memory key
            memory_type: Type of memory
        
        Returns:
            MemoryItem or None if not found
        """
        cache_key = f"{memory_type}:{key}"
        
        # Check cache first
        if cache_key in self._cache:
            item = self._cache[cache_key]
            if not item.is_expired:
                return item
            else:
                del self._cache[cache_key]
        
        # Check database for long-term memories
        if memory_type == "long_term" and self._conn:
            cursor = self._conn.execute(
                "SELECT * FROM memories WHERE key = ? AND memory_type = ?",
                (key, memory_type),
            )
            row = cursor.fetchone()
            if row:
                item = MemoryItem(
                    id=UUID(row["id"]),
                    key=row["key"],
                    value=json.loads(row["value"]),
                    memory_type=row["memory_type"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                )
                self._cache[cache_key] = item
                return item
        
        return None
    
    async def query(
        self,
        query: str,
        memory_type: str = "all",
        limit: int = 10,
    ) -> List[MemoryItem]:
        """
        Search memories by query.
        
        Currently does simple substring matching.
        
        TODO: Add full-text search
        TODO: Add semantic search with embeddings
        
        Args:
            query: Search query
            memory_type: Type to search ('all' for all types)
            limit: Maximum results
        
        Returns:
            List of matching MemoryItems
        """
        results = []
        
        # Search cache
        for cache_key, item in self._cache.items():
            if item.is_expired:
                continue
            
            if memory_type != "all" and item.memory_type != memory_type:
                continue
            
            # Simple substring search
            if query.lower() in item.key.lower():
                results.append(item)
            elif isinstance(item.value, str) and query.lower() in item.value.lower():
                results.append(item)
            
            if len(results) >= limit:
                break
        
        # Search database for long-term memories
        if len(results) < limit and self._conn:
            type_filter = "" if memory_type == "all" else f"AND memory_type = '{memory_type}'"
            cursor = self._conn.execute(f"""
                SELECT * FROM memories 
                WHERE (key LIKE ? OR value LIKE ?) {type_filter}
                LIMIT ?
            """, (f"%{query}%", f"%{query}%", limit - len(results)))
            
            for row in cursor:
                item = MemoryItem(
                    id=UUID(row["id"]),
                    key=row["key"],
                    value=json.loads(row["value"]),
                    memory_type=row["memory_type"],
                    created_at=datetime.fromisoformat(row["created_at"]),
                    expires_at=datetime.fromisoformat(row["expires_at"]) if row["expires_at"] else None,
                    metadata=json.loads(row["metadata"]) if row["metadata"] else {},
                )
                results.append(item)
        
        return results[:limit]
    
    async def delete(self, key: str, memory_type: str = "short_term") -> bool:
        """Delete a memory by key."""
        cache_key = f"{memory_type}:{key}"
        
        deleted = False
        
        if cache_key in self._cache:
            del self._cache[cache_key]
            deleted = True
        
        if memory_type == "long_term" and self._conn:
            cursor = self._conn.execute(
                "DELETE FROM memories WHERE key = ? AND memory_type = ?",
                (key, memory_type),
            )
            self._conn.commit()
            deleted = deleted or cursor.rowcount > 0
        
        return deleted
    
    async def clear_expired(self) -> int:
        """Remove all expired memories."""
        count = 0
        
        # Clear from cache
        expired_keys = [
            k for k, v in self._cache.items()
            if v.is_expired
        ]
        for key in expired_keys:
            del self._cache[key]
            count += 1
        
        # Clear from database
        if self._conn:
            now = datetime.utcnow().isoformat()
            cursor = self._conn.execute(
                "DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at < ?",
                (now,),
            )
            self._conn.commit()
            count += cursor.rowcount
        
        return count


# =============================================================================
# Memory Agent
# =============================================================================

class MemoryAgent(BaseAgent):
    """
    Agent responsible for memory and context management.
    
    This agent:
        - Stores and retrieves memories
        - Maintains conversation history
        - Provides context to other agents
        - Handles user preferences
    
    Configuration:
        memory.backend: Storage backend ('sqlite', 'memory')
        memory.sqlite.database_path: Path to SQLite database
        memory.short_term.ttl_seconds: TTL for short-term memory
        memory.conversation.max_turns: Max conversation turns to keep
    
    Events Consumed:
        - MemoryStoreEvent: Store a memory
        - MemoryQueryEvent: Query memories
        - VoiceInputEvent: Track user input
        - ResponseGeneratedEvent: Track assistant responses
    
    Events Produced:
        - MemoryQueryResultEvent: Query results
        - ConversationContextEvent: Conversation context
    
    TODO: Add memory pruning strategies
    TODO: Add memory importance scoring
    TODO: Add user profile management
    """
    
    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(name="MemoryAgent", event_bus=event_bus, config=config)
        
        # Storage
        self._store: Optional[MemoryStore] = None
        
        # Conversation history
        self._conversation: List[ConversationTurn] = []
        self._max_turns = self._get_config("memory.conversation.max_turns", 20)
        self._session_id = uuid4()
        
        # TTL settings
        self._short_term_ttl = self._get_config(
            "memory.short_term.ttl_seconds", 3600
        )
        
        # Cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
    
    @property
    def capabilities(self) -> List[AgentCapability]:
        """Define memory agent capabilities."""
        return [
            AgentCapability(
                name="memory_storage",
                description="Store and retrieve memories",
                input_events=["MemoryStoreEvent", "MemoryQueryEvent"],
                output_events=["MemoryQueryResultEvent"],
            ),
            AgentCapability(
                name="conversation_tracking",
                description="Track conversation history",
                input_events=["VoiceInputEvent", "ResponseGeneratedEvent"],
                output_events=["ConversationContextEvent"],
            ),
        ]
    
    async def _setup(self) -> None:
        """Initialize memory agent resources."""
        # Initialize storage
        db_path = self._get_config(
            "memory.sqlite.database_path",
            "data/memory.db",
        )
        self._store = MemoryStore(db_path)
        await self._store.initialize()
        
        # Subscribe to events
        self._subscribe(MemoryStoreEvent, self._handle_store)
        self._subscribe(MemoryQueryEvent, self._handle_query)
        self._subscribe(VoiceInputEvent, self._handle_voice_input)
        self._subscribe(ResponseGeneratedEvent, self._handle_response)
        
        # Start cleanup task
        self._cleanup_task = self._create_task(self._cleanup_loop())
        
        self._logger.info("Memory agent initialized")
    
    async def _teardown(self) -> None:
        """Cleanup memory agent resources."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        if self._store:
            await self._store.shutdown()
        
        self._logger.info("Memory agent shutdown complete")
    
    async def _handle_store(self, event: MemoryStoreEvent) -> None:
        """Handle memory store requests."""
        expires_at = None
        if event.ttl:
            expires_at = datetime.utcnow() + timedelta(seconds=event.ttl)
        elif event.memory_type == "short_term":
            expires_at = datetime.utcnow() + timedelta(seconds=self._short_term_ttl)
        
        item = MemoryItem(
            key=event.key,
            value=event.value,
            memory_type=event.memory_type,
            expires_at=expires_at,
        )
        
        await self._store.store(item)
        self._logger.debug(f"Stored memory: {event.key} ({event.memory_type})")
    
    async def _handle_query(self, event: MemoryQueryEvent) -> None:
        """Handle memory query requests."""
        results = await self._store.query(
            query=event.query,
            memory_type=event.memory_type,
            limit=event.limit,
        )
        
        await self._emit(MemoryQueryResultEvent(
            results=[item.to_dict() for item in results],
            query=event.query,
            total_matches=len(results),
            source=self._name,
            correlation_id=event.correlation_id,
        ))
    
    async def _handle_voice_input(self, event: VoiceInputEvent) -> None:
        """Track user input in conversation history."""
        turn = ConversationTurn(
            role="user",
            content=event.text,
        )
        self._add_turn(turn)
    
    async def _handle_response(self, event: ResponseGeneratedEvent) -> None:
        """Track assistant response in conversation history."""
        turn = ConversationTurn(
            role="assistant",
            content=event.text,
        )
        self._add_turn(turn)
    
    def _add_turn(self, turn: ConversationTurn) -> None:
        """Add a turn to conversation history."""
        self._conversation.append(turn)
        
        # Trim if too long
        if len(self._conversation) > self._max_turns:
            self._conversation = self._conversation[-self._max_turns:]
    
    async def get_conversation_context(self) -> ConversationContextEvent:
        """
        Get current conversation context.
        
        Returns:
            ConversationContextEvent with recent history
        """
        messages = [turn.to_dict() for turn in self._conversation]
        
        # Generate summary if conversation is long
        summary = ""
        if len(self._conversation) > 10:
            # TODO: Use LLM to generate summary
            summary = f"Conversation with {len(self._conversation)} turns"
        
        return ConversationContextEvent(
            messages=messages,
            summary=summary,
            session_id=self._session_id,
            source=self._name,
        )
    
    async def _cleanup_loop(self) -> None:
        """Periodically clean up expired memories."""
        while self.is_running:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                count = await self._store.clear_expired()
                if count > 0:
                    self._logger.debug(f"Cleaned up {count} expired memories")
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Error in cleanup loop: {e}")
    
    # =========================================================================
    # Convenience Methods
    # =========================================================================
    
    async def remember(
        self,
        key: str,
        value: Any,
        memory_type: str = "short_term",
        ttl: Optional[int] = None,
    ) -> None:
        """
        Store a memory directly.
        
        Convenience method for other agents to use.
        """
        await self._handle_store(MemoryStoreEvent(
            key=key,
            value=value,
            memory_type=memory_type,
            ttl=ttl,
        ))
    
    async def recall(
        self,
        key: str,
        memory_type: str = "short_term",
    ) -> Optional[Any]:
        """
        Retrieve a memory directly.
        
        Convenience method for other agents to use.
        """
        item = await self._store.get(key, memory_type)
        return item.value if item else None
    
    async def search(
        self,
        query: str,
        memory_type: str = "all",
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search memories directly.
        
        Convenience method for other agents to use.
        """
        items = await self._store.query(query, memory_type, limit)
        return [item.to_dict() for item in items]
    
    def get_session_id(self) -> UUID:
        """Get current session ID."""
        return self._session_id
    
    def clear_conversation(self) -> None:
        """Clear conversation history and start fresh."""
        self._conversation.clear()
        self._session_id = uuid4()
        self._logger.info("Conversation cleared")
