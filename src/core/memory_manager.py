"""
Eva Live Memory Management Module

This module handles session context, conversation memory, and user preferences
for maintaining coherent and personalized interactions across Eva Live sessions.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import hashlib

# Redis for session storage
import redis
import pickle

from ..shared.config import get_config
from ..shared.models import PerformanceMetric, Conversation, SessionState
from ..input.nlu import NLUResult

@dataclass
class ConversationTurn:
    """A single turn in a conversation"""
    turn_id: str
    timestamp: datetime
    user_input: str
    eva_response: str
    nlu_result: Optional[NLUResult] = None
    context: Dict[str, Any] = None
    metadata: Dict[str, Any] = None

@dataclass
class SessionMemory:
    """Memory for a single session"""
    session_id: str
    user_id: Optional[str]
    created_at: datetime
    last_updated: datetime
    conversation_turns: List[ConversationTurn]
    session_context: Dict[str, Any]
    user_preferences: Dict[str, Any]
    presentation_state: Dict[str, Any]
    total_turns: int
    
@dataclass
class UserProfile:
    """Long-term user profile and preferences"""
    user_id: str
    name: Optional[str]
    preferences: Dict[str, Any]
    conversation_history_summary: str
    interaction_patterns: Dict[str, Any]
    created_at: datetime
    last_updated: datetime

class ConversationMemory:
    """Manages conversation context and history"""
    
    def __init__(self, max_turns: int = 50):
        self.max_turns = max_turns
        self.logger = logging.getLogger(__name__)
    
    def add_turn(self, memory: SessionMemory, user_input: str, eva_response: str, 
                 nlu_result: Optional[NLUResult] = None, context: Optional[Dict[str, Any]] = None) -> ConversationTurn:
        """Add a new conversation turn to memory"""
        turn_id = f"turn_{len(memory.conversation_turns) + 1}_{int(time.time())}"
        
        turn = ConversationTurn(
            turn_id=turn_id,
            timestamp=datetime.utcnow(),
            user_input=user_input,
            eva_response=eva_response,
            nlu_result=nlu_result,
            context=context or {},
            metadata={
                'turn_number': len(memory.conversation_turns) + 1,
                'session_id': memory.session_id
            }
        )
        
        memory.conversation_turns.append(turn)
        memory.total_turns += 1
        memory.last_updated = datetime.utcnow()
        
        # Maintain sliding window of conversations
        if len(memory.conversation_turns) > self.max_turns:
            memory.conversation_turns = memory.conversation_turns[-self.max_turns:]
        
        self.logger.debug(f"Added conversation turn {turn_id} to session {memory.session_id}")
        
        return turn
    
    def get_recent_context(self, memory: SessionMemory, num_turns: int = 5) -> str:
        """Get recent conversation context as a string"""
        recent_turns = memory.conversation_turns[-num_turns:] if memory.conversation_turns else []
        
        context_parts = []
        for turn in recent_turns:
            context_parts.append(f"User: {turn.user_input}")
            context_parts.append(f"Eva: {turn.eva_response}")
        
        return "\n".join(context_parts)
    
    def get_conversation_summary(self, memory: SessionMemory) -> str:
        """Generate a summary of the conversation so far"""
        if not memory.conversation_turns:
            return "No conversation history."
        
        # Simple summary - could be enhanced with AI summarization
        total_turns = len(memory.conversation_turns)
        duration = memory.last_updated - memory.created_at
        
        # Extract key topics from NLU results
        topics = set()
        intents = set()
        
        for turn in memory.conversation_turns:
            if turn.nlu_result:
                intents.add(turn.nlu_result.intent.value)
                for entity in turn.nlu_result.entities:
                    if entity.label in ['PRODUCT', 'FEATURE', 'TOPIC']:
                        topics.add(entity.text)
        
        summary = f"Conversation with {total_turns} turns over {duration}. "
        if topics:
            summary += f"Discussed topics: {', '.join(list(topics)[:5])}. "
        if intents:
            summary += f"Main intents: {', '.join(list(intents)[:3])}."
        
        return summary

class ContextManager:
    """Manages session context and state"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def update_presentation_context(self, memory: SessionMemory, slide_number: int, 
                                   content: str, user_engagement: float) -> None:
        """Update presentation context"""
        memory.presentation_state.update({
            'current_slide': slide_number,
            'current_content': content,
            'user_engagement_score': user_engagement,
            'slide_start_time': datetime.utcnow().isoformat(),
            'questions_on_slide': memory.presentation_state.get('questions_on_slide', 0)
        })
        
        memory.last_updated = datetime.utcnow()
    
    def add_presentation_question(self, memory: SessionMemory, question: str) -> None:
        """Track questions asked during presentation"""
        questions = memory.presentation_state.get('slide_questions', [])
        questions.append({
            'question': question,
            'timestamp': datetime.utcnow().isoformat(),
            'slide_number': memory.presentation_state.get('current_slide', 0)
        })
        
        memory.presentation_state['slide_questions'] = questions
        memory.presentation_state['questions_on_slide'] = memory.presentation_state.get('questions_on_slide', 0) + 1
    
    def get_context_for_ai(self, memory: SessionMemory) -> Dict[str, Any]:
        """Get formatted context for AI processing"""
        return {
            'session_info': {
                'session_id': memory.session_id,
                'duration_minutes': (memory.last_updated - memory.created_at).total_seconds() / 60,
                'total_turns': memory.total_turns
            },
            'presentation_state': memory.presentation_state,
            'user_preferences': memory.user_preferences,
            'conversation_summary': memory.session_context.get('conversation_summary', ''),
            'recent_topics': self._extract_recent_topics(memory),
            'user_engagement': memory.session_context.get('engagement_score', 0.5)
        }
    
    def _extract_recent_topics(self, memory: SessionMemory) -> List[str]:
        """Extract recent conversation topics"""
        topics = []
        recent_turns = memory.conversation_turns[-10:] if memory.conversation_turns else []
        
        for turn in recent_turns:
            if turn.nlu_result:
                for entity in turn.nlu_result.entities:
                    if entity.label in ['PRODUCT', 'FEATURE', 'TOPIC', 'ORGANIZATION']:
                        topics.append(entity.text)
        
        # Return unique topics, most recent first
        return list(dict.fromkeys(reversed(topics)))[:5]

class RedisMemoryStore:
    """Redis-based storage for session memory"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize Redis
        self.redis_client = redis.Redis(
            host=self.config.database.redis_host,
            port=self.config.database.redis_port,
            db=self.config.database.redis_database,
            password=self.config.database.redis_password,
            decode_responses=False
        )
        
        # Storage settings
        self.session_ttl = self.config.get('session.default_duration', 180) * 60  # Convert minutes to seconds
        self.user_profile_ttl = 30 * 24 * 3600  # 30 days for user profiles
    
    async def save_session_memory(self, memory: SessionMemory) -> None:
        """Save session memory to Redis"""
        try:
            # Serialize memory object
            memory_data = self._serialize_session_memory(memory)
            
            # Save to Redis with TTL
            key = f"session_memory:{memory.session_id}"
            self.redis_client.setex(key, self.session_ttl, memory_data)
            
            self.logger.debug(f"Saved session memory for {memory.session_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to save session memory: {e}")
            raise
    
    async def load_session_memory(self, session_id: str) -> Optional[SessionMemory]:
        """Load session memory from Redis"""
        try:
            key = f"session_memory:{session_id}"
            memory_data = self.redis_client.get(key)
            
            if memory_data:
                memory = self._deserialize_session_memory(memory_data)
                self.logger.debug(f"Loaded session memory for {session_id}")
                return memory
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to load session memory: {e}")
            return None
    
    async def delete_session_memory(self, session_id: str) -> None:
        """Delete session memory from Redis"""
        try:
            key = f"session_memory:{session_id}"
            self.redis_client.delete(key)
            
            self.logger.debug(f"Deleted session memory for {session_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to delete session memory: {e}")
    
    async def save_user_profile(self, profile: UserProfile) -> None:
        """Save user profile to Redis"""
        try:
            # Serialize profile
            profile_data = self._serialize_user_profile(profile)
            
            # Save to Redis with longer TTL
            key = f"user_profile:{profile.user_id}"
            self.redis_client.setex(key, self.user_profile_ttl, profile_data)
            
            self.logger.debug(f"Saved user profile for {profile.user_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to save user profile: {e}")
            raise
    
    async def load_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Load user profile from Redis"""
        try:
            key = f"user_profile:{user_id}"
            profile_data = self.redis_client.get(key)
            
            if profile_data:
                profile = self._deserialize_user_profile(profile_data)
                self.logger.debug(f"Loaded user profile for {user_id}")
                return profile
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to load user profile: {e}")
            return None
    
    def _serialize_session_memory(self, memory: SessionMemory) -> bytes:
        """Serialize session memory for storage"""
        # Convert to dict and handle datetime serialization
        memory_dict = asdict(memory)
        
        # Convert datetime objects to ISO strings
        memory_dict['created_at'] = memory.created_at.isoformat()
        memory_dict['last_updated'] = memory.last_updated.isoformat()
        
        for turn in memory_dict['conversation_turns']:
            turn['timestamp'] = turn['timestamp'].isoformat()
            # Handle NLU result serialization
            if turn['nlu_result']:
                turn['nlu_result'] = self._serialize_nlu_result(turn['nlu_result'])
        
        return pickle.dumps(memory_dict)
    
    def _deserialize_session_memory(self, data: bytes) -> SessionMemory:
        """Deserialize session memory from storage"""
        memory_dict = pickle.loads(data)
        
        # Convert ISO strings back to datetime
        memory_dict['created_at'] = datetime.fromisoformat(memory_dict['created_at'])
        memory_dict['last_updated'] = datetime.fromisoformat(memory_dict['last_updated'])
        
        # Deserialize conversation turns
        turns = []
        for turn_dict in memory_dict['conversation_turns']:
            turn_dict['timestamp'] = datetime.fromisoformat(turn_dict['timestamp'])
            
            # Reconstruct NLU result if present
            if turn_dict['nlu_result']:
                turn_dict['nlu_result'] = self._deserialize_nlu_result(turn_dict['nlu_result'])
            
            turns.append(ConversationTurn(**turn_dict))
        
        memory_dict['conversation_turns'] = turns
        
        return SessionMemory(**memory_dict)
    
    def _serialize_nlu_result(self, nlu_result: NLUResult) -> Dict[str, Any]:
        """Serialize NLU result for storage"""
        return {
            'text': nlu_result.text,
            'intent': nlu_result.intent.value,
            'intent_confidence': nlu_result.intent_confidence,
            'entities': [asdict(entity) for entity in nlu_result.entities],
            'sentiment': nlu_result.sentiment.value,
            'sentiment_confidence': nlu_result.sentiment_confidence,
            'processing_time_ms': nlu_result.processing_time_ms,
            'metadata': nlu_result.metadata
        }
    
    def _deserialize_nlu_result(self, data: Dict[str, Any]) -> NLUResult:
        """Deserialize NLU result from storage"""
        # This is a simplified reconstruction - in practice you'd want
        # to properly reconstruct the full NLUResult object
        return data  # Return as dict for now
    
    def _serialize_user_profile(self, profile: UserProfile) -> bytes:
        """Serialize user profile for storage"""
        profile_dict = asdict(profile)
        profile_dict['created_at'] = profile.created_at.isoformat()
        profile_dict['last_updated'] = profile.last_updated.isoformat()
        
        return pickle.dumps(profile_dict)
    
    def _deserialize_user_profile(self, data: bytes) -> UserProfile:
        """Deserialize user profile from storage"""
        profile_dict = pickle.loads(data)
        profile_dict['created_at'] = datetime.fromisoformat(profile_dict['created_at'])
        profile_dict['last_updated'] = datetime.fromisoformat(profile_dict['last_updated'])
        
        return UserProfile(**profile_dict)

class MemoryManager:
    """Main memory management class for Eva Live"""
    
    def __init__(self, session_id: Optional[str] = None):
        self.config = get_config()
        self.session_id = session_id
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.conversation_memory = ConversationMemory()
        self.context_manager = ContextManager()
        self.storage = RedisMemoryStore()
        
        # Current session memory
        self._current_memory: Optional[SessionMemory] = None
        
        # Performance tracking
        self.metrics: List[PerformanceMetric] = []
    
    async def initialize_session(self, session_id: str, user_id: Optional[str] = None) -> SessionMemory:
        """Initialize or load session memory"""
        start_time = time.time()
        
        try:
            # Try to load existing session
            existing_memory = await self.storage.load_session_memory(session_id)
            
            if existing_memory:
                self._current_memory = existing_memory
                self.logger.info(f"Loaded existing session memory for {session_id}")
            else:
                # Create new session memory
                self._current_memory = SessionMemory(
                    session_id=session_id,
                    user_id=user_id,
                    created_at=datetime.utcnow(),
                    last_updated=datetime.utcnow(),
                    conversation_turns=[],
                    session_context={},
                    user_preferences={},
                    presentation_state={
                        'current_slide': 0,
                        'questions_asked': 0,
                        'engagement_score': 0.5
                    },
                    total_turns=0
                )
                
                # Load user preferences if available
                if user_id:
                    user_profile = await self.storage.load_user_profile(user_id)
                    if user_profile:
                        self._current_memory.user_preferences = user_profile.preferences
                
                # Save initial state
                await self.storage.save_session_memory(self._current_memory)
                
                self.logger.info(f"Created new session memory for {session_id}")
            
            processing_time = int((time.time() - start_time) * 1000)
            await self._record_metric("session_init_time_ms", processing_time, "memory_manager")
            
            return self._current_memory
            
        except Exception as e:
            self.logger.error(f"Failed to initialize session memory: {e}")
            raise
    
    async def add_conversation_turn(self, user_input: str, eva_response: str, 
                                   nlu_result: Optional[NLUResult] = None) -> ConversationTurn:
        """Add a new conversation turn"""
        if not self._current_memory:
            raise RuntimeError("Session memory not initialized")
        
        try:
            # Add turn to conversation memory
            turn = self.conversation_memory.add_turn(
                self._current_memory,
                user_input,
                eva_response,
                nlu_result
            )
            
            # Update session context
            self._update_session_context()
            
            # Save to storage
            await self.storage.save_session_memory(self._current_memory)
            
            await self._record_metric("conversation_turns", self._current_memory.total_turns, "memory_manager")
            
            return turn
            
        except Exception as e:
            self.logger.error(f"Failed to add conversation turn: {e}")
            raise
    
    async def get_context_for_response(self, query: str) -> Dict[str, Any]:
        """Get relevant context for generating a response"""
        if not self._current_memory:
            return {}
        
        try:
            # Get conversation context
            recent_context = self.conversation_memory.get_recent_context(self._current_memory, num_turns=5)
            
            # Get AI context
            ai_context = self.context_manager.get_context_for_ai(self._current_memory)
            
            # Get conversation summary
            conversation_summary = self.conversation_memory.get_conversation_summary(self._current_memory)
            
            context = {
                'recent_conversation': recent_context,
                'conversation_summary': conversation_summary,
                'presentation_state': self._current_memory.presentation_state,
                'user_preferences': self._current_memory.user_preferences,
                'session_info': ai_context['session_info'],
                'current_query': query
            }
            
            return context
            
        except Exception as e:
            self.logger.error(f"Failed to get context for response: {e}")
            return {}
    
    async def update_presentation_state(self, slide_number: int, content: str, engagement_score: float) -> None:
        """Update presentation state"""
        if not self._current_memory:
            return
        
        try:
            self.context_manager.update_presentation_context(
                self._current_memory,
                slide_number,
                content,
                engagement_score
            )
            
            await self.storage.save_session_memory(self._current_memory)
            
        except Exception as e:
            self.logger.error(f"Failed to update presentation state: {e}")
    
    async def cleanup_session(self) -> None:
        """Clean up session memory"""
        if not self._current_memory:
            return
        
        try:
            # Update user profile with session data
            if self._current_memory.user_id:
                await self._update_user_profile()
            
            # Delete session memory
            await self.storage.delete_session_memory(self._current_memory.session_id)
            
            self.logger.info(f"Cleaned up session memory for {self._current_memory.session_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup session: {e}")
    
    def _update_session_context(self) -> None:
        """Update session context based on conversation"""
        if not self._current_memory:
            return
        
        # Update conversation summary
        summary = self.conversation_memory.get_conversation_summary(self._current_memory)
        self._current_memory.session_context['conversation_summary'] = summary
        
        # Calculate engagement score based on recent interactions
        recent_turns = self._current_memory.conversation_turns[-5:]
        if recent_turns:
            avg_confidence = sum(
                turn.nlu_result.intent_confidence 
                for turn in recent_turns 
                if turn.nlu_result
            ) / len([turn for turn in recent_turns if turn.nlu_result])
            
            self._current_memory.session_context['engagement_score'] = avg_confidence if avg_confidence > 0 else 0.5
    
    async def _update_user_profile(self) -> None:
        """Update long-term user profile based on session"""
        if not self._current_memory or not self._current_memory.user_id:
            return
        
        try:
            # Load existing profile or create new one
            profile = await self.storage.load_user_profile(self._current_memory.user_id)
            
            if not profile:
                profile = UserProfile(
                    user_id=self._current_memory.user_id,
                    name=None,
                    preferences={},
                    conversation_history_summary="",
                    interaction_patterns={},
                    created_at=datetime.utcnow(),
                    last_updated=datetime.utcnow()
                )
            
            # Update profile with session data
            profile.last_updated = datetime.utcnow()
            
            # Update conversation history summary
            session_summary = self.conversation_memory.get_conversation_summary(self._current_memory)
            profile.conversation_history_summary += f"\n{session_summary}"
            
            # Update interaction patterns
            profile.interaction_patterns['total_sessions'] = profile.interaction_patterns.get('total_sessions', 0) + 1
            profile.interaction_patterns['total_turns'] = profile.interaction_patterns.get('total_turns', 0) + self._current_memory.total_turns
            
            # Save updated profile
            await self.storage.save_user_profile(profile)
            
        except Exception as e:
            self.logger.error(f"Failed to update user profile: {e}")
    
    async def _record_metric(self, metric_type: str, value: float, component: str) -> None:
        """Record performance metric"""
        metric = PerformanceMetric(
            metric_type=metric_type,
            metric_value=value,
            component=component
        )
        self.metrics.append(metric)
    
    def get_metrics(self) -> List[PerformanceMetric]:
        """Get recorded metrics"""
        return self.metrics.copy()
    
    @property
    def current_memory(self) -> Optional[SessionMemory]:
        """Get current session memory"""
        return self._current_memory

# Utility functions
async def create_memory_manager(session_id: str, user_id: Optional[str] = None) -> MemoryManager:
    """Create and initialize a memory manager"""
    manager = MemoryManager(session_id)
    await manager.initialize_session(session_id, user_id)
    return manager

async def test_memory_manager():
    """Test function for memory manager"""
    try:
        # Create memory manager
        manager = MemoryManager()
        
        # Initialize session
        memory = await manager.initialize_session("test_session", "test_user")
        print(f"Initialized session: {memory.session_id}")
        
        # Add conversation turn
        turn = await manager.add_conversation_turn(
            "Hello, I'd like to learn about your product",
            "Hello! I'd be happy to tell you about our AI avatar system. What would you like to know?"
        )
        print(f"Added conversation turn: {turn.turn_id}")
        
        # Get context
        context = await manager.get_context_for_response("What are the pricing options?")
        print(f"Generated context with {len(context)} keys")
        
        # Cleanup
        await manager.cleanup_session()
        print("Session cleaned up successfully")
        
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_memory_manager())
