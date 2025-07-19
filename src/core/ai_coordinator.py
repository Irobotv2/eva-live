"""
Eva Live AI Coordinator Module

This module orchestrates all AI components to create a cohesive, intelligent
virtual presenter experience. It manages the pipeline from input to output.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import json

from ..shared.config import get_config
from ..shared.models import PerformanceMetric
from ..input.speech_recognition import SpeechRecognitionModule, TranscriptionResult
from ..input.nlu import NLUModule, NLUResult
from .knowledge_base import KnowledgeBase
from .memory_manager import MemoryManager
from .response_generator import ResponseGenerator, ResponseContext, ResponseType, ResponseTone, GeneratedResponse

class PipelineStage(str, Enum):
    """Stages in the AI processing pipeline"""
    INPUT_PROCESSING = "input_processing"
    CONTEXT_GATHERING = "context_gathering"
    RESPONSE_GENERATION = "response_generation"
    OUTPUT_PREPARATION = "output_preparation"
    COMPLETE = "complete"

class ProcessingMode(str, Enum):
    """Processing modes for different scenarios"""
    REAL_TIME = "real_time"
    BATCH = "batch"
    EMERGENCY = "emergency"
    FALLBACK = "fallback"

@dataclass
class ProcessingRequest:
    """Request for AI processing"""
    request_id: str
    user_input: str
    input_type: str  # 'speech', 'text', 'gesture'
    session_context: Dict[str, Any]
    processing_mode: ProcessingMode
    priority: int = 1  # 1-5, 5 being highest
    metadata: Dict[str, Any] = None

@dataclass
class ProcessingResult:
    """Result of AI processing"""
    request_id: str
    success: bool
    eva_response: GeneratedResponse
    pipeline_metrics: Dict[str, Any]
    total_processing_time_ms: int
    errors: List[str]
    fallback_used: bool
    metadata: Dict[str, Any]

@dataclass
class PipelineMetrics:
    """Metrics for the entire processing pipeline"""
    total_time_ms: int
    stage_times: Dict[str, int]
    quality_scores: Dict[str, float]
    cache_hits: Dict[str, bool]
    component_status: Dict[str, str]
    errors_encountered: List[str]

class ComponentHealthMonitor:
    """Monitors health and performance of AI components"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.component_status = {}
        self.performance_history = {}
        self.error_counts = {}
        
    def update_component_status(self, component: str, status: str, metrics: Dict[str, Any] = None):
        """Update status of a component"""
        self.component_status[component] = {
            'status': status,
            'last_update': time.time(),
            'metrics': metrics or {}
        }
        
        # Track performance history
        if component not in self.performance_history:
            self.performance_history[component] = []
        
        if metrics:
            self.performance_history[component].append({
                'timestamp': time.time(),
                'metrics': metrics
            })
            
            # Keep only last 100 entries
            if len(self.performance_history[component]) > 100:
                self.performance_history[component] = self.performance_history[component][-100:]
    
    def record_error(self, component: str, error: str):
        """Record an error for a component"""
        if component not in self.error_counts:
            self.error_counts[component] = []
        
        self.error_counts[component].append({
            'timestamp': time.time(),
            'error': error
        })
        
        # Keep only errors from last hour
        cutoff = time.time() - 3600
        self.error_counts[component] = [
            err for err in self.error_counts[component] 
            if err['timestamp'] > cutoff
        ]
    
    def get_component_health(self, component: str) -> Dict[str, Any]:
        """Get health status of a component"""
        status_info = self.component_status.get(component, {})
        error_count = len(self.error_counts.get(component, []))
        
        # Calculate health score
        health_score = 1.0
        if error_count > 0:
            health_score = max(0.0, 1.0 - (error_count / 10))  # Degrade based on errors
        
        # Check last update time
        last_update = status_info.get('last_update', 0)
        if time.time() - last_update > 300:  # 5 minutes
            health_score *= 0.5  # Reduce score for stale status
        
        return {
            'component': component,
            'status': status_info.get('status', 'unknown'),
            'health_score': health_score,
            'error_count': error_count,
            'last_update': last_update,
            'recent_metrics': status_info.get('metrics', {})
        }
    
    def get_overall_health(self) -> Dict[str, Any]:
        """Get overall system health"""
        components = ['speech_recognition', 'nlu', 'knowledge_base', 'memory_manager', 'response_generator']
        
        component_healths = {}
        total_health = 0
        
        for component in components:
            health = self.get_component_health(component)
            component_healths[component] = health
            total_health += health['health_score']
        
        overall_score = total_health / len(components) if components else 0
        
        return {
            'overall_health_score': overall_score,
            'status': 'healthy' if overall_score > 0.8 else 'degraded' if overall_score > 0.5 else 'critical',
            'components': component_healths
        }

class FallbackManager:
    """Manages fallback strategies when components fail"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.fallback_responses = {
            'speech_recognition_fail': [
                "I'm having trouble hearing you clearly. Could you please type your question?",
                "There seems to be an audio issue. Let me help you another way."
            ],
            'nlu_fail': [
                "I want to make sure I understand you correctly. Could you rephrase your question?",
                "Let me help you with that. What specific information are you looking for?"
            ],
            'knowledge_base_fail': [
                "I'm accessing the information you need. In the meantime, let me share what I can tell you.",
                "Let me provide you with some general information while I look up the specifics."
            ],
            'response_generation_fail': [
                "Thank you for your question. Let me gather the right information to help you.",
                "I appreciate your patience while I formulate the best response for you."
            ],
            'general_fail': [
                "I'm experiencing a brief technical issue. Let me help you in another way.",
                "Thank you for your patience. How else can I assist you today?"
            ]
        }
    
    def get_fallback_response(self, component: str, context: Dict[str, Any] = None) -> str:
        """Get an appropriate fallback response"""
        fallback_key = f"{component}_fail"
        responses = self.fallback_responses.get(fallback_key, self.fallback_responses['general_fail'])
        
        # Simple selection (could be enhanced with context-aware selection)
        import random
        return random.choice(responses)
    
    def should_use_fallback(self, component: str, error_count: int, health_score: float) -> bool:
        """Determine if fallback should be used"""
        # Use fallback if component is unhealthy or has many recent errors
        return health_score < 0.5 or error_count > 5

class AICoordinator:
    """Main AI coordinator that orchestrates all components"""
    
    def __init__(self, session_id: str):
        self.config = get_config()
        self.session_id = session_id
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.speech_recognition = SpeechRecognitionModule(session_id)
        self.nlu = NLUModule(session_id)
        self.knowledge_base = KnowledgeBase(session_id)
        self.memory_manager = MemoryManager(session_id)
        self.response_generator = ResponseGenerator(session_id)
        
        # Initialize support systems
        self.health_monitor = ComponentHealthMonitor()
        self.fallback_manager = FallbackManager()
        
        # Processing queue and metrics
        self.processing_queue = asyncio.Queue()
        self.active_requests = {}
        self.metrics: List[PerformanceMetric] = []
        
        # Performance targets
        self.target_latency_ms = self.config.get('performance.latency_targets.total_pipeline', 500)
        self.quality_threshold = 0.7
        
    async def initialize(self, user_id: Optional[str] = None) -> None:
        """Initialize the AI coordinator and all components"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Initializing AI coordinator for session {self.session_id}")
            
            # Initialize session memory
            await self.memory_manager.initialize_session(self.session_id, user_id)
            
            # Update component health
            self.health_monitor.update_component_status('memory_manager', 'initialized')
            self.health_monitor.update_component_status('speech_recognition', 'ready')
            self.health_monitor.update_component_status('nlu', 'ready')
            self.health_monitor.update_component_status('knowledge_base', 'ready')
            self.health_monitor.update_component_status('response_generator', 'ready')
            
            init_time = int((time.time() - start_time) * 1000)
            await self._record_metric("coordinator_init_time_ms", init_time, "ai_coordinator")
            
            self.logger.info(f"AI coordinator initialized in {init_time}ms")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize AI coordinator: {e}")
            raise
    
    async def process_user_input(
        self,
        user_input: str,
        input_type: str = "text",
        processing_mode: ProcessingMode = ProcessingMode.REAL_TIME,
        priority: int = 1
    ) -> ProcessingResult:
        """Process user input through the complete AI pipeline"""
        
        request_id = f"req_{int(time.time() * 1000)}"
        start_time = time.time()
        
        # Create processing request
        request = ProcessingRequest(
            request_id=request_id,
            user_input=user_input,
            input_type=input_type,
            session_context={},  # Will be filled during processing
            processing_mode=processing_mode,
            priority=priority,
            metadata={}
        )
        
        self.logger.info(f"Processing request {request_id}: '{user_input[:50]}...'")
        
        errors = []
        fallback_used = False
        stage_times = {}
        quality_scores = {}
        
        try:
            # Stage 1: NLU Processing
            stage_start = time.time()
            nlu_result = await self._process_nlu(user_input, request)
            stage_times['nlu'] = int((time.time() - stage_start) * 1000)
            quality_scores['nlu'] = nlu_result.intent_confidence if nlu_result else 0.0
            
            # Stage 2: Context Gathering
            stage_start = time.time()
            context = await self._gather_context(user_input, nlu_result, request)
            stage_times['context_gathering'] = int((time.time() - stage_start) * 1000)
            
            # Stage 3: Response Generation
            stage_start = time.time()
            response = await self._generate_response(context, nlu_result, request)
            stage_times['response_generation'] = int((time.time() - stage_start) * 1000)
            quality_scores['response'] = response.confidence_score
            
            # Stage 4: Memory Update
            stage_start = time.time()
            await self._update_memory(user_input, response, nlu_result)
            stage_times['memory_update'] = int((time.time() - stage_start) * 1000)
            
            total_time = int((time.time() - start_time) * 1000)
            
            # Create pipeline metrics
            pipeline_metrics = {
                'total_time_ms': total_time,
                'stage_times': stage_times,
                'quality_scores': quality_scores,
                'target_latency_ms': self.target_latency_ms,
                'within_target': total_time <= self.target_latency_ms
            }
            
            # Record metrics
            await self._record_metric("pipeline_total_time_ms", total_time, "ai_coordinator")
            await self._record_metric("pipeline_quality_score", quality_scores.get('response', 0), "ai_coordinator")
            
            # Create result
            result = ProcessingResult(
                request_id=request_id,
                success=True,
                eva_response=response,
                pipeline_metrics=pipeline_metrics,
                total_processing_time_ms=total_time,
                errors=errors,
                fallback_used=fallback_used,
                metadata={
                    'nlu_result': nlu_result.__dict__ if nlu_result else None,
                    'context_tokens': len(str(context).split()) if context else 0
                }
            )
            
            self.logger.info(f"Request {request_id} processed successfully in {total_time}ms")
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing request {request_id}: {e}")
            errors.append(str(e))
            
            # Generate fallback result
            return await self._generate_fallback_result(request, errors, start_time)
    
    async def _process_nlu(self, user_input: str, request: ProcessingRequest) -> Optional[NLUResult]:
        """Process input through NLU"""
        try:
            # Check component health
            nlu_health = self.health_monitor.get_component_health('nlu')
            
            if self.fallback_manager.should_use_fallback('nlu', 0, nlu_health['health_score']):
                self.logger.warning("NLU component unhealthy, using simplified processing")
                return None
            
            # Process with NLU
            nlu_result = await self.nlu.process(user_input)
            
            # Update health status
            self.health_monitor.update_component_status('nlu', 'healthy', {
                'last_processing_time_ms': nlu_result.processing_time_ms,
                'confidence': nlu_result.intent_confidence
            })
            
            return nlu_result
            
        except Exception as e:
            self.logger.error(f"NLU processing failed: {e}")
            self.health_monitor.record_error('nlu', str(e))
            return None
    
    async def _gather_context(self, user_input: str, nlu_result: Optional[NLUResult], request: ProcessingRequest) -> ResponseContext:
        """Gather context from memory and knowledge base"""
        try:
            # Get memory context
            memory_context = await self.memory_manager.get_context_for_response(user_input)
            
            # Get relevant knowledge
            knowledge_context = await self.knowledge_base.get_relevant_context(user_input, max_tokens=1500)
            
            # Extract entities and intent from NLU result
            intent = nlu_result.intent.value if nlu_result else "general"
            sentiment = nlu_result.sentiment.value if nlu_result else "neutral"
            entities = [entity.__dict__ for entity in nlu_result.entities] if nlu_result else []
            
            # Create response context
            context = ResponseContext(
                user_query=user_input,
                conversation_history=memory_context.get('recent_conversation', ''),
                relevant_knowledge=knowledge_context,
                presentation_state=memory_context.get('presentation_state', {}),
                user_preferences=memory_context.get('user_preferences', {}),
                session_info=memory_context.get('session_info', {}),
                intent=intent,
                sentiment=sentiment,
                entities=entities,
                metadata={}
            )
            
            return context
            
        except Exception as e:
            self.logger.error(f"Context gathering failed: {e}")
            # Return minimal context
            return ResponseContext(
                user_query=user_input,
                conversation_history="",
                relevant_knowledge="",
                presentation_state={},
                user_preferences={},
                session_info={},
                intent="general",
                sentiment="neutral",
                entities=[],
                metadata={}
            )
    
    async def _generate_response(self, context: ResponseContext, nlu_result: Optional[NLUResult], request: ProcessingRequest) -> GeneratedResponse:
        """Generate response using the response generator"""
        try:
            # Determine response type based on intent
            response_type = self._determine_response_type(context.intent)
            
            # Determine tone based on user preferences and context
            tone = self._determine_response_tone(context)
            
            # Generate response
            response = await self.response_generator.generate_response(
                context,
                response_type,
                tone
            )
            
            # Update health status
            self.health_monitor.update_component_status('response_generator', 'healthy', {
                'last_processing_time_ms': response.processing_time_ms,
                'quality_score': response.confidence_score
            })
            
            return response
            
        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
            self.health_monitor.record_error('response_generator', str(e))
            
            # Generate emergency fallback
            fallback_text = self.fallback_manager.get_fallback_response('response_generation', context.__dict__)
            
            return GeneratedResponse(
                text=fallback_text,
                response_type=ResponseType.ERROR_RECOVERY,
                tone=ResponseTone.PROFESSIONAL,
                confidence_score=0.5,
                knowledge_sources=[],
                suggested_actions=[],
                presentation_updates={},
                processing_time_ms=50,
                metadata={'fallback': True, 'error': str(e)}
            )
    
    async def _update_memory(self, user_input: str, response: GeneratedResponse, nlu_result: Optional[NLUResult]) -> None:
        """Update memory with the conversation turn"""
        try:
            await self.memory_manager.add_conversation_turn(
                user_input,
                response.text,
                nlu_result
            )
            
            # Update presentation state if needed
            if response.presentation_updates:
                for key, value in response.presentation_updates.items():
                    if key == 'engagement_boost':
                        engagement_score = 0.8  # Boost engagement
                        await self.memory_manager.update_presentation_state(
                            self.memory_manager.current_memory.presentation_state.get('current_slide', 0),
                            response.text[:100],  # First 100 chars as content summary
                            engagement_score
                        )
            
        except Exception as e:
            self.logger.error(f"Memory update failed: {e}")
            self.health_monitor.record_error('memory_manager', str(e))
    
    def _determine_response_type(self, intent: str) -> ResponseType:
        """Determine appropriate response type based on intent"""
        intent_mapping = {
            'question': ResponseType.QUESTION_ANSWER,
            'clarification': ResponseType.CLARIFICATION,
            'greeting': ResponseType.ENGAGEMENT,
            'farewell': ResponseType.TRANSITION,
            'confirmation': ResponseType.INFORMATIONAL,
            'objection': ResponseType.CLARIFICATION,
            'request_demo': ResponseType.PRESENTATION_CONTENT,
            'request_pricing': ResponseType.INFORMATIONAL,
            'technical_question': ResponseType.QUESTION_ANSWER,
            'business_question': ResponseType.QUESTION_ANSWER,
            'feedback': ResponseType.ENGAGEMENT
        }
        
        return intent_mapping.get(intent, ResponseType.QUESTION_ANSWER)
    
    def _determine_response_tone(self, context: ResponseContext) -> ResponseTone:
        """Determine appropriate response tone"""
        # Default to professional
        tone = ResponseTone.PROFESSIONAL
        
        # Adjust based on user preferences
        preferred_tone = context.user_preferences.get('tone', 'professional')
        
        tone_mapping = {
            'professional': ResponseTone.PROFESSIONAL,
            'friendly': ResponseTone.FRIENDLY,
            'enthusiastic': ResponseTone.ENTHUSIASTIC,
            'confident': ResponseTone.CONFIDENT,
            'empathetic': ResponseTone.EMPATHETIC
        }
        
        tone = tone_mapping.get(preferred_tone, ResponseTone.PROFESSIONAL)
        
        # Adjust based on sentiment
        if context.sentiment == 'negative':
            tone = ResponseTone.EMPATHETIC
        elif context.sentiment == 'positive':
            tone = ResponseTone.ENTHUSIASTIC
        
        return tone
    
    async def _generate_fallback_result(self, request: ProcessingRequest, errors: List[str], start_time: float) -> ProcessingResult:
        """Generate a fallback result when processing fails"""
        fallback_text = self.fallback_manager.get_fallback_response('general', request.__dict__)
        
        fallback_response = GeneratedResponse(
            text=fallback_text,
            response_type=ResponseType.ERROR_RECOVERY,
            tone=ResponseTone.PROFESSIONAL,
            confidence_score=0.3,
            knowledge_sources=[],
            suggested_actions=[],
            presentation_updates={},
            processing_time_ms=50,
            metadata={'fallback': True, 'errors': errors}
        )
        
        total_time = int((time.time() - start_time) * 1000)
        
        return ProcessingResult(
            request_id=request.request_id,
            success=False,
            eva_response=fallback_response,
            pipeline_metrics={'total_time_ms': total_time},
            total_processing_time_ms=total_time,
            errors=errors,
            fallback_used=True,
            metadata={'emergency_fallback': True}
        )
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        health = self.health_monitor.get_overall_health()
        
        # Get component metrics
        component_metrics = {}
        for component in ['speech_recognition', 'nlu', 'knowledge_base', 'memory_manager', 'response_generator']:
            component_metrics[component] = self.health_monitor.get_component_health(component)
        
        # Get recent performance
        recent_metrics = self.metrics[-10:] if self.metrics else []
        
        return {
            'session_id': self.session_id,
            'overall_health': health,
            'component_status': component_metrics,
            'recent_performance': recent_metrics,
            'active_requests': len(self.active_requests),
            'target_latency_ms': self.target_latency_ms
        }
    
    async def cleanup(self) -> None:
        """Clean up resources"""
        try:
            # Clean up session memory
            await self.memory_manager.cleanup_session()
            
            # Clear processing queue
            while not self.processing_queue.empty():
                self.processing_queue.get_nowait()
            
            self.logger.info(f"AI coordinator cleaned up for session {self.session_id}")
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    async def _record_metric(self, metric_type: str, value: float, component: str) -> None:
        """Record performance metric"""
        metric = PerformanceMetric(
            metric_type=metric_type,
            metric_value=value,
            component=component
        )
        self.metrics.append(metric)
        
        # Keep only recent metrics
        if len(self.metrics) > 1000:
            self.metrics = self.metrics[-500:]
    
    def get_metrics(self) -> List[PerformanceMetric]:
        """Get recorded metrics"""
        return self.metrics.copy()

# Utility functions
async def create_ai_coordinator(session_id: str, user_id: Optional[str] = None) -> AICoordinator:
    """Create and initialize an AI coordinator"""
    coordinator = AICoordinator(session_id)
    await coordinator.initialize(user_id)
    return coordinator

async def test_ai_coordinator():
    """Test function for AI coordinator"""
    try:
        # Create coordinator
        coordinator = AICoordinator("test_session")
        await coordinator.initialize("test_user")
        
        # Test processing
        result = await coordinator.process_user_input("What are the main features of Eva Live?")
        
        print(f"Processing result: {result.success}")
        print(f"Response: {result.eva_response.text}")
        print(f"Total time: {result.total_processing_time_ms}ms")
        print(f"Quality score: {result.eva_response.confidence_score:.2f}")
        
        # Get system status
        status = await coordinator.get_system_status()
        print(f"System health: {status['overall_health']['status']}")
        
        # Cleanup
        await coordinator.cleanup()
        
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_ai_coordinator())
