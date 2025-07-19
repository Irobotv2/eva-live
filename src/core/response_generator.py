"""
Eva Live Response Generation Module

This module handles intelligent response generation using GPT-4, integrating with
the knowledge base and memory systems to create contextual, coherent responses.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import re

import openai
from openai import AsyncOpenAI

from ..shared.config import get_config
from ..shared.models import PerformanceMetric
from .knowledge_base import KnowledgeBase, SearchResult
from .memory_manager import MemoryManager

class ResponseType(str, Enum):
    """Types of responses Eva can generate"""
    INFORMATIONAL = "informational"
    QUESTION_ANSWER = "question_answer"
    PRESENTATION_CONTENT = "presentation_content"
    CLARIFICATION = "clarification"
    ENGAGEMENT = "engagement"
    TRANSITION = "transition"
    ERROR_RECOVERY = "error_recovery"

class ResponseTone(str, Enum):
    """Tone options for responses"""
    PROFESSIONAL = "professional"
    FRIENDLY = "friendly"
    ENTHUSIASTIC = "enthusiastic"
    CONFIDENT = "confident"
    EMPATHETIC = "empathetic"

@dataclass
class ResponseContext:
    """Context for response generation"""
    user_query: str
    conversation_history: str
    relevant_knowledge: str
    presentation_state: Dict[str, Any]
    user_preferences: Dict[str, Any]
    session_info: Dict[str, Any]
    intent: str
    sentiment: str
    entities: List[Dict[str, Any]]

@dataclass
class GeneratedResponse:
    """Generated response from Eva"""
    text: str
    response_type: ResponseType
    tone: ResponseTone
    confidence_score: float
    knowledge_sources: List[str]
    suggested_actions: List[Dict[str, Any]]
    presentation_updates: Dict[str, Any]
    processing_time_ms: int
    metadata: Dict[str, Any]

class PromptTemplateManager:
    """Manages prompt templates for different response scenarios"""
    
    def __init__(self):
        self.templates = {
            ResponseType.INFORMATIONAL: self._get_informational_template(),
            ResponseType.QUESTION_ANSWER: self._get_qa_template(),
            ResponseType.PRESENTATION_CONTENT: self._get_presentation_template(),
            ResponseType.CLARIFICATION: self._get_clarification_template(),
            ResponseType.ENGAGEMENT: self._get_engagement_template(),
            ResponseType.TRANSITION: self._get_transition_template(),
            ResponseType.ERROR_RECOVERY: self._get_error_recovery_template()
        }
    
    def _get_informational_template(self) -> str:
        return """
You are Eva, an AI-powered virtual presenter with expertise in your presentation topic.

Context:
- User Query: {user_query}
- Conversation History: {conversation_history}
- Relevant Knowledge: {relevant_knowledge}
- Current Slide: {current_slide}
- Presentation Topic: {presentation_topic}
- User Preferences: {user_preferences}

Instructions:
1. Provide accurate, helpful information based on the knowledge provided
2. Maintain a {tone} tone throughout your response
3. Reference specific sources when possible
4. Keep responses concise but comprehensive
5. Naturally transition back to presentation flow when appropriate

Response should be natural, engaging, and focused on helping the user understand the topic better.

Generate your response:
"""
    
    def _get_qa_template(self) -> str:
        return """
You are Eva, an AI virtual presenter answering a specific question.

Context:
- Question: {user_query}
- Conversation History: {conversation_history}
- Relevant Knowledge: {relevant_knowledge}
- Intent: {intent}
- Sentiment: {sentiment}
- Current Context: {session_info}

Instructions:
1. Directly answer the user's question using the provided knowledge
2. If you don't have enough information, acknowledge this honestly
3. Provide examples or analogies when helpful
4. Maintain a {tone} tone
5. Suggest follow-up questions or related topics if relevant

Be accurate, helpful, and engaging in your response.

Generate your answer:
"""
    
    def _get_presentation_template(self) -> str:
        return """
You are Eva, delivering presentation content in an engaging way.

Context:
- Current Slide: {current_slide}
- Slide Content: {slide_content}
- Presentation Flow: {presentation_flow}
- Audience Engagement: {engagement_level}
- User Interaction: {user_query}

Instructions:
1. Present the content in an engaging, conversational manner
2. Adapt your delivery based on audience engagement level
3. Include relevant examples and explanations
4. Use a {tone} tone appropriate for the audience
5. Prepare smooth transitions to next topics

Deliver the content naturally and engagingly:
"""
    
    def _get_clarification_template(self) -> str:
        return """
You are Eva, seeking clarification to better help the user.

Context:
- User Input: {user_query}
- Ambiguity Detected: {ambiguity_reason}
- Possible Interpretations: {possible_meanings}
- Conversation Context: {conversation_history}

Instructions:
1. Acknowledge the user's input
2. Explain what clarification you need
3. Provide 2-3 specific options for the user to choose from
4. Maintain a helpful and {tone} tone
5. Show that you want to provide the best possible help

Ask for clarification in a helpful way:
"""
    
    def _get_engagement_template(self) -> str:
        return """
You are Eva, engaging with the audience to maintain interest and participation.

Context:
- Engagement Level: {engagement_level}
- Presentation Progress: {presentation_progress}
- Recent Interactions: {recent_interactions}
- Topic: {current_topic}

Instructions:
1. Create engaging content that draws the audience in
2. Ask thought-provoking questions
3. Use interactive elements and examples
4. Maintain a {tone} and enthusiastic tone
5. Encourage participation and questions

Generate engaging content:
"""
    
    def _get_transition_template(self) -> str:
        return """
You are Eva, smoothly transitioning between presentation topics.

Context:
- Previous Topic: {previous_topic}
- Next Topic: {next_topic}
- Transition Reason: {transition_reason}
- Audience State: {audience_state}

Instructions:
1. Smoothly bridge the previous and next topics
2. Maintain flow and logical connection
3. Re-engage the audience for the new topic
4. Use a {tone} tone
5. Create anticipation for the upcoming content

Create a smooth transition:
"""
    
    def _get_error_recovery_template(self) -> str:
        return """
You are Eva, gracefully handling an error or unexpected situation.

Context:
- Error Type: {error_type}
- User Context: {user_query}
- Fallback Information: {fallback_info}
- Session State: {session_state}

Instructions:
1. Acknowledge the issue without technical details
2. Provide alternative helpful information
3. Redirect to related topics you can discuss
4. Maintain confidence and a {tone} tone
5. Suggest ways to continue the presentation

Handle the situation gracefully:
"""

class ResponseQualityAnalyzer:
    """Analyzes and scores response quality"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_response(self, response: str, context: ResponseContext) -> Tuple[float, Dict[str, Any]]:
        """Analyze response quality and return score with details"""
        scores = {}
        
        # Relevance score (0-1)
        scores['relevance'] = self._calculate_relevance(response, context)
        
        # Coherence score (0-1)
        scores['coherence'] = self._calculate_coherence(response)
        
        # Completeness score (0-1)
        scores['completeness'] = self._calculate_completeness(response, context)
        
        # Engagement score (0-1)
        scores['engagement'] = self._calculate_engagement(response)
        
        # Length appropriateness (0-1)
        scores['length'] = self._calculate_length_score(response)
        
        # Overall score (weighted average)
        weights = {
            'relevance': 0.3,
            'coherence': 0.25,
            'completeness': 0.2,
            'engagement': 0.15,
            'length': 0.1
        }
        
        overall_score = sum(scores[key] * weights[key] for key in scores)
        
        return overall_score, scores
    
    def _calculate_relevance(self, response: str, context: ResponseContext) -> float:
        """Calculate how relevant the response is to the query"""
        # Simple keyword overlap analysis
        query_words = set(context.user_query.lower().split())
        response_words = set(response.lower().split())
        
        if not query_words:
            return 0.5
        
        overlap = len(query_words.intersection(response_words))
        relevance = min(overlap / len(query_words), 1.0)
        
        # Boost score if response addresses the query type
        if any(word in response.lower() for word in ['what', 'how', 'why', 'when', 'where'] 
               if word in context.user_query.lower()):
            relevance += 0.2
        
        return min(relevance, 1.0)
    
    def _calculate_coherence(self, response: str) -> float:
        """Calculate response coherence and flow"""
        sentences = response.split('.')
        if len(sentences) < 2:
            return 0.8  # Short responses are generally coherent
        
        # Check for logical flow indicators
        coherence_indicators = [
            'therefore', 'however', 'additionally', 'furthermore',
            'for example', 'in contrast', 'as a result', 'meanwhile'
        ]
        
        indicator_count = sum(1 for indicator in coherence_indicators 
                            if indicator in response.lower())
        
        # Base score with bonus for coherence indicators
        base_score = 0.7
        indicator_bonus = min(indicator_count * 0.1, 0.3)
        
        return min(base_score + indicator_bonus, 1.0)
    
    def _calculate_completeness(self, response: str, context: ResponseContext) -> float:
        """Calculate if response completely addresses the query"""
        # Check response length relative to query complexity
        query_complexity = len(context.user_query.split()) + len(context.entities)
        response_length = len(response.split())
        
        # Expect more detailed response for complex queries
        expected_length = max(20, query_complexity * 3)
        length_ratio = min(response_length / expected_length, 1.0)
        
        # Check if key entities from query are addressed
        entity_coverage = 0
        if context.entities:
            addressed_entities = sum(1 for entity in context.entities 
                                   if entity.get('text', '').lower() in response.lower())
            entity_coverage = addressed_entities / len(context.entities)
        
        return (length_ratio * 0.6) + (entity_coverage * 0.4)
    
    def _calculate_engagement(self, response: str) -> float:
        """Calculate how engaging the response is"""
        engagement_features = {
            'questions': len(re.findall(r'\?', response)) * 0.1,
            'examples': len(re.findall(r'for example|such as|like', response.lower())) * 0.15,
            'personal_pronouns': len(re.findall(r'\byou\b|\byour\b', response.lower())) * 0.1,
            'active_voice': 0.2 if any(word in response.lower() for word in ['let\'s', 'we can', 'you can']) else 0,
            'enthusiasm': 0.15 if any(punct in response for punct in ['!']) else 0
        }
        
        total_engagement = sum(engagement_features.values())
        return min(total_engagement, 1.0)
    
    def _calculate_length_score(self, response: str) -> float:
        """Calculate if response length is appropriate"""
        word_count = len(response.split())
        
        # Optimal range: 20-150 words
        if 20 <= word_count <= 150:
            return 1.0
        elif word_count < 20:
            return word_count / 20
        else:  # > 150 words
            return max(0.5, 150 / word_count)

class ResponseGenerator:
    """Main response generation engine"""
    
    def __init__(self, session_id: Optional[str] = None):
        self.config = get_config()
        self.session_id = session_id
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI client
        self.openai_client = AsyncOpenAI(api_key=self.config.ai_services.openai_api_key)
        self.model = self.config.ai_services.openai_model
        
        # Initialize components
        self.prompt_manager = PromptTemplateManager()
        self.quality_analyzer = ResponseQualityAnalyzer()
        
        # Response caching
        self.response_cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Performance tracking
        self.metrics: List[PerformanceMetric] = []
    
    async def generate_response(
        self,
        context: ResponseContext,
        response_type: ResponseType = ResponseType.QUESTION_ANSWER,
        tone: ResponseTone = ResponseTone.PROFESSIONAL,
        max_retries: int = 2
    ) -> GeneratedResponse:
        """Generate a contextual response"""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = self._generate_cache_key(context, response_type, tone)
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                return cached_response
            
            # Generate response
            response_text = await self._generate_with_retries(
                context, response_type, tone, max_retries
            )
            
            # Analyze quality
            quality_score, quality_details = self.quality_analyzer.analyze_response(
                response_text, context
            )
            
            # Extract knowledge sources
            knowledge_sources = self._extract_knowledge_sources(context)
            
            # Generate suggested actions
            suggested_actions = self._generate_suggested_actions(context, response_text)
            
            # Generate presentation updates
            presentation_updates = self._generate_presentation_updates(context, response_text)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            # Create response object
            generated_response = GeneratedResponse(
                text=response_text,
                response_type=response_type,
                tone=tone,
                confidence_score=quality_score,
                knowledge_sources=knowledge_sources,
                suggested_actions=suggested_actions,
                presentation_updates=presentation_updates,
                processing_time_ms=processing_time,
                metadata={
                    'quality_details': quality_details,
                    'model_used': self.model,
                    'context_tokens': len(str(context).split()),
                    'cache_hit': False
                }
            )
            
            # Cache the response
            self._cache_response(cache_key, generated_response)
            
            # Record metrics
            await self._record_metric("response_generation_time_ms", processing_time, "response_generator")
            await self._record_metric("response_quality_score", quality_score, "response_generator")
            await self._record_metric("response_length_words", len(response_text.split()), "response_generator")
            
            self.logger.info(f"Generated response: {len(response_text)} chars, quality: {quality_score:.2f}")
            
            return generated_response
            
        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
            
            # Generate fallback response
            return await self._generate_fallback_response(context, response_type, tone)
    
    async def _generate_with_retries(
        self,
        context: ResponseContext,
        response_type: ResponseType,
        tone: ResponseTone,
        max_retries: int
    ) -> str:
        """Generate response with retry logic"""
        
        for attempt in range(max_retries + 1):
            try:
                # Get appropriate prompt template
                template = self.prompt_manager.templates[response_type]
                
                # Fill template with context
                prompt = template.format(
                    user_query=context.user_query,
                    conversation_history=context.conversation_history,
                    relevant_knowledge=context.relevant_knowledge,
                    current_slide=context.presentation_state.get('current_slide', 0),
                    presentation_topic=context.presentation_state.get('topic', 'General Topic'),
                    slide_content=context.presentation_state.get('current_content', ''),
                    presentation_flow=context.presentation_state.get('flow', ''),
                    engagement_level=context.session_info.get('engagement_score', 0.5),
                    user_preferences=json.dumps(context.user_preferences),
                    intent=context.intent,
                    sentiment=context.sentiment,
                    session_info=json.dumps(context.session_info),
                    tone=tone.value,
                    # Additional context fields
                    ambiguity_reason=context.metadata.get('ambiguity_reason', ''),
                    possible_meanings=context.metadata.get('possible_meanings', ''),
                    presentation_progress=context.presentation_state.get('progress', ''),
                    recent_interactions=context.conversation_history[-500:],  # Last 500 chars
                    current_topic=context.presentation_state.get('current_topic', ''),
                    previous_topic=context.presentation_state.get('previous_topic', ''),
                    next_topic=context.presentation_state.get('next_topic', ''),
                    transition_reason=context.metadata.get('transition_reason', ''),
                    audience_state=context.session_info.get('audience_state', 'attentive'),
                    error_type=context.metadata.get('error_type', ''),
                    fallback_info=context.metadata.get('fallback_info', ''),
                    session_state=json.dumps(context.session_info)
                )
                
                # Call OpenAI API
                response = await self.openai_client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are Eva, a professional AI virtual presenter. Always respond naturally and helpfully."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    max_tokens=self.config.ai_services.openai_max_tokens,
                    temperature=self.config.ai_services.openai_temperature,
                    timeout=30
                )
                
                response_text = response.choices[0].message.content.strip()
                
                # Validate response
                if self._validate_response(response_text, context):
                    return response_text
                else:
                    self.logger.warning(f"Response validation failed on attempt {attempt + 1}")
                    if attempt == max_retries:
                        return response_text  # Return anyway on final attempt
                
            except Exception as e:
                self.logger.warning(f"Response generation attempt {attempt + 1} failed: {e}")
                if attempt == max_retries:
                    raise
                
                # Wait before retry
                await asyncio.sleep(1 * (attempt + 1))
        
        raise RuntimeError("All response generation attempts failed")
    
    def _validate_response(self, response: str, context: ResponseContext) -> bool:
        """Validate generated response"""
        # Basic validation checks
        if not response or len(response.strip()) < 10:
            return False
        
        # Check for inappropriate content
        inappropriate_markers = ['sorry, i cannot', 'i cannot assist', 'i am not able']
        if any(marker in response.lower() for marker in inappropriate_markers):
            return False
        
        # Check for reasonable length
        word_count = len(response.split())
        if word_count > 300:  # Too long
            return False
        
        return True
    
    def _extract_knowledge_sources(self, context: ResponseContext) -> List[str]:
        """Extract knowledge sources used in response"""
        sources = []
        
        # Extract from relevant knowledge
        if context.relevant_knowledge:
            # Simple heuristic to find source markers
            lines = context.relevant_knowledge.split('\n')
            for line in lines:
                if line.startswith('Source:'):
                    source = line.replace('Source:', '').strip()
                    if source not in sources:
                        sources.append(source)
        
        return sources[:3]  # Limit to top 3 sources
    
    def _generate_suggested_actions(self, context: ResponseContext, response: str) -> List[Dict[str, Any]]:
        """Generate suggested actions based on context and response"""
        actions = []
        
        # Suggest slide navigation based on content
        if 'next' in response.lower() and 'slide' in response.lower():
            actions.append({
                'type': 'navigate_slide',
                'direction': 'next',
                'description': 'Move to next slide'
            })
        
        # Suggest follow-up questions
        if '?' in context.user_query:
            actions.append({
                'type': 'suggest_followup',
                'questions': [
                    'Would you like me to explain this in more detail?',
                    'Do you have any specific questions about this topic?',
                    'Shall we move on to the next section?'
                ]
            })
        
        # Suggest engagement activities
        if context.session_info.get('engagement_score', 0.5) < 0.6:
            actions.append({
                'type': 'engagement_boost',
                'suggestions': [
                    'Ask the audience a question',
                    'Share a relevant example',
                    'Invite questions or comments'
                ]
            })
        
        return actions
    
    def _generate_presentation_updates(self, context: ResponseContext, response: str) -> Dict[str, Any]:
        """Generate presentation state updates based on response"""
        updates = {}
        
        # Update current topic if mentioned
        if 'topic' in response.lower():
            updates['last_topic_mentioned'] = time.time()
        
        # Update engagement based on response type
        if any(marker in response for marker in ['?', '!', 'example']):
            updates['engagement_boost'] = True
        
        # Track questions answered
        if context.intent == 'question':
            updates['questions_answered'] = context.session_info.get('questions_answered', 0) + 1
        
        return updates
    
    async def _generate_fallback_response(
        self,
        context: ResponseContext,
        response_type: ResponseType,
        tone: ResponseTone
    ) -> GeneratedResponse:
        """Generate a fallback response when main generation fails"""
        
        fallback_responses = {
            ResponseType.QUESTION_ANSWER: "I understand your question about {topic}. Let me provide some helpful information based on what I know.",
            ResponseType.INFORMATIONAL: "Thank you for your interest in {topic}. This is an important aspect of our discussion.",
            ResponseType.PRESENTATION_CONTENT: "Let's continue with our presentation on {topic}. This next section covers some key points.",
            ResponseType.CLARIFICATION: "I want to make sure I understand your question correctly. Could you help me clarify what specific aspect you're interested in?",
            ResponseType.ENGAGEMENT: "Great question! This is exactly the kind of topic that's important to discuss in detail.",
            ResponseType.TRANSITION: "Now let's move on to our next topic, which builds on what we've just covered.",
            ResponseType.ERROR_RECOVERY: "Let me help you with that. While I gather the specific information you need, let me share what I can tell you about this topic."
        }
        
        base_text = fallback_responses.get(
            response_type,
            "Thank you for your question. Let me provide you with some helpful information."
        )
        
        # Simple topic extraction
        topic = context.presentation_state.get('current_topic', 'this topic')
        response_text = base_text.format(topic=topic)
        
        return GeneratedResponse(
            text=response_text,
            response_type=response_type,
            tone=tone,
            confidence_score=0.5,
            knowledge_sources=[],
            suggested_actions=[],
            presentation_updates={},
            processing_time_ms=50,
            metadata={'fallback': True}
        )
    
    def _generate_cache_key(self, context: ResponseContext, response_type: ResponseType, tone: ResponseTone) -> str:
        """Generate cache key for response"""
        import hashlib
        
        key_data = f"{context.user_query}_{response_type.value}_{tone.value}_{context.intent}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[GeneratedResponse]:
        """Get cached response if available and not expired"""
        if cache_key in self.response_cache:
            cached_item = self.response_cache[cache_key]
            if time.time() - cached_item['timestamp'] < self.cache_ttl:
                cached_item['response'].metadata['cache_hit'] = True
                return cached_item['response']
            else:
                del self.response_cache[cache_key]
        
        return None
    
    def _cache_response(self, cache_key: str, response: GeneratedResponse) -> None:
        """Cache response with timestamp"""
        self.response_cache[cache_key] = {
            'response': response,
            'timestamp': time.time()
        }
        
        # Simple cache size management
        if len(self.response_cache) > 100:
            # Remove oldest entries
            oldest_keys = sorted(
                self.response_cache.keys(),
                key=lambda k: self.response_cache[k]['timestamp']
            )[:20]
            
            for key in oldest_keys:
                del self.response_cache[key]
    
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

# Utility functions
async def create_response_context(
    user_query: str,
    memory_manager: MemoryManager,
    knowledge_base: KnowledgeBase,
    intent: str = "question",
    sentiment: str = "neutral",
    entities: List[Dict[str, Any]] = None
) -> ResponseContext:
    """Create response context from available information"""
    
    # Get conversation context
    context_dict = await memory_manager.get_context_for_response(user_query)
    
    # Get relevant knowledge
    relevant_knowledge = await knowledge_base.get_relevant_context(user_query, max_tokens=1500)
    
    return ResponseContext(
        user_query=user_query,
        conversation_history=context_dict.get('recent_conversation', ''),
        relevant_knowledge=relevant_knowledge,
        presentation_state=context_dict.get('presentation_state', {}),
        user_preferences=context_dict.get('user_preferences', {}),
        session_info=context_dict.get('session_info', {}),
        intent=intent,
        sentiment=sentiment,
        entities=entities or [],
        metadata={}
    )

async def test_response_generator():
    """Test function for response generator"""
    try:
        # Create test context
        test_context = ResponseContext(
            user_query="What are the main features of Eva Live?",
            conversation_history="User: Hello\nEva: Hi there! How can I help you today?",
            relevant_knowledge="Eva Live features include speech recognition, NLU, avatar rendering, and voice synthesis.",
            presentation_state={'current_slide': 1, 'topic': 'Eva Live Features'},
            user_preferences={'tone': 'professional'},
            session_info={'engagement_score': 0.8},
            intent="question",
            sentiment="neutral",
            entities=[{'text': 'Eva Live', 'label': 'PRODUCT'}],
            metadata={}
        )
        
        # Initialize generator
        generator = ResponseGenerator()
        
        # Generate response
        response = await generator.generate_response(
            test_context,
            ResponseType.QUESTION_ANSWER,
            ResponseTone.PROFESSIONAL
        )
        
        print(f"Generated response: {response.text}")
        print(f"Quality score: {response.confidence_score:.2f}")
        print(f"Processing time: {response.processing_time_ms}ms")
        print(f"Sources: {response.knowledge_sources}")
        
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_response_generator())
