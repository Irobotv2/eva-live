"""
Eva Live Natural Language Understanding Module

This module handles natural language processing for understanding user intent,
extracting entities, and providing context-aware interpretation of conversations.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import re

import openai
from sentence_transformers import SentenceTransformer
import numpy as np

from ..shared.config import get_config
from ..shared.models import PerformanceMetric

class IntentType(str, Enum):
    """Predefined intent types"""
    QUESTION = "question"
    CLARIFICATION = "clarification"
    GREETING = "greeting"
    FAREWELL = "farewell"
    CONFIRMATION = "confirmation"
    OBJECTION = "objection"
    REQUEST_DEMO = "request_demo"
    REQUEST_PRICING = "request_pricing"
    TECHNICAL_QUESTION = "technical_question"
    BUSINESS_QUESTION = "business_question"
    FEEDBACK = "feedback"
    UNKNOWN = "unknown"

class SentimentType(str, Enum):
    """Sentiment analysis results"""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

@dataclass
class Entity:
    """Extracted entity"""
    text: str
    label: str
    confidence: float
    start_pos: int
    end_pos: int

@dataclass
class NLUResult:
    """Natural language understanding result"""
    text: str
    intent: IntentType
    intent_confidence: float
    entities: List[Entity]
    sentiment: SentimentType
    sentiment_confidence: float
    context_embeddings: Optional[np.ndarray]
    processing_time_ms: int
    metadata: Dict[str, Any]

class IntentClassifier:
    """Intent classification using GPT-4 and rule-based approaches"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI client
        openai.api_key = self.config.ai_services.openai_api_key
        
        # Define intent patterns
        self.intent_patterns = {
            IntentType.GREETING: [
                r'\b(hello|hi|hey|good morning|good afternoon|good evening)\b',
                r'\b(greetings|welcome)\b'
            ],
            IntentType.FAREWELL: [
                r'\b(goodbye|bye|farewell|see you|thank you)\b',
                r'\b(thanks|appreciate)\b.*\b(time|presentation)\b'
            ],
            IntentType.CONFIRMATION: [
                r'\b(yes|yeah|correct|right|exactly|absolutely)\b',
                r'\b(i agree|that\'s right)\b'
            ],
            IntentType.OBJECTION: [
                r'\b(no|disagree|wrong|incorrect)\b',
                r'\b(i don\'t think|not sure about)\b'
            ],
            IntentType.REQUEST_DEMO: [
                r'\b(demo|demonstration|show me|can you show)\b',
                r'\b(example|sample|how it works)\b'
            ],
            IntentType.REQUEST_PRICING: [
                r'\b(price|pricing|cost|how much|expensive)\b',
                r'\b(budget|afford|payment|subscription)\b'
            ],
            IntentType.TECHNICAL_QUESTION: [
                r'\b(technical|technology|api|integration|system)\b',
                r'\b(how does|architecture|platform|security)\b'
            ],
            IntentType.BUSINESS_QUESTION: [
                r'\b(business|roi|revenue|profit|market)\b',
                r'\b(customers|clients|use case|benefits)\b'
            ]
        }
    
    async def classify_intent(self, text: str, context: Optional[Dict[str, Any]] = None) -> Tuple[IntentType, float]:
        """Classify user intent using hybrid approach"""
        start_time = time.time()
        
        try:
            # First try rule-based classification
            rule_based_intent, rule_confidence = self._rule_based_classification(text)
            
            if rule_confidence > 0.8:
                return rule_based_intent, rule_confidence
            
            # Use GPT-4 for complex classification
            gpt_intent, gpt_confidence = await self._gpt_classification(text, context)
            
            # Combine results (prefer GPT if confidence is high)
            if gpt_confidence > 0.7:
                return gpt_intent, gpt_confidence
            elif rule_confidence > 0.5:
                return rule_based_intent, rule_confidence
            else:
                return gpt_intent, gpt_confidence
                
        except Exception as e:
            self.logger.error(f"Intent classification error: {e}")
            return IntentType.UNKNOWN, 0.0
    
    def _rule_based_classification(self, text: str) -> Tuple[IntentType, float]:
        """Rule-based intent classification using regex patterns"""
        text_lower = text.lower()
        
        # Check patterns for each intent
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    # Calculate confidence based on pattern match strength
                    confidence = min(0.9, 0.6 + len(re.findall(pattern, text_lower)) * 0.1)
                    return intent, confidence
        
        # Default to question if contains question markers
        if '?' in text or any(word in text_lower for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            return IntentType.QUESTION, 0.7
        
        return IntentType.UNKNOWN, 0.0
    
    async def _gpt_classification(self, text: str, context: Optional[Dict[str, Any]] = None) -> Tuple[IntentType, float]:
        """GPT-4 based intent classification"""
        try:
            # Build context string
            context_str = ""
            if context:
                context_str = f"Context: {json.dumps(context, indent=2)}\n\n"
            
            # Create prompt
            prompt = f"""
Analyze the following user message and classify its intent. Consider the context if provided.

{context_str}User Message: "{text}"

Available intents:
- question: User is asking for information
- clarification: User wants clarification on something said
- greeting: User is greeting or starting conversation
- farewell: User is ending conversation or saying goodbye
- confirmation: User is agreeing or confirming something
- objection: User is disagreeing or objecting
- request_demo: User wants to see a demonstration
- request_pricing: User is asking about pricing or costs
- technical_question: User has technical questions
- business_question: User has business-related questions
- feedback: User is providing feedback
- unknown: Intent cannot be determined

Respond with JSON format:
{{
    "intent": "intent_name",
    "confidence": 0.85,
    "reasoning": "Brief explanation"
}}
"""
            
            response = await openai.ChatCompletion.acreate(
                model=self.config.ai_services.openai_model,
                messages=[
                    {"role": "system", "content": "You are an expert at understanding user intent in business conversations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=200
            )
            
            # Parse response
            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            
            intent_str = result.get('intent', 'unknown')
            confidence = result.get('confidence', 0.0)
            
            # Convert to enum
            try:
                intent = IntentType(intent_str)
            except ValueError:
                intent = IntentType.UNKNOWN
                confidence = 0.0
            
            return intent, confidence
            
        except Exception as e:
            self.logger.error(f"GPT intent classification error: {e}")
            return IntentType.UNKNOWN, 0.0

class EntityExtractor:
    """Named Entity Recognition for extracting important entities"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Define entity patterns
        self.entity_patterns = {
            'MONEY': [
                r'\$[\d,]+(?:\.\d{2})?',
                r'\b\d+(?:,\d{3})*(?:\.\d{2})?\s*(?:dollars?|usd|euros?|pounds?)\b'
            ],
            'DATE': [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                r'\b(?:january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2},?\s+\d{4}\b'
            ],
            'TIME': [
                r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:am|pm)?\b'
            ],
            'COMPANY': [
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+(?:Inc|Corp|LLC|Ltd|Company)\b'
            ],
            'PRODUCT': [
                r'\bEva Live\b',
                r'\bEva\b(?!\s+Live)'
            ],
            'FEATURE': [
                r'\b(?:virtual camera|voice synthesis|avatar|ai|machine learning|nlp)\b'
            ]
        }
    
    async def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities from text using rule-based and AI approaches"""
        entities = []
        
        try:
            # Rule-based extraction
            rule_entities = self._rule_based_extraction(text)
            entities.extend(rule_entities)
            
            # GPT-4 based extraction for complex entities
            gpt_entities = await self._gpt_extraction(text)
            entities.extend(gpt_entities)
            
            # Remove duplicates and overlaps
            entities = self._remove_duplicate_entities(entities)
            
            return entities
            
        except Exception as e:
            self.logger.error(f"Entity extraction error: {e}")
            return []
    
    def _rule_based_extraction(self, text: str) -> List[Entity]:
        """Rule-based entity extraction using regex patterns"""
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entities.append(Entity(
                        text=match.group(),
                        label=entity_type,
                        confidence=0.8,
                        start_pos=match.start(),
                        end_pos=match.end()
                    ))
        
        return entities
    
    async def _gpt_extraction(self, text: str) -> List[Entity]:
        """GPT-4 based entity extraction"""
        try:
            prompt = f"""
Extract named entities from the following text. Focus on business-relevant entities.

Text: "{text}"

Entity types to look for:
- PERSON: Names of people
- ORGANIZATION: Company/organization names
- PRODUCT: Product or service names
- TECHNOLOGY: Technical terms or technologies
- METRIC: Numbers, percentages, measurements
- LOCATION: Places, cities, countries

Respond with JSON format:
{{
    "entities": [
        {{
            "text": "entity text",
            "label": "ENTITY_TYPE",
            "confidence": 0.9
        }}
    ]
}}
"""
            
            response = await openai.ChatCompletion.acreate(
                model=self.config.ai_services.openai_model,
                messages=[
                    {"role": "system", "content": "You are an expert at named entity recognition."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=300
            )
            
            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            
            entities = []
            for entity_data in result.get('entities', []):
                # Find entity position in text
                entity_text = entity_data['text']
                start_pos = text.lower().find(entity_text.lower())
                
                if start_pos >= 0:
                    entities.append(Entity(
                        text=entity_text,
                        label=entity_data['label'],
                        confidence=entity_data.get('confidence', 0.7),
                        start_pos=start_pos,
                        end_pos=start_pos + len(entity_text)
                    ))
            
            return entities
            
        except Exception as e:
            self.logger.error(f"GPT entity extraction error: {e}")
            return []
    
    def _remove_duplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Remove duplicate and overlapping entities"""
        # Sort by position
        entities.sort(key=lambda e: e.start_pos)
        
        filtered = []
        for entity in entities:
            # Check for overlaps with existing entities
            overlap = False
            for existing in filtered:
                if (entity.start_pos < existing.end_pos and 
                    entity.end_pos > existing.start_pos):
                    # Keep entity with higher confidence
                    if entity.confidence > existing.confidence:
                        filtered.remove(existing)
                        break
                    else:
                        overlap = True
                        break
            
            if not overlap:
                filtered.append(entity)
        
        return filtered

class SentimentAnalyzer:
    """Sentiment analysis for understanding emotional tone"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Sentiment keywords
        self.positive_words = {
            'excellent', 'great', 'amazing', 'wonderful', 'fantastic', 'perfect',
            'love', 'like', 'impressed', 'satisfied', 'happy', 'pleased'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike',
            'disappointed', 'frustrated', 'angry', 'concerned', 'worried'
        }
    
    async def analyze_sentiment(self, text: str) -> Tuple[SentimentType, float]:
        """Analyze sentiment using hybrid approach"""
        try:
            # Rule-based sentiment
            rule_sentiment, rule_confidence = self._rule_based_sentiment(text)
            
            # GPT-4 sentiment analysis
            gpt_sentiment, gpt_confidence = await self._gpt_sentiment(text)
            
            # Combine results
            if gpt_confidence > 0.8:
                return gpt_sentiment, gpt_confidence
            elif rule_confidence > 0.6:
                return rule_sentiment, rule_confidence
            else:
                return gpt_sentiment, max(gpt_confidence, 0.5)
                
        except Exception as e:
            self.logger.error(f"Sentiment analysis error: {e}")
            return SentimentType.NEUTRAL, 0.0
    
    def _rule_based_sentiment(self, text: str) -> Tuple[SentimentType, float]:
        """Rule-based sentiment analysis"""
        text_lower = set(text.lower().split())
        
        positive_count = len(text_lower.intersection(self.positive_words))
        negative_count = len(text_lower.intersection(self.negative_words))
        
        total_words = len(text.split())
        
        if positive_count > negative_count:
            confidence = min(0.9, 0.6 + (positive_count / total_words) * 2)
            return SentimentType.POSITIVE, confidence
        elif negative_count > positive_count:
            confidence = min(0.9, 0.6 + (negative_count / total_words) * 2)
            return SentimentType.NEGATIVE, confidence
        else:
            return SentimentType.NEUTRAL, 0.5
    
    async def _gpt_sentiment(self, text: str) -> Tuple[SentimentType, float]:
        """GPT-4 based sentiment analysis"""
        try:
            prompt = f"""
Analyze the sentiment of the following text in a business context.

Text: "{text}"

Classify the sentiment as:
- positive: User is expressing satisfaction, agreement, or positive emotions
- negative: User is expressing dissatisfaction, disagreement, or negative emotions  
- neutral: User is expressing neutral, factual, or mixed emotions

Respond with JSON format:
{{
    "sentiment": "positive|negative|neutral",
    "confidence": 0.85,
    "reasoning": "Brief explanation"
}}
"""
            
            response = await openai.ChatCompletion.acreate(
                model=self.config.ai_services.openai_model,
                messages=[
                    {"role": "system", "content": "You are an expert at sentiment analysis in business conversations."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=150
            )
            
            result_text = response.choices[0].message.content
            result = json.loads(result_text)
            
            sentiment_str = result.get('sentiment', 'neutral')
            confidence = result.get('confidence', 0.0)
            
            try:
                sentiment = SentimentType(sentiment_str)
            except ValueError:
                sentiment = SentimentType.NEUTRAL
                confidence = 0.5
            
            return sentiment, confidence
            
        except Exception as e:
            self.logger.error(f"GPT sentiment analysis error: {e}")
            return SentimentType.NEUTRAL, 0.0

class NLUModule:
    """Main Natural Language Understanding module"""
    
    def __init__(self, session_id: Optional[str] = None):
        self.config = get_config()
        self.session_id = session_id
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Initialize embedding model for context
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Performance tracking
        self.metrics: List[PerformanceMetric] = []
    
    async def process(self, text: str, context: Optional[Dict[str, Any]] = None) -> NLUResult:
        """Process text through NLU pipeline"""
        start_time = time.time()
        
        try:
            # Run all analyses in parallel
            intent_task = self.intent_classifier.classify_intent(text, context)
            entities_task = self.entity_extractor.extract_entities(text)
            sentiment_task = self.sentiment_analyzer.analyze_sentiment(text)
            
            # Wait for all tasks to complete
            (intent, intent_confidence), entities, (sentiment, sentiment_confidence) = await asyncio.gather(
                intent_task, entities_task, sentiment_task
            )
            
            # Generate context embeddings
            embeddings = self.embedding_model.encode([text])[0]
            
            processing_time = int((time.time() - start_time) * 1000)
            
            # Create result
            result = NLUResult(
                text=text,
                intent=intent,
                intent_confidence=intent_confidence,
                entities=entities,
                sentiment=sentiment,
                sentiment_confidence=sentiment_confidence,
                context_embeddings=embeddings,
                processing_time_ms=processing_time,
                metadata={
                    'context': context,
                    'entity_count': len(entities),
                    'text_length': len(text)
                }
            )
            
            # Record metrics
            await self._record_metric("processing_time_ms", processing_time, "nlu")
            await self._record_metric("intent_confidence", intent_confidence, "nlu")
            await self._record_metric("sentiment_confidence", sentiment_confidence, "nlu")
            await self._record_metric("entity_count", len(entities), "nlu")
            
            return result
            
        except Exception as e:
            self.logger.error(f"NLU processing error: {e}")
            raise
    
    async def _record_metric(self, metric_type: str, value: float, component: str) -> None:
        """Record performance metric"""
        metric = PerformanceMetric(
            metric_type=metric_type,
            metric_value=value,
            component=component
        )
        
        if self.session_id:
            # In a real implementation, this would be saved to database
            pass
        
        self.metrics.append(metric)
    
    def get_metrics(self) -> List[PerformanceMetric]:
        """Get recorded metrics"""
        return self.metrics.copy()

# Utility functions
def extract_question_type(text: str) -> str:
    """Extract the type of question being asked"""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ['what', 'which']):
        return 'what'
    elif any(word in text_lower for word in ['how']):
        return 'how'
    elif any(word in text_lower for word in ['why']):
        return 'why'
    elif any(word in text_lower for word in ['when']):
        return 'when'
    elif any(word in text_lower for word in ['where']):
        return 'where'
    elif any(word in text_lower for word in ['who']):
        return 'who'
    else:
        return 'general'

async def test_nlu():
    """Test function for NLU module"""
    # Initialize module
    nlu = NLUModule()
    
    # Test cases
    test_texts = [
        "Hello, I'm interested in learning more about Eva Live",
        "What are the pricing options for your product?",
        "Can you show me a demo of the avatar technology?",
        "I'm not sure about the security aspects of this solution",
        "This looks amazing! When can we get started?"
    ]
    
    for text in test_texts:
        print(f"\nTesting: '{text}'")
        try:
            result = await nlu.process(text)
            print(f"Intent: {result.intent} (confidence: {result.intent_confidence:.2f})")
            print(f"Sentiment: {result.sentiment} (confidence: {result.sentiment_confidence:.2f})")
            print(f"Entities: {[f'{e.text} ({e.label})' for e in result.entities]}")
            print(f"Processing time: {result.processing_time_ms}ms")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_nlu())
