"""
Eva Live Core AI Processing Layer

This module contains the core AI processing components that form the brain
of the Eva Live system, including knowledge management, response generation,
content adaptation, and AI coordination.
"""

from .knowledge_base import KnowledgeBase, SearchResult, initialize_knowledge_base
from .memory_manager import MemoryManager, SessionMemory, ConversationTurn, create_memory_manager
from .document_processor import DocumentProcessor, ProcessedDocument, DocumentChunk, process_document_async
from .response_generator import ResponseGenerator, ResponseContext, GeneratedResponse, ResponseType, ResponseTone, create_response_context
from .ai_coordinator import AICoordinator, ProcessingResult, ProcessingRequest, create_ai_coordinator

__all__ = [
    'KnowledgeBase',
    'SearchResult',
    'initialize_knowledge_base',
    'MemoryManager', 
    'SessionMemory',
    'ConversationTurn',
    'create_memory_manager',
    'DocumentProcessor',
    'ProcessedDocument',
    'DocumentChunk',
    'process_document_async',
    'ResponseGenerator',
    'ResponseContext',
    'GeneratedResponse',
    'ResponseType',
    'ResponseTone',
    'create_response_context',
    'AICoordinator',
    'ProcessingResult',
    'ProcessingRequest',
    'create_ai_coordinator'
]
