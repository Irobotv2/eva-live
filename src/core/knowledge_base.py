"""
Eva Live Knowledge Base Module

This module handles vector database operations for semantic search and knowledge
retrieval. It integrates with Pinecone for vector storage and OpenAI for embeddings.
"""

import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import hashlib
import json

# Vector database and embeddings
import pinecone
import openai
import numpy as np
from sentence_transformers import SentenceTransformer

# Caching
import redis
import pickle

from ..shared.config import get_config
from ..shared.models import PerformanceMetric
from .document_processor import DocumentChunk, ProcessedDocument

@dataclass
class SearchResult:
    """Search result from knowledge base"""
    id: str
    content: str
    score: float
    metadata: Dict[str, Any]
    source: str
    chunk_index: int

@dataclass
class KnowledgeEntry:
    """Entry in the knowledge base"""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    created_at: float
    updated_at: float

class EmbeddingGenerator:
    """Handles embedding generation using OpenAI and local models"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI
        openai.api_key = self.config.ai_services.openai_api_key
        self.openai_model = self.config.ai_services.openai_embedding_model
        
        # Initialize local embedding model as fallback
        try:
            self.local_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            self.logger.warning(f"Failed to load local embedding model: {e}")
            self.local_model = None
    
    async def generate_embeddings(self, texts: List[str], use_openai: bool = True) -> List[List[float]]:
        """Generate embeddings for a list of texts"""
        if use_openai:
            try:
                return await self._generate_openai_embeddings(texts)
            except Exception as e:
                self.logger.warning(f"OpenAI embeddings failed, using fallback: {e}")
                return await self._generate_local_embeddings(texts)
        else:
            return await self._generate_local_embeddings(texts)
    
    async def _generate_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API"""
        try:
            # Process in batches to avoid rate limits
            batch_size = 100
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                response = await openai.Embedding.acreate(
                    model=self.openai_model,
                    input=batch
                )
                
                batch_embeddings = [item['embedding'] for item in response['data']]
                all_embeddings.extend(batch_embeddings)
                
                # Small delay to respect rate limits
                if len(texts) > batch_size:
                    await asyncio.sleep(0.1)
            
            return all_embeddings
            
        except Exception as e:
            self.logger.error(f"OpenAI embedding generation failed: {e}")
            raise
    
    async def _generate_local_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using local model"""
        if not self.local_model:
            raise RuntimeError("No local embedding model available")
        
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None, 
                lambda: self.local_model.encode(texts).tolist()
            )
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Local embedding generation failed: {e}")
            raise

class VectorDatabase:
    """Vector database interface for Pinecone"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize Pinecone
        pinecone.init(
            api_key=self.config.database.pinecone_api_key,
            environment=self.config.database.pinecone_environment
        )
        
        self.index_name = self.config.database.pinecone_index_name
        self.dimension = 1536  # OpenAI embedding dimension
        
        # Connect to index
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize or connect to Pinecone index"""
        try:
            # Check if index exists
            if self.index_name not in pinecone.list_indexes():
                self.logger.info(f"Creating Pinecone index: {self.index_name}")
                pinecone.create_index(
                    name=self.index_name,
                    dimension=self.dimension,
                    metric="cosine",
                    pod_type="p1.x1"
                )
                
                # Wait for index to be ready
                time.sleep(30)
            
            # Connect to index
            self.index = pinecone.Index(self.index_name)
            self.logger.info(f"Connected to Pinecone index: {self.index_name}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Pinecone index: {e}")
            raise
    
    async def upsert_vectors(self, vectors: List[Tuple[str, List[float], Dict[str, Any]]]) -> None:
        """Upsert vectors to the database"""
        try:
            # Process in batches
            batch_size = 100
            
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                
                # Convert to Pinecone format
                upsert_data = []
                for vector_id, embedding, metadata in batch:
                    upsert_data.append({
                        'id': vector_id,
                        'values': embedding,
                        'metadata': metadata
                    })
                
                # Upsert to Pinecone
                self.index.upsert(vectors=upsert_data)
                
                self.logger.debug(f"Upserted batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")
            
            self.logger.info(f"Upserted {len(vectors)} vectors to knowledge base")
            
        except Exception as e:
            self.logger.error(f"Failed to upsert vectors: {e}")
            raise
    
    async def search_vectors(self, query_embedding: List[float], top_k: int = 10, filter_dict: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search for similar vectors"""
        try:
            # Perform search
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )
            
            # Convert to SearchResult objects
            search_results = []
            for match in results['matches']:
                result = SearchResult(
                    id=match['id'],
                    content=match['metadata'].get('content', ''),
                    score=match['score'],
                    metadata=match['metadata'],
                    source=match['metadata'].get('source_file', ''),
                    chunk_index=match['metadata'].get('chunk_index', 0)
                )
                search_results.append(result)
            
            return search_results
            
        except Exception as e:
            self.logger.error(f"Vector search failed: {e}")
            raise
    
    async def delete_vectors(self, vector_ids: List[str]) -> None:
        """Delete vectors from the database"""
        try:
            self.index.delete(ids=vector_ids)
            self.logger.info(f"Deleted {len(vector_ids)} vectors from knowledge base")
            
        except Exception as e:
            self.logger.error(f"Failed to delete vectors: {e}")
            raise
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        try:
            stats = self.index.describe_index_stats()
            return stats
        except Exception as e:
            self.logger.error(f"Failed to get index stats: {e}")
            return {}

class KnowledgeCache:
    """Redis-based caching for knowledge base operations"""
    
    def __init__(self):
        self.config = get_config()
        self.logger = logging.getLogger(__name__)
        
        # Initialize Redis
        self.redis_client = redis.Redis(
            host=self.config.database.redis_host,
            port=self.config.database.redis_port,
            db=self.config.database.redis_database,
            password=self.config.database.redis_password,
            decode_responses=False  # We'll handle encoding ourselves
        )
        
        # Cache settings
        self.cache_timeout = self.config.get('features.cache_timeout', 1800)  # 30 minutes
        self.max_cache_size = 1000  # Maximum cached items
    
    async def get_search_results(self, query_hash: str) -> Optional[List[SearchResult]]:
        """Get cached search results"""
        try:
            cache_key = f"search:{query_hash}"
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                search_results = pickle.loads(cached_data)
                self.logger.debug(f"Cache hit for query hash: {query_hash}")
                return search_results
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Cache get failed: {e}")
            return None
    
    async def set_search_results(self, query_hash: str, results: List[SearchResult]) -> None:
        """Cache search results"""
        try:
            cache_key = f"search:{query_hash}"
            serialized_data = pickle.dumps(results)
            
            self.redis_client.setex(
                cache_key,
                self.cache_timeout,
                serialized_data
            )
            
            self.logger.debug(f"Cached search results for query hash: {query_hash}")
            
        except Exception as e:
            self.logger.warning(f"Cache set failed: {e}")
    
    async def get_embeddings(self, text_hash: str) -> Optional[List[float]]:
        """Get cached embeddings"""
        try:
            cache_key = f"embedding:{text_hash}"
            cached_data = self.redis_client.get(cache_key)
            
            if cached_data:
                embedding = pickle.loads(cached_data)
                return embedding
            
            return None
            
        except Exception as e:
            self.logger.warning(f"Embedding cache get failed: {e}")
            return None
    
    async def set_embeddings(self, text_hash: str, embedding: List[float]) -> None:
        """Cache embeddings"""
        try:
            cache_key = f"embedding:{text_hash}"
            serialized_data = pickle.dumps(embedding)
            
            self.redis_client.setex(
                cache_key,
                self.cache_timeout * 2,  # Cache embeddings longer
                serialized_data
            )
            
        except Exception as e:
            self.logger.warning(f"Embedding cache set failed: {e}")
    
    def _generate_query_hash(self, query: str, filters: Optional[Dict[str, Any]] = None) -> str:
        """Generate hash for search query"""
        query_data = {
            'query': query,
            'filters': filters or {}
        }
        query_str = json.dumps(query_data, sort_keys=True)
        return hashlib.md5(query_str.encode()).hexdigest()

class KnowledgeBase:
    """Main knowledge base class integrating all components"""
    
    def __init__(self, session_id: Optional[str] = None):
        self.config = get_config()
        self.session_id = session_id
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.embedding_generator = EmbeddingGenerator()
        self.vector_db = VectorDatabase()
        self.cache = KnowledgeCache()
        
        # Performance tracking
        self.metrics: List[PerformanceMetric] = []
    
    async def add_document(self, processed_doc: ProcessedDocument) -> None:
        """Add a processed document to the knowledge base"""
        start_time = time.time()
        
        try:
            # Generate embeddings for all chunks
            chunk_texts = [chunk.content for chunk in processed_doc.chunks]
            embeddings = await self.embedding_generator.generate_embeddings(chunk_texts)
            
            # Prepare vectors for upsert
            vectors = []
            for chunk, embedding in zip(processed_doc.chunks, embeddings):
                metadata = {
                    **chunk.metadata,
                    'content': chunk.content,
                    'document_title': processed_doc.title,
                    'token_count': chunk.token_count,
                    'total_chunks': chunk.total_chunks
                }
                
                vectors.append((chunk.id, embedding, metadata))
            
            # Upsert to vector database
            await self.vector_db.upsert_vectors(vectors)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            # Record metrics
            await self._record_metric("document_indexing_time_ms", processing_time, "knowledge_base")
            await self._record_metric("chunks_indexed", len(processed_doc.chunks), "knowledge_base")
            
            self.logger.info(f"Added document {processed_doc.document_id} with {len(processed_doc.chunks)} chunks to knowledge base")
            
        except Exception as e:
            self.logger.error(f"Failed to add document to knowledge base: {e}")
            raise
    
    async def search(self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Search the knowledge base for relevant content"""
        start_time = time.time()
        
        try:
            # Generate query hash for caching
            query_hash = self.cache._generate_query_hash(query, filters)
            
            # Check cache first
            cached_results = await self.cache.get_search_results(query_hash)
            if cached_results:
                search_time = int((time.time() - start_time) * 1000)
                await self._record_metric("search_time_ms", search_time, "knowledge_base_cache")
                return cached_results[:top_k]
            
            # Generate query embedding
            query_embeddings = await self.embedding_generator.generate_embeddings([query])
            query_embedding = query_embeddings[0]
            
            # Search vector database
            results = await self.vector_db.search_vectors(
                query_embedding=query_embedding,
                top_k=top_k,
                filter_dict=filters
            )
            
            # Cache results
            await self.cache.set_search_results(query_hash, results)
            
            search_time = int((time.time() - start_time) * 1000)
            
            # Record metrics
            await self._record_metric("search_time_ms", search_time, "knowledge_base")
            await self._record_metric("results_found", len(results), "knowledge_base")
            
            self.logger.info(f"Search completed: {len(results)} results found in {search_time}ms")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Knowledge base search failed: {e}")
            raise
    
    async def get_relevant_context(self, query: str, max_tokens: int = 2000) -> str:
        """Get relevant context for a query, optimized for token count"""
        try:
            # Search for relevant chunks
            results = await self.search(query, top_k=20)
            
            # Select best chunks within token limit
            context_parts = []
            current_tokens = 0
            
            for result in results:
                chunk_tokens = result.metadata.get('token_count', 0)
                
                if current_tokens + chunk_tokens <= max_tokens:
                    context_parts.append(f"Source: {result.source}\n{result.content}")
                    current_tokens += chunk_tokens
                else:
                    break
            
            context = "\n\n---\n\n".join(context_parts)
            
            self.logger.debug(f"Generated context with {current_tokens} tokens from {len(context_parts)} chunks")
            
            return context
            
        except Exception as e:
            self.logger.error(f"Failed to get relevant context: {e}")
            return ""
    
    async def remove_document(self, document_id: str) -> None:
        """Remove a document from the knowledge base"""
        try:
            # Search for all chunks of this document
            results = await self.search("", top_k=1000, filters={'document_id': document_id})
            
            # Extract chunk IDs
            chunk_ids = [result.id for result in results]
            
            if chunk_ids:
                # Delete from vector database
                await self.vector_db.delete_vectors(chunk_ids)
                self.logger.info(f"Removed document {document_id} ({len(chunk_ids)} chunks) from knowledge base")
            
        except Exception as e:
            self.logger.error(f"Failed to remove document {document_id}: {e}")
            raise
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        try:
            vector_stats = self.vector_db.get_index_stats()
            
            return {
                'total_vectors': vector_stats.get('totalVectorCount', 0),
                'index_fullness': vector_stats.get('indexFullness', 0.0),
                'namespaces': vector_stats.get('namespaces', {}),
                'cache_info': {
                    'redis_connected': self.cache.redis_client.ping()
                }
            }
        except Exception as e:
            self.logger.error(f"Failed to get knowledge base stats: {e}")
            return {}
    
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
async def initialize_knowledge_base() -> KnowledgeBase:
    """Initialize and return a knowledge base instance"""
    kb = KnowledgeBase()
    return kb

async def test_knowledge_base():
    """Test function for knowledge base"""
    try:
        # Initialize knowledge base
        kb = KnowledgeBase()
        
        # Test search
        results = await kb.search("test query", top_k=5)
        print(f"Search returned {len(results)} results")
        
        # Test context generation
        context = await kb.get_relevant_context("test query")
        print(f"Generated context with {len(context)} characters")
        
        # Get stats
        stats = kb.get_stats()
        print(f"Knowledge base stats: {stats}")
        
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_knowledge_base())
