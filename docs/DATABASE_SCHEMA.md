# Eva Live Database Schema

## 1. Overview

Eva Live uses a hybrid database architecture combining PostgreSQL for relational data, Redis for caching and session management, and a vector database (Pinecone/Weaviate) for semantic search and knowledge management.

## 2. PostgreSQL Schema

### 2.1 User Management

```sql
-- Users table for authentication and authorization
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    company VARCHAR(255),
    role VARCHAR(50) DEFAULT 'user' CHECK (role IN ('admin', 'operator', 'user')),
    subscription_tier VARCHAR(50) DEFAULT 'basic' CHECK (subscription_tier IN ('basic', 'professional', 'enterprise')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    email_verified BOOLEAN DEFAULT false,
    api_quota_monthly INTEGER DEFAULT 10000,
    api_calls_used INTEGER DEFAULT 0,
    concurrent_sessions_limit INTEGER DEFAULT 5
);

-- API Keys for programmatic access
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    key_hash VARCHAR(255) NOT NULL UNIQUE,
    name VARCHAR(100) NOT NULL,
    scopes TEXT[] DEFAULT ARRAY['eva:read'],
    last_used TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true
);

-- OAuth applications
CREATE TABLE oauth_applications (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    client_id VARCHAR(255) UNIQUE NOT NULL,
    client_secret_hash VARCHAR(255) NOT NULL,
    redirect_uris TEXT[],
    scopes TEXT[],
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true
);
```

### 2.2 Avatar Management

```sql
-- Avatar definitions and configurations
CREATE TABLE avatars (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    avatar_type VARCHAR(50) DEFAULT 'standard' CHECK (avatar_type IN ('standard', 'custom', 'premium')),
    model_version VARCHAR(50) DEFAULT 'v1',
    appearance_config JSONB NOT NULL DEFAULT '{}',
    personality_config JSONB NOT NULL DEFAULT '{}',
    voice_config JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_public BOOLEAN DEFAULT false,
    is_active BOOLEAN DEFAULT true,
    training_status VARCHAR(50) DEFAULT 'ready' CHECK (training_status IN ('pending', 'training', 'ready', 'failed')),
    quality_score DECIMAL(3,2) DEFAULT 0.0
);

-- Voice samples for custom avatars
CREATE TABLE voice_samples (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    avatar_id UUID NOT NULL REFERENCES avatars(id) ON DELETE CASCADE,
    file_path VARCHAR(500) NOT NULL,
    duration_seconds DECIMAL(8,2),
    quality_score DECIMAL(3,2),
    processing_status VARCHAR(50) DEFAULT 'pending' CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### 2.3 Session Management

```sql
-- Eva presentation sessions
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    avatar_id UUID NOT NULL REFERENCES avatars(id),
    name VARCHAR(255),
    description TEXT,
    status VARCHAR(50) DEFAULT 'created' CHECK (status IN ('created', 'starting', 'active', 'paused', 'stopping', 'stopped', 'failed')),
    config JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    started_at TIMESTAMP WITH TIME ZONE,
    ended_at TIMESTAMP WITH TIME ZONE,
    total_duration_seconds INTEGER DEFAULT 0,
    participant_count INTEGER DEFAULT 0,
    questions_received INTEGER DEFAULT 0,
    operator_interventions INTEGER DEFAULT 0,
    quality_metrics JSONB DEFAULT '{}',
    error_count INTEGER DEFAULT 0,
    last_error TEXT
);

-- Session participants tracking
CREATE TABLE session_participants (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    participant_id VARCHAR(255) NOT NULL,
    participant_name VARCHAR(255),
    joined_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    left_at TIMESTAMP WITH TIME ZONE,
    questions_asked INTEGER DEFAULT 0,
    engagement_score DECIMAL(3,2)
);

-- Real-time session state
CREATE TABLE session_state (
    session_id UUID PRIMARY KEY REFERENCES sessions(id) ON DELETE CASCADE,
    current_slide INTEGER DEFAULT 0,
    presentation_mode VARCHAR(50) DEFAULT 'auto',
    is_speaking BOOLEAN DEFAULT false,
    awaiting_questions BOOLEAN DEFAULT false,
    operator_connected BOOLEAN DEFAULT false,
    last_interaction TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    current_topic VARCHAR(255),
    conversation_context JSONB DEFAULT '{}',
    system_metrics JSONB DEFAULT '{}'
);
```

### 2.4 Content Management

```sql
-- Presentation content
CREATE TABLE presentations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    file_path VARCHAR(500),
    file_type VARCHAR(50),
    file_size_bytes BIGINT,
    slide_count INTEGER DEFAULT 0,
    processing_status VARCHAR(50) DEFAULT 'pending' CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed')),
    extraction_config JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processed_at TIMESTAMP WITH TIME ZONE
);

-- Individual slides
CREATE TABLE presentation_slides (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    presentation_id UUID NOT NULL REFERENCES presentations(id) ON DELETE CASCADE,
    slide_number INTEGER NOT NULL,
    title VARCHAR(500),
    content TEXT,
    speaker_notes TEXT,
    slide_type VARCHAR(50) DEFAULT 'content',
    duration_seconds INTEGER,
    image_path VARCHAR(500),
    animations JSONB DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Knowledge base documents
CREATE TABLE knowledge_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    title VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    document_type VARCHAR(50) DEFAULT 'general' CHECK (document_type IN ('faq', 'general', 'technical', 'marketing')),
    priority INTEGER DEFAULT 1 CHECK (priority BETWEEN 1 AND 10),
    tags TEXT[],
    embedding_id VARCHAR(255), -- Reference to vector database
    confidence_score DECIMAL(3,2) DEFAULT 0.0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true
);
```

### 2.5 Conversation Management

```sql
-- Conversation history
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    participant_id VARCHAR(255),
    message_type VARCHAR(50) NOT NULL CHECK (message_type IN ('question', 'response', 'system', 'operator')),
    content TEXT NOT NULL,
    speaker VARCHAR(100),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    confidence_score DECIMAL(3,2),
    response_time_ms INTEGER,
    context JSONB DEFAULT '{}',
    metadata JSONB DEFAULT '{}'
);

-- Question analysis and responses
CREATE TABLE question_responses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    question_text TEXT NOT NULL,
    response_text TEXT NOT NULL,
    intent_classification VARCHAR(100),
    entities_extracted JSONB DEFAULT '{}',
    knowledge_sources TEXT[],
    confidence_score DECIMAL(3,2) NOT NULL,
    response_time_ms INTEGER NOT NULL,
    operator_reviewed BOOLEAN DEFAULT false,
    quality_rating INTEGER CHECK (quality_rating BETWEEN 1 AND 5),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### 2.6 Monitoring and Analytics

```sql
-- System performance metrics
CREATE TABLE performance_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    metric_type VARCHAR(100) NOT NULL,
    metric_value DECIMAL(10,4) NOT NULL,
    unit VARCHAR(50),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    component VARCHAR(100),
    additional_data JSONB DEFAULT '{}'
);

-- System alerts and notifications
CREATE TABLE system_alerts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID REFERENCES sessions(id) ON DELETE CASCADE,
    alert_type VARCHAR(100) NOT NULL,
    severity VARCHAR(20) DEFAULT 'info' CHECK (severity IN ('info', 'warning', 'error', 'critical')),
    title VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    component VARCHAR(100),
    metadata JSONB DEFAULT '{}',
    acknowledged BOOLEAN DEFAULT false,
    acknowledged_by UUID REFERENCES users(id),
    acknowledged_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    resolved_at TIMESTAMP WITH TIME ZONE
);

-- Operator interventions
CREATE TABLE operator_interventions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    operator_id UUID NOT NULL REFERENCES users(id),
    intervention_type VARCHAR(50) NOT NULL CHECK (intervention_type IN ('takeover', 'suggestion', 'correction', 'emergency_stop')),
    trigger_reason VARCHAR(255),
    action_taken TEXT NOT NULL,
    duration_seconds INTEGER,
    effectiveness_rating INTEGER CHECK (effectiveness_rating BETWEEN 1 AND 5),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    ended_at TIMESTAMP WITH TIME ZONE
);
```

### 2.7 Integrations

```sql
-- Platform integrations
CREATE TABLE platform_integrations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    platform VARCHAR(50) NOT NULL CHECK (platform IN ('zoom', 'teams', 'google_meet', 'webex')),
    config JSONB NOT NULL DEFAULT '{}',
    credentials_encrypted TEXT,
    status VARCHAR(50) DEFAULT 'inactive' CHECK (status IN ('active', 'inactive', 'error', 'expired')),
    last_sync TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Webhook configurations
CREATE TABLE webhooks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    url VARCHAR(500) NOT NULL,
    events TEXT[] NOT NULL,
    secret_hash VARCHAR(255),
    is_active BOOLEAN DEFAULT true,
    retry_count INTEGER DEFAULT 3,
    timeout_seconds INTEGER DEFAULT 30,
    last_triggered TIMESTAMP WITH TIME ZONE,
    failure_count INTEGER DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```

### 2.8 Indexes and Performance

```sql
-- Primary indexes for performance
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_active ON users(is_active) WHERE is_active = true;
CREATE INDEX idx_api_keys_hash ON api_keys(key_hash);
CREATE INDEX idx_api_keys_user ON api_keys(user_id) WHERE is_active = true;

CREATE INDEX idx_sessions_user ON sessions(user_id);
CREATE INDEX idx_sessions_status ON sessions(status);
CREATE INDEX idx_sessions_created ON sessions(created_at);

CREATE INDEX idx_conversations_session ON conversations(session_id);
CREATE INDEX idx_conversations_timestamp ON conversations(timestamp);
CREATE INDEX idx_conversations_type ON conversations(message_type);

CREATE INDEX idx_performance_metrics_session ON performance_metrics(session_id);
CREATE INDEX idx_performance_metrics_timestamp ON performance_metrics(timestamp);
CREATE INDEX idx_performance_metrics_type ON performance_metrics(metric_type);

CREATE INDEX idx_presentations_user ON presentations(user_id);
CREATE INDEX idx_slides_presentation ON presentation_slides(presentation_id);
CREATE INDEX idx_knowledge_documents_session ON knowledge_documents(session_id) WHERE session_id IS NOT NULL;

-- Composite indexes for common queries
CREATE INDEX idx_sessions_user_status ON sessions(user_id, status);
CREATE INDEX idx_conversations_session_timestamp ON conversations(session_id, timestamp);
CREATE INDEX idx_performance_session_type_timestamp ON performance_metrics(session_id, metric_type, timestamp);
```

## 3. Redis Schema

### 3.1 Session Management

```redis
# Active session data (TTL: session duration + 1 hour)
eva:session:{session_id}:state = {
    "status": "active",
    "current_slide": 5,
    "is_speaking": false,
    "operator_connected": true,
    "last_heartbeat": "2025-01-18T20:30:00Z"
}

# Session metrics cache (TTL: 1 hour)
eva:session:{session_id}:metrics = {
    "latency_ms": 234,
    "cpu_usage": 0.68,
    "memory_usage": 0.71,
    "gpu_usage": 0.84,
    "active_connections": 1
}

# Conversation context (TTL: 24 hours)
eva:session:{session_id}:context = {
    "recent_questions": ["What's the timeline?", "Who's the target market?"],
    "current_topic": "product_features",
    "user_preferences": {},
    "conversation_flow": "presentation"
}
```

### 3.2 Caching

```redis
# User session cache (TTL: 30 minutes)
eva:user:{user_id}:session = {
    "id": "uuid",
    "email": "user@example.com",
    "role": "user",
    "subscription_tier": "professional",
    "permissions": ["eva:read", "eva:write"]
}

# API rate limiting (TTL: 1 minute)
eva:ratelimit:{api_key}:{endpoint} = current_count

# Knowledge base cache (TTL: 1 hour)
eva:kb:{session_id}:search:{query_hash} = {
    "results": [...],
    "confidence_scores": [...],
    "cached_at": "2025-01-18T20:30:00Z"
}

# Avatar configuration cache (TTL: 6 hours)
eva:avatar:{avatar_id}:config = {
    "appearance": {...},
    "personality": {...},
    "voice_settings": {...}
}
```

### 3.3 Real-time Communication

```redis
# WebSocket connection tracking
eva:ws:sessions = Set of active session IDs
eva:ws:session:{session_id}:connections = Set of connection IDs

# Message queues for real-time updates
eva:queue:session:{session_id}:updates = List of pending updates
eva:queue:alerts = List of system alerts
eva:queue:operator:{operator_id} = List of operator notifications
```

## 4. Vector Database Schema (Pinecone/Weaviate)

### 4.1 Knowledge Embeddings

```python
# Document embeddings structure
{
    "id": "doc_12345_chunk_001",
    "vector": [0.1, 0.2, ...], # 1536-dimensional embedding
    "metadata": {
        "document_id": "doc_12345",
        "document_type": "faq",
        "session_id": "sess_abc123",
        "user_id": "user_789",
        "chunk_index": 1,
        "content": "Text content of the chunk",
        "title": "Document title",
        "priority": 5,
        "tags": ["product", "features"],
        "created_at": "2025-01-18T20:30:00Z",
        "confidence_score": 0.92
    }
}

# Conversation embeddings for context
{
    "id": "conv_67890_turn_003",
    "vector": [0.3, 0.1, ...],
    "metadata": {
        "conversation_id": "conv_67890",
        "session_id": "sess_abc123",
        "turn_number": 3,
        "speaker": "participant",
        "content": "Question or response text",
        "intent": "product_inquiry",
        "entities": {"product": "eva-live", "feature": "voice"},
        "timestamp": "2025-01-18T20:30:00Z"
    }
}
```

### 4.2 Search Indexes

```python
# Semantic search configuration
index_config = {
    "dimension": 1536,  # OpenAI text-embedding-ada-002
    "metric": "cosine",
    "pod_type": "p1.x1",
    "replicas": 2,
    "shards": 1
}

# Namespace organization
namespaces = {
    "knowledge_base": "General knowledge documents",
    "presentations": "Presentation content",
    "conversations": "Historical conversations",
    "faqs": "Frequently asked questions"
}
```

## 5. Database Operations

### 5.1 Session Lifecycle

```sql
-- Create new session
INSERT INTO sessions (user_id, avatar_id, name, config) VALUES (...);
INSERT INTO session_state (session_id) VALUES (...);

-- Start session
UPDATE sessions SET status = 'active', started_at = NOW() WHERE id = ?;
UPDATE session_state SET operator_connected = true WHERE session_id = ?;

-- End session
UPDATE sessions 
SET status = 'stopped', ended_at = NOW(), 
    total_duration_seconds = EXTRACT(EPOCH FROM (NOW() - started_at))
WHERE id = ?;
```

### 5.2 Real-time Metrics

```sql
-- Insert performance metric
INSERT INTO performance_metrics (session_id, metric_type, metric_value, component)
VALUES (?, 'latency_ms', ?, 'speech_recognition');

-- Get recent metrics
SELECT metric_type, AVG(metric_value) as avg_value, 
       MIN(metric_value) as min_value, MAX(metric_value) as max_value
FROM performance_metrics 
WHERE session_id = ? AND timestamp > NOW() - INTERVAL '1 hour'
GROUP BY metric_type;
```

### 5.3 Knowledge Search

```python
# Vector similarity search
def search_knowledge_base(query_embedding, session_id, top_k=5):
    results = pinecone_index.query(
        vector=query_embedding,
        filter={"session_id": session_id},
        top_k=top_k,
        include_metadata=True
    )
    return results

# Hybrid search (vector + keyword)
def hybrid_search(query, session_id, alpha=0.7):
    vector_results = search_knowledge_base(embed_query(query), session_id)
    keyword_results = postgres_fts_search(query, session_id)
    return combine_results(vector_results, keyword_results, alpha)
```

## 6. Data Retention and Archival

### 6.1 Retention Policies

```sql
-- Archive old sessions (after 1 year)
CREATE TABLE sessions_archive (LIKE sessions INCLUDING ALL);

-- Move old performance metrics to time-series database
-- Delete metrics older than 90 days from main database

-- Clean up temporary files and cache
DELETE FROM redis WHERE key LIKE 'eva:temp:*' AND ttl < 0;
```

### 6.2 Backup Strategy

```bash
# Daily PostgreSQL backup
pg_dump eva_live_db > eva_live_backup_$(date +%Y%m%d).sql

# Vector database backup
pinecone backup create --index eva-live-knowledge

# Redis persistence
redis-cli BGSAVE
```

---

This comprehensive database schema provides the foundation for all data storage and retrieval operations in the Eva Live system, ensuring data integrity, performance, and scalability.
