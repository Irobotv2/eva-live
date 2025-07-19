# Eva Live API Specifications

## 1. Overview

Eva Live exposes a comprehensive set of APIs for system integration, monitoring, and control. The API architecture follows RESTful principles with WebSocket support for real-time communication and GraphQL for complex queries.

## 2. API Architecture

### 2.1 Base Configuration

```yaml
Base URL: https://api.eva-live.com/v1
Authentication: Bearer Token (JWT)
Content-Type: application/json
Rate Limiting: 1000 requests/minute per API key
WebSocket: wss://ws.eva-live.com/v1
```

### 2.2 Authentication

All API requests require authentication via JWT tokens obtained through OAuth 2.0 flow.

```http
POST /auth/token
Content-Type: application/json

{
  "grant_type": "client_credentials",
  "client_id": "your_client_id",
  "client_secret": "your_client_secret",
  "scope": "eva:read eva:write eva:admin"
}
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "Bearer",
  "expires_in": 3600,
  "scope": "eva:read eva:write eva:admin"
}
```

## 3. Core APIs

### 3.1 Session Management API

#### Create Eva Session

```http
POST /sessions
Authorization: Bearer {token}
Content-Type: application/json

{
  "presenter_config": {
    "avatar_id": "eva-default-v2",
    "voice_id": "elevenlabs-female-professional",
    "language": "en-US",
    "personality_traits": {
      "enthusiasm": 0.8,
      "formality": 0.7,
      "empathy": 0.9
    }
  },
  "content_config": {
    "presentation_url": "https://docs.google.com/presentation/d/...",
    "knowledge_base_files": ["faq.json", "product_info.pdf"],
    "fallback_responses": {
      "technical_issues": "I'm experiencing a technical difficulty. Let me get back to you.",
      "unknown_question": "That's an excellent question. Let me research that for you."
    }
  },
  "integration_config": {
    "platform": "zoom",
    "meeting_id": "123-456-789",
    "virtual_camera_enabled": true,
    "screen_sharing_enabled": true
  },
  "monitoring_config": {
    "human_operator": {
      "operator_id": "op_12345",
      "intervention_level": "moderate",
      "notifications_enabled": true
    },
    "quality_settings": {
      "video_quality": "1080p",
      "audio_quality": "high",
      "latency_priority": "low"
    }
  }
}
```

**Response:**
```json
{
  "session_id": "sess_abc123",
  "status": "created",
  "endpoints": {
    "control_ws": "wss://ws.eva-live.com/v1/sessions/sess_abc123/control",
    "monitoring_ws": "wss://ws.eva-live.com/v1/sessions/sess_abc123/monitor",
    "virtual_camera_url": "eva-camera://sess_abc123",
    "virtual_microphone_url": "eva-mic://sess_abc123"
  },
  "created_at": "2025-01-18T20:30:00Z",
  "expires_at": "2025-01-18T23:30:00Z"
}
```

#### Start Eva Session

```http
POST /sessions/{session_id}/start
Authorization: Bearer {token}

{
  "initial_message": "Hello everyone! I'm Eva, and I'm excited to present to you today.",
  "auto_start_presentation": true,
  "operator_ready": true
}
```

#### Stop Eva Session

```http
POST /sessions/{session_id}/stop
Authorization: Bearer {token}

{
  "graceful_shutdown": true,
  "closing_message": "Thank you for your attention. Have a great day!"
}
```

#### Get Session Status

```http
GET /sessions/{session_id}
Authorization: Bearer {token}
```

**Response:**
```json
{
  "session_id": "sess_abc123",
  "status": "active",
  "current_state": {
    "presentation_slide": 5,
    "is_speaking": false,
    "awaiting_questions": true,
    "audience_count": 24
  },
  "performance_metrics": {
    "latency_ms": 145,
    "audio_quality_score": 0.94,
    "video_quality_score": 0.91,
    "ai_confidence_score": 0.87
  },
  "uptime_seconds": 1847
}
```

### 3.2 Content Management API

#### Upload Presentation Content

```http
POST /sessions/{session_id}/content
Authorization: Bearer {token}
Content-Type: multipart/form-data

file: presentation.pptx
metadata: {
  "title": "Q4 Product Roadmap",
  "auto_extract_text": true,
  "generate_faqs": true,
  "slide_timing": "auto"
}
```

#### Update Knowledge Base

```http
PUT /sessions/{session_id}/knowledge-base
Authorization: Bearer {token}
Content-Type: application/json

{
  "documents": [
    {
      "id": "doc_001",
      "title": "Product FAQ",
      "content": "...",
      "type": "faq",
      "priority": "high"
    }
  ],
  "embeddings_update": true,
  "immediate_sync": true
}
```

#### Get Content Status

```http
GET /sessions/{session_id}/content/status
Authorization: Bearer {token}
```

**Response:**
```json
{
  "presentation": {
    "total_slides": 15,
    "processed_slides": 15,
    "extraction_complete": true
  },
  "knowledge_base": {
    "document_count": 23,
    "embedding_count": 15647,
    "last_updated": "2025-01-18T20:25:00Z"
  },
  "faqs": {
    "total_questions": 87,
    "confidence_scores": {
      "high": 65,
      "medium": 18,
      "low": 4
    }
  }
}
```

### 3.3 Real-Time Interaction API

#### Send Message to Eva

```http
POST /sessions/{session_id}/interact
Authorization: Bearer {token}
Content-Type: application/json

{
  "message": {
    "type": "question",
    "content": "What are the key features of the new product release?",
    "speaker_id": "participant_456",
    "timestamp": "2025-01-18T20:30:15Z"
  },
  "context": {
    "current_slide": 7,
    "previous_questions": ["What's the timeline?", "Who's the target market?"]
  }
}
```

**Response:**
```json
{
  "response_id": "resp_789",
  "eva_response": {
    "text": "Great question! The key features include advanced AI integration, real-time collaboration tools, and enhanced security protocols. Let me show you the details on the next slide.",
    "actions": [
      {
        "type": "navigate_slide",
        "slide_number": 8
      },
      {
        "type": "highlight_section",
        "section": "key_features"
      }
    ],
    "confidence_score": 0.92,
    "response_time_ms": 287
  }
}
```

#### Get Conversation History

```http
GET /sessions/{session_id}/conversation
Authorization: Bearer {token}
```

### 3.4 Operator Control API

#### Send Operator Command

```http
POST /sessions/{session_id}/operator/command
Authorization: Bearer {token}
Content-Type: application/json

{
  "command": {
    "type": "suggest_response",
    "priority": "high",
    "content": "Actually, let me clarify that the release date has been moved to Q2.",
    "override_current": false
  },
  "operator_id": "op_12345"
}
```

#### Operator Takeover

```http
POST /sessions/{session_id}/operator/takeover
Authorization: Bearer {token}
Content-Type: application/json

{
  "takeover_message": "This is the human presenter taking over. Let me address that question directly.",
  "duration_minutes": 5,
  "auto_return": true
}
```

### 3.5 Monitoring and Analytics API

#### Get Real-Time Metrics

```http
GET /sessions/{session_id}/metrics
Authorization: Bearer {token}
```

**Response:**
```json
{
  "performance": {
    "latency": {
      "speech_recognition": 89,
      "ai_processing": 156,
      "voice_synthesis": 134,
      "total_pipeline": 379
    },
    "quality_scores": {
      "audio_clarity": 0.94,
      "video_quality": 0.89,
      "lip_sync_accuracy": 0.96,
      "expression_naturalness": 0.87
    },
    "system_resources": {
      "cpu_usage": 0.67,
      "memory_usage": 0.72,
      "gpu_usage": 0.84,
      "network_bandwidth": 15.6
    }
  },
  "engagement": {
    "questions_asked": 12,
    "average_response_time": 2.4,
    "audience_attention_score": 0.78,
    "interaction_frequency": 0.8
  }
}
```

#### Get Session Analytics

```http
GET /sessions/{session_id}/analytics
Authorization: Bearer {token}
```

## 4. WebSocket APIs

### 4.1 Real-Time Control WebSocket

**Connection:**
```javascript
const ws = new WebSocket('wss://ws.eva-live.com/v1/sessions/sess_abc123/control');
```

**Message Types:**

#### Eva Status Updates
```json
{
  "type": "status_update",
  "timestamp": "2025-01-18T20:30:00Z",
  "data": {
    "status": "speaking",
    "current_slide": 5,
    "estimated_completion": "2025-01-18T20:32:15Z"
  }
}
```

#### Question Received
```json
{
  "type": "question_received",
  "data": {
    "question_id": "q_456",
    "content": "What's the pricing model?",
    "speaker": "participant_789",
    "confidence": 0.95,
    "suggested_responses": [
      {
        "response": "Our pricing starts at $99/month for the basic plan...",
        "confidence": 0.91,
        "source": "pricing_doc.pdf"
      }
    ]
  }
}
```

#### System Alert
```json
{
  "type": "system_alert",
  "severity": "warning",
  "data": {
    "alert_type": "high_latency",
    "message": "Response latency above threshold (500ms)",
    "current_value": 567,
    "threshold": 500,
    "suggested_action": "reduce_quality_settings"
  }
}
```

### 4.2 Monitoring WebSocket

**Connection:**
```javascript
const ws = new WebSocket('wss://ws.eva-live.com/v1/sessions/sess_abc123/monitor');
```

**Real-time Metrics Stream:**
```json
{
  "type": "metrics_update",
  "timestamp": "2025-01-18T20:30:05Z",
  "data": {
    "latency_ms": 234,
    "cpu_usage": 0.68,
    "memory_usage": 0.71,
    "active_connections": 1,
    "questions_per_minute": 2.4,
    "ai_confidence_avg": 0.89
  }
}
```

## 5. Avatar Customization API

### 5.1 Avatar Configuration

```http
PUT /avatars/{avatar_id}/config
Authorization: Bearer {token}
Content-Type: application/json

{
  "appearance": {
    "skin_tone": "medium",
    "hair_color": "brown",
    "hair_style": "professional_short",
    "eye_color": "hazel",
    "clothing": {
      "style": "business_casual",
      "color_scheme": "blue_gray",
      "accessories": ["glasses"]
    }
  },
  "personality": {
    "enthusiasm_level": 0.8,
    "formality_level": 0.7,
    "gesture_frequency": 0.6,
    "expression_range": "full"
  },
  "voice_settings": {
    "pitch": 0.0,
    "speed": 1.0,
    "emotion_responsiveness": 0.8
  }
}
```

### 5.2 Create Custom Avatar

```http
POST /avatars/custom
Authorization: Bearer {token}
Content-Type: multipart/form-data

reference_images: [image1.jpg, image2.jpg, image3.jpg]
voice_sample: voice_sample.wav
config: {
  "name": "Custom Executive Avatar",
  "description": "Professional presenter for executive meetings",
  "training_duration": "standard"
}
```

## 6. Integration APIs

### 6.1 Platform Integration

#### Zoom Integration
```http
POST /integrations/zoom/connect
Authorization: Bearer {token}
Content-Type: application/json

{
  "zoom_api_key": "your_zoom_api_key",
  "zoom_api_secret": "your_zoom_api_secret",
  "meeting_settings": {
    "auto_join": true,
    "camera_enabled": true,
    "microphone_enabled": true,
    "screen_share_enabled": true
  }
}
```

#### Microsoft Teams Integration
```http
POST /integrations/teams/connect
Authorization: Bearer {token}
Content-Type: application/json

{
  "tenant_id": "your_tenant_id",
  "client_id": "your_client_id",
  "client_secret": "your_client_secret",
  "permissions": ["Meetings.ReadWrite", "OnlineMeetings.ReadWrite"]
}
```

### 6.2 Webhook Configuration

```http
POST /webhooks
Authorization: Bearer {token}
Content-Type: application/json

{
  "url": "https://your-app.com/webhooks/eva-events",
  "events": [
    "session.started",
    "session.ended",
    "question.received",
    "system.alert",
    "operator.intervention"
  ],
  "secret": "your_webhook_secret"
}
```

## 7. Error Handling

### 7.1 Error Response Format

```json
{
  "error": {
    "code": "INVALID_SESSION",
    "message": "Session not found or expired",
    "details": {
      "session_id": "sess_abc123",
      "error_timestamp": "2025-01-18T20:30:00Z"
    },
    "suggestion": "Create a new session or check session ID"
  }
}
```

### 7.2 Common Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| AUTHENTICATION_FAILED | 401 | Invalid or expired token |
| INSUFFICIENT_PERMISSIONS | 403 | Missing required scope |
| SESSION_NOT_FOUND | 404 | Session doesn't exist |
| RATE_LIMIT_EXCEEDED | 429 | Too many requests |
| CONTENT_TOO_LARGE | 413 | Upload size exceeds limit |
| AI_SERVICE_UNAVAILABLE | 503 | AI service temporarily down |
| INVALID_AUDIO_FORMAT | 400 | Unsupported audio format |
| CONCURRENT_SESSION_LIMIT | 409 | Max sessions reached |

## 8. Rate Limits and Quotas

### 8.1 API Rate Limits

| Endpoint Category | Limit | Window |
|------------------|-------|---------|
| Authentication | 10 requests | 1 minute |
| Session Management | 100 requests | 1 minute |
| Content Upload | 10 requests | 1 minute |
| Real-time Interaction | 1000 requests | 1 minute |
| Monitoring | 500 requests | 1 minute |

### 8.2 Resource Quotas

| Resource | Limit | Billing Tier |
|----------|-------|--------------|
| Concurrent Sessions | 5 | Basic |
| Concurrent Sessions | 25 | Professional |
| Concurrent Sessions | 100 | Enterprise |
| Monthly API Calls | 10,000 | Basic |
| Monthly API Calls | 100,000 | Professional |
| Monthly API Calls | 1,000,000 | Enterprise |

## 9. SDK and Libraries

### 9.1 Official SDKs

- **Python**: `pip install eva-live-sdk`
- **JavaScript/Node.js**: `npm install eva-live-sdk`
- **C#/.NET**: `dotnet add package EvaLive.SDK`
- **Java**: Maven/Gradle dependency available

### 9.2 Community Libraries

- **Go**: `github.com/community/eva-live-go`
- **Ruby**: `gem install eva-live-ruby`
- **PHP**: `composer require eva-live/php-sdk`

---

This comprehensive API specification provides all the necessary interfaces for integrating with and controlling the Eva Live system, ensuring developers can build robust applications and integrations.
