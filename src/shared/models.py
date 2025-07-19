"""
Eva Live Core Data Models

This module defines all the core data models and schemas used throughout
the Eva Live system using Pydantic for validation and serialization.
"""

from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, validator, ConfigDict


# Enums for type safety
class SessionStatus(str, Enum):
    CREATED = "created"
    STARTING = "starting"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    FAILED = "failed"


class MessageType(str, Enum):
    QUESTION = "question"
    RESPONSE = "response"
    SYSTEM = "system"
    OPERATOR = "operator"


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class InterventionType(str, Enum):
    TAKEOVER = "takeover"
    SUGGESTION = "suggestion"
    CORRECTION = "correction"
    EMERGENCY_STOP = "emergency_stop"


class AvatarType(str, Enum):
    STANDARD = "standard"
    CUSTOM = "custom"
    PREMIUM = "premium"


class DocumentType(str, Enum):
    FAQ = "faq"
    GENERAL = "general"
    TECHNICAL = "technical"
    MARKETING = "marketing"


class ProcessingStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


# Core Models
class User(BaseModel):
    """User model"""
    id: UUID = Field(default_factory=uuid4)
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    company: Optional[str] = None
    role: str = "user"
    subscription_tier: str = "basic"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_login: Optional[datetime] = None
    is_active: bool = True
    email_verified: bool = False
    api_quota_monthly: int = 10000
    api_calls_used: int = 0
    concurrent_sessions_limit: int = 5

    model_config = ConfigDict(from_attributes=True)


class Avatar(BaseModel):
    """Avatar model"""
    id: UUID = Field(default_factory=uuid4)
    user_id: Optional[UUID] = None
    name: str
    description: Optional[str] = None
    avatar_type: AvatarType = AvatarType.STANDARD
    model_version: str = "v1"
    appearance_config: Dict[str, Any] = Field(default_factory=dict)
    personality_config: Dict[str, Any] = Field(default_factory=dict)
    voice_config: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_public: bool = False
    is_active: bool = True
    training_status: str = "ready"
    quality_score: float = 0.0

    model_config = ConfigDict(from_attributes=True)


class Session(BaseModel):
    """Eva presentation session model"""
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    avatar_id: UUID
    name: Optional[str] = None
    description: Optional[str] = None
    status: SessionStatus = SessionStatus.CREATED
    config: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    total_duration_seconds: int = 0
    participant_count: int = 0
    questions_received: int = 0
    operator_interventions: int = 0
    quality_metrics: Dict[str, float] = Field(default_factory=dict)
    error_count: int = 0
    last_error: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)

    @property
    def is_active(self) -> bool:
        """Check if session is currently active"""
        return self.status in [SessionStatus.STARTING, SessionStatus.ACTIVE, SessionStatus.PAUSED]

    @property
    def duration(self) -> Optional[timedelta]:
        """Get session duration if ended"""
        if self.started_at and self.ended_at:
            return self.ended_at - self.started_at
        return None


class SessionState(BaseModel):
    """Real-time session state model"""
    session_id: UUID
    current_slide: int = 0
    presentation_mode: str = "auto"
    is_speaking: bool = False
    awaiting_questions: bool = False
    operator_connected: bool = False
    last_interaction: datetime = Field(default_factory=datetime.utcnow)
    current_topic: Optional[str] = None
    conversation_context: Dict[str, Any] = Field(default_factory=dict)
    system_metrics: Dict[str, float] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


class Conversation(BaseModel):
    """Conversation message model"""
    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    participant_id: Optional[str] = None
    message_type: MessageType
    content: str
    speaker: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    confidence_score: Optional[float] = None
    response_time_ms: Optional[int] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


class QuestionResponse(BaseModel):
    """Question analysis and response model"""
    id: UUID = Field(default_factory=uuid4)
    conversation_id: UUID
    question_text: str
    response_text: str
    intent_classification: Optional[str] = None
    entities_extracted: Dict[str, Any] = Field(default_factory=dict)
    knowledge_sources: List[str] = Field(default_factory=list)
    confidence_score: float
    response_time_ms: int
    operator_reviewed: bool = False
    quality_rating: Optional[int] = Field(None, ge=1, le=5)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(from_attributes=True)


class KnowledgeDocument(BaseModel):
    """Knowledge base document model"""
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    session_id: Optional[UUID] = None
    title: str
    content: str
    document_type: DocumentType = DocumentType.GENERAL
    priority: int = Field(1, ge=1, le=10)
    tags: List[str] = Field(default_factory=list)
    embedding_id: Optional[str] = None
    confidence_score: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    is_active: bool = True

    model_config = ConfigDict(from_attributes=True)


class Presentation(BaseModel):
    """Presentation content model"""
    id: UUID = Field(default_factory=uuid4)
    user_id: UUID
    session_id: Optional[UUID] = None
    title: str
    description: Optional[str] = None
    file_path: Optional[str] = None
    file_type: Optional[str] = None
    file_size_bytes: Optional[int] = None
    slide_count: int = 0
    processing_status: ProcessingStatus = ProcessingStatus.PENDING
    extraction_config: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class PresentationSlide(BaseModel):
    """Individual presentation slide model"""
    id: UUID = Field(default_factory=uuid4)
    presentation_id: UUID
    slide_number: int
    title: Optional[str] = None
    content: Optional[str] = None
    speaker_notes: Optional[str] = None
    slide_type: str = "content"
    duration_seconds: Optional[int] = None
    image_path: Optional[str] = None
    animations: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(from_attributes=True)


class PerformanceMetric(BaseModel):
    """System performance metric model"""
    id: UUID = Field(default_factory=uuid4)
    session_id: Optional[UUID] = None
    metric_type: str
    metric_value: float
    unit: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    component: Optional[str] = None
    additional_data: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


class SystemAlert(BaseModel):
    """System alert model"""
    id: UUID = Field(default_factory=uuid4)
    session_id: Optional[UUID] = None
    alert_type: str
    severity: AlertSeverity = AlertSeverity.INFO
    title: str
    message: str
    component: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    acknowledged: bool = False
    acknowledged_by: Optional[UUID] = None
    acknowledged_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class OperatorIntervention(BaseModel):
    """Operator intervention model"""
    id: UUID = Field(default_factory=uuid4)
    session_id: UUID
    operator_id: UUID
    intervention_type: InterventionType
    trigger_reason: Optional[str] = None
    action_taken: str
    duration_seconds: Optional[int] = None
    effectiveness_rating: Optional[int] = Field(None, ge=1, le=5)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


# API Request/Response Models
class SessionCreateRequest(BaseModel):
    """Request model for creating a new session"""
    presenter_config: Dict[str, Any]
    content_config: Dict[str, Any]
    integration_config: Dict[str, Any]
    monitoring_config: Dict[str, Any]

    model_config = ConfigDict(extra='forbid')


class SessionCreateResponse(BaseModel):
    """Response model for session creation"""
    session_id: UUID
    status: SessionStatus
    endpoints: Dict[str, str]
    created_at: datetime
    expires_at: datetime

    model_config = ConfigDict(from_attributes=True)


class SessionStatusResponse(BaseModel):
    """Response model for session status"""
    session_id: UUID
    status: SessionStatus
    current_state: Dict[str, Any]
    performance_metrics: Dict[str, float]
    uptime_seconds: int

    model_config = ConfigDict(from_attributes=True)


class InteractionRequest(BaseModel):
    """Request model for real-time interaction"""
    message: Dict[str, Any]
    context: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra='forbid')


class InteractionResponse(BaseModel):
    """Response model for interaction"""
    response_id: UUID
    eva_response: Dict[str, Any]

    model_config = ConfigDict(from_attributes=True)


class MetricsResponse(BaseModel):
    """Response model for real-time metrics"""
    performance: Dict[str, Any]
    engagement: Dict[str, Any]

    model_config = ConfigDict(from_attributes=True)


class OperatorCommand(BaseModel):
    """Operator command model"""
    command: Dict[str, Any]
    operator_id: UUID

    model_config = ConfigDict(extra='forbid')


class WebSocketMessage(BaseModel):
    """WebSocket message model"""
    type: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Dict[str, Any]

    model_config = ConfigDict(from_attributes=True)


class VoiceSettings(BaseModel):
    """Voice configuration model"""
    provider: str = "elevenlabs"
    voice_id: str
    model_id: str = "eleven_monolingual_v1"
    stability: float = Field(0.75, ge=0.0, le=1.0)
    similarity_boost: float = Field(0.75, ge=0.0, le=1.0)
    pitch: float = Field(0.0, ge=-1.0, le=1.0)
    speed: float = Field(1.0, ge=0.25, le=4.0)
    emotion_responsiveness: float = Field(0.8, ge=0.0, le=1.0)

    model_config = ConfigDict(from_attributes=True)


class AvatarAppearance(BaseModel):
    """Avatar appearance configuration"""
    skin_tone: str = "medium"
    hair_color: str = "brown"
    hair_style: str = "professional_short"
    eye_color: str = "hazel"
    clothing: Dict[str, str] = Field(default_factory=lambda: {
        "style": "business_casual",
        "color_scheme": "blue_gray"
    })
    accessories: List[str] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)


class AvatarPersonality(BaseModel):
    """Avatar personality configuration"""
    enthusiasm_level: float = Field(0.8, ge=0.0, le=1.0)
    formality_level: float = Field(0.7, ge=0.0, le=1.0)
    gesture_frequency: float = Field(0.6, ge=0.0, le=1.0)
    expression_range: str = "full"
    empathy_level: float = Field(0.9, ge=0.0, le=1.0)

    model_config = ConfigDict(from_attributes=True)


class RenderingConfig(BaseModel):
    """Avatar rendering configuration"""
    quality: str = "high"
    frame_rate: int = Field(30, ge=15, le=60)
    resolution: str = "1080p"
    gpu_acceleration: bool = True
    real_time_optimization: bool = True

    model_config = ConfigDict(from_attributes=True)


# Utility functions
def create_session_endpoints(session_id: UUID, base_url: str) -> Dict[str, str]:
    """Create session endpoint URLs"""
    return {
        "control_ws": f"{base_url}/sessions/{session_id}/control",
        "monitoring_ws": f"{base_url}/sessions/{session_id}/monitor",
        "virtual_camera_url": f"eva-camera://{session_id}",
        "virtual_microphone_url": f"eva-mic://{session_id}"
    }


def calculate_session_uptime(session: Session) -> int:
    """Calculate session uptime in seconds"""
    if not session.started_at:
        return 0
    
    end_time = session.ended_at or datetime.utcnow()
    return int((end_time - session.started_at).total_seconds())


def validate_confidence_score(score: float) -> float:
    """Validate and normalize confidence score"""
    return max(0.0, min(1.0, score))


def generate_correlation_id() -> str:
    """Generate a correlation ID for request tracing"""
    return str(uuid4())[:8]
