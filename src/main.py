"""
Eva Live Main Application

This is the main entry point for the Eva Live system. It initializes all components,
starts the FastAPI server, and manages the overall application lifecycle.
"""

import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from .shared.config import init_config, get_config
from .shared.models import (
    SessionCreateRequest, SessionCreateResponse, SessionStatusResponse,
    InteractionRequest, InteractionResponse, MetricsResponse,
    OperatorCommand, create_session_endpoints, calculate_session_uptime
)

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global application state
app_state: Dict[str, Any] = {
    "sessions": {},
    "active_sessions": set(),
    "metrics_collector": None,
    "shutdown_event": asyncio.Event()
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting Eva Live application...")
    
    # Initialize configuration
    config = init_config()
    logger.info(f"Loaded configuration from: {config.config_path}")
    
    # Initialize database connections
    # await init_database()
    
    # Initialize AI services
    # await init_ai_services()
    
    # Start background tasks
    # await start_background_tasks()
    
    logger.info("Eva Live application started successfully")
    
    yield
    
    # Cleanup on shutdown
    logger.info("Shutting down Eva Live application...")
    app_state["shutdown_event"].set()
    
    # Stop all active sessions
    for session_id in list(app_state["active_sessions"]):
        await stop_session(session_id)
    
    # Close database connections
    # await close_database()
    
    logger.info("Eva Live application shut down complete")

# Create FastAPI application
app = FastAPI(
    title="Eva Live API",
    description="Hyper-Realistic AI-Driven Virtual Presenter System",
    version="1.0.0",
    lifespan=lifespan
)

# Add middleware
config = get_config()

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get("security.cors.allowed_origins", ["*"]),
    allow_credentials=True,
    allow_methods=config.get("security.cors.allowed_methods", ["*"]),
    allow_headers=config.get("security.cors.allowed_headers", ["*"]),
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Session management endpoints
@app.post("/sessions", response_model=SessionCreateResponse)
async def create_session(
    request: SessionCreateRequest,
    background_tasks: BackgroundTasks
) -> SessionCreateResponse:
    """Create a new Eva Live session"""
    try:
        config = get_config()
        
        # Create session ID
        from uuid import uuid4
        from datetime import datetime, timedelta
        
        session_id = uuid4()
        
        # Create session endpoints
        endpoints = create_session_endpoints(session_id, config.websocket_url)
        
        # Initialize session state
        session_data = {
            "id": session_id,
            "config": request.dict(),
            "status": "created",
            "created_at": datetime.utcnow(),
            "expires_at": datetime.utcnow() + timedelta(hours=3)
        }
        
        app_state["sessions"][str(session_id)] = session_data
        
        # Start session initialization in background
        background_tasks.add_task(initialize_session, session_id, request)
        
        logger.info(f"Created session {session_id}")
        
        return SessionCreateResponse(
            session_id=session_id,
            status="created",
            endpoints=endpoints,
            created_at=session_data["created_at"],
            expires_at=session_data["expires_at"]
        )
        
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sessions/{session_id}/start")
async def start_session(session_id: str, background_tasks: BackgroundTasks):
    """Start an Eva Live session"""
    try:
        if session_id not in app_state["sessions"]:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = app_state["sessions"][session_id]
        session["status"] = "starting"
        
        # Start session in background
        background_tasks.add_task(run_session, session_id)
        
        logger.info(f"Starting session {session_id}")
        
        return {"status": "starting", "message": "Session is starting"}
        
    except Exception as e:
        logger.error(f"Error starting session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sessions/{session_id}/stop")
async def stop_session_endpoint(session_id: str):
    """Stop an Eva Live session"""
    try:
        if session_id not in app_state["sessions"]:
            raise HTTPException(status_code=404, detail="Session not found")
        
        await stop_session(session_id)
        
        logger.info(f"Stopped session {session_id}")
        
        return {"status": "stopped", "message": "Session stopped successfully"}
        
    except Exception as e:
        logger.error(f"Error stopping session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}", response_model=SessionStatusResponse)
async def get_session_status(session_id: str) -> SessionStatusResponse:
    """Get session status and metrics"""
    try:
        if session_id not in app_state["sessions"]:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = app_state["sessions"][session_id]
        
        # Calculate uptime
        from datetime import datetime
        uptime = 0
        if session.get("started_at"):
            uptime = int((datetime.utcnow() - session["started_at"]).total_seconds())
        
        return SessionStatusResponse(
            session_id=session_id,
            status=session["status"],
            current_state={
                "presentation_slide": session.get("current_slide", 0),
                "is_speaking": session.get("is_speaking", False),
                "awaiting_questions": session.get("awaiting_questions", True),
                "audience_count": session.get("audience_count", 0)
            },
            performance_metrics={
                "latency_ms": session.get("latency_ms", 0),
                "audio_quality_score": session.get("audio_quality", 0.0),
                "video_quality_score": session.get("video_quality", 0.0),
                "ai_confidence_score": session.get("ai_confidence", 0.0)
            },
            uptime_seconds=uptime
        )
        
    except Exception as e:
        logger.error(f"Error getting session status {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sessions/{session_id}/interact", response_model=InteractionResponse)
async def interact_with_eva(
    session_id: str, 
    request: InteractionRequest
) -> InteractionResponse:
    """Send a message to Eva and get a response"""
    try:
        if session_id not in app_state["sessions"]:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = app_state["sessions"][session_id]
        
        if session["status"] != "active":
            raise HTTPException(status_code=400, detail="Session is not active")
        
        # Process the interaction (this would integrate with the AI processing layer)
        response_id = str(uuid4())
        
        # Mock response for now
        eva_response = {
            "text": f"Thank you for your question: '{request.message.get('content', '')}'",
            "actions": [],
            "confidence_score": 0.85,
            "response_time_ms": 250
        }
        
        logger.info(f"Processed interaction for session {session_id}")
        
        return InteractionResponse(
            response_id=response_id,
            eva_response=eva_response
        )
        
    except Exception as e:
        logger.error(f"Error processing interaction for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/metrics", response_model=MetricsResponse)
async def get_session_metrics(session_id: str) -> MetricsResponse:
    """Get real-time session metrics"""
    try:
        if session_id not in app_state["sessions"]:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = app_state["sessions"][session_id]
        
        return MetricsResponse(
            performance={
                "latency": {
                    "speech_recognition": session.get("speech_latency", 0),
                    "ai_processing": session.get("ai_latency", 0),
                    "voice_synthesis": session.get("voice_latency", 0),
                    "total_pipeline": session.get("total_latency", 0)
                },
                "quality_scores": {
                    "audio_clarity": session.get("audio_quality", 0.0),
                    "video_quality": session.get("video_quality", 0.0),
                    "lip_sync_accuracy": session.get("lip_sync_quality", 0.0),
                    "expression_naturalness": session.get("expression_quality", 0.0)
                },
                "system_resources": {
                    "cpu_usage": session.get("cpu_usage", 0.0),
                    "memory_usage": session.get("memory_usage", 0.0),
                    "gpu_usage": session.get("gpu_usage", 0.0),
                    "network_bandwidth": session.get("network_usage", 0.0)
                }
            },
            engagement={
                "questions_asked": session.get("questions_count", 0),
                "average_response_time": session.get("avg_response_time", 0.0),
                "audience_attention_score": session.get("attention_score", 0.0),
                "interaction_frequency": session.get("interaction_rate", 0.0)
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting metrics for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Eva Live",
        "version": "1.0.0",
        "active_sessions": len(app_state["active_sessions"])
    }

# Metrics endpoint
@app.get("/metrics")
async def get_system_metrics():
    """Get system-wide metrics"""
    return {
        "total_sessions": len(app_state["sessions"]),
        "active_sessions": len(app_state["active_sessions"]),
        "system_status": "operational"
    }

# Background task functions
async def initialize_session(session_id: str, request: SessionCreateRequest):
    """Initialize session components in background"""
    try:
        session = app_state["sessions"][str(session_id)]
        
        # Initialize complete AI coordinator
        from .core.ai_coordinator import AICoordinator
        from .output.voice_synthesis import VoiceSynthesizer
        from .output.audio_processor import AudioProcessor
        
        # Create AI coordinator (handles all AI components)
        ai_coordinator = AICoordinator(str(session_id))
        await ai_coordinator.initialize(request.presenter_config.get('user_id'))
        
        # Initialize output components
        voice_synthesizer = VoiceSynthesizer(str(session_id))
        audio_processor = AudioProcessor(str(session_id))
        
        # Store components in session
        session["components"] = {
            "ai_coordinator": ai_coordinator,
            "voice_synthesizer": voice_synthesizer,
            "audio_processor": audio_processor
        }
        
        # Store individual component references for backward compatibility
        session["ai_components"] = {
            "speech_recognition": ai_coordinator.speech_recognition,
            "nlu": ai_coordinator.nlu,
            "memory_manager": ai_coordinator.memory_manager,
            "knowledge_base": ai_coordinator.knowledge_base,
            "response_generator": ai_coordinator.response_generator
        }
        
        session["status"] = "initialized"
        session["capabilities"] = {
            "speech_recognition": True,
            "natural_language_understanding": True,
            "knowledge_retrieval": True,
            "response_generation": True,
            "voice_synthesis": True,
            "audio_processing": True
        }
        
        logger.info(f"Session {session_id} initialized successfully with complete AI pipeline")
        
    except Exception as e:
        logger.error(f"Error initializing session {session_id}: {e}")
        session = app_state["sessions"][str(session_id)]
        session["status"] = "failed"
        session["error"] = str(e)

async def run_session(session_id: str):
    """Run an active Eva Live session"""
    try:
        session = app_state["sessions"][session_id]
        session["status"] = "active"
        session["started_at"] = datetime.utcnow()
        
        app_state["active_sessions"].add(session_id)
        
        logger.info(f"Session {session_id} is now active")
        
        # Main session loop would go here
        # This would coordinate between all the AI components
        while session["status"] == "active":
            await asyncio.sleep(1)
            
            # Update session metrics
            session["latency_ms"] = 250  # Mock data
            session["audio_quality"] = 0.92
            session["video_quality"] = 0.89
            session["ai_confidence"] = 0.87
            
            # Check for shutdown
            if app_state["shutdown_event"].is_set():
                break
                
    except Exception as e:
        logger.error(f"Error running session {session_id}: {e}")
        session = app_state["sessions"][session_id]
        session["status"] = "failed"
        session["error"] = str(e)
    finally:
        app_state["active_sessions"].discard(session_id)

async def stop_session(session_id: str):
    """Stop a session gracefully"""
    try:
        if session_id in app_state["sessions"]:
            session = app_state["sessions"][session_id]
            session["status"] = "stopped"
            session["ended_at"] = datetime.utcnow()
            
            app_state["active_sessions"].discard(session_id)
            
            # Cleanup session resources
            # Close AI components, cleanup memory, etc.
            
    except Exception as e:
        logger.error(f"Error stopping session {session_id}: {e}")

# Signal handlers for graceful shutdown
def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    app_state["shutdown_event"].set()

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def main():
    """Main entry point"""
    config = get_config()
    
    logger.info(f"Starting Eva Live server on {config.host}:{config.port}")
    
    uvicorn.run(
        "src.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="info" if not config.debug else "debug",
        access_log=True
    )

if __name__ == "__main__":
    main()
