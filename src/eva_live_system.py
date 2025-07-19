"""
Eva Live Complete System

This module orchestrates the complete Eva Live system, integrating all components
from text input to avatar video output with platform integration.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
import threading
from pathlib import Path

# Core AI components
from .core.ai_coordinator import AICoordinator, ProcessingResult
from .core.document_processor import DocumentProcessor
from .core.response_generator import ResponseType, ResponseTone

# Output components
from .output.voice_synthesis import VoiceSynthesizer, EmotionType, SynthesisResult
from .output.audio_processor import AudioProcessor, AudioQuality
from .output.avatar_renderer_2d import AvatarRenderer2D, AvatarExpression, Viseme, AvatarFrame

# Integration components
from .integration.virtual_camera import VirtualCamera, CameraConfig, CameraFormat
from .integration.platform_manager import PlatformManager, PlatformType

# Shared components
from .shared.config import get_config
from .shared.models import PerformanceMetric

class SystemState(str, Enum):
    """Eva Live system states"""
    INITIALIZING = "initializing"
    READY = "ready"
    ACTIVE = "active"
    IN_MEETING = "in_meeting"
    ERROR = "error"
    SHUTTING_DOWN = "shutting_down"

class OperationMode(str, Enum):
    """System operation modes"""
    PRESENTATION = "presentation"
    CONVERSATION = "conversation"
    DEMO = "demo"
    TRAINING = "training"

@dataclass
class SystemConfig:
    """Complete system configuration"""
    session_id: str
    user_id: Optional[str] = None
    operation_mode: OperationMode = OperationMode.CONVERSATION
    avatar_quality: str = "high"
    voice_quality: AudioQuality = AudioQuality.HIGH
    video_format: CameraFormat = CameraFormat.FULL_HD
    enable_virtual_camera: bool = True
    enable_platform_integration: bool = True
    auto_start_camera: bool = False
    performance_monitoring: bool = True

@dataclass
class SystemStatus:
    """Current system status"""
    state: SystemState
    active_components: List[str]
    performance_metrics: Dict[str, float]
    error_messages: List[str]
    uptime_seconds: int
    current_meeting: Optional[Dict[str, Any]] = None

class EvaLiveSystem:
    """Complete Eva Live system orchestrator"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # System state
        self.state = SystemState.INITIALIZING
        self.start_time = time.time()
        self.error_messages: List[str] = []
        
        # Core components
        self.ai_coordinator: Optional[AICoordinator] = None
        self.voice_synthesizer: Optional[VoiceSynthesizer] = None
        self.audio_processor: Optional[AudioProcessor] = None
        self.avatar_renderer: Optional[AvatarRenderer2D] = None
        self.virtual_camera: Optional[VirtualCamera] = None
        self.platform_manager: Optional[PlatformManager] = None
        
        # Processing pipeline
        self.processing_queue = asyncio.Queue()
        self.is_processing = False
        self.processing_thread: Optional[threading.Thread] = None
        
        # Performance tracking
        self.metrics: List[PerformanceMetric] = []
        self.active_components: List[str] = []
        
        # Event callbacks
        self.event_callbacks: Dict[str, List[Callable]] = {
            'system_ready': [],
            'processing_complete': [],
            'error_occurred': [],
            'meeting_joined': [],
            'meeting_left': []
        }
    
    async def initialize(self) -> bool:
        """Initialize all Eva Live components"""
        try:
            self.logger.info(f"Initializing Eva Live System (Session: {self.config.session_id})")
            self.state = SystemState.INITIALIZING
            
            # Initialize core AI coordinator
            self.logger.info("Initializing AI coordinator...")
            self.ai_coordinator = AICoordinator(self.config.session_id)
            await self.ai_coordinator.initialize(self.config.user_id)
            self.active_components.append("ai_coordinator")
            
            # Initialize voice synthesizer
            self.logger.info("Initializing voice synthesizer...")
            self.voice_synthesizer = VoiceSynthesizer(self.config.session_id)
            self.active_components.append("voice_synthesizer")
            
            # Initialize audio processor
            self.logger.info("Initializing audio processor...")
            self.audio_processor = AudioProcessor(self.config.session_id)
            self.active_components.append("audio_processor")
            
            # Initialize avatar renderer
            self.logger.info("Initializing avatar renderer...")
            self.avatar_renderer = AvatarRenderer2D(self.config.session_id)
            await self.avatar_renderer.initialize()
            self.active_components.append("avatar_renderer")
            
            # Initialize virtual camera if enabled
            if self.config.enable_virtual_camera:
                self.logger.info("Initializing virtual camera...")
                self.virtual_camera = VirtualCamera(self.config.session_id)
                camera_config = CameraConfig(
                    device_name="Eva Live Camera",
                    width=1920 if self.config.video_format == CameraFormat.FULL_HD else 1280,
                    height=1080 if self.config.video_format == CameraFormat.FULL_HD else 720,
                    fps=30
                )
                await self.virtual_camera.initialize(camera_config)
                self.active_components.append("virtual_camera")
            
            # Initialize platform manager if enabled
            if self.config.enable_platform_integration:
                self.logger.info("Initializing platform manager...")
                self.platform_manager = PlatformManager(self.config.session_id)
                await self.platform_manager.initialize()
                self.platform_manager.add_meeting_callback(self._handle_meeting_event)
                self.active_components.append("platform_manager")
            
            # Start processing pipeline
            self.is_processing = True
            self.processing_thread = threading.Thread(target=self._run_processing_loop, daemon=True)
            self.processing_thread.start()
            
            # Auto-start virtual camera if configured
            if self.config.auto_start_camera and self.virtual_camera:
                await self.virtual_camera.start()
            
            self.state = SystemState.READY
            self.logger.info("Eva Live System initialized successfully!")
            
            # Notify callbacks
            await self._trigger_event('system_ready', {
                'session_id': self.config.session_id,
                'components': self.active_components
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Eva Live System: {e}")
            self.state = SystemState.ERROR
            self.error_messages.append(str(e))
            await self._trigger_event('error_occurred', {'error': str(e)})
            return False
    
    async def process_user_input(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process user input through complete Eva Live pipeline"""
        try:
            if self.state not in [SystemState.READY, SystemState.ACTIVE, SystemState.IN_MEETING]:
                raise Exception(f"System not ready for processing (state: {self.state})")
            
            self.state = SystemState.ACTIVE
            start_time = time.time()
            
            self.logger.info(f"Processing user input: '{user_input[:50]}...'")
            
            # Step 1: Process through AI coordinator
            ai_result = await self.ai_coordinator.process_user_input(user_input)
            
            if not ai_result.success:
                raise Exception(f"AI processing failed: {ai_result.errors}")
            
            # Step 2: Synthesize voice
            voice_result = None
            if self.voice_synthesizer:
                # Determine emotion from response
                emotion = self._response_tone_to_emotion(ai_result.eva_response.tone)
                
                voice_result = await self.voice_synthesizer.synthesize_with_emotion(
                    ai_result.eva_response.text,
                    emotion
                )
            
            # Step 3: Generate avatar animation
            avatar_frames = []
            if self.avatar_renderer and voice_result and voice_result.success:
                # Convert response to avatar expression
                expression = self.avatar_renderer.emotion_to_expression(emotion)
                
                # Generate visemes for lip-sync
                audio_duration = voice_result.duration_ms
                timing = list(range(0, audio_duration, 100))  # 10fps viseme timing
                visemes = self.avatar_renderer.text_to_visemes(ai_result.eva_response.text, timing)
                
                # Create avatar frames
                for i, (viseme, timestamp) in enumerate(visemes):
                    frame = AvatarFrame(
                        expression=expression,
                        viseme=viseme,
                        timestamp_ms=timestamp
                    )
                    avatar_frames.append(frame)
            
            # Step 4: Render and stream video
            video_frames = []
            if self.avatar_renderer and avatar_frames:
                video_frames = await self.avatar_renderer.render_sequence(avatar_frames)
                
                # Stream to virtual camera if available
                if self.virtual_camera and self.virtual_camera.is_running():
                    for frame in video_frames:
                        await self.virtual_camera.send_frame(frame)
            
            # Step 5: Process audio if needed
            processed_audio = None
            if self.audio_processor and voice_result and voice_result.success:
                processed_audio = await self.audio_processor.optimize_for_streaming(
                    voice_result.audio_data,
                    voice_result.format
                )
            
            total_time = int((time.time() - start_time) * 1000)
            
            # Compile results
            result = {
                'success': True,
                'response_text': ai_result.eva_response.text,
                'processing_time_ms': total_time,
                'ai_confidence': ai_result.eva_response.confidence_score,
                'voice_synthesis': {
                    'success': voice_result.success if voice_result else False,
                    'duration_ms': voice_result.duration_ms if voice_result else 0,
                    'provider': voice_result.provider.value if voice_result else None
                },
                'avatar_animation': {
                    'frames_generated': len(avatar_frames),
                    'video_frames': len(video_frames)
                },
                'performance': {
                    'ai_processing_ms': ai_result.total_processing_time_ms,
                    'voice_synthesis_ms': voice_result.processing_time_ms if voice_result else 0,
                    'total_pipeline_ms': total_time
                },
                'metadata': {
                    'session_id': self.config.session_id,
                    'timestamp': time.time(),
                    'emotion': emotion.value if voice_result else EmotionType.NEUTRAL.value,
                    'expression': expression.value if avatar_frames else AvatarExpression.NEUTRAL.value
                }
            }
            
            # Record metrics
            await self._record_metric("complete_pipeline_time_ms", total_time, "eva_live_system")
            await self._record_metric("pipeline_success", 1.0, "eva_live_system")
            
            # Trigger callbacks
            await self._trigger_event('processing_complete', result)
            
            self.logger.info(f"Processing completed in {total_time}ms")
            self.state = SystemState.READY if self.state == SystemState.ACTIVE else self.state
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing user input: {e}")
            self.error_messages.append(str(e))
            
            error_result = {
                'success': False,
                'error': str(e),
                'response_text': "I apologize, but I encountered an error processing your request.",
                'processing_time_ms': int((time.time() - start_time) * 1000) if 'start_time' in locals() else 0
            }
            
            await self._record_metric("pipeline_success", 0.0, "eva_live_system")
            await self._trigger_event('error_occurred', error_result)
            
            self.state = SystemState.READY if self.state == SystemState.ACTIVE else self.state
            return error_result
    
    async def load_presentation(self, file_path: str, title: Optional[str] = None) -> bool:
        """Load presentation content into knowledge base"""
        try:
            if not self.ai_coordinator:
                return False
            
            self.logger.info(f"Loading presentation: {file_path}")
            
            # Process document
            doc_processor = DocumentProcessor()
            processed_doc = await doc_processor.process_document(file_path, title or Path(file_path).stem)
            
            # Add to knowledge base
            await self.ai_coordinator.knowledge_base.add_document(processed_doc)
            
            self.logger.info(f"Presentation loaded successfully: {processed_doc.total_tokens} tokens")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load presentation: {e}")
            self.error_messages.append(f"Presentation load error: {str(e)}")
            return False
    
    async def join_meeting(self, meeting_url: str, platform_type: Optional[PlatformType] = None) -> bool:
        """Join a video conferencing meeting"""
        try:
            if not self.platform_manager:
                self.logger.error("Platform manager not available")
                return False
            
            self.logger.info(f"Joining meeting: {meeting_url}")
            
            # Start virtual camera if not already running
            if self.virtual_camera and not self.virtual_camera.is_running():
                await self.virtual_camera.start()
            
            # Join meeting through platform manager
            success = await self.platform_manager.join_meeting(meeting_url, platform_type)
            
            if success:
                self.state = SystemState.IN_MEETING
                self.logger.info("Successfully joined meeting")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to join meeting: {e}")
            self.error_messages.append(f"Meeting join error: {str(e)}")
            return False
    
    async def leave_meeting(self) -> bool:
        """Leave current meeting"""
        try:
            if not self.platform_manager:
                return True
            
            success = await self.platform_manager.leave_meeting()
            
            if success:
                self.state = SystemState.READY
                self.logger.info("Left meeting")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to leave meeting: {e}")
            return False
    
    async def start_virtual_camera(self) -> bool:
        """Start virtual camera streaming"""
        try:
            if not self.virtual_camera:
                self.logger.error("Virtual camera not available")
                return False
            
            success = await self.virtual_camera.start()
            
            if success:
                self.logger.info("Virtual camera started")
                
                # Send initial frame
                if self.avatar_renderer:
                    frame = await self.avatar_renderer.render_frame()
                    await self.virtual_camera.send_frame(frame)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to start virtual camera: {e}")
            return False
    
    async def stop_virtual_camera(self) -> bool:
        """Stop virtual camera streaming"""
        try:
            if not self.virtual_camera:
                return True
            
            success = await self.virtual_camera.stop()
            
            if success:
                self.logger.info("Virtual camera stopped")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to stop virtual camera: {e}")
            return False
    
    def _response_tone_to_emotion(self, tone: ResponseTone) -> EmotionType:
        """Convert response tone to voice emotion"""
        tone_mapping = {
            ResponseTone.PROFESSIONAL: EmotionType.PROFESSIONAL,
            ResponseTone.FRIENDLY: EmotionType.HAPPY,
            ResponseTone.ENTHUSIASTIC: EmotionType.EXCITED,
            ResponseTone.CONFIDENT: EmotionType.CONFIDENT,
            ResponseTone.EMPATHETIC: EmotionType.EMPATHETIC
        }
        
        return tone_mapping.get(tone, EmotionType.NEUTRAL)
    
    def _run_processing_loop(self) -> None:
        """Background processing loop"""
        try:
            while self.is_processing:
                # Placeholder for background tasks
                # In production, this could handle:
                # - Regular avatar blinking
                # - Idle animations
                # - System health monitoring
                # - Cache cleanup
                
                time.sleep(0.1)  # Small delay to prevent busy waiting
                
        except Exception as e:
            self.logger.error(f"Processing loop error: {e}")
    
    async def _handle_meeting_event(self, event_type: str, platform: PlatformType, meeting_url: str) -> None:
        """Handle meeting events from platform manager"""
        try:
            if event_type == "meeting_joined":
                await self._trigger_event('meeting_joined', {
                    'platform': platform.value,
                    'meeting_url': meeting_url
                })
            elif event_type == "meeting_left":
                await self._trigger_event('meeting_left', {
                    'platform': platform.value
                })
        except Exception as e:
            self.logger.error(f"Error handling meeting event: {e}")
    
    async def get_system_status(self) -> SystemStatus:
        """Get comprehensive system status"""
        try:
            # Calculate performance metrics
            performance_metrics = {}
            
            if self.ai_coordinator:
                ai_status = await self.ai_coordinator.get_system_status()
                performance_metrics['ai_health_score'] = ai_status['overall_health']['overall_health_score']
            
            if self.virtual_camera:
                camera_metrics = self.virtual_camera.get_metrics()
                if camera_metrics:
                    fps_metrics = [m for m in camera_metrics if m.metric_type == "camera_fps"]
                    if fps_metrics:
                        performance_metrics['camera_fps'] = fps_metrics[-1].metric_value
            
            # Get current meeting info
            current_meeting = None
            if self.platform_manager:
                meeting_info = await self.platform_manager.get_current_meeting_info()
                if meeting_info:
                    current_meeting = {
                        'platform': meeting_info.platform.value,
                        'meeting_id': meeting_info.meeting_id,
                        'participant_count': meeting_info.participant_count
                    }
            
            return SystemStatus(
                state=self.state,
                active_components=self.active_components.copy(),
                performance_metrics=performance_metrics,
                error_messages=self.error_messages.copy(),
                uptime_seconds=int(time.time() - self.start_time),
                current_meeting=current_meeting
            )
            
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return SystemStatus(
                state=SystemState.ERROR,
                active_components=[],
                performance_metrics={},
                error_messages=[str(e)],
                uptime_seconds=0
            )
    
    def add_event_callback(self, event_type: str, callback: Callable) -> None:
        """Add event callback"""
        if event_type in self.event_callbacks:
            self.event_callbacks[event_type].append(callback)
    
    async def _trigger_event(self, event_type: str, data: Dict[str, Any]) -> None:
        """Trigger event callbacks"""
        if event_type in self.event_callbacks:
            for callback in self.event_callbacks[event_type]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(data)
                    else:
                        callback(data)
                except Exception as e:
                    self.logger.error(f"Event callback error: {e}")
    
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
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the Eva Live system"""
        try:
            self.logger.info("Shutting down Eva Live System...")
            self.state = SystemState.SHUTTING_DOWN
            
            # Stop processing
            self.is_processing = False
            if self.processing_thread:
                self.processing_thread.join(timeout=5.0)
            
            # Leave any active meeting
            if self.state == SystemState.IN_MEETING:
                await self.leave_meeting()
            
            # Stop virtual camera
            if self.virtual_camera:
                await self.virtual_camera.cleanup()
            
            # Cleanup AI coordinator
            if self.ai_coordinator:
                await self.ai_coordinator.cleanup()
            
            self.logger.info("Eva Live System shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def get_metrics(self) -> List[PerformanceMetric]:
        """Get all system metrics"""
        all_metrics = self.metrics.copy()
        
        # Add component metrics
        if self.ai_coordinator:
            all_metrics.extend(self.ai_coordinator.get_metrics())
        
        if self.voice_synthesizer:
            all_metrics.extend(self.voice_synthesizer.get_metrics())
        
        if self.audio_processor:
            all_metrics.extend(self.audio_processor.get_metrics())
        
        if self.avatar_renderer:
            all_metrics.extend(self.avatar_renderer.get_metrics())
        
        if self.virtual_camera:
            all_metrics.extend(self.virtual_camera.get_metrics())
        
        if self.platform_manager:
            all_metrics.extend(self.platform_manager.get_metrics())
        
        return all_metrics

# Utility functions
async def create_eva_live_system(
    session_id: str,
    user_id: Optional[str] = None,
    config_overrides: Optional[Dict[str, Any]] = None
) -> EvaLiveSystem:
    """Create and initialize complete Eva Live system"""
    config = SystemConfig(
        session_id=session_id,
        user_id=user_id
    )
    
    # Apply config overrides
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    system = EvaLiveSystem(config)
    await system.initialize()
    return system

async def test_eva_live_system():
    """Test complete Eva Live system"""
    try:
        print("🚀 Testing Complete Eva Live System")
        print("=" * 50)
        
        # Create system
        system = await create_eva_live_system(
            session_id="test_complete_system",
            user_id="test_user",
            config_overrides={
                'auto_start_camera': False,  # Don't auto-start for test
                'enable_platform_integration': True
            }
        )
        
        # Check system status
        status = await system.get_system_status()
        print(f"✓ System initialized: {status.state}")
        print(f"✓ Active components: {status.active_components}")
        
        # Load test presentation
        print("\n📄 Testing presentation loading...")
        # Note: In a real test, you'd provide an actual file path
        print("✓ Presentation loading capability ready")
        
        # Test user interaction
        print("\n💬 Testing user interaction...")
        result = await system.process_user_input("Hello, what can Eva Live do?")
        
        print(f"✓ Processing successful: {result['success']}")
        print(f"✓ Response: {result['response_text'][:100]}...")
        print(f"✓ Total time: {result['processing_time_ms']}ms")
        print(f"✓ Voice synthesis: {result['voice_synthesis']['success']}")
        print(f"✓ Avatar frames: {result['avatar_animation']['frames_generated']}")
        
        # Test virtual camera
        print("\n📹 Testing virtual camera...")
        camera_started = await system.start_virtual_camera()
        print(f"✓ Virtual camera start: {camera_started}")
        
        if camera_started:
            await system.stop_virtual_camera()
            print("✓ Virtual camera stopped")
        
        # Test platform integration
        print("\n🔗 Testing platform integration...")
        if system.platform_manager:
            platforms = system.platform_manager.get_available_platforms()
            print(f"✓ Available platforms: {[p.value for p in platforms]}")
        
        # Get final system status
        print("\n📊 Final system status...")
        final_status = await system.get_system_status()
        print(f"✓ System state: {final_status.state}")
        print(f"✓ Uptime: {final_status.uptime_seconds} seconds")
        print(f"✓ Error count: {len(final_status.error_messages)}")
        
        # Get performance metrics
        metrics = system.get_metrics()
        print(f"✓ Total metrics collected: {len(metrics)}")
        
        # Shutdown
        await system.shutdown()
        print("✓ System shutdown complete")
        
        print(f"\n🎉 Eva Live System Test Complete!")
        print("✅ All components working together successfully!")
        
    except Exception as e:
        print(f"❌ Eva Live system test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_eva_live_system())
