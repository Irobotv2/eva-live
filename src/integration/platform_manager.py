"""
Eva Live Platform Manager

This module manages integrations with various video conferencing platforms
like Zoom, Microsoft Teams, Google Meet, and generic WebRTC applications.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import json
import webbrowser
from urllib.parse import urlencode

from ..shared.config import get_config
from ..shared.models import PerformanceMetric

class PlatformType(str, Enum):
    """Supported platform types"""
    ZOOM = "zoom"
    TEAMS = "teams"
    GOOGLE_MEET = "meet"
    WEBEX = "webex"
    GENERIC_WEBRTC = "webrtc"
    BROWSER_BASED = "browser"

class PlatformState(str, Enum):
    """Platform integration states"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    IN_MEETING = "in_meeting"
    ERROR = "error"

@dataclass
class PlatformConfig:
    """Platform-specific configuration"""
    platform_type: PlatformType
    api_key: Optional[str] = None
    api_secret: Optional[str] = None
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    redirect_uri: Optional[str] = None
    webhook_url: Optional[str] = None
    custom_settings: Dict[str, Any] = None

@dataclass
class MeetingInfo:
    """Meeting information"""
    meeting_id: str
    meeting_url: str
    host_name: str
    participant_count: int
    duration_seconds: int
    platform: PlatformType
    metadata: Dict[str, Any] = None

class PlatformIntegration(ABC):
    """Abstract base class for platform integrations"""
    
    @abstractmethod
    async def initialize(self, config: PlatformConfig) -> bool:
        """Initialize platform integration"""
        pass
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to platform"""
        pass
    
    @abstractmethod
    async def join_meeting(self, meeting_url: str) -> bool:
        """Join a meeting"""
        pass
    
    @abstractmethod
    async def start_video_stream(self) -> bool:
        """Start video streaming to platform"""
        pass
    
    @abstractmethod
    async def stop_video_stream(self) -> bool:
        """Stop video streaming"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> bool:
        """Disconnect from platform"""
        pass
    
    @abstractmethod
    async def get_meeting_info(self) -> Optional[MeetingInfo]:
        """Get current meeting information"""
        pass

class ZoomIntegration(PlatformIntegration):
    """Zoom platform integration"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config: Optional[PlatformConfig] = None
        self.state = PlatformState.DISCONNECTED
        self.meeting_info: Optional[MeetingInfo] = None
        
    async def initialize(self, config: PlatformConfig) -> bool:
        """Initialize Zoom integration"""
        try:
            self.config = config
            
            # Validate Zoom configuration
            if not config.api_key or not config.api_secret:
                self.logger.error("Zoom API key and secret required")
                return False
            
            self.logger.info("Zoom integration initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Zoom integration: {e}")
            return False
    
    async def connect(self) -> bool:
        """Connect to Zoom"""
        try:
            self.state = PlatformState.CONNECTING
            
            # In a real implementation, this would:
            # 1. Authenticate with Zoom API
            # 2. Get access token
            # 3. Set up webhooks
            
            # For demo purposes, we'll simulate connection
            await asyncio.sleep(1)
            
            self.state = PlatformState.CONNECTED
            self.logger.info("Connected to Zoom")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Zoom: {e}")
            self.state = PlatformState.ERROR
            return False
    
    async def join_meeting(self, meeting_url: str) -> bool:
        """Join Zoom meeting"""
        try:
            if self.state != PlatformState.CONNECTED:
                self.logger.error("Not connected to Zoom")
                return False
            
            # Extract meeting ID from URL
            meeting_id = self._extract_meeting_id(meeting_url)
            if not meeting_id:
                self.logger.error("Invalid Zoom meeting URL")
                return False
            
            # In real implementation:
            # 1. Join meeting via Zoom SDK
            # 2. Set up video/audio streams
            # 3. Handle meeting events
            
            self.meeting_info = MeetingInfo(
                meeting_id=meeting_id,
                meeting_url=meeting_url,
                host_name="Meeting Host",
                participant_count=1,
                duration_seconds=0,
                platform=PlatformType.ZOOM
            )
            
            self.state = PlatformState.IN_MEETING
            self.logger.info(f"Joined Zoom meeting: {meeting_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to join Zoom meeting: {e}")
            return False
    
    def _extract_meeting_id(self, meeting_url: str) -> Optional[str]:
        """Extract meeting ID from Zoom URL"""
        try:
            # Simple extraction for demo
            if "zoom.us/j/" in meeting_url:
                return meeting_url.split("/j/")[1].split("?")[0]
            return None
        except Exception:
            return None
    
    async def start_video_stream(self) -> bool:
        """Start video streaming to Zoom"""
        try:
            if self.state != PlatformState.IN_MEETING:
                return False
            
            # In real implementation:
            # 1. Initialize Zoom video SDK
            # 2. Set Eva Live as video source
            # 3. Start streaming
            
            self.logger.info("Started video streaming to Zoom")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start Zoom video stream: {e}")
            return False
    
    async def stop_video_stream(self) -> bool:
        """Stop video streaming"""
        try:
            self.logger.info("Stopped video streaming to Zoom")
            return True
        except Exception as e:
            self.logger.error(f"Failed to stop Zoom video stream: {e}")
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Zoom"""
        try:
            await self.stop_video_stream()
            self.state = PlatformState.DISCONNECTED
            self.meeting_info = None
            self.logger.info("Disconnected from Zoom")
            return True
        except Exception as e:
            self.logger.error(f"Failed to disconnect from Zoom: {e}")
            return False
    
    async def get_meeting_info(self) -> Optional[MeetingInfo]:
        """Get current meeting information"""
        return self.meeting_info

class TeamsIntegration(PlatformIntegration):
    """Microsoft Teams platform integration"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config: Optional[PlatformConfig] = None
        self.state = PlatformState.DISCONNECTED
        self.meeting_info: Optional[MeetingInfo] = None
        
    async def initialize(self, config: PlatformConfig) -> bool:
        """Initialize Teams integration"""
        try:
            self.config = config
            
            # Validate Teams configuration
            if not config.client_id or not config.client_secret:
                self.logger.error("Teams client ID and secret required")
                return False
            
            self.logger.info("Teams integration initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Teams integration: {e}")
            return False
    
    async def connect(self) -> bool:
        """Connect to Teams"""
        try:
            self.state = PlatformState.CONNECTING
            
            # In real implementation:
            # 1. OAuth flow with Microsoft Graph
            # 2. Get access token
            # 3. Set up Teams bot/app
            
            await asyncio.sleep(1)
            
            self.state = PlatformState.CONNECTED
            self.logger.info("Connected to Microsoft Teams")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Teams: {e}")
            self.state = PlatformState.ERROR
            return False
    
    async def join_meeting(self, meeting_url: str) -> bool:
        """Join Teams meeting"""
        try:
            if self.state != PlatformState.CONNECTED:
                return False
            
            meeting_id = self._extract_meeting_id(meeting_url)
            
            self.meeting_info = MeetingInfo(
                meeting_id=meeting_id or "teams_meeting",
                meeting_url=meeting_url,
                host_name="Teams Host",
                participant_count=1,
                duration_seconds=0,
                platform=PlatformType.TEAMS
            )
            
            self.state = PlatformState.IN_MEETING
            self.logger.info("Joined Teams meeting")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to join Teams meeting: {e}")
            return False
    
    def _extract_meeting_id(self, meeting_url: str) -> Optional[str]:
        """Extract meeting ID from Teams URL"""
        try:
            # Teams URLs are complex, simplified for demo
            if "teams.microsoft.com" in meeting_url:
                return "teams_meeting_id"
            return None
        except Exception:
            return None
    
    async def start_video_stream(self) -> bool:
        """Start video streaming to Teams"""
        try:
            if self.state != PlatformState.IN_MEETING:
                return False
            
            self.logger.info("Started video streaming to Teams")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start Teams video stream: {e}")
            return False
    
    async def stop_video_stream(self) -> bool:
        """Stop video streaming"""
        try:
            self.logger.info("Stopped video streaming to Teams")
            return True
        except Exception as e:
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Teams"""
        try:
            await self.stop_video_stream()
            self.state = PlatformState.DISCONNECTED
            self.meeting_info = None
            self.logger.info("Disconnected from Teams")
            return True
        except Exception:
            return False
    
    async def get_meeting_info(self) -> Optional[MeetingInfo]:
        """Get current meeting information"""
        return self.meeting_info

class GoogleMeetIntegration(PlatformIntegration):
    """Google Meet platform integration"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config: Optional[PlatformConfig] = None
        self.state = PlatformState.DISCONNECTED
        self.meeting_info: Optional[MeetingInfo] = None
        
    async def initialize(self, config: PlatformConfig) -> bool:
        """Initialize Google Meet integration"""
        try:
            self.config = config
            self.logger.info("Google Meet integration initialized")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize Google Meet integration: {e}")
            return False
    
    async def connect(self) -> bool:
        """Connect to Google Meet"""
        try:
            self.state = PlatformState.CONNECTING
            
            # In real implementation:
            # 1. Google OAuth flow
            # 2. Meet API setup
            # 3. WebRTC configuration
            
            await asyncio.sleep(1)
            
            self.state = PlatformState.CONNECTED
            self.logger.info("Connected to Google Meet")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to Google Meet: {e}")
            self.state = PlatformState.ERROR
            return False
    
    async def join_meeting(self, meeting_url: str) -> bool:
        """Join Google Meet meeting"""
        try:
            if self.state != PlatformState.CONNECTED:
                return False
            
            meeting_id = self._extract_meeting_id(meeting_url)
            
            self.meeting_info = MeetingInfo(
                meeting_id=meeting_id or "meet_meeting",
                meeting_url=meeting_url,
                host_name="Meet Host",
                participant_count=1,
                duration_seconds=0,
                platform=PlatformType.GOOGLE_MEET
            )
            
            self.state = PlatformState.IN_MEETING
            self.logger.info("Joined Google Meet meeting")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to join Google Meet meeting: {e}")
            return False
    
    def _extract_meeting_id(self, meeting_url: str) -> Optional[str]:
        """Extract meeting ID from Meet URL"""
        try:
            if "meet.google.com/" in meeting_url:
                return meeting_url.split("meet.google.com/")[1].split("?")[0]
            return None
        except Exception:
            return None
    
    async def start_video_stream(self) -> bool:
        """Start video streaming to Google Meet"""
        try:
            if self.state != PlatformState.IN_MEETING:
                return False
            
            self.logger.info("Started video streaming to Google Meet")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start Meet video stream: {e}")
            return False
    
    async def stop_video_stream(self) -> bool:
        """Stop video streaming"""
        try:
            self.logger.info("Stopped video streaming to Google Meet")
            return True
        except Exception:
            return False
    
    async def disconnect(self) -> bool:
        """Disconnect from Google Meet"""
        try:
            await self.stop_video_stream()
            self.state = PlatformState.DISCONNECTED
            self.meeting_info = None
            self.logger.info("Disconnected from Google Meet")
            return True
        except Exception:
            return False
    
    async def get_meeting_info(self) -> Optional[MeetingInfo]:
        """Get current meeting information"""
        return self.meeting_info

class BrowserBasedIntegration(PlatformIntegration):
    """Browser-based integration using virtual camera"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config: Optional[PlatformConfig] = None
        self.state = PlatformState.DISCONNECTED
        self.meeting_info: Optional[MeetingInfo] = None
        
    async def initialize(self, config: PlatformConfig) -> bool:
        """Initialize browser-based integration"""
        try:
            self.config = config
            self.logger.info("Browser-based integration initialized")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize browser integration: {e}")
            return False
    
    async def connect(self) -> bool:
        """Connect for browser-based use"""
        try:
            self.state = PlatformState.CONNECTED
            self.logger.info("Browser-based integration ready")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect browser integration: {e}")
            return False
    
    async def join_meeting(self, meeting_url: str) -> bool:
        """Open meeting URL in browser"""
        try:
            # Open the meeting URL in default browser
            webbrowser.open(meeting_url)
            
            self.meeting_info = MeetingInfo(
                meeting_id="browser_meeting",
                meeting_url=meeting_url,
                host_name="Browser Host",
                participant_count=1,
                duration_seconds=0,
                platform=PlatformType.BROWSER_BASED
            )
            
            self.state = PlatformState.IN_MEETING
            self.logger.info(f"Opened meeting in browser: {meeting_url}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to open meeting in browser: {e}")
            return False
    
    async def start_video_stream(self) -> bool:
        """Video streaming handled by virtual camera"""
        try:
            self.logger.info("Video streaming via virtual camera (browser will use Eva Live Camera)")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start browser video stream: {e}")
            return False
    
    async def stop_video_stream(self) -> bool:
        """Stop video streaming"""
        return True
    
    async def disconnect(self) -> bool:
        """Disconnect browser integration"""
        try:
            self.state = PlatformState.DISCONNECTED
            self.meeting_info = None
            return True
        except Exception:
            return False
    
    async def get_meeting_info(self) -> Optional[MeetingInfo]:
        """Get current meeting information"""
        return self.meeting_info

class PlatformManager:
    """Main platform manager that coordinates all integrations"""
    
    def __init__(self, session_id: Optional[str] = None):
        self.config = get_config()
        self.session_id = session_id
        self.logger = logging.getLogger(__name__)
        
        # Platform integrations
        self.integrations: Dict[PlatformType, PlatformIntegration] = {}
        self.active_platform: Optional[PlatformType] = None
        
        # Performance tracking
        self.metrics: List[PerformanceMetric] = []
        
        # Event callbacks
        self.meeting_callbacks: List[Callable] = []
        
    async def initialize(self) -> bool:
        """Initialize platform manager and available integrations"""
        try:
            # Initialize available platform integrations
            await self._initialize_zoom()
            await self._initialize_teams()
            await self._initialize_google_meet()
            await self._initialize_browser_based()
            
            self.logger.info(f"Initialized {len(self.integrations)} platform integrations")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize platform manager: {e}")
            return False
    
    async def _initialize_zoom(self) -> None:
        """Initialize Zoom integration if configured"""
        try:
            config = PlatformConfig(
                platform_type=PlatformType.ZOOM,
                api_key=self.config.get('platform_integrations.zoom.api_key'),
                api_secret=self.config.get('platform_integrations.zoom.api_secret')
            )
            
            if config.api_key and config.api_secret:
                integration = ZoomIntegration()
                if await integration.initialize(config):
                    self.integrations[PlatformType.ZOOM] = integration
                    self.logger.info("Zoom integration available")
            
        except Exception as e:
            self.logger.warning(f"Zoom integration not available: {e}")
    
    async def _initialize_teams(self) -> None:
        """Initialize Teams integration if configured"""
        try:
            config = PlatformConfig(
                platform_type=PlatformType.TEAMS,
                client_id=self.config.get('platform_integrations.teams.client_id'),
                client_secret=self.config.get('platform_integrations.teams.client_secret')
            )
            
            if config.client_id and config.client_secret:
                integration = TeamsIntegration()
                if await integration.initialize(config):
                    self.integrations[PlatformType.TEAMS] = integration
                    self.logger.info("Teams integration available")
            
        except Exception as e:
            self.logger.warning(f"Teams integration not available: {e}")
    
    async def _initialize_google_meet(self) -> None:
        """Initialize Google Meet integration"""
        try:
            config = PlatformConfig(
                platform_type=PlatformType.GOOGLE_MEET,
                client_id=self.config.get('platform_integrations.google.client_id'),
                client_secret=self.config.get('platform_integrations.google.client_secret')
            )
            
            integration = GoogleMeetIntegration()
            if await integration.initialize(config):
                self.integrations[PlatformType.GOOGLE_MEET] = integration
                self.logger.info("Google Meet integration available")
            
        except Exception as e:
            self.logger.warning(f"Google Meet integration not available: {e}")
    
    async def _initialize_browser_based(self) -> None:
        """Initialize browser-based integration (always available)"""
        try:
            config = PlatformConfig(platform_type=PlatformType.BROWSER_BASED)
            integration = BrowserBasedIntegration()
            
            if await integration.initialize(config):
                self.integrations[PlatformType.BROWSER_BASED] = integration
                self.logger.info("Browser-based integration available")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize browser integration: {e}")
    
    async def connect_platform(self, platform_type: PlatformType) -> bool:
        """Connect to a specific platform"""
        try:
            if platform_type not in self.integrations:
                self.logger.error(f"Platform {platform_type} not available")
                return False
            
            integration = self.integrations[platform_type]
            success = await integration.connect()
            
            if success:
                self.active_platform = platform_type
                await self._record_metric("platform_connect_success", 1.0, f"platform_{platform_type}")
                self.logger.info(f"Connected to {platform_type}")
            else:
                await self._record_metric("platform_connect_success", 0.0, f"platform_{platform_type}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to connect to {platform_type}: {e}")
            return False
    
    async def join_meeting(self, meeting_url: str, platform_type: Optional[PlatformType] = None) -> bool:
        """Join a meeting on specified or auto-detected platform"""
        try:
            # Auto-detect platform if not specified
            if not platform_type:
                platform_type = self._detect_platform_from_url(meeting_url)
            
            # Fallback to browser-based if detection fails
            if not platform_type or platform_type not in self.integrations:
                platform_type = PlatformType.BROWSER_BASED
            
            # Connect to platform if not already connected
            if self.active_platform != platform_type:
                if not await self.connect_platform(platform_type):
                    return False
            
            # Join meeting
            integration = self.integrations[platform_type]
            success = await integration.join_meeting(meeting_url)
            
            if success:
                # Start video streaming
                await integration.start_video_stream()
                
                # Notify callbacks
                for callback in self.meeting_callbacks:
                    try:
                        await callback("meeting_joined", platform_type, meeting_url)
                    except Exception as e:
                        self.logger.error(f"Callback error: {e}")
                
                await self._record_metric("meeting_join_success", 1.0, f"platform_{platform_type}")
                self.logger.info(f"Joined meeting on {platform_type}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to join meeting: {e}")
            return False
    
    def _detect_platform_from_url(self, meeting_url: str) -> Optional[PlatformType]:
        """Auto-detect platform from meeting URL"""
        url_lower = meeting_url.lower()
        
        if "zoom.us" in url_lower:
            return PlatformType.ZOOM
        elif "teams.microsoft.com" in url_lower:
            return PlatformType.TEAMS
        elif "meet.google.com" in url_lower:
            return PlatformType.GOOGLE_MEET
        elif "webex.com" in url_lower:
            return PlatformType.WEBEX
        
        return None
    
    async def leave_meeting(self) -> bool:
        """Leave current meeting"""
        try:
            if not self.active_platform:
                return True
            
            integration = self.integrations[self.active_platform]
            success = await integration.disconnect()
            
            if success:
                # Notify callbacks
                for callback in self.meeting_callbacks:
                    try:
                        await callback("meeting_left", self.active_platform, "")
                    except Exception:
                        pass
                
                self.logger.info("Left meeting")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to leave meeting: {e}")
            return False
    
    async def get_current_meeting_info(self) -> Optional[MeetingInfo]:
        """Get information about current meeting"""
        try:
            if not self.active_platform:
                return None
            
            integration = self.integrations[self.active_platform]
            return await integration.get_meeting_info()
            
        except Exception as e:
            self.logger.error(f"Failed to get meeting info: {e}")
            return None
    
    def get_available_platforms(self) -> List[PlatformType]:
        """Get list of available platform integrations"""
        return list(self.integrations.keys())
    
    def add_meeting_callback(self, callback: Callable) -> None:
        """Add callback for meeting events"""
        self.meeting_callbacks.append(callback)
    
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
async def create_platform_manager(session_id: Optional[str] = None) -> PlatformManager:
    """Create and initialize platform manager"""
    manager = PlatformManager(session_id)
    await manager.initialize()
    return manager

async def test_platform_manager():
    """Test platform manager functionality"""
    try:
        # Create platform manager
        manager = PlatformManager()
        
        # Initialize
        success = await manager.initialize()
        print(f"Platform manager initialization: {'✓' if success else '✗'}")
        
        # Get available platforms
        platforms = manager.get_available_platforms()
        print(f"Available platforms: {[p.value for p in platforms]}")
        
        # Test browser-based integration
        if PlatformType.BROWSER_BASED in platforms:
            success = await manager.connect_platform(PlatformType.BROWSER_BASED)
            print(f"Browser integration connect: {'✓' if success else '✗'}")
            
            # Test joining a meeting
            test_url = "https://meet.google.com/test-meeting"
            success = await manager.join_meeting(test_url)
            print(f"Join meeting test: {'✓' if success else '✗'}")
            
            # Get meeting info
            info = await manager.get_current_meeting_info()
            if info:
                print(f"Meeting info: {info.platform} - {info.meeting_url}")
            
            # Leave meeting
            success = await manager.leave_meeting()
            print(f"Leave meeting: {'✓' if success else '✗'}")
        
        print("Platform manager test completed!")
        
    except Exception as e:
        print(f"Platform manager test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_platform_manager())
