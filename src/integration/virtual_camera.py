"""
Eva Live Virtual Camera System

This module provides virtual camera functionality to stream Eva Live avatar
video to video conferencing applications like Zoom, Teams, and Google Meet.
"""

import asyncio
import logging
import time
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod
import platform
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from ..shared.config import get_config
from ..shared.models import PerformanceMetric

class CameraFormat(str, Enum):
    """Video format options for virtual camera"""
    HD_720P = "1280x720"
    FULL_HD = "1920x1080"
    UHD_4K = "3840x2160"

class CameraState(str, Enum):
    """Virtual camera states"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"

@dataclass
class CameraConfig:
    """Virtual camera configuration"""
    device_name: str = "Eva Live Camera"
    width: int = 1920
    height: int = 1080
    fps: int = 30
    format: str = "MJPG"
    device_id: Optional[int] = None

class VirtualCameraBackend(ABC):
    """Abstract base class for platform-specific virtual camera implementations"""
    
    @abstractmethod
    async def initialize(self, config: CameraConfig) -> bool:
        """Initialize the virtual camera backend"""
        pass
    
    @abstractmethod
    async def start_streaming(self) -> bool:
        """Start video streaming"""
        pass
    
    @abstractmethod
    async def send_frame(self, frame: np.ndarray) -> bool:
        """Send a video frame"""
        pass
    
    @abstractmethod
    async def stop_streaming(self) -> bool:
        """Stop video streaming"""
        pass
    
    @abstractmethod
    async def cleanup(self) -> bool:
        """Cleanup resources"""
        pass

class WindowsVirtualCamera(VirtualCameraBackend):
    """Windows DirectShow virtual camera implementation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config: Optional[CameraConfig] = None
        self.obs_virtual_cam = None
        self.is_streaming = False
        
    async def initialize(self, config: CameraConfig) -> bool:
        """Initialize Windows virtual camera"""
        try:
            self.config = config
            
            # Check if OBS Virtual Camera is available
            if await self._check_obs_virtual_cam():
                self.logger.info("Using OBS Virtual Camera for Windows")
                return True
            
            # Fallback to FFmpeg virtual camera
            if await self._setup_ffmpeg_virtual_cam():
                self.logger.info("Using FFmpeg virtual camera for Windows")
                return True
            
            self.logger.error("No virtual camera backend available on Windows")
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Windows virtual camera: {e}")
            return False
    
    async def _check_obs_virtual_cam(self) -> bool:
        """Check if OBS Virtual Camera is installed"""
        try:
            # Look for OBS Studio installation
            obs_paths = [
                "C:\\Program Files\\obs-studio\\bin\\64bit\\obs64.exe",
                "C:\\Program Files (x86)\\obs-studio\\bin\\32bit\\obs32.exe"
            ]
            
            for path in obs_paths:
                if Path(path).exists():
                    return True
            
            return False
            
        except Exception:
            return False
    
    async def _setup_ffmpeg_virtual_cam(self) -> bool:
        """Setup FFmpeg-based virtual camera"""
        try:
            # Create named pipe for video data
            self.pipe_name = "\\\\.\\pipe\\eva_live_video"
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup FFmpeg virtual camera: {e}")
            return False
    
    async def start_streaming(self) -> bool:
        """Start streaming on Windows"""
        try:
            if self.obs_virtual_cam:
                return await self._start_obs_streaming()
            else:
                return await self._start_ffmpeg_streaming()
                
        except Exception as e:
            self.logger.error(f"Failed to start Windows streaming: {e}")
            return False
    
    async def _start_ffmpeg_streaming(self) -> bool:
        """Start FFmpeg-based streaming"""
        try:
            # Start FFmpeg process to create virtual camera
            cmd = [
                "ffmpeg",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{self.config.width}x{self.config.height}",
                "-r", str(self.config.fps),
                "-i", "-",
                "-f", "dshow",
                "-vcodec", "mjpeg",
                f"video={self.config.device_name}"
            ]
            
            self.ffmpeg_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            self.is_streaming = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start FFmpeg streaming: {e}")
            return False
    
    async def send_frame(self, frame: np.ndarray) -> bool:
        """Send frame to Windows virtual camera"""
        try:
            if not self.is_streaming:
                return False
            
            # Resize frame if needed
            if frame.shape[:2] != (self.config.height, self.config.width):
                frame = cv2.resize(frame, (self.config.width, self.config.height))
            
            # Send to FFmpeg process
            if hasattr(self, 'ffmpeg_process') and self.ffmpeg_process.poll() is None:
                self.ffmpeg_process.stdin.write(frame.tobytes())
                self.ffmpeg_process.stdin.flush()
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to send frame on Windows: {e}")
            return False
    
    async def stop_streaming(self) -> bool:
        """Stop Windows streaming"""
        try:
            self.is_streaming = False
            
            if hasattr(self, 'ffmpeg_process'):
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop Windows streaming: {e}")
            return False
    
    async def cleanup(self) -> bool:
        """Cleanup Windows resources"""
        return await self.stop_streaming()

class MacOSVirtualCamera(VirtualCameraBackend):
    """macOS AVFoundation virtual camera implementation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config: Optional[CameraConfig] = None
        self.is_streaming = False
        
    async def initialize(self, config: CameraConfig) -> bool:
        """Initialize macOS virtual camera"""
        try:
            self.config = config
            
            # Check for obs-mac-virtualcam
            if await self._check_obs_mac_virtualcam():
                self.logger.info("Using OBS Virtual Camera for macOS")
                return True
            
            # Check for other virtual camera solutions
            if await self._setup_ffmpeg_virtual_cam():
                self.logger.info("Using FFmpeg virtual camera for macOS")
                return True
            
            self.logger.warning("Limited virtual camera support on macOS")
            return True  # Continue with basic implementation
            
        except Exception as e:
            self.logger.error(f"Failed to initialize macOS virtual camera: {e}")
            return False
    
    async def _check_obs_mac_virtualcam(self) -> bool:
        """Check for OBS Virtual Camera on macOS"""
        try:
            obs_plugin_path = "/Library/CoreMediaIO/Plug-Ins/DAL/obs-mac-virtualcam.plugin"
            return Path(obs_plugin_path).exists()
            
        except Exception:
            return False
    
    async def _setup_ffmpeg_virtual_cam(self) -> bool:
        """Setup FFmpeg virtual camera for macOS"""
        try:
            # Create FIFO pipe for video data
            import tempfile
            self.pipe_path = f"{tempfile.gettempdir()}/eva_live_video.pipe"
            
            # Create named pipe
            subprocess.run(["mkfifo", self.pipe_path], check=True)
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup macOS virtual camera: {e}")
            return False
    
    async def start_streaming(self) -> bool:
        """Start streaming on macOS"""
        try:
            # Start FFmpeg to stream to virtual camera
            cmd = [
                "ffmpeg",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{self.config.width}x{self.config.height}",
                "-r", str(self.config.fps),
                "-i", self.pipe_path,
                "-f", "avfoundation",
                "-pixel_format", "uyvy422",
                "-video_size", f"{self.config.width}x{self.config.height}",
                "-framerate", str(self.config.fps),
                "-t", "0",
                "Eva Live Camera"
            ]
            
            self.ffmpeg_process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # Open pipe for writing
            self.pipe_file = open(self.pipe_path, 'wb')
            self.is_streaming = True
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start macOS streaming: {e}")
            return False
    
    async def send_frame(self, frame: np.ndarray) -> bool:
        """Send frame to macOS virtual camera"""
        try:
            if not self.is_streaming or not hasattr(self, 'pipe_file'):
                return False
            
            # Resize frame if needed
            if frame.shape[:2] != (self.config.height, self.config.width):
                frame = cv2.resize(frame, (self.config.width, self.config.height))
            
            # Write frame to pipe
            self.pipe_file.write(frame.tobytes())
            self.pipe_file.flush()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send frame on macOS: {e}")
            return False
    
    async def stop_streaming(self) -> bool:
        """Stop macOS streaming"""
        try:
            self.is_streaming = False
            
            if hasattr(self, 'pipe_file'):
                self.pipe_file.close()
            
            if hasattr(self, 'ffmpeg_process'):
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait()
            
            # Clean up pipe
            if hasattr(self, 'pipe_path') and Path(self.pipe_path).exists():
                Path(self.pipe_path).unlink()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop macOS streaming: {e}")
            return False
    
    async def cleanup(self) -> bool:
        """Cleanup macOS resources"""
        return await self.stop_streaming()

class LinuxVirtualCamera(VirtualCameraBackend):
    """Linux V4L2 virtual camera implementation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config: Optional[CameraConfig] = None
        self.is_streaming = False
        
    async def initialize(self, config: CameraConfig) -> bool:
        """Initialize Linux virtual camera"""
        try:
            self.config = config
            
            # Check for v4l2loopback
            if await self._check_v4l2loopback():
                self.logger.info("Using v4l2loopback for Linux virtual camera")
                return True
            
            self.logger.error("v4l2loopback not found. Please install: sudo apt install v4l2loopback-dkms")
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Linux virtual camera: {e}")
            return False
    
    async def _check_v4l2loopback(self) -> bool:
        """Check if v4l2loopback is available"""
        try:
            # Check if module is loaded
            result = subprocess.run(
                ["lsmod", "|", "grep", "v4l2loopback"],
                shell=True,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return True
            
            # Try to load module
            result = subprocess.run(
                ["sudo", "modprobe", "v4l2loopback"],
                capture_output=True
            )
            
            return result.returncode == 0
            
        except Exception:
            return False
    
    async def start_streaming(self) -> bool:
        """Start streaming on Linux"""
        try:
            # Find available v4l2loopback device
            device_path = await self._find_loopback_device()
            if not device_path:
                return False
            
            self.device_path = device_path
            
            # Start FFmpeg to stream to virtual camera
            cmd = [
                "ffmpeg",
                "-f", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{self.config.width}x{self.config.height}",
                "-r", str(self.config.fps),
                "-i", "-",
                "-f", "v4l2",
                "-pix_fmt", "yuyv422",
                device_path
            ]
            
            self.ffmpeg_process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            self.is_streaming = True
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start Linux streaming: {e}")
            return False
    
    async def _find_loopback_device(self) -> Optional[str]:
        """Find available v4l2loopback device"""
        try:
            for i in range(10):  # Check /dev/video0 through /dev/video9
                device_path = f"/dev/video{i}"
                if Path(device_path).exists():
                    # Check if it's a loopback device
                    result = subprocess.run(
                        ["v4l2-ctl", "--device", device_path, "--info"],
                        capture_output=True,
                        text=True
                    )
                    
                    if "Dummy video device" in result.stdout:
                        return device_path
            
            return None
            
        except Exception:
            return None
    
    async def send_frame(self, frame: np.ndarray) -> bool:
        """Send frame to Linux virtual camera"""
        try:
            if not self.is_streaming:
                return False
            
            # Resize frame if needed
            if frame.shape[:2] != (self.config.height, self.config.width):
                frame = cv2.resize(frame, (self.config.width, self.config.height))
            
            # Send to FFmpeg process
            if hasattr(self, 'ffmpeg_process') and self.ffmpeg_process.poll() is None:
                self.ffmpeg_process.stdin.write(frame.tobytes())
                self.ffmpeg_process.stdin.flush()
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to send frame on Linux: {e}")
            return False
    
    async def stop_streaming(self) -> bool:
        """Stop Linux streaming"""
        try:
            self.is_streaming = False
            
            if hasattr(self, 'ffmpeg_process'):
                self.ffmpeg_process.terminate()
                self.ffmpeg_process.wait()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop Linux streaming: {e}")
            return False
    
    async def cleanup(self) -> bool:
        """Cleanup Linux resources"""
        return await self.stop_streaming()

class VirtualCamera:
    """Main virtual camera interface that handles platform-specific implementations"""
    
    def __init__(self, session_id: Optional[str] = None):
        self.config = get_config()
        self.session_id = session_id
        self.logger = logging.getLogger(__name__)
        
        # Platform detection
        self.platform = platform.system().lower()
        self.backend: Optional[VirtualCameraBackend] = None
        
        # Camera state
        self.state = CameraState.STOPPED
        self.camera_config = CameraConfig()
        
        # Frame buffer for smooth streaming
        self.frame_buffer = []
        self.buffer_size = 5
        
        # Performance tracking
        self.metrics: List[PerformanceMetric] = []
        self.frame_count = 0
        self.last_fps_time = time.time()
        
        # Streaming thread
        self.streaming_thread: Optional[threading.Thread] = None
        self.stop_streaming_event = threading.Event()
    
    async def initialize(self, config: Optional[CameraConfig] = None) -> bool:
        """Initialize virtual camera with platform-specific backend"""
        try:
            if config:
                self.camera_config = config
            
            # Initialize platform-specific backend
            if self.platform == "windows":
                self.backend = WindowsVirtualCamera()
            elif self.platform == "darwin":  # macOS
                self.backend = MacOSVirtualCamera()
            elif self.platform == "linux":
                self.backend = LinuxVirtualCamera()
            else:
                self.logger.error(f"Unsupported platform: {self.platform}")
                return False
            
            # Initialize backend
            success = await self.backend.initialize(self.camera_config)
            
            if success:
                self.state = CameraState.STOPPED
                self.logger.info(f"Virtual camera initialized for {self.platform}")
            else:
                self.state = CameraState.ERROR
                self.logger.error("Failed to initialize virtual camera backend")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to initialize virtual camera: {e}")
            self.state = CameraState.ERROR
            return False
    
    async def start(self) -> bool:
        """Start virtual camera streaming"""
        try:
            if self.state != CameraState.STOPPED:
                self.logger.warning(f"Cannot start camera in state: {self.state}")
                return False
            
            self.state = CameraState.STARTING
            
            # Start backend streaming
            if not await self.backend.start_streaming():
                self.state = CameraState.ERROR
                return False
            
            # Start streaming thread
            self.stop_streaming_event.clear()
            self.streaming_thread = threading.Thread(target=self._streaming_loop, daemon=True)
            self.streaming_thread.start()
            
            self.state = CameraState.RUNNING
            self.logger.info("Virtual camera streaming started")
            
            await self._record_metric("camera_start_success", 1.0, "virtual_camera")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start virtual camera: {e}")
            self.state = CameraState.ERROR
            await self._record_metric("camera_start_success", 0.0, "virtual_camera")
            return False
    
    async def send_frame(self, frame: np.ndarray) -> bool:
        """Send video frame to virtual camera"""
        try:
            if self.state != CameraState.RUNNING:
                return False
            
            # Add frame to buffer
            if len(self.frame_buffer) >= self.buffer_size:
                self.frame_buffer.pop(0)  # Remove oldest frame
            
            self.frame_buffer.append(frame.copy())
            
            # Update performance metrics
            self.frame_count += 1
            current_time = time.time()
            
            if current_time - self.last_fps_time >= 1.0:
                fps = self.frame_count / (current_time - self.last_fps_time)
                await self._record_metric("camera_fps", fps, "virtual_camera")
                
                self.frame_count = 0
                self.last_fps_time = current_time
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending frame to virtual camera: {e}")
            return False
    
    def _streaming_loop(self) -> None:
        """Background streaming loop"""
        try:
            target_frame_time = 1.0 / self.camera_config.fps
            last_frame_time = time.time()
            
            while not self.stop_streaming_event.is_set():
                current_time = time.time()
                
                # Check if it's time for the next frame
                if current_time - last_frame_time >= target_frame_time:
                    if self.frame_buffer:
                        # Get latest frame
                        frame = self.frame_buffer[-1]
                        
                        # Send frame to backend
                        try:
                            asyncio.run(self.backend.send_frame(frame))
                        except Exception as e:
                            self.logger.error(f"Error in streaming loop: {e}")
                        
                        last_frame_time = current_time
                
                # Small sleep to prevent busy waiting
                time.sleep(0.001)
                
        except Exception as e:
            self.logger.error(f"Streaming loop error: {e}")
    
    async def stop(self) -> bool:
        """Stop virtual camera streaming"""
        try:
            if self.state != CameraState.RUNNING:
                return True
            
            # Stop streaming thread
            if self.streaming_thread:
                self.stop_streaming_event.set()
                self.streaming_thread.join(timeout=5.0)
            
            # Stop backend
            await self.backend.stop_streaming()
            
            # Clear frame buffer
            self.frame_buffer.clear()
            
            self.state = CameraState.STOPPED
            self.logger.info("Virtual camera streaming stopped")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop virtual camera: {e}")
            return False
    
    async def cleanup(self) -> bool:
        """Cleanup virtual camera resources"""
        try:
            await self.stop()
            
            if self.backend:
                await self.backend.cleanup()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup virtual camera: {e}")
            return False
    
    def get_state(self) -> CameraState:
        """Get current camera state"""
        return self.state
    
    def is_running(self) -> bool:
        """Check if camera is currently running"""
        return self.state == CameraState.RUNNING
    
    async def get_available_formats(self) -> List[CameraFormat]:
        """Get available camera formats"""
        return [CameraFormat.HD_720P, CameraFormat.FULL_HD, CameraFormat.UHD_4K]
    
    def set_format(self, format: CameraFormat) -> None:
        """Set camera format"""
        if format == CameraFormat.HD_720P:
            self.camera_config.width = 1280
            self.camera_config.height = 720
        elif format == CameraFormat.FULL_HD:
            self.camera_config.width = 1920
            self.camera_config.height = 1080
        elif format == CameraFormat.UHD_4K:
            self.camera_config.width = 3840
            self.camera_config.height = 2160
    
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
async def create_virtual_camera(session_id: Optional[str] = None, config: Optional[CameraConfig] = None) -> VirtualCamera:
    """Create and initialize virtual camera"""
    camera = VirtualCamera(session_id)
    await camera.initialize(config)
    return camera

async def test_virtual_camera():
    """Test virtual camera functionality"""
    try:
        # Create virtual camera
        camera = VirtualCamera()
        
        # Initialize
        success = await camera.initialize()
        print(f"Camera initialization: {'✓' if success else '✗'}")
        
        if success:
            # Start streaming
            success = await camera.start()
            print(f"Camera streaming start: {'✓' if success else '✗'}")
            
            if success:
                # Send test frames
                for i in range(30):  # 1 second at 30fps
                    # Create test frame
                    frame = np.zeros((1080, 1920, 3), dtype=np.uint8)
                    cv2.putText(frame, f"Eva Live Test Frame {i}", 
                              (500, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                    
                    await camera.send_frame(frame)
                    await asyncio.sleep(1/30)  # 30fps timing
                
                print("✓ Sent 30 test frames")
                
                # Stop streaming
                await camera.stop()
                print("✓ Camera stopped")
        
        # Cleanup
        await camera.cleanup()
        print("✓ Camera cleanup completed")
        
        print("Virtual camera test completed successfully!")
        
    except Exception as e:
        print(f"Virtual camera test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_virtual_camera())
