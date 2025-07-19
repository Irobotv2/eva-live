"""
Eva Live Integration Layer

This module contains platform integrations and virtual device drivers
for seamless deployment across different video conferencing platforms.
"""

from .virtual_camera import VirtualCamera, CameraFormat, create_virtual_camera
from .platform_manager import PlatformManager, PlatformType, create_platform_manager

__all__ = [
    'VirtualCamera',
    'CameraFormat', 
    'create_virtual_camera',
    'PlatformManager',
    'PlatformType',
    'create_platform_manager'
]
