"""
CV Development Environment

A development environment for building and testing CV apps using pre-recorded video
instead of live camera feeds. Enables rapid iteration and testing of CV algorithms.
"""

from .video_buffer_service import VideoBufferService, get_video_buffer_service

__all__ = ['VideoBufferService', 'get_video_buffer_service']
