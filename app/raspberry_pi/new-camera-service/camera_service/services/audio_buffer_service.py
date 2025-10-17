import threading
import time
import logging
import os
import numpy as np
from threading import Lock
from collections import deque
from ..config.settings import Settings
import subprocess
import pyaudio
import wave

logger = logging.getLogger(__name__)

class AudioBufferService:
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AudioBufferService, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
        
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        # Audio settings
        self.sample_rate = 44100  # Standard CD quality
        self.channels = 1  # Mono
        self.format = pyaudio.paInt16  # 16-bit
        self.chunk_size = 1024  # Buffer size
        self.buffer_seconds = 30  # Increased to 30 seconds to match video
        self.buffer_size = int((self.sample_rate * self.buffer_seconds) / self.chunk_size)
        
        # Core buffer components
        self.audio_buffer = deque(maxlen=self.buffer_size)
        self.buffer_lock = Lock()
        self.is_running = False
        self.shutdown_event = threading.Event()
        
        # PyAudio components
        self.pyaudio = None
        self.stream = None
        
        # Health monitoring
        self.last_audio_time = 0
        
        logger.info("Audio Buffer Service initialized")
        
        # Don't auto-start, let the application control this
    
    def start_buffering(self):
        """Start audio capture and buffering"""
        with self._lock:
            if self.is_running:
                return False
                
            try:
                # Initialize PyAudio
                self.pyaudio = pyaudio.PyAudio()
                
                # Find the USB microphone device
                device_index = None
                for i in range(self.pyaudio.get_device_count()):
                    device_info = self.pyaudio.get_device_info_by_index(i)
                    logger.info(f"Audio device {i}: {device_info['name']}")
                    if "USB PnP Sound Device" in device_info["name"] or "USB Audio" in device_info["name"]:
                        device_index = i
                        logger.info(f"Found USB microphone at device index {i}")
                        break
                
                # If device not found, use default
                if device_index is None:
                    logger.warning("USB microphone not found, using default audio input")
                
                # Start the audio stream
                self.stream = self.pyaudio.open(
                    format=self.format,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    input_device_index=device_index,
                    frames_per_buffer=self.chunk_size,
                    stream_callback=self._audio_callback
                )
                
                self.is_running = True
                self.shutdown_event.clear()
                
                logger.info(f"Audio buffering started with device: {device_index}")
                return True
                
            except Exception as e:
                logger.error(f"Failed to start audio buffer: {e}")
                self.cleanup()
                return False
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Process audio data from the stream"""
        if self.shutdown_event.is_set():
            return (None, pyaudio.paComplete)
            
        # Store audio frame with timestamp
        timestamp = time.time()
        with self.buffer_lock:
            self.audio_buffer.append((in_data, timestamp))
            
        self.last_audio_time = timestamp
        return (None, pyaudio.paContinue)
    
    def get_audio_segment(self, start_time, end_time):
        """Extract audio segment matching a specific time range"""
        with self.buffer_lock:
            # Find audio frames that fall within the time range
            segment = []
            for frame, timestamp in self.audio_buffer:
                if start_time <= timestamp <= end_time:
                    segment.append(frame)
            
            # Concatenate frames if any were found
            if segment:
                return b''.join(segment)
            return None
    
    def save_audio_segment(self, start_time, end_time, output_path):
        """Save audio segment to a WAV file"""
        audio_data = self.get_audio_segment(start_time, end_time)
        if not audio_data:
            logger.warning("No audio data available for the specified time range")
            return False
            
        try:
            # Create a WAV file
            with wave.open(output_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(self.pyaudio.get_sample_size(self.format))
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio_data)
                
            logger.info(f"Audio segment saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving audio segment: {e}")
            return False
    
    def stop_buffering(self):
        """Stop audio buffering"""
        with self._lock:
            if not self.is_running:
                return
                
            self.shutdown_event.set()
            
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
                
            if self.pyaudio:
                self.pyaudio.terminate()
                self.pyaudio = None
                
            self.is_running = False
            logger.info("Audio buffering stopped")
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_buffering()
        with self.buffer_lock:
            self.audio_buffer.clear()
            
    def get_buffer_status(self):
        """Get information about the audio buffer"""
        with self.buffer_lock:
            return {
                "running": self.is_running,
                "buffer_size": len(self.audio_buffer),
                "last_timestamp": self.last_audio_time,
                "device_info": "USB Audio Device" if self.is_running else "Not connected"
            } 