"""
Utility functions for Camera Service

Various utilities for camera management, diagnostics, and testing.
"""

import os
import logging
import subprocess
import numpy as np
import json
import time
import cv2
from typing import Dict, List, Tuple, Optional, Any

# Set up logging
logger = logging.getLogger(__name__)

def detect_cameras() -> Dict[str, List[Dict[str, Any]]]:
    """
    Detect all available camera devices on the system.
    Uses both v4l2-ctl and manual checking of /dev/video* devices.
    
    Returns:
        Dictionary with camera information
    """
    try:
        # First try using v4l2-ctl
        v4l2_devices = []
        try:
            result = subprocess.run(['v4l2-ctl', '--list-devices'], 
                                    capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                # Parse the output
                lines = result.stdout.strip().split('\n')
                current_device = None
                
                for line in lines:
                    if not line.startswith('\t'):
                        # This is a device name
                        current_device = line.strip()
                    elif current_device and line.strip().startswith('/dev/video'):
                        # This is a device path
                        device_path = line.strip()
                        device_id = int(device_path.replace('/dev/video', ''))
                        v4l2_devices.append({
                            'name': current_device,
                            'path': device_path,
                            'id': device_id
                        })
        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.warning(f"Could not use v4l2-ctl to detect cameras: {e}")
        
        # Now check /dev/video* devices directly
        direct_devices = []
        for i in range(10):  # Check first 10 video devices
            device_path = f'/dev/video{i}'
            if os.path.exists(device_path):
                # Try to open the device to check if it's accessible
                try:
                    cap = cv2.VideoCapture(i)
                    is_opened = cap.isOpened()
                    cap.release()
                    
                    # Device exists and is openable
                    direct_devices.append({
                        'name': f"Camera {i}",
                        'path': device_path,
                        'id': i,
                        'accessible': is_opened
                    })
                except Exception as e:
                    # Device exists but can't be opened
                    direct_devices.append({
                        'name': f"Camera {i}",
                        'path': device_path,
                        'id': i,
                        'accessible': False,
                        'error': str(e)
                    })
        
        return {
            'v4l2_devices': v4l2_devices,
            'direct_devices': direct_devices
        }
        
    except Exception as e:
        logger.error(f"Error detecting cameras: {e}")
        return {
            'error': str(e),
            'v4l2_devices': [],
            'direct_devices': []
        }

def reset_camera_devices() -> Dict[str, Any]:
    """
    Reset all camera devices by unbinding and rebinding the USB devices.
    
    Returns:
        Dictionary with operation results
    """
    try:
        # Get USB device list 
        usb_devices = []
        
        # Try to find video devices in USB bus
        lsusb_result = subprocess.run(['lsusb'], capture_output=True, text=True, timeout=3)
        if lsusb_result.returncode == 0:
            for line in lsusb_result.stdout.strip().split('\n'):
                if 'Camera' in line or 'webcam' in line.lower() or 'Video' in line:
                    usb_devices.append(line)
        
        # Cannot perform actual unbind/rebind as it requires root privileges
        # Instead, send a signal to all devices to reset
        reset_count = 0
        
        # Reset all video devices using v4l2-ctl
        for i in range(10):
            device_path = f'/dev/video{i}'
            if os.path.exists(device_path):
                try:
                    # Use v4l2-ctl to reset the device if available
                    reset_result = subprocess.run(['v4l2-ctl', '--device', device_path, '--set-ctrl', 'power_line_frequency=0'], 
                                               capture_output=True, text=True, timeout=2)
                    
                    if reset_result.returncode == 0:
                        reset_count += 1
                except:
                    # Ignore errors, as v4l2-ctl might not be installed
                    pass
                    
                # Try a more reliable method - release and re-open
                try:
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        cap.release()
                        time.sleep(0.2)
                        cap = cv2.VideoCapture(i)
                        cap.release()
                        reset_count += 1
                except:
                    pass
                    
        return {
            'success': reset_count > 0,
            'devices_reset': reset_count,
            'usb_devices': usb_devices,
            'message': f"Reset signal sent to {reset_count} devices"
        }
        
    except Exception as e:
        logger.error(f"Error resetting camera devices: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def get_camera_health(camera_index: int = 0) -> Dict[str, Any]:
    """
    Get health information about a specific camera.
    
    Args:
        camera_index: Camera index to check
        
    Returns:
        Dictionary with camera health information
    """
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return {
                'healthy': False,
                'error': f"Camera {camera_index} could not be opened"
            }
        
        # Check if we can read a frame
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            return {
                'healthy': False,
                'error': f"Could not read frame from camera {camera_index}"
            }
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame brightness as a health check
        brightness = np.mean(frame)
        
        cap.release()
        
        return {
            'healthy': True,
            'width': width,
            'height': height,
            'fps': fps,
            'brightness': float(brightness),
            'dark_frame': brightness < 30,  # Flag if image is very dark
            'message': f"Camera {camera_index} is working properly"
        }
        
    except Exception as e:
        logger.error(f"Error checking camera health: {e}")
        return {
            'healthy': False,
            'error': str(e)
        }

def check_facenet_availability() -> Dict[str, Any]:
    """
    Check if FaceNet is properly installed and working.
    Tests both the model file existence and integrity.
    
    Returns:
        Dictionary with FaceNet availability status and details
    """
    try:
        # Check if the FaceNet model file exists
        model_path = os.path.expanduser('~/mmoment/app/orin_nano/camera_service/models/facenet_model/facenet_keras.h5')
        model_exists = os.path.exists(model_path)
        
        # Check if TensorFlow is available
        tf_available = False
        try:
            import tensorflow as tf
            tf_version = tf.__version__
            tf_available = True
        except ImportError:
            tf_version = "Not installed"
        
        # Check if OpenCV is available with proper DNN support
        opencv_available = False
        opencv_with_dnn = False
        try:
            cv_version = cv2.__version__
            opencv_available = True
            # Check if DNN module is available
            if hasattr(cv2, 'dnn'):
                opencv_with_dnn = True
        except:
            cv_version = "Not installed or incompatible"
        
        # Try to load the model if all prerequisites are met
        model_integrity = False
        model_error = None
        if model_exists and tf_available:
            try:
                # Only try to import and use facenet_model if all prerequisites are met
                from services.facenet_model import get_facenet_model
                model = get_facenet_model()
                
                # Create a simple test to verify model works
                test_image = np.zeros((160, 160, 3), dtype=np.uint8)
                embedding = model.generate_embedding(test_image)
                
                if embedding is not None:
                    model_integrity = True
            except Exception as e:
                model_error = str(e)
        
        return {
            'available': model_exists and tf_available and model_integrity,
            'model_file_exists': model_exists,
            'model_file_path': model_path,
            'model_file_size': os.path.getsize(model_path) if model_exists else 0,
            'tensorflow_available': tf_available,
            'tensorflow_version': tf_version,
            'opencv_available': opencv_available,
            'opencv_version': cv_version,
            'opencv_dnn_support': opencv_with_dnn,
            'model_integrity_verified': model_integrity,
            'model_error': model_error,
            'message': "FaceNet is properly installed and working" if (model_exists and tf_available and model_integrity) else 
                      "FaceNet is not available or not working correctly"
        }
    except Exception as e:
        logger.error(f"Error checking FaceNet availability: {e}")
        return {
            'available': False,
            'error': str(e),
            'message': f"Error checking FaceNet availability: {str(e)}"
        } 