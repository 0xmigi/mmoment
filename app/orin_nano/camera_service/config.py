"""
Configuration Module

Contains configuration constants and utility functions.
"""

import os
import sys
import datetime
import time

# Debug flag
DEBUG = False

# PDA of this camera - used for API endpoints and Solana program
CAMERA_PDA = os.environ.get('CAMERA_PDA', 'WT9oJrL7sbNip8Rc2w5LoWFpwsUcZJnnJE2zZjMuVD')

# Program ID for the Solana program
PROGRAM_ID = os.environ.get('PROGRAM_ID', 'Hx5JaUCZXQqvcYzTcdgm9ZE3sqhMWqwAhNXZBrzWm4S')

# Directory paths
BASE_DIR = os.path.expanduser("~/jetson_system")
DATA_DIR = os.path.join(BASE_DIR, "data")
CAMERA_IMAGES_DIR = os.path.join(DATA_DIR, "images")
CAMERA_VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
FACES_DIR = os.path.join(DATA_DIR, "faces")
FACE_EMBEDDINGS_DIR = os.path.join(DATA_DIR, "face_embeddings")
CONFIG_DIR = os.path.join(DATA_DIR, "config")

# Alternate directory to ensure videos are saved in multiple locations
ALT_VIDEOS_DIR = os.path.join(DATA_DIR, "videos_backup")

# Feature flags (these can be dynamically changed at runtime)
enable_face_detection = True
enable_face_visualization = True
enable_gesture_visualization = True
disable_all_header_visualization = False

# MediaPipe hand detection settings
min_detection_confidence = 0.5
min_tracking_confidence = 0.5

# Recording settings
MAX_RECORDING_TIME = 60  # Maximum duration of recordings in seconds

# Add timestamps to log output
def log(message):
    """Print a log message with timestamp"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

# Create required directories
def create_directories():
    """Create required directories if they don't exist"""
    for directory in [DATA_DIR, CAMERA_IMAGES_DIR, CAMERA_VIDEOS_DIR, FACES_DIR, FACE_EMBEDDINGS_DIR, CONFIG_DIR, ALT_VIDEOS_DIR]:
        os.makedirs(directory, exist_ok=True)

# Save a configuration value to persistent storage
def save_config(key, value):
    """Save a configuration value to a file"""
    # Make sure CONFIG_DIR exists
    os.makedirs(CONFIG_DIR, exist_ok=True)
    
    # Write to file
    with open(os.path.join(CONFIG_DIR, f"{key}.conf"), 'w') as f:
        f.write(str(value))

# Load a configuration value from persistent storage
def load_config(key, default_value=None):
    """Load a configuration value from a file"""
    # Check if file exists
    file_path = os.path.join(CONFIG_DIR, f"{key}.conf")
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                value = f.read().strip()
                print(f"Loaded config {key}: '{value}'")
                return value
        except Exception as e:
            print(f"Error loading config {key}: {e}")
    
    return default_value

# Dedicated function to load a boolean config value
def load_bool_config(key, default_value=False):
    """Load a boolean configuration value"""
    value = load_config(key, str(default_value))
    
    # If it's already a boolean, return it
    if isinstance(value, bool):
        return value
    
    # Otherwise, convert string to boolean
    if value is not None:
        is_true = value.lower() in ('true', '1', 'yes', 'on', 't')
        print(f"Converting config {key} from '{value}' to boolean: {is_true}")
        return is_true
    
    return default_value

# Dictionary of gesture labels for UI display
gesture_labels = {
    "none": "No gesture",
    "thumbs_up": "Thumbs Up (start recording)",
    "peace": "Peace Sign (take photo)",
    "open_palm": "Open Palm (stop recording)",
    "pointing": "Pointing",
    "fist": "Fist"
}

# List of allowed gestures for detection
allowed_gestures = ["thumbs_up", "peace", "open_palm", "pointing", "fist"]

# Load configuration values at startup
def load_all_configs():
    """Load all configuration values from files"""
    global enable_face_detection, enable_face_visualization, enable_gesture_visualization
    
    # Create directories if they don't exist
    create_directories()
    
    # Load feature flags
    face_detection = load_bool_config("enable_face_detection")
    if face_detection is not None:
        enable_face_detection = face_detection
    
    face_vis = load_bool_config("enable_face_visualization")
    if face_vis is not None:
        enable_face_visualization = face_vis
    
    gesture_vis = load_bool_config("enable_gesture_visualization")
    if gesture_vis is not None:
        enable_gesture_visualization = gesture_vis
    
    # Log loaded configurations
    log(f"Loaded configuration values:")
    log(f"  enable_face_detection: {enable_face_detection}")
    log(f"  enable_face_visualization: {enable_face_visualization}")
    log(f"  enable_gesture_visualization: {enable_gesture_visualization}")

# Save all configuration values
def save_all_configs():
    """Save all configuration values to files"""
    # Save feature flags
    save_config("enable_face_detection", str(enable_face_detection))
    save_config("enable_face_visualization", str(enable_face_visualization))
    save_config("enable_gesture_visualization", str(enable_gesture_visualization))
    
    # Log saved configurations
    log(f"Saved configuration values:")
    log(f"  enable_face_detection: {enable_face_detection}")
    log(f"  enable_face_visualization: {enable_face_visualization}")
    log(f"  enable_gesture_visualization: {enable_gesture_visualization}")

# Initialize at import time
create_directories()
load_all_configs() 