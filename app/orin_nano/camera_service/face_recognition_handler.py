"""
Face Recognition Handler Module (Standalone)

Handles face recognition and enrollment functions with encryption support.
"""

import os
import sys
import time
import uuid
import json
import cv2
import numpy as np
import traceback
import base64
import hashlib
from cryptography.fernet import Fernet

# Add paths to find system libraries
python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
potential_paths = [
    os.path.expanduser(f'~/.local/lib/python{python_version}/site-packages'),
    os.path.expanduser('~/.local/lib/python3.10/site-packages'),
    os.path.expanduser('~/.local/lib/python3.9/site-packages'),
    os.path.expanduser('~/.local/lib/python3.8/site-packages'),
    '/usr/local/lib/python3.10/dist-packages',
    '/usr/lib/python3/dist-packages'
]

# Add all potential paths to sys.path
for path in potential_paths:
    if os.path.exists(path) and path not in sys.path:
        sys.path.append(path)
        print(f"Added {path} to Python path")

# Directly import face_recognition
try:
    import face_recognition as fr
    FACE_RECOGNITION_AVAILABLE = True
    print("Face recognition library successfully imported in handler module")
except ImportError as e:
    print(f"Error importing face_recognition in handler module: {e}")
    fr = None
    FACE_RECOGNITION_AVAILABLE = False
    print("WARNING: face_recognition library not available in handler module")

# Try different import approaches for camera module
try:
    # First try relative import
    from . import config
    from . import camera
    print("Using relative imports for config and camera modules")
except ImportError:
    try:
        # Then try absolute import
        import jetson_system.camera_service.config as config
        import jetson_system.camera_service.camera as camera
        print("Using absolute imports for config and camera modules")
    except ImportError:
        try:
            # Direct import as last resort
            import config
            import camera
            print("Using direct imports for config and camera modules")
        except ImportError as e:
            print(f"Failed to import config or camera: {e}")
            # Create stub objects if needed
            if 'config' not in globals():
                print("Creating stub config object")
                import types
                config = types.SimpleNamespace()
                config.DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
                config.create_directories = lambda: None

# Dictionary to store enrolled faces with embeddings
enrolled_faces = {}

# Path to save/load enrolled faces
FACES_FILE = os.path.join(config.DATA_DIR, 'enrolled_faces.json')
FACES_DIR = os.path.join(config.DATA_DIR, 'faces')
KEY_FILE = os.path.join(config.DATA_DIR, 'face_encryption.key')

# Create directory if it doesn't exist
if not os.path.exists(FACES_DIR):
    os.makedirs(FACES_DIR, exist_ok=True)

# Get or create encryption key
def get_encryption_key():
    """Get or create the encryption key for face embeddings"""
    if os.path.exists(KEY_FILE):
        with open(KEY_FILE, 'rb') as f:
            return f.read()
    else:
        # Generate a key from a random password
        password = os.urandom(16)
        key = base64.urlsafe_b64encode(hashlib.sha256(password).digest())
        with open(KEY_FILE, 'wb') as f:
            f.write(key)
        return key

# Initialize encryption
try:
    encryption_key = get_encryption_key()
    cipher = Fernet(encryption_key)
    print("Facial embedding encryption initialized")
    ENCRYPTION_AVAILABLE = True
except Exception as e:
    print(f"Error initializing encryption: {e}")
    ENCRYPTION_AVAILABLE = False
    cipher = None

# Encrypt and decrypt face embeddings
def encrypt_embedding(embedding):
    """Encrypt face embedding"""
    if not ENCRYPTION_AVAILABLE:
        return embedding.tolist()  # Return as-is if encryption not available
    
    try:
        # Convert to JSON-compatible format first
        embedding_bytes = json.dumps(embedding.tolist()).encode()
        encrypted = cipher.encrypt(embedding_bytes)
        return base64.b64encode(encrypted).decode()  # Store as base64 string
    except Exception as e:
        print(f"Error encrypting embedding: {e}")
        return embedding.tolist()  # Fallback to unencrypted

def decrypt_embedding(encrypted_data):
    """Decrypt face embedding"""
    if not ENCRYPTION_AVAILABLE or not isinstance(encrypted_data, str):
        return np.array(encrypted_data)  # Handle unencrypted data
    
    try:
        encrypted_bytes = base64.b64decode(encrypted_data)
        decrypted_bytes = cipher.decrypt(encrypted_bytes)
        embedding_list = json.loads(decrypted_bytes.decode())
        return np.array(embedding_list)
    except Exception as e:
        print(f"Error decrypting embedding: {e}")
        # If decryption fails, try treating it as unencrypted data
        if isinstance(encrypted_data, list):
            return np.array(encrypted_data)
        return None

# Load enrolled faces from file
def load_enrolled_faces():
    """Load enrolled faces from JSON file"""
    global enrolled_faces
    
    if not FACE_RECOGNITION_AVAILABLE:
        print("Cannot load enrolled faces: face_recognition library not available")
        return
    
    if os.path.exists(FACES_FILE):
        try:
            with open(FACES_FILE, 'r') as f:
                enrolled_faces = json.load(f)
            print(f"Loaded {len(enrolled_faces)} enrolled faces")
        except Exception as e:
            print(f"Error loading enrolled faces: {e}")
            enrolled_faces = {}
    else:
        print("No enrolled faces file found. Starting with empty database.")
        enrolled_faces = {}

# Save enrolled faces to file
def save_enrolled_faces():
    """Save enrolled faces to JSON file"""
    try:
        with open(FACES_FILE, 'w') as f:
            json.dump(enrolled_faces, f)
        print(f"Saved {len(enrolled_faces)} enrolled faces")
    except Exception as e:
        print(f"Error saving enrolled faces: {e}")

# Enroll a face
def enroll_face(display_name="Unknown"):
    """Enroll a face for future recognition"""
    if not FACE_RECOGNITION_AVAILABLE:
        print("Face recognition is not available, cannot enroll face")
        return False, "Face recognition library not available"
    
    try:
        # Capture current frame
        frame = camera.get_raw_frame()
        if frame is None:
            return False, "No camera frame available"
        
        # Convert to RGB (face_recognition uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = fr.face_locations(rgb_frame)
        
        if not face_locations:
            return False, "No face detected in the frame"
        
        # Use the first face if multiple are detected
        face_location = face_locations[0]
        
        # Generate face encoding
        face_encoding = fr.face_encodings(rgb_frame, [face_location])[0]
        
        # Create unique face ID
        face_id = str(uuid.uuid4())
        
        # Encrypt the face encoding before saving
        encrypted_encoding = encrypt_embedding(face_encoding)
        
        # Save face data
        enrolled_faces[face_id] = {
            "name": display_name,
            "encoding": encrypted_encoding,  # Store encrypted encoding
            "enrolled_at": int(time.time()),
            "face_id": face_id,
            "encrypted": ENCRYPTION_AVAILABLE
        }
        
        # Save updated face database
        save_enrolled_faces()
        
        # Save face image
        face_img_path = os.path.join(FACES_DIR, f"{face_id}.jpg")
        top, right, bottom, left = face_location
        face_img = frame[top:bottom, left:right]
        cv2.imwrite(face_img_path, face_img)
        
        return True, {"face_id": face_id, "name": display_name}
    except Exception as e:
        traceback.print_exc()
        return False, f"Error enrolling face: {str(e)}"

# Recognize a face
def recognize_face():
    """Recognize a face against enrolled faces"""
    global enrolled_faces
    
    if not FACE_RECOGNITION_AVAILABLE:
        print("Face recognition is not available, cannot recognize face")
        return False, "Face recognition library not available"
    
    if not enrolled_faces:
        return False, "No faces enrolled yet"
    
    try:
        # Capture current frame
        frame = camera.get_raw_frame()
        if frame is None:
            return False, "No camera frame available"
        
        # Convert to RGB (face_recognition uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = fr.face_locations(rgb_frame)
        
        if not face_locations:
            return False, "No face detected in the frame"
        
        # Use the first face if multiple are detected
        face_location = face_locations[0]
        
        # Generate face encoding
        face_encoding = fr.face_encodings(rgb_frame, [face_location])[0]
        
        # Compare with enrolled faces
        best_match = None
        best_distance = 0.6  # Threshold for face recognition (lower is more strict)
        
        for face_id, face_data in enrolled_faces.items():
            try:
                # Decrypt stored encoding
                stored_encoding = decrypt_embedding(face_data["encoding"])
                
                if stored_encoding is None:
                    print(f"Failed to decrypt encoding for face {face_id}")
                    continue
                
                # Compute face distance
                distance = fr.face_distance([stored_encoding], face_encoding)[0]
                
                # Check if this is a better match
                if distance < best_distance:
                    best_distance = distance
                    best_match = face_data
            except Exception as e:
                print(f"Error comparing with face {face_id}: {e}")
        
        if best_match:
            # Update the identified_user variable in the camera module
            camera.identified_user = best_match["name"]
            return True, {
                "face_id": best_match["face_id"], 
                "name": best_match["name"], 
                "confidence": 1.0 - best_distance,
                "face_recognized": True
            }
        else:
            # Reset the identified_user variable
            camera.identified_user = None
            return False, "No matching face found"
    except Exception as e:
        traceback.print_exc()
        return False, f"Error recognizing face: {str(e)}"

# List all enrolled faces
def list_enrolled_faces():
    """List all enrolled faces"""
    if not FACE_RECOGNITION_AVAILABLE:
        return {"success": False, "error": "Face recognition library not available"}
        
    faces_list = []
    
    for face_id, face_data in enrolled_faces.items():
        # Create a copy without the encoding (too large for JSON response)
        face_info = {
            "face_id": face_id,
            "name": face_data["name"],
            "enrolled_at": face_data.get("enrolled_at", 0),
            "encrypted": face_data.get("encrypted", False)
        }
        
        # Add face image path if it exists
        face_img_path = os.path.join(FACES_DIR, f"{face_id}.jpg")
        if os.path.exists(face_img_path):
            face_info["image_path"] = face_img_path
            
        faces_list.append(face_info)
    
    return {"success": True, "faces": faces_list}

# Clear all enrolled faces
def clear_all_faces():
    """Clear all enrolled faces"""
    global enrolled_faces
    
    if not FACE_RECOGNITION_AVAILABLE:
        return {"success": False, "error": "Face recognition library not available"}
    
    try:
        # Count before clearing
        count_before = len(enrolled_faces)
        
        # Store face IDs to delete
        face_ids = list(enrolled_faces.keys())
        
        # Clear enrolled faces dictionary
        enrolled_faces = {}
        
        # Save empty database immediately
        save_enrolled_faces()
        
        # Delete face images
        deleted_count = 0
        for face_id in face_ids:
            face_img_path = os.path.join(FACES_DIR, f"{face_id}.jpg")
            try:
                if os.path.exists(face_img_path):
                    os.remove(face_img_path)
                    deleted_count += 1
            except Exception as e:
                print(f"Error deleting face image {face_id}: {e}")
        
        # Check if the enrolled_faces.json file exists and ensure it's empty
        if os.path.exists(FACES_FILE):
            try:
                # Verify the file is empty or contains only {}
                with open(FACES_FILE, 'w') as f:
                    f.write("{}")
            except Exception as e:
                print(f"Error writing empty faces file: {e}")
        
        # Reset the identified_user in camera module
        if 'camera' in globals() and hasattr(camera, 'identified_user'):
            camera.identified_user = None
            
        print(f"Successfully cleared {count_before} faces, deleted {deleted_count} face images")
        
        return {
            "success": True, 
            "cleared_count": count_before,
            "deleted_images": deleted_count
        }
    except Exception as e:
        print(f"Error in clear_all_faces: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

# Check if user is currently recognized
def is_user_recognized():
    """Check if a user face is currently recognized"""
    return camera.identified_user is not None, camera.identified_user 