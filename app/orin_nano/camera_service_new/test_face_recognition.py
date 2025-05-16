#!/usr/bin/env python3

"""
Test Face Recognition

This script tests the face recognition functionality by:
1. Connecting to a session
2. Enrolling a test wallet with your face
3. Attempting to recognize your face
4. Displaying the results

Run this script to verify that face recognition is working properly.
"""

import requests
import json
import time
import os

# Configuration
API_URL = "http://localhost:5003"
TEST_WALLET = "TEST_WALLET_123"
SESSION_ID = None

def print_section(title):
    """Print a section title"""
    print("\n" + "=" * 60)
    print(f" {title} ".center(60, "="))
    print("=" * 60)

def check_service_status():
    """Check if the camera service is running"""
    print_section("Checking Camera Service Status")
    
    try:
        response = requests.get(f"{API_URL}/health")
        status = response.json()
        
        print(f"Camera Service Status: {status.get('status', 'unknown')}")
        print(f"Buffer FPS: {status.get('buffer_fps', 'unknown')}")
        
        # For this service, we consider it healthy if we get a response
        return True
    except Exception as e:
        print(f"Error checking service status: {e}")
        return False

def connect_session():
    """Connect to a session"""
    print_section("Connecting to Session")
    global SESSION_ID
    
    try:
        # Connect to a session with the test wallet address
        response = requests.post(f"{API_URL}/connect", json={"wallet_address": TEST_WALLET})
        result = response.json()
        
        if result.get('success', False):
            SESSION_ID = result.get('session_id')
            print(f"Connected to session: {SESSION_ID}")
            print(f"Wallet Address: {result.get('wallet_address')}")
            return True
        else:
            print(f"Failed to connect: {result.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        print(f"Error connecting to session: {e}")
        return False

def check_face_recognition_status():
    """Check if face recognition is available"""
    print_section("Checking Face Recognition Status")
    
    try:
        # Visit the test page to see if face recognition is working
        response = requests.get(f"{API_URL}/test-page")
        
        if response.status_code == 200:
            print("Test page is accessible, which suggests the service is running")
            print("Face recognition status will be checked visually")
            print("Please visit http://localhost:5003/test-page in your browser")
            return True
        else:
            print(f"Error accessing test page: {response.status_code}")
            return False
    except Exception as e:
        print(f"Error checking face recognition status: {e}")
        return False

def enroll_face(wallet_address):
    """Enroll a face for the given wallet address"""
    print_section(f"Enrolling Face for {wallet_address}")
    print("Please look at the camera...")
    
    if not SESSION_ID:
        print("No active session, cannot enroll face")
        return False
    
    try:
        # Wait a moment for user to position themselves
        time.sleep(2)
        
        # Enroll face (this is a POST request with a JSON body)
        response = requests.post(f"{API_URL}/enroll_face", json={"wallet_address": wallet_address, "session_id": SESSION_ID})
        result = response.json()
        
        print(f"Enrollment Result: {result.get('success', False)}")
        if result.get('success', False):
            print(f"Wallet Address: {result.get('wallet_address', 'unknown')}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
        
        return result.get('success', False)
    except Exception as e:
        print(f"Error enrolling face: {e}")
        return False

def test_recognition():
    """Test face recognition by recognizing your face"""
    print_section("Testing Face Recognition")
    print("Please look at the camera...")
    
    if not SESSION_ID:
        print("No active session, cannot test recognition")
        return False
    
    try:
        # Wait a moment for recognition to happen
        time.sleep(3)
        
        # Recognize face
        response = requests.get(f"{API_URL}/recognize_face", params={"session_id": SESSION_ID})
        result = response.json()
        
        print(f"Recognition Result: {result.get('success', False)}")
        
        if result.get('recognized_count', 0) > 0:
            print("\nRecognized Faces:")
            for wallet, face_data in result.get('recognized_faces', {}).items():
                confidence = face_data[4] if len(face_data) > 4 else 0
                print(f"  - Wallet: {wallet}, Confidence: {confidence:.2f}")
            return True
        else:
            print("\nNo faces recognized.")
            return False
    except Exception as e:
        print(f"Error testing recognition: {e}")
        return False

def disconnect_session():
    """Disconnect from the session"""
    print_section("Disconnecting Session")
    
    if not SESSION_ID:
        print("No active session to disconnect")
        return True
    
    try:
        # Disconnect from the session
        response = requests.post(f"{API_URL}/disconnect", json={"session_id": SESSION_ID})
        result = response.json()
        
        print(f"Disconnection Result: {result.get('success', False)}")
        return result.get('success', False)
    except Exception as e:
        print(f"Error disconnecting session: {e}")
        return False

def main():
    """Main test function"""
    print_section("Face Recognition Test")
    print("This script will test the face recognition functionality.")
    
    # Check if service is running
    if not check_service_status():
        print("Camera service is not ready. Please make sure it's running.")
        return
    
    # Check face recognition status
    if not check_face_recognition_status():
        print("Face recognition status check failed.")
        return
    
    # Connect to a session
    if not connect_session():
        print("Failed to connect to a session. Please try again.")
        return
    
    # Enroll test face
    if not enroll_face(TEST_WALLET):
        print("Face enrollment failed. Please try again.")
        disconnect_session()
        return
    
    print("\nFace enrolled successfully! Now testing recognition...")
    time.sleep(2)
    
    # Test recognition
    if test_recognition():
        print("\nSuccess! Your face was recognized correctly.")
    else:
        print("\nFace recognition test failed. Your face was not recognized.")
    
    # Disconnect from the session
    disconnect_session()
    
    print_section("Test Complete")
    print("You can now use the face recognition in your application.")
    print("Visit http://localhost:5003/test-page to see the camera feed.")

if __name__ == "__main__":
    main() 