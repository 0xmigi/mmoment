// API Test JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize state
    let connected = false;
    let username = '';
    let faceBoxesEnabled = true;
    let gestureTagsEnabled = true;
    
    // Face recognition functions
    document.getElementById('recognizeBtn').addEventListener('click', function() {
        recognizeFace();
    });
    
    document.getElementById('enrollBtn').addEventListener('click', function() {
        enrollFace();
    });
    
    document.getElementById('listFacesBtn').addEventListener('click', function() {
        listEnrolledFaces();
    });
    
    document.getElementById('clearFacesBtn').addEventListener('click', function() {
        clearAllFaces();
    });
    
    // User connection functions
    document.getElementById('connectBtn').addEventListener('click', function() {
        connectUser();
    });
    
    document.getElementById('disconnectBtn').addEventListener('click', function() {
        disconnectUser();
    });
    
    // System control functions
    document.getElementById('Enable').addEventListener('click', function() {
        toggleFaceBoxes(true);
    });
    
    document.getElementById('Disable').addEventListener('click', function() {
        toggleFaceBoxes(false);
    });
    
    document.getElementById('gestureTagsEnableBtn').addEventListener('click', function() {
        toggleGestureTags(true);
    });
    
    document.getElementById('gestureTagsDisableBtn').addEventListener('click', function() {
        toggleGestureTags(false);
    });
    
    document.getElementById('healthBtn').addEventListener('click', function() {
        testHealth();
    });
    
    document.getElementById('streamStatusBtn').addEventListener('click', function() {
        checkStreamStatus();
    });
    
    // Add refresh stream button handler
    document.getElementById('refreshStreamBtn').addEventListener('click', function() {
        refreshStream();
    });
    
    // Capture functions
    document.getElementById('captureBtn').addEventListener('click', function() {
        captureMoment();
    });
    
    document.getElementById('clearGalleryBtn').addEventListener('click', function() {
        clearGallery();
    });
    
    document.getElementById('startRecordingBtn').addEventListener('click', function() {
        startRecording();
    });
    
    document.getElementById('stopRecordingBtn').addEventListener('click', function() {
        stopRecording();
    });
    
    document.getElementById('resetRecordingBtn').addEventListener('click', function() {
        resetRecording();
    });
    
    document.getElementById('viewVideosBtn').addEventListener('click', function() {
        viewAllVideos();
    });
    
    document.getElementById('clearVideosBtn').addEventListener('click', function() {
        clearVideos();
    });
    
    // Function to toggle face boxes with enhanced reliability
    function toggleFaceBoxes(enable) {
        // Directly set URL parameters for clarity
        const url = `/toggle-face-visualization?enable=${enable}`;
        
        console.log(`Toggling face boxes: ${enable ? 'ENABLE' : 'DISABLE'}`);
        
        // Show immediate feedback
        const statusEl = document.getElementById('controlResponses');
        if (statusEl) {
            statusEl.textContent = `Sending request to ${enable ? 'enable' : 'disable'} face boxes...`;
        }
        
        // Update UI state immediately for responsiveness
        document.getElementById('enableFaceBoxes').classList.toggle('active', enable);
        document.getElementById('disableFaceBoxes').classList.toggle('active', !enable);
        
        // Update active-button class for visual feedback
        document.getElementById('enableFaceBoxes').classList.toggle('active-button', enable);
        document.getElementById('disableFaceBoxes').classList.toggle('active-button', !enable);
        
        // Make direct fetch with cache busting to prevent cached responses
        fetch(url + '&t=' + new Date().getTime(), { 
            method: 'GET',
            cache: 'no-cache',
            headers: {
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            }
        })
        .then(response => {
            if (!response.ok) {
                // If GET fails, try POST instead as fallback
                console.log('GET request failed, trying POST...');
                return fetch('/toggle-face-visualization', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        state: enable,
                        enable: enable 
                    })
                });
            }
            return response;
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Network response was not ok: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Toggle face boxes response:', data);
            
            // Check if the toggle was successful
            if (data.success) {
                // Use the returned state to update our UI
                const actualState = data.face_visualization || data.state || enable;
                
                // Update status message
                if (statusEl) {
                    statusEl.textContent = actualState ? 
                        'Face boxes enabled successfully' : 
                        'Face boxes disabled successfully';
                }
                    
                // Update UI to match actual server state
                document.getElementById('enableFaceBoxes').classList.toggle('active', actualState);
                document.getElementById('disableFaceBoxes').classList.toggle('active', !actualState);
                
                // Update active-button class to match actual state
                document.getElementById('enableFaceBoxes').classList.toggle('active-button', actualState);
                document.getElementById('disableFaceBoxes').classList.toggle('active-button', !actualState);
                
                // Store the current state globally
                window.showFaceBoxes = actualState;
                
                // Force refresh the stream to apply changes immediately
                refreshStream();
            } else {
                // Handle error
                if (statusEl) {
                    statusEl.textContent = `Error: ${data.error || 'Failed to toggle face boxes'}`;
                }
                
                // Revert UI changes
                document.getElementById('enableFaceBoxes').classList.toggle('active', !enable);
                document.getElementById('disableFaceBoxes').classList.toggle('active', enable);
                document.getElementById('enableFaceBoxes').classList.toggle('active-button', !enable);
                document.getElementById('disableFaceBoxes').classList.toggle('active-button', enable);
            }
        })
        .catch(error => {
            console.error('Error toggling face boxes:', error);
            if (statusEl) {
                statusEl.textContent = `Error: ${error.message}`;
            }
            
            // Try a more direct approach as fallback
            console.log('Trying fallback approach via config endpoint...');
            fetch('/set-config', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ enable_face_visualization: enable })
            })
            .then(response => response.json())
            .then(data => {
                console.log('Set config directly as fallback, result:', data);
                if (data.success) {
                    if (statusEl) {
                        statusEl.textContent = enable ? 
                            'Face boxes enabled (via fallback)' : 
                            'Face boxes disabled (via fallback)';
                    }
                    window.showFaceBoxes = enable;
                    
                    // Update UI classes
                    document.getElementById('enableFaceBoxes').classList.toggle('active-button', enable);
                    document.getElementById('disableFaceBoxes').classList.toggle('active-button', !enable);
                }
                // Force refresh the stream to apply changes
                setTimeout(refreshStream, 200);
            })
            .catch(err => {
                console.error('Fallback also failed:', err);
                if (statusEl) {
                    statusEl.textContent = `Error: ${err.message}`;
                }
                // Revert UI state on complete failure
                document.getElementById('enableFaceBoxes').classList.toggle('active-button', !enable);
                document.getElementById('disableFaceBoxes').classList.toggle('active-button', enable);
            });
        });
    }
    
    // Function to refresh the stream with reliable forcing
    function refreshStream() {
        const streamImg = document.getElementById('stream');
        if (streamImg) {
            try {
                // First, try to force refresh on server
                fetch('/refresh-stream', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        _timestamp: Date.now() // Add timestamp to prevent caching
                    })
                })
                .then(response => response.json())
                .then(data => {
                    console.log('Refresh stream response:', data);
                })
                .catch(err => {
                    console.log('Refresh stream request failed silently:', err);
                });
                
                // Then, update the client-side image with cache busting
                const timestamp = new Date().getTime();
                const currentSrc = streamImg.src;
                
                // Reset the source to force a full reload
                streamImg.src = '';
                
                // Use setTimeout to ensure the browser processes the empty src first
                setTimeout(() => {
                    // Create a new URL, removing any existing timestamp parameter
                    let newSrc = currentSrc.split('?')[0]; 
                    newSrc = `${newSrc}?t=${timestamp}`;
                    streamImg.src = newSrc;
                    console.log('Stream refreshed with timestamp:', timestamp);
                }, 100);
                
                // Additional forced refresh after a short delay
                setTimeout(() => {
                    const newTimestamp = new Date().getTime();
                    streamImg.src = `/stream?t=${newTimestamp}`;
                }, 500);
            } catch (e) {
                console.error('Error refreshing stream:', e);
                // Fallback simple refresh
                streamImg.src = `/stream?t=${new Date().getTime()}`;
            }
        } else {
            console.warn('No stream image found to refresh');
        }
    }
    
    // Function to toggle gesture tags
    function toggleGestureTags(enable) {
        const url = enable ? '/toggle-gesture-visualization?enable=true' : '/toggle-gesture-visualization?disable=true';
        
        fetch(url)
            .then(response => response.json())
            .then(data => {
                console.log('Toggle gesture tags response:', data);
                gestureTagsEnabled = enable;
                
                const statusEl = document.getElementById('controlResponses');
                statusEl.textContent = enable ? 
                    'Gesture tags enabled' : 
                    'Gesture tags disabled';
                
                // Update UI state
                document.getElementById('gestureTagsEnableBtn').classList.toggle('active', enable);
                document.getElementById('gestureTagsDisableBtn').classList.toggle('active', !enable);
            })
            .catch(error => {
                console.error('Error toggling gesture tags:', error);
            });
    }
    
    // Function to test health endpoint
    function testHealth() {
        fetch('/health')
            .then(response => response.json())
            .then(data => {
                console.log('Health response:', data);
                
                const statusEl = document.getElementById('controlResponses');
                statusEl.textContent = JSON.stringify(data, null, 2);
            })
            .catch(error => {
                console.error('Error checking health:', error);
            });
    }
    
    // Function to check stream status
    function checkStreamStatus() {
        fetch('/camera-info')
            .then(response => response.json())
            .then(data => {
                console.log('Stream status response:', data);
                
                const statusEl = document.getElementById('controlResponses');
                statusEl.textContent = JSON.stringify(data, null, 2);
            })
            .catch(error => {
                console.error('Error checking stream status:', error);
            });
    }
    
    // Function to connect user
    function connectUser() {
        const usernameInput = document.getElementById('usernameInput').value.trim();
        if (!usernameInput) {
            alert('Please enter a username');
            return;
        }
        
        const url = `/connect?display_name=${encodeURIComponent(usernameInput)}`;
        
        fetch(url)
            .then(response => response.json())
            .then(data => {
                console.log('Connect response:', data);
                
                if (data.success) {
                    connected = true;
                    username = usernameInput;
                    
                    // Update UI
                    document.getElementById('connectionStatus').textContent = 'Connected';
                    document.getElementById('connectionStatus').className = '';
                    document.getElementById('connectBtn').disabled = true;
                    document.getElementById('disconnectBtn').disabled = false;
                }
            })
            .catch(error => {
                console.error('Error connecting user:', error);
            });
    }
    
    // Function to disconnect user
    function disconnectUser() {
        fetch('/disconnect')
            .then(response => response.json())
            .then(data => {
                console.log('Disconnect response:', data);
                
                connected = false;
                username = '';
                
                // Update UI
                document.getElementById('connectionStatus').textContent = 'Not Connected';
                document.getElementById('connectionStatus').className = 'not-connected';
                document.getElementById('connectBtn').disabled = false;
                document.getElementById('disconnectBtn').disabled = true;
            })
            .catch(error => {
                console.error('Error disconnecting user:', error);
            });
    }
    
    // Function to recognize face with enhanced error handling
    function recognizeFace() {
        const statusEl = document.getElementById('faceRecognitionStatus');
        statusEl.textContent = "Recognizing face...";
        statusEl.className = "status in-progress";
        
        // Make POST request instead of GET for better reliability
        fetch('/recognize-face', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: 'api-test-session'
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Recognize face response:', data);
            
            if (data.success && data.face_recognized) {
                statusEl.textContent = `Face recognized: ${data.name} (${Math.round(data.confidence * 100)}% confidence)`;
                statusEl.className = "status success";
            } else {
                const errorMessage = data.error || "No face recognized";
                statusEl.textContent = `Error: ${errorMessage}`;
                statusEl.className = "status error";
                
                // If the error is about the library not being available, give helpful instructions
                if (errorMessage.includes("Face recognition library not available")) {
                    statusEl.textContent = "Error: Face recognition library not available. Please contact system administrator.";
                }
            }
        })
        .catch(error => {
            console.error('Error recognizing face:', error);
            statusEl.textContent = `Error: ${error.message}`;
            statusEl.className = "status error";
        });
    }
    
    // Function to enroll face with enhanced error handling
    function enrollFace() {
        const nameInput = document.getElementById('usernameInput').value.trim() || 'User';
        const statusEl = document.getElementById('faceRecognitionStatus');
        statusEl.textContent = "Enrolling face...";
        statusEl.className = "status in-progress";
        
        // Make POST request instead of GET for better reliability
        fetch('/enroll-face', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                display_name: nameInput,
                session_id: 'api-test-session'
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Enroll face response:', data);
            
            if (data.success) {
                statusEl.textContent = `Face enrolled successfully as: ${data.name || nameInput}`;
                statusEl.className = "status success";
                // Force refresh the stream to show face boxes with name
                refreshStream();
            } else {
                const errorMessage = data.error || "Face enrollment failed";
                statusEl.textContent = `Error: ${errorMessage}`;
                statusEl.className = "status error";
                
                // If the error is about the library not being available, give helpful instructions
                if (errorMessage.includes("Face recognition library not available")) {
                    statusEl.textContent = "Error: Face recognition library not available. Please contact system administrator.";
                } else if (errorMessage.includes("No face detected")) {
                    statusEl.textContent = "No face detected. Make sure your face is visible in the camera view.";
                }
            }
        })
        .catch(error => {
            console.error('Error enrolling face:', error);
            statusEl.textContent = `Error: ${error.message}`;
            statusEl.className = "status error";
        });
    }
    
    // Function to list enrolled faces
    function listEnrolledFaces() {
        fetch('/list-enrolled-faces')
            .then(response => response.json())
            .then(data => {
                console.log('List faces response:', data);
                
                const statusEl = document.getElementById('faceRecognitionStatus');
                if (data.success) {
                    if (data.faces.length > 0) {
                        statusEl.textContent = `${data.faces.length} face(s) enrolled: ${data.faces.map(f => f.name).join(', ')}`;
                    } else {
                        statusEl.textContent = 'No faces enrolled yet';
                    }
                } else {
                    statusEl.textContent = 'Failed to list faces';
                }
            })
            .catch(error => {
                console.error('Error listing faces:', error);
            });
    }
    
    // Function to clear all faces
    function clearAllFaces() {
        fetch('/clear-all-faces')
            .then(response => response.json())
            .then(data => {
                console.log('Clear faces response:', data);
                
                const statusEl = document.getElementById('faceRecognitionStatus');
                if (data.success) {
                    statusEl.textContent = 'All faces cleared';
                } else {
                    statusEl.textContent = 'Failed to clear faces';
                }
            })
            .catch(error => {
                console.error('Error clearing faces:', error);
            });
    }
    
    // Function to capture moment
    function captureMoment() {
        // Implementation for capturing images
    }
    
    // Function to clear gallery
    function clearGallery() {
        // Implementation for clearing gallery
    }
    
    // Function to start recording
    function startRecording() {
        // Implementation for starting recording
    }
    
    // Function to stop recording
    function stopRecording() {
        // Implementation for stopping recording
    }
    
    // Function to reset recording
    function resetRecording() {
        // Implementation for resetting recording
    }
    
    // Function to view all videos
    function viewAllVideos() {
        // Implementation for viewing all videos
    }
    
    // Function to clear videos
    function clearVideos() {
        // Implementation for clearing videos
    }
    
    // Fetch initial status on page load
    function checkInitialStatus() {
        // Check face boxes status
        fetch('/visual-controls-status')
            .then(response => response.json())
            .then(data => {
                console.log('Visual controls status:', data);
                
                if (data.success) {
                    const faceBoxesEnabled = data.face_visualization;
                    const gestureTagsEnabled = data.gesture_visualization;
                    
                    // Store for global access
                    window.showFaceBoxes = faceBoxesEnabled;
                    
                    // Update UI state for face boxes
                    document.getElementById('enableFaceBoxes').classList.toggle('active-button', faceBoxesEnabled);
                    document.getElementById('disableFaceBoxes').classList.toggle('active-button', !faceBoxesEnabled);
                    
                    // Update UI state for gesture tags
                    document.getElementById('gestureTagsEnableBtn').classList.toggle('active-button', gestureTagsEnabled);
                    document.getElementById('gestureTagsDisableBtn').classList.toggle('active-button', !gestureTagsEnabled);
                    
                    // Update the face boxes status text if available
                    const faceBoxesStatus = document.getElementById('faceBoxesStatus');
                    if (faceBoxesStatus) {
                        faceBoxesStatus.textContent = faceBoxesEnabled ? 
                            "Face boxes enabled" : "Face boxes disabled";
                        faceBoxesStatus.style.color = faceBoxesEnabled ? "#4CAF50" : "#ff6347";
                    }
                    
                    console.log(`Initial state: face boxes ${faceBoxesEnabled ? 'enabled' : 'disabled'}, gesture tags ${gestureTagsEnabled ? 'enabled' : 'disabled'}`);
                }
            })
            .catch(error => {
                console.error('Error checking initial status:', error);
            });
    }
    
    // Call initial status check immediately on load
    document.addEventListener('DOMContentLoaded', function() {
        // Update face box toggle button references
        const enableFaceBtn = document.getElementById('Enable');
        const disableFaceBtn = document.getElementById('Disable');
        
        if (enableFaceBtn && disableFaceBtn) {
            // Set initial event listeners
            enableFaceBtn.addEventListener('click', function() {
                toggleFaceBoxes(true);
            });
            
            disableFaceBtn.addEventListener('click', function() {
                toggleFaceBoxes(false);
            });
            
            console.log('Face box toggle buttons initialized');
        } else {
            console.error('Could not find face box toggle buttons');
        }
        
        // Check initial status
        checkInitialStatus();
        
        // Set up regular refresh of the stream
        setInterval(refreshStream, 10000); // Refresh every 10 seconds to keep connection alive
    });
}); 