/**
 * Pi5 Camera Implementation
 * 
 * This implementation wraps the existing Pi5 camera service to follow
 * the standardized camera interface.
 */

import { 
  ICamera, 
  CameraCapabilities, 
  CameraStatus, 
  CameraStreamInfo, 
  CameraActionResponse, 
  CameraMediaResponse, 
  CameraGestureResponse, 
  CameraSession 
} from '../camera-interface';

export class Pi5Camera implements ICamera {
  public readonly cameraId: string;
  public readonly cameraType: string = 'pi5';
  public readonly apiUrl: string;
  
  private debugMode = true;
  private currentSession: CameraSession | null = null;

  constructor(cameraId: string, apiUrl: string) {
    this.cameraId = cameraId;
    this.apiUrl = apiUrl;
    this.log('Pi5Camera initialized', { cameraId, apiUrl });
  }

  private log(...args: any[]) {
    if (this.debugMode) {
      console.log(`[Pi5Camera:${this.cameraId.slice(0, 8)}]`, ...args);
    }
  }

  private async makeApiCall(endpoint: string, method: string, data?: any): Promise<Response> {
    const url = `${this.apiUrl}${endpoint}`;
    this.log(`Making ${method} request to: ${url}`);
    
    try {
      const response = await fetch(url, {
        method,
        headers: {
          'Content-Type': 'application/json'
        },
        mode: 'cors',
        credentials: 'omit',
        body: method !== 'GET' && data ? JSON.stringify(data) : undefined
      });
      
      this.log(`Response status: ${response.status} ${response.statusText}`);
      return response;
    } catch (error) {
      this.log('Fetch error:', error);
      throw error;
    }
  }

  getCapabilities(): CameraCapabilities {
    return {
      canTakePhotos: true,
      canRecordVideos: true,
      canStream: true,
      canDetectGestures: false,
      canRecognizeFaces: false,
      hasLivepeerStreaming: false,
      supportedStreamFormats: ['mjpeg']
    };
  }

  async connect(walletAddress?: string): Promise<CameraActionResponse<CameraSession>> {
    try {
      this.log('Connecting to Pi5 camera', { walletAddress });
      
      // Pi5 cameras don't have session management, so we simulate it
      this.currentSession = {
        sessionId: `pi5_${Date.now()}`,
        walletAddress: walletAddress || 'anonymous',
        cameraPda: this.cameraId,
        timestamp: Date.now(),
        isActive: true
      };
      
      // Test connection to make sure camera is available
      const testResult = await this.testConnection();
      if (!testResult.success) {
        this.currentSession = null;
        return {
          success: false,
          error: testResult.error || 'Failed to connect to Pi5 camera'
        };
      }
      
      this.log('Connected successfully', this.currentSession);
      return {
        success: true,
        data: this.currentSession
      };
    } catch (error) {
      this.log('Connection error:', error);
      this.currentSession = null;
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to connect'
      };
    }
  }

  async disconnect(): Promise<CameraActionResponse> {
    try {
      this.log('Disconnecting from Pi5 camera');
      this.currentSession = null;
      this.log('Disconnected successfully');
      return { success: true };
    } catch (error) {
      this.log('Disconnect error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to disconnect'
      };
    }
  }

  async testConnection(): Promise<CameraActionResponse<{ message: string; url: string }>> {
    const url = `${this.apiUrl}/api/health`;
    
    try {
      const response = await fetch(url, {
        method: 'GET',
        mode: 'cors',
        credentials: 'omit'
      });
      
      if (response.ok) {
        const data = await response.json();
        return {
          success: true,
          data: {
            message: `Pi5 camera service is healthy (${data.status || 'ok'})`,
            url
          }
        };
      } else {
        return {
          success: false,
          error: `HTTP ${response.status}: ${response.statusText}`,
          data: { message: '', url }
        };
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Connection failed',
        data: { message: '', url }
      };
    }
  }

  isConnected(): boolean {
    return this.currentSession !== null;
  }

  async getStatus(): Promise<CameraActionResponse<CameraStatus>> {
    try {
      const response = await this.makeApiCall('/status', 'GET');
      
      if (response.ok) {
        const data = await response.json();
        return {
          success: true,
          data: {
            isOnline: true,
            isStreaming: data.streaming || false,
            isRecording: data.recording || false,
            lastSeen: Date.now(),
            owner: this.currentSession?.walletAddress
          }
        };
      } else {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get status',
        data: {
          isOnline: false,
          isStreaming: false,
          isRecording: false,
          lastSeen: Date.now()
        }
      };
    }
  }

  async takePhoto(): Promise<CameraActionResponse<CameraMediaResponse>> {
    if (!this.currentSession) {
      return {
        success: false,
        error: 'No active session. Please connect first.'
      };
    }

    try {
      this.log('Taking photo with Pi5 DePIN verification');
      
      // Create blockchain transaction for DePIN verification
      const sendSimpleTransaction = (window as any).sendSimpleTransaction;
      if (!sendSimpleTransaction) {
        return {
          success: false,
          error: 'Pi5 camera requires blockchain transaction verification. Transaction system not available.'
        };
      }

      const txSignature = await sendSimpleTransaction('photo_captured');
      if (!txSignature) {
        return {
          success: false,
          error: 'Failed to create blockchain transaction for photo capture'
        };
      }

      this.log(`Created transaction: ${txSignature}`);
      
      // Call Pi5 middleware with transaction verification
      const response = await fetch(`${this.apiUrl}/api/capture`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        mode: 'cors',
        credentials: 'omit',
        body: JSON.stringify({
          tx_signature: txSignature,
          wallet_address: this.currentSession.walletAddress
        })
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        this.log(`Camera API Error: ${response.status} - ${errorText}`);
        return {
          success: false,
          error: `Camera API error: ${response.status} ${errorText}`
        };
      }
      
      const imageBlob = await response.blob();
      this.log(`Photo captured successfully: ${imageBlob.size} bytes, ${imageBlob.type}`);
      
      // Upload to IPFS and handle storage like the working camera-action-service.ts
      try {
        const { unifiedIpfsService } = await import('../../storage/ipfs/unified-ipfs-service');
        const results = await unifiedIpfsService.uploadFile(imageBlob, this.currentSession.walletAddress, 'image');
        
        if (results.length === 0) {
          return { 
            success: false, 
            error: 'Failed to upload image to IPFS' 
          };
        }

        // Save transaction ID with the media object
        results.forEach(media => {
          media.transactionId = txSignature;
          media.cameraId = this.cameraId;
        });
        
        // Store the transaction ID in localStorage
        try {
          const mediaTransactionsKey = `mediaTransactions_${this.currentSession.walletAddress}`;
          const existingDataStr = localStorage.getItem(mediaTransactionsKey) || '{}';
          const existingData = JSON.parse(existingDataStr);
          
          results.forEach(media => {
            existingData[media.id] = {
              transactionId: txSignature,
              cameraId: this.cameraId,
              timestamp: media.timestamp,
              type: 'photo'
            };
          });
          
          localStorage.setItem(mediaTransactionsKey, JSON.stringify(existingData));
          this.log(`Saved transaction ID ${txSignature} for media ID ${results[0].id}`);
        } catch (e) {
          this.log(`Error saving transaction ID to localStorage: ${e}`);
        }
        
        return {
          success: true,
          data: { 
            blob: imageBlob,
            timestamp: Date.now(),
            size: imageBlob.size
          }
        };
      } catch (uploadError) {
        this.log('IPFS upload error:', uploadError);
        // Still return success with the blob even if IPFS fails
        return {
          success: true,
          data: { 
            blob: imageBlob,
            timestamp: Date.now(),
            size: imageBlob.size
          }
        };
      }
    } catch (error) {
      this.log('Photo capture error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to take photo'
      };
    }
  }

  async startVideoRecording(): Promise<CameraActionResponse<CameraMediaResponse>> {
    if (!this.currentSession) {
      return {
        success: false,
        error: 'No active session. Please connect first.'
      };
    }

    try {
      this.log('Starting video recording with Pi5 DePIN verification');
      
      // Create blockchain transaction for DePIN verification
      const sendSimpleTransaction = (window as any).sendSimpleTransaction;
      if (!sendSimpleTransaction) {
        return {
          success: false,
          error: 'Pi5 camera requires blockchain transaction verification. Transaction system not available.'
        };
      }

      const txSignature = await sendSimpleTransaction('video_started');
      if (!txSignature) {
        return {
          success: false,
          error: 'Failed to create blockchain transaction for video recording'
        };
      }

      this.log(`Created transaction: ${txSignature}`);
      
      // Call Pi5 middleware with transaction verification
      const response = await fetch(`${this.apiUrl}/api/record`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        mode: 'cors',
        credentials: 'omit',
        body: JSON.stringify({
          tx_signature: txSignature,
          wallet_address: this.currentSession.walletAddress,
          duration: 30 // 30-second duration
        })
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        this.log(`Camera API Error: ${response.status} - ${errorText}`);
        return {
          success: false,
          error: `Camera API error: ${response.status} ${errorText}`
        };
      }
      
      const recordingData = await response.json();
      this.log(`Recording completed, status: ${recordingData.status}`);
      
      if (recordingData.status !== 'recorded') {
        return {
          success: false,
          error: `Recording failed: ${recordingData.status}`
        };
      }
      
      // Pi5 records immediately, so we can try to get the video file now
      if (recordingData.filename) {
        try {
          const videoResponse = await this.getMostRecentVideo();
          if (videoResponse.success && videoResponse.data?.blob) {
            // Upload to IPFS like the photo capture
            try {
              const { unifiedIpfsService } = await import('../../storage/ipfs/unified-ipfs-service');
              const results = await unifiedIpfsService.uploadFile(
                videoResponse.data.blob,
                this.currentSession.walletAddress,
                'video'
              );
              
              if (results.length > 0) {
                // Save transaction ID with the media object
                results.forEach(media => {
                  media.transactionId = txSignature;
                  media.cameraId = this.cameraId;
                });
                
                // Store the transaction ID in localStorage
                try {
                  const mediaTransactionsKey = `mediaTransactions_${this.currentSession.walletAddress}`;
                  const existingDataStr = localStorage.getItem(mediaTransactionsKey) || '{}';
                  const existingData = JSON.parse(existingDataStr);
                  
                  results.forEach(media => {
                    existingData[media.id] = {
                      transactionId: txSignature,
                      cameraId: this.cameraId,
                      timestamp: media.timestamp,
                      type: 'video'
                    };
                  });
                  
                  localStorage.setItem(mediaTransactionsKey, JSON.stringify(existingData));
                  this.log(`Saved transaction ID ${txSignature} for video media ID ${results[0].id}`);
                } catch (e) {
                  this.log(`Error saving transaction ID to localStorage: ${e}`);
                }
              }
            } catch (uploadError) {
              this.log('IPFS upload error for video:', uploadError);
            }
          }
        } catch (videoError) {
          this.log('Error getting recorded video:', videoError);
        }
      }
      
      return {
        success: true,
        data: { 
          filename: recordingData.filename,
          timestamp: Date.now()
        }
      };
    } catch (error) {
      this.log('Start recording error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to start recording'
      };
    }
  }

  async stopVideoRecording(): Promise<CameraActionResponse<CameraMediaResponse>> {
    try {
      this.log('Stopping video recording');
      
      const response = await this.makeApiCall('/record/stop', 'POST');
      
      if (response.ok) {
        const data = await response.json();
        this.log('Video recording stopped:', data);
        
        return {
          success: true,
          data: { 
            filename: data.filename,
            path: data.path,
            size: data.size,
            timestamp: Date.now()
          }
        };
      } else {
        throw new Error(`Failed to stop recording: ${response.status}`);
      }
    } catch (error) {
      this.log('Stop recording error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to stop recording'
      };
    }
  }

  async getRecordedVideo(filename: string): Promise<CameraActionResponse<CameraMediaResponse>> {
    try {
      this.log('Getting recorded video:', filename);
      
      const response = await fetch(`${this.apiUrl}/api/video/download/${filename}`, {
        method: 'GET',
        mode: 'cors',
        credentials: 'omit'
      });
      
      if (response.ok) {
        const videoBlob = await response.blob();
        this.log('Video retrieved successfully:', videoBlob.size, 'bytes');
        
        return {
          success: true,
          data: { 
            blob: videoBlob,
            filename,
            size: videoBlob.size
          }
        };
      } else {
        throw new Error(`Failed to get video: ${response.status}`);
      }
    } catch (error) {
      this.log('Get video error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get video'
      };
    }
  }

  async listVideos(): Promise<CameraActionResponse<{ videos: any[] }>> {
    try {
      this.log('Listing available videos');
      
      const response = await this.makeApiCall('/videos', 'GET');
      
      if (response.ok) {
        const data = await response.json();
        this.log('Videos listed successfully:', data);
        
        return {
          success: true,
          data: { videos: data.videos || data }
        };
      } else {
        throw new Error(`Failed to list videos: ${response.status}`);
      }
    } catch (error) {
      this.log('List videos error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to list videos'
      };
    }
  }

  async startStream(): Promise<CameraActionResponse<CameraStreamInfo>> {
    if (!this.currentSession) {
      return {
        success: false,
        error: 'No active session. Please connect first.'
      };
    }

    try {
      this.log('Starting stream with Pi5 DePIN verification');
      
      // Create blockchain transaction for DePIN verification
      const sendSimpleTransaction = (window as any).sendSimpleTransaction;
      if (!sendSimpleTransaction) {
        return {
          success: false,
          error: 'Pi5 camera requires blockchain transaction verification. Transaction system not available.'
        };
      }

      const txSignature = await sendSimpleTransaction('stream_started');
      if (!txSignature) {
        return {
          success: false,
          error: 'Failed to create blockchain transaction for stream start'
        };
      }

      this.log(`Created transaction: ${txSignature}`);
      
      // Call Pi5 middleware with transaction verification
      const response = await fetch(`${this.apiUrl}/api/stream/start`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        mode: 'cors',
        credentials: 'omit',
        body: JSON.stringify({
          tx_signature: txSignature,
          wallet_address: this.currentSession.walletAddress
        })
      });
      
      if (!response.ok) {
        const errorText = await response.text();
        this.log(`Stream API Error: ${response.status} - ${errorText}`);
        return {
          success: false,
          error: `Stream API error: ${response.status} ${errorText}`
        };
      }
      
      const streamInfo = await response.json();
      this.log('Stream started successfully:', streamInfo);
      
      return {
        success: true,
        data: {
          isActive: true,
          streamUrl: streamInfo.streamUrl || `${this.apiUrl}/stream`,
          format: 'mjpeg'
        }
      };
    } catch (error) {
      this.log('Start stream error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to start stream'
      };
    }
  }

  async stopStream(): Promise<CameraActionResponse<CameraStreamInfo>> {
    try {
      this.log('Stopping MJPEG stream');
      
      const response = await fetch(`${this.apiUrl}/api/stream/stop`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        mode: 'cors',
        credentials: 'omit',
        body: JSON.stringify({
          wallet_address: this.currentSession?.walletAddress
        })
      });
      
      if (response.ok) {
        const data = await response.json();
        this.log('MJPEG stream stopped:', data);
        
        return {
          success: true,
          data: {
            isActive: false,
            format: 'mjpeg'
          }
        };
      } else {
        throw new Error(`Failed to stop stream: ${response.status}`);
      }
    } catch (error) {
      this.log('Stop stream error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to stop stream'
      };
    }
  }

  async getStreamInfo(): Promise<CameraActionResponse<CameraStreamInfo>> {
    try {
      const response = await fetch(`${this.apiUrl}/api/stream/info`, {
        method: 'GET',
        mode: 'cors',
        credentials: 'omit'
      });
      
      if (response.ok) {
        const data = await response.json();
        
        return {
          success: true,
          data: {
            isActive: data.isActive || false,
            streamUrl: data.isActive ? `${this.apiUrl}/stream` : undefined,
            format: 'mjpeg'
          }
        };
      } else {
        throw new Error(`Failed to get stream info: ${response.status}`);
      }
    } catch (error) {
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get stream info',
        data: {
          isActive: false,
          format: 'mjpeg'
        }
      };
    }
  }

  getStreamUrl(): string {
    return `${this.apiUrl}/stream`;
  }

  // Pi5 cameras don't support gesture detection
  async getCurrentGesture(): Promise<CameraActionResponse<CameraGestureResponse>> {
    return {
      success: false,
      error: 'Gesture detection not supported on Pi5 cameras'
    };
  }

  async toggleGestureControls(enabled: boolean): Promise<CameraActionResponse<{ enabled: boolean }>> {
    return {
      success: false,
      error: 'Gesture controls not supported on Pi5 cameras'
    };
  }

  async toggleFaceVisualization(enabled: boolean): Promise<CameraActionResponse<{ enabled: boolean }>> {
    return {
      success: false,
      error: 'Face visualization not supported on Pi5 cameras'
    };
  }

  getCurrentSession(): CameraSession | null {
    return this.currentSession;
  }

  isCurrentlyRecording(): boolean {
    // Don't track recording state in frontend - always return false
    // Let the Pi5 camera handle its own state
    return false;
  }

  async getMostRecentVideo(): Promise<CameraActionResponse<CameraMediaResponse>> {
    try {
      this.log('Getting most recent video from Pi5');
      
      const response = await fetch(`${this.apiUrl}/api/record/latest`, {
        method: 'GET',
        mode: 'cors',
        credentials: 'omit'
      });
      
      if (response.ok) {
        const videoBlob = await response.blob();
        this.log('Most recent video retrieved successfully:', videoBlob.size, 'bytes');
        
        return {
          success: true,
          data: { 
            blob: videoBlob,
            timestamp: Date.now(),
            size: videoBlob.size
          }
        };
      } else {
        const errorText = await response.text();
        this.log(`Failed to get most recent video: ${response.status} - ${errorText}`);
        return {
          success: false,
          error: `Failed to get most recent video: ${response.status} ${errorText}`
        };
      }
    } catch (error) {
      this.log('Get most recent video error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get most recent video'
      };
    }
  }
} 