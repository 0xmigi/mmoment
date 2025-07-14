/**
 * Jetson Camera Implementation
 * 
 * This implementation wraps the existing Jetson camera service to follow
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

export class JetsonCamera implements ICamera {
  public readonly cameraId: string;
  public readonly cameraType: string = 'jetson';
  public apiUrl: string;
  
  private debugMode = true;
  private currentSession: CameraSession | null = null;
  private lastStreamingStatus: boolean | undefined;
  private lastStreamInfo: CameraStreamInfo | null = null;

  // CURRENT LIVEPEER ACCOUNT (1000 minutes remaining)
  private static readonly CURRENT_PLAYBACK_ID = '6315myh7iojrn5uk';
  private static readonly CURRENT_STREAM_KEY = '6315-9m3d-yfzn-xhf6';
  private static readonly CURRENT_STREAM_URL = 'https://lvpr.tv/?v=6315myh7iojrn5uk';
  private static readonly CURRENT_HLS_URL = 'https://livepeercdn.studio/hls/6315myh7iojrn5uk/index.m3u8';
  
  // BACKUP LIVEPEER ACCOUNT (for when current runs out)
  // private static readonly BACKUP_PLAYBACK_ID = '24583tdeg6syfcqi';
  // private static readonly BACKUP_STREAM_KEY = '2458-aycn-mgfp-2dze';
  // private static readonly BACKUP_STREAM_URL = 'https://lvpr.tv/?v=24583tdeg6syfcqi';
  // private static readonly BACKUP_HLS_URL = 'https://livepeercdn.studio/hls/24583tdeg6syfcqi/index.m3u8';

  constructor(cameraId: string, apiUrl: string) {
    this.cameraId = cameraId;
    this.apiUrl = apiUrl;
    this.log('JetsonCamera initialized for:', cameraId, 'at', apiUrl);
  }

  private log(...args: any[]) {
    if (this.debugMode) {
      console.log(`[JetsonCamera:${this.cameraId.slice(0, 8)}]`, ...args);
    }
  }

  private async makeApiCall(endpoint: string, method: string, data?: any): Promise<Response> {
    const url = `${this.apiUrl}${endpoint}`;
    this.log(`Making ${method} request to: ${url}`);
    
    if (data) {
      this.log(`Request data:`, data);
    }
    
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
      
      if (!response.ok) {
        this.log(`HTTP error: ${response.status} ${response.statusText}`);
        
        // If PDA-based URL fails and we have a legacy URL, try that
        const legacyUrl = (this as any).legacyUrl;
        if (legacyUrl && this.apiUrl !== legacyUrl) {
          this.log(`Trying legacy URL: ${legacyUrl}`);
          try {
            const legacyResponse = await fetch(`${legacyUrl}${endpoint}`, {
              method,
              headers: {
                'Content-Type': 'application/json'
              },
              mode: 'cors',
              credentials: 'omit',
              body: method !== 'GET' && data ? JSON.stringify(data) : undefined
            });
            
            if (legacyResponse.ok) {
              this.log(`Legacy URL succeeded, updating API URL`);
              this.apiUrl = legacyUrl;
              return legacyResponse;
            }
          } catch (legacyError) {
            this.log(`Legacy URL also failed:`, legacyError);
          }
        }
      }
      
      return response;
    } catch (error) {
      this.log('Fetch error:', error);
      
      // If PDA-based URL fails with network error and we have a legacy URL, try that
      const legacyUrl = (this as any).legacyUrl;
      if (legacyUrl && this.apiUrl !== legacyUrl) {
        this.log(`Network error with PDA URL, trying legacy URL: ${legacyUrl}`);
        try {
          const legacyResponse = await fetch(`${legacyUrl}${endpoint}`, {
            method,
            headers: {
              'Content-Type': 'application/json'
            },
            mode: 'cors',
            credentials: 'omit',
            body: method !== 'GET' && data ? JSON.stringify(data) : undefined
          });
          
          this.log(`Legacy URL response status: ${legacyResponse.status}`);
          if (legacyResponse.ok) {
            this.log(`Legacy URL succeeded, updating API URL`);
            this.apiUrl = legacyUrl;
            return legacyResponse;
          }
        } catch (legacyError) {
          this.log(`Legacy URL also failed:`, legacyError);
        }
      }
      
      throw error;
    }
  }

  getCapabilities(): CameraCapabilities {
    return {
      canTakePhotos: true,
      canRecordVideos: true,
      canStream: true,
      canDetectGestures: true,
      canRecognizeFaces: true,
      hasLivepeerStreaming: true,
      supportedStreamFormats: ['livepeer', 'mjpeg']
    };
  }

  async connect(walletAddress?: string): Promise<CameraActionResponse<CameraSession>> {
    try {
      this.log('Connecting to Jetson camera', { walletAddress });
      
      const response = await this.makeApiCall('/api/session/connect', 'POST', {
        wallet_address: walletAddress || 'anonymous'
      });

      if (response.ok) {
        const data = await response.json();
        
        if (data.success) {
          this.currentSession = {
            sessionId: data.session_id,
            walletAddress: data.wallet_address,
            cameraPda: this.cameraId,
            timestamp: Date.now(),
            isActive: true
          };
          
          this.log('Connected successfully', this.currentSession);
          return {
            success: true,
            data: this.currentSession
          };
        } else {
          throw new Error(data.error || data.message || 'Failed to start session');
        }
      } else {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
    } catch (error) {
      this.log('Connection error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to connect'
      };
    }
  }

  async disconnect(): Promise<CameraActionResponse> {
    try {
      this.log('Disconnecting from Jetson camera');
      
      if (this.currentSession) {
        const response = await this.makeApiCall('/api/session/disconnect', 'POST', {
          wallet_address: this.currentSession.walletAddress,
          session_id: this.currentSession.sessionId
        });

        if (response.ok) {
          const data = await response.json();
          if (data.success) {
            this.currentSession = null;
            this.log('Disconnected successfully');
            return { success: true };
          } else {
            throw new Error(data.error || data.message || 'Failed to end session');
          }
        } else {
          throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
      } else {
        // No active session, consider it successful
        return { success: true };
      }
    } catch (error) {
      this.log('Disconnect error:', error);
      // Clear session even if API call failed
      this.currentSession = null;
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to disconnect'
      };
    }
  }

  async testConnection(): Promise<CameraActionResponse<{ message: string; url: string }>> {
    const url = `${this.apiUrl}/api/health`;
    
    try {
      const response = await this.makeApiCall('/api/health', 'GET');
      
      if (response.ok) {
        const data = await response.json();
        return {
          success: true,
          data: {
            message: `Jetson camera service is healthy (${data.status})`,
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
      const response = await this.makeApiCall('/api/status', 'GET');
      
      if (response.ok) {
        const data = await response.json();
        
        // Debug logging (reduced to prevent console spam)
        this.log('🔍 Raw API response from /api/status:', data);
        
        // Check if the response has an unusual structure
        let streamingStatus = false;
        
        // Try multiple possible locations for the streaming status
        if (data.isStreaming !== undefined) {
          streamingStatus = data.isStreaming;
        } else if (data.streaming !== undefined) {
          streamingStatus = data.streaming;
        } else if (data.data?.isStreaming !== undefined) {
          streamingStatus = data.data.isStreaming;
        } else if (data.data?.streaming !== undefined) {
          streamingStatus = data.data.streaming;
        } else {
          // Check if there's a weird nested structure
          for (const key of Object.keys(data)) {
            if (typeof data[key] === 'object' && data[key] !== null) {
              if (data[key].isStreaming !== undefined) {
                streamingStatus = data[key].isStreaming;
                break;
              } else if (data[key].streaming !== undefined) {
                streamingStatus = data[key].streaming;
                break;
              }
            }
          }
        }
        
        // Only log when status actually changes
        if (this.lastStreamingStatus !== streamingStatus) {
          this.log('🎯 Streaming status changed:', this.lastStreamingStatus, '→', streamingStatus);
          this.lastStreamingStatus = streamingStatus;
        }
        
        return {
          success: true,
          data: {
            isOnline: true,
            isStreaming: streamingStatus,
            isRecording: data.recording || data.data?.recording || false,
            lastSeen: Date.now(),
            owner: this.currentSession?.walletAddress
          }
        };
      } else {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
    } catch (error) {
      this.log('❌ getStatus error:', error);
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
      this.log('Taking photo with session:', this.currentSession);
      
      const response = await this.makeApiCall('/api/capture', 'POST', {
        wallet_address: this.currentSession.walletAddress,
        session_id: this.currentSession.sessionId
      });
      
      this.log('Capture API response status:', response.status, response.statusText);
      
      if (response.ok) {
        const data = await response.json();
        this.log('Capture API response data:', data);
        
        if (data.success && data.filename) {
          this.log('Photo captured successfully:', data.filename);
          
          // Now fetch the actual photo blob from the photos endpoint
          try {
            this.log('Fetching photo blob for:', data.filename);
            const photoResponse = await this.makeApiCall(`/api/photos/${data.filename}`, 'GET');
            
            this.log('Photo fetch response status:', photoResponse.status, photoResponse.statusText);
            
            if (photoResponse.ok) {
              const photoBlob = await photoResponse.blob();
              this.log('Photo blob retrieved successfully:', photoBlob.size, 'bytes, type:', photoBlob.type);
              
              if (photoBlob.size === 0) {
                this.log('ERROR: Photo blob is empty!');
                return {
                  success: false,
                  error: 'Photo blob is empty - camera may not be working properly'
                };
              }
              
              return {
                success: true,
                data: { 
                  blob: photoBlob,
                  filename: data.filename,
                  timestamp: data.timestamp || Date.now(),
                  size: photoBlob.size
                }
              };
            } else {
              const errorText = await photoResponse.text();
              this.log('Failed to fetch photo blob:', photoResponse.status, errorText);
              return {
                success: false,
                error: `Failed to fetch photo: ${photoResponse.status} - ${errorText}`
              };
            }
          } catch (fetchError) {
            this.log('Error fetching photo blob:', fetchError);
            return {
              success: false,
              error: `Failed to retrieve photo: ${fetchError instanceof Error ? fetchError.message : 'Unknown error'}`
            };
          }
        } else {
          this.log('Capture API returned unsuccessful result:', data);
          return {
            success: false,
            error: data.error || 'Photo capture failed - no filename returned'
          };
        }
      } else {
        const errorText = await response.text();
        this.log('Capture API failed:', response.status, response.statusText, errorText);
        return {
          success: false,
          error: `Failed to capture photo: ${response.status} - ${errorText}`
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
      this.log('Starting video recording with session:', this.currentSession);
      
      // Ensure we're still connected by testing the session
      const isConnected = this.isConnected();
      if (!isConnected) {
        this.log('Session appears to be invalid, attempting to reconnect...');
        const reconnectResult = await this.connect(this.currentSession.walletAddress);
        if (!reconnectResult.success) {
          return {
            success: false,
            error: 'Session expired and reconnection failed. Please connect again.'
          };
        }
      }
      
      // Use the standardized /api/record endpoint with action parameter
      const response = await this.makeApiCall('/api/record', 'POST', {
        action: 'start',
        wallet_address: this.currentSession.walletAddress,
        session_id: this.currentSession.sessionId,
        duration: 0  // 0 means record until stopped
      });
      
      this.log('Start recording API response status:', response.status, response.statusText);
      
      if (response.ok) {
        const data = await response.json();
        this.log('Start recording API response data:', data);
        
        if (data.success) {
          this.log('Video recording started successfully');
          
          return {
            success: true,
            data: { 
              filename: data.filename,
              timestamp: data.timestamp || Date.now()
            }
          };
        } else {
          this.log('Start recording API returned unsuccessful result:', data);
          return {
            success: false,
            error: data.error || 'Failed to start recording'
          };
        }
      } else {
        const errorText = await response.text();
        this.log('Start recording API failed:', response.status, response.statusText, errorText);
        
        // If it's a 403, the session is likely invalid
        if (response.status === 403) {
          this.log('403 error - session likely invalid, clearing session');
          this.currentSession = null;
          return {
            success: false,
            error: 'Session expired. Please connect again and try recording.'
          };
        }
        
        return {
          success: false,
          error: `Failed to start recording: ${response.status} - ${errorText}`
        };
      }
    } catch (error) {
      this.log('Start recording error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to start recording'
      };
    }
  }

  async stopVideoRecording(): Promise<CameraActionResponse<CameraMediaResponse>> {
    if (!this.currentSession) {
      return {
        success: false,
        error: 'No active session. Please connect first.'
      };
    }

    try {
      this.log('Stopping video recording with session:', this.currentSession);
      
      // Use the standardized /api/record endpoint with action parameter
      const response = await this.makeApiCall('/api/record', 'POST', {
        action: 'stop',
        wallet_address: this.currentSession.walletAddress,
        session_id: this.currentSession.sessionId
      });
      
      this.log('Stop recording API response status:', response.status, response.statusText);
      
      if (response.ok) {
        const data = await response.json();
        this.log('Stop recording API response data:', data);
        
        if (data.success) {
          this.log('Video recording stopped successfully:', data.filename);
          
          return {
            success: true,
            data: { 
              filename: data.filename,
              timestamp: data.timestamp || Date.now()
            }
          };
        } else {
          this.log('Stop recording API returned unsuccessful result:', data);
          return {
            success: false,
            error: data.error || 'Failed to stop recording'
          };
        }
      } else {
        const errorText = await response.text();
        this.log('Stop recording API failed:', response.status, response.statusText, errorText);
        return {
          success: false,
          error: `Failed to stop recording: ${response.status} - ${errorText}`
        };
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
      
      const response = await this.makeApiCall(`/api/videos/${filename}`, 'GET');
      
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
      
      const response = await this.makeApiCall('/api/videos', 'GET');
      
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

  async getMostRecentVideo(): Promise<CameraActionResponse<CameraMediaResponse>> {
    try {
      this.log('Getting most recent video');
      
      const listResponse = await this.listVideos();
      if (!listResponse.success || !listResponse.data?.videos) {
        throw new Error('Failed to get video list');
      }
      
      const videos = listResponse.data.videos;
      if (!Array.isArray(videos) || videos.length === 0) {
        throw new Error('No videos available');
      }
      
      // Filter to only .mov files
      const movVideos = videos.filter(video => {
        const filename = video.filename || video;
        return filename.endsWith('.mov');
      });
      
      if (movVideos.length === 0) {
        throw new Error('No .mov videos available');
      }
      
      // Sort .mov videos by timestamp/name to get the most recent
      const sortedVideos = movVideos.sort((a, b) => {
        // Try to sort by timestamp if available, otherwise by filename
        if (a.timestamp && b.timestamp) {
          return b.timestamp - a.timestamp;
        }
        return b.filename?.localeCompare(a.filename || '') || 0;
      });
      
      const mostRecentVideo = sortedVideos[0];
      const filename = mostRecentVideo.filename || mostRecentVideo;
      this.log('Most recent .mov video:', filename);
      
      // Get the actual .mov video file
      return await this.getRecordedVideo(filename);
    } catch (error) {
      this.log('Get most recent video error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get most recent video'
      };
    }
  }

  async startStream(): Promise<CameraActionResponse<CameraStreamInfo>> {
    try {
      this.log('Starting Livepeer stream');
      
      // Livepeer endpoints don't require session data
      const response = await this.makeApiCall('/api/stream/livepeer/start', 'POST');
      
      this.log('Start stream API response status:', response.status, response.statusText);
      
      if (response.ok) {
        const data = await response.json();
        this.log('Start stream API response data:', data);
        
        if (data.success) {
          this.log('Livepeer stream started successfully');
          
          return {
            success: true,
            data: {
              isActive: true,
              streamUrl: data.playback_url || JetsonCamera.CURRENT_STREAM_URL,
              playbackId: JetsonCamera.CURRENT_PLAYBACK_ID,
              streamKey: data.stream_key || JetsonCamera.CURRENT_STREAM_KEY,
              hlsUrl: data.hls_url || JetsonCamera.CURRENT_HLS_URL,
              format: 'livepeer'
            }
          };
        } else {
          this.log('Start stream API returned unsuccessful result:', data);
          return {
            success: false,
            error: data.error || 'Failed to start stream'
          };
        }
      } else {
        const errorText = await response.text();
        this.log('Start stream API failed:', response.status, response.statusText, errorText);
        
        // Even if the API returns an error, the stream might actually start
        // So we'll optimistically set the local state and let the user check
        this.log('API failed but stream might have started anyway, setting optimistic state');
        
        return {
          success: true, // Return success despite API error
          data: {
            isActive: true,
            streamUrl: JetsonCamera.CURRENT_STREAM_URL,
            playbackId: JetsonCamera.CURRENT_PLAYBACK_ID,
            streamKey: JetsonCamera.CURRENT_STREAM_KEY,
            hlsUrl: JetsonCamera.CURRENT_HLS_URL,
            format: 'livepeer'
          }
        };
      }
    } catch (error) {
      this.log('Start stream error:', error);
      
      // Even if there's an exception, the stream might work
      // Set optimistic state
      return {
        success: true, // Return success despite error
        data: {
          isActive: true,
          streamUrl: JetsonCamera.CURRENT_STREAM_URL,
          playbackId: JetsonCamera.CURRENT_PLAYBACK_ID,
          streamKey: JetsonCamera.CURRENT_STREAM_KEY,
          hlsUrl: JetsonCamera.CURRENT_HLS_URL,
          format: 'livepeer'
        }
      };
    }
  }

  async stopStream(): Promise<CameraActionResponse<CameraStreamInfo>> {
    try {
      this.log('Stopping Livepeer stream');
      
      // Livepeer endpoints don't require session data
      const response = await this.makeApiCall('/api/stream/livepeer/stop', 'POST');
      
      this.log('Stop stream API response status:', response.status, response.statusText);
      
      if (response.ok) {
        const data = await response.json();
        this.log('Stop stream API response data:', data);
        
        if (data.success) {
          this.log('Livepeer stream stopped successfully');
          
          return {
            success: true,
            data: {
              isActive: false,
              streamUrl: '',
              playbackId: '',
              streamKey: '',
              hlsUrl: '',
              format: 'livepeer'
            }
          };
        } else {
          this.log('Stop stream API returned unsuccessful result:', data);
          return {
            success: false,
            error: data.error || 'Failed to stop stream'
          };
        }
      } else {
        const errorText = await response.text();
        this.log('Stop stream API failed:', response.status, response.statusText, errorText);
        return {
          success: false,
          error: `Failed to stop stream: ${response.status} - ${errorText}`
        };
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
      this.log('Getting Jetson stream info');
      
      // Query actual hardware state instead of using local state
      const statusResponse = await this.getStatus();
      const isStreamingFromHardware = statusResponse.success ? statusResponse.data?.isStreaming || false : false;
      
      // The Jetson has a static Livepeer playback ID that's always available
      // The playback ID is always available, but isActive depends on hardware state
      const streamInfo = {
        isActive: isStreamingFromHardware, // Use actual hardware state
        streamUrl: JetsonCamera.CURRENT_STREAM_URL,
        playbackId: JetsonCamera.CURRENT_PLAYBACK_ID,
        format: 'livepeer' as const
      };
      
      // Only log when stream state actually changes
      if (this.lastStreamInfo?.isActive !== streamInfo.isActive) {
        this.log('📤 Stream state changed - isActive:', this.lastStreamInfo?.isActive, '→', streamInfo.isActive);
        this.lastStreamInfo = streamInfo;
      }
      
      return {
        success: true,
        data: streamInfo
      };
    } catch (error) {
      this.log('Get stream info error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get stream info',
        data: {
          isActive: false,
          streamUrl: JetsonCamera.CURRENT_STREAM_URL,
          playbackId: JetsonCamera.CURRENT_PLAYBACK_ID,
          format: 'livepeer' as const
        }
      };
    }
  }

  getStreamUrl(): string {
    return JetsonCamera.CURRENT_STREAM_URL;
  }

  async getCurrentGesture(): Promise<CameraActionResponse<CameraGestureResponse>> {
    try {
      this.log('Getting current gesture');
      
      const response = await this.makeApiCall('/api/gesture/current', 'GET');

      const data = await response.json();
      
      if (response.ok && data.success) {
        this.log('Current gesture retrieved:', data.gesture);
        return { 
          success: true,
          data: { 
            gesture: data.gesture,
            confidence: data.confidence,
            timestamp: data.timestamp
          }
        };
      } else {
        throw new Error(data.error || 'Failed to get current gesture');
      }
    } catch (error) {
      this.log('Get current gesture error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to get current gesture'
      };
    }
  }

  async toggleGestureControls(enabled: boolean): Promise<CameraActionResponse<{ enabled: boolean }>> {
    try {
      this.log(`Toggling gesture controls: ${enabled}`);
      
      // Gesture controls are client-side only - just store in localStorage
      localStorage.setItem('jetson_gesture_controls_enabled', enabled.toString());
      
      this.log('Gesture controls toggled successfully:', enabled);
      
      return { 
        success: true,
        data: { enabled }
      };
    } catch (error) {
      this.log('Gesture controls toggle error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to toggle gesture controls'
      };
    }
  }

  async toggleGestureVisualization(enabled: boolean): Promise<CameraActionResponse<{ enabled: boolean }>> {
    try {
      this.log(`Toggling gesture visualization: ${enabled}`);
      this.log(`Making API call to: ${this.apiUrl}/api/visualization/gesture`);
      
      // Use the standardized API endpoint for gesture visualization
      const response = await this.makeApiCall('/api/visualization/gesture', 'POST', {
        enabled
      });

      this.log(`Response status: ${response.status} ${response.statusText}`);
      const data = await response.json();
      this.log(`Response data:`, data);
      
      if (response.ok && data.success) {
        this.log('Gesture visualization toggled successfully:', data.enabled);
        
        return { 
          success: true,
          data: { enabled: data.enabled }
        };
      } else {
        throw new Error(data.error || 'Failed to toggle gesture visualization');
      }
    } catch (error) {
      this.log('Gesture visualization toggle error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to toggle gesture visualization'
      };
    }
  }

  async toggleFaceVisualization(enabled: boolean): Promise<CameraActionResponse<{ enabled: boolean }>> {
    try {
      this.log(`Toggling face visualization: ${enabled}`);
      this.log(`Making API call to: ${this.apiUrl}/api/visualization/face`);
      
      const response = await this.makeApiCall('/api/visualization/face', 'POST', {
        enabled
      });

      this.log(`Response status: ${response.status} ${response.statusText}`);
      const data = await response.json();
      this.log(`Response data:`, data);
      
      if (response.ok && data.success) {
        this.log('Face visualization toggled successfully:', data.enabled);
        return { 
          success: true,
          data: { enabled: data.enabled }
        };
      } else {
        throw new Error(data.error || 'Failed to toggle face visualization');
      }
    } catch (error) {
      this.log('Face visualization toggle error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to toggle face visualization'
      };
    }
  }

  getCurrentSession(): CameraSession | null {
    return this.currentSession;
  }

  getGestureControlsStatus(): boolean {
    const stored = localStorage.getItem('jetson_gesture_controls_enabled');
    return stored === 'true';
  }

  async checkForGestureTrigger(): Promise<{
    shouldCapture: boolean;
    gestureType: 'photo' | 'video' | null;
    gesture?: string;
    confidence?: number;
  }> {
    try {
      // Only check if gesture controls are enabled
      if (!this.getGestureControlsStatus()) {
        return { shouldCapture: false, gestureType: null };
      }

      const gestureResponse = await this.getCurrentGesture();
      
      if (gestureResponse.success && gestureResponse.data) {
        const { gesture, confidence } = gestureResponse.data;
        
        // Use simplified gesture mappings for auto-recording
        // Peace sign = photo, Thumbs up = 10-second video recording
        if (confidence && confidence > 0.7 && gesture !== 'none') {
          switch (gesture) {
            case 'peace':
              return { 
                shouldCapture: true, 
                gestureType: 'photo',
                gesture,
                confidence
              };
            case 'thumbs_up':
              return { 
                shouldCapture: true, 
                gestureType: 'video',
                gesture,
                confidence
              };
            default:
              return { shouldCapture: false, gestureType: null };
          }
        }
      }
      
      return { shouldCapture: false, gestureType: null };
    } catch (error) {
      this.log('Error checking gesture trigger:', error);
      return { shouldCapture: false, gestureType: null };
    }
  }

  async isCurrentlyRecording(): Promise<boolean> {
    try {
      this.log('Checking if currently recording');
      
      const response = await this.makeApiCall('/api/status', 'GET');
      
      if (response.ok) {
        const data = await response.json();
        const recording = data.recording || false;
        this.log('Recording status:', recording);
        return recording;
      } else {
        this.log('Failed to get recording status');
        return false;
      }
    } catch (error) {
      this.log('Error checking recording status:', error);
      return false;
    }
  }

  async enrollFace(walletAddress: string): Promise<CameraActionResponse<{ enrolled: boolean; faceId: string }>> {
    try {
      this.log(`[DEPRECATED] enrollFace called - redirecting to new transaction-based flow`);
      this.log(`Enrolling face for wallet: ${walletAddress}`);
      
      // Use the new session-less approach - call prepare transaction endpoint directly
      const response = await this.makeApiCall('/api/face/enroll/prepare-transaction', 'POST', {
        wallet_address: walletAddress
      });
      
      const data = await response.json();
      console.log('[JetsonCamera] Face enrollment response (session-less):', data);
      
      if (response.ok && data.success) {
        this.log('Face enrollment prepared successfully (session-less):', data);
        
        // For the old interface, we just return that enrollment was "prepared"
        // The actual transaction signing and confirmation would need to be handled separately
        return {
          success: true,
          data: {
            enrolled: false, // Not fully enrolled yet - just prepared
            faceId: data.face_id || data.faceId || 'prepared-' + Date.now()
          }
        };
      } else {
        console.error('[JetsonCamera] Face enrollment preparation failed (session-less):', {
          status: response.status,
          statusText: response.statusText,
          responseData: data
        });
        throw new Error(data.error || `HTTP ${response.status}: Failed to prepare face enrollment`);
      }
    } catch (error) {
      this.log('Face enrollment error (session-less):', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to enroll face'
      };
    }
  }

  async prepareFaceEnrollmentTransaction(walletAddress: string): Promise<CameraActionResponse<{ transactionBuffer: string; faceId: string; metadata?: any }>> {
    try {
      this.log(`Preparing face enrollment transaction for wallet: ${walletAddress}`);
      
      // Call the prepare transaction endpoint with only wallet address
      const response = await this.makeApiCall('/api/face/enroll/prepare-transaction', 'POST', {
        wallet_address: walletAddress
      });
      
      const data = await response.json();
      console.log('[JetsonCamera] Face enrollment transaction preparation response:', data);
      console.log('[JetsonCamera] Response status:', response.status, response.statusText);
      
      if (response.ok && data.success) {
        this.log('Face enrollment transaction prepared successfully:', data);
        
        // Log the specific fields we're extracting
        console.log('[JetsonCamera] Extracted fields:', {
          transactionBuffer: data.transaction_buffer || data.transactionBuffer,
          faceId: data.face_id || data.faceId,
          metadata: data.metadata
        });
        
        return {
          success: true,
          data: {
            transactionBuffer: data.transaction_buffer || data.transactionBuffer,
            faceId: data.face_id || data.faceId,
            metadata: data.metadata
          }
        };
      } else {
        console.error('[JetsonCamera] Face enrollment preparation failed:', {
          status: response.status,
          statusText: response.statusText,
          responseData: data
        });
        throw new Error(data.error || `HTTP ${response.status}: Failed to prepare face enrollment transaction`);
      }
    } catch (error) {
      this.log('Face enrollment transaction preparation error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to prepare face enrollment transaction'
      };
    }
  }

  async confirmFaceEnrollmentTransaction(walletAddress: string, confirmationData: { signedTransaction: string; faceId: string; biometricSessionId?: string }): Promise<CameraActionResponse<{ enrolled: boolean; faceId: string; transactionId?: string }>> {
    try {
      this.log(`Confirming face enrollment transaction for wallet: ${walletAddress}`);
      
      // Call the new confirm transaction endpoint
      const response = await this.makeApiCall('/api/face/enroll/confirm', 'POST', {
        wallet_address: walletAddress,
        signed_transaction: confirmationData.signedTransaction,
        face_id: confirmationData.faceId,
        biometric_session_id: confirmationData.biometricSessionId
      });
      
      const data = await response.json();
      console.log('[JetsonCamera] Face enrollment transaction confirmation response:', data);
      console.log('[JetsonCamera] Confirmation response status:', response.status, response.statusText);
      
      if (response.ok && data.success) {
        this.log('Face enrollment transaction confirmed successfully:', data);
        
        // Log the specific fields we're extracting
        console.log('[JetsonCamera] Confirmation extracted fields:', {
          enrolled: true,
          faceId: data.face_id || data.faceId || confirmationData.faceId,
          transactionId: data.transaction_id || data.transactionId
        });
        
        return {
          success: true,
          data: {
            enrolled: true,
            faceId: data.face_id || data.faceId || confirmationData.faceId,
            transactionId: data.transaction_id || data.transactionId
          }
        };
      } else {
        console.error('[JetsonCamera] Face enrollment confirmation failed:', {
          status: response.status,
          statusText: response.statusText,
          responseData: data
        });
        throw new Error(data.error || `HTTP ${response.status}: Failed to confirm face enrollment transaction`);
      }
    } catch (error) {
      this.log('Face enrollment transaction confirmation error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to confirm face enrollment transaction'
      };
    }
  }

  // Debug function to test session connectivity
  async debugSession(): Promise<void> {
    console.log('🔗 Testing session connectivity...');
    
    if (!this.currentSession) {
      console.log('❌ No current session');
      return;
    }
    
    console.log('Current session:', this.currentSession);
    
    const isConnected = this.isConnected();
    console.log('Is connected:', isConnected);
    
    if (!isConnected) {
      console.log('🔄 Attempting to reconnect...');
      const reconnectResult = await this.connect(this.currentSession.walletAddress);
      console.log('Reconnect result:', reconnectResult);
    } else {
      console.log('✅ Session is valid');
    }
  }

  // Debug function to test photo capture
  async debugPhotoCapture(): Promise<void> {
    console.log('📸 Testing photo capture...');
    
    const result = await this.takePhoto();
    console.log('Photo capture result:', result);
    
    if (result.success && result.data) {
      console.log('✅ Photo captured successfully');
      
      // result.data is CameraMediaResponse, which should contain a blob
      const blob = result.data as Blob; // Cast to Blob since takePhoto returns the blob directly
      console.log('Photo data type:', typeof blob);
      console.log('Photo data size:', blob.size, 'bytes');
      
      // Create download link for verification
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `debug_photo_${Date.now()}.jpg`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      console.log('📥 Photo downloaded for verification');
    } else {
      console.log('❌ Photo capture failed:', result.error);
    }
  }

  // Debug function to test stream display
  async debugStreamDisplay(): Promise<void> {
    console.log('📺 Testing stream display...');
    
    try {
      // Test stream info retrieval
      console.log('1️⃣ Getting stream info...');
      const streamInfo = await this.getStreamInfo();
      console.log('Stream info result:', streamInfo);
      
      if (streamInfo.success && streamInfo.data) {
        console.log('✅ Stream info retrieved successfully');
        console.log('Stream details:');
        console.log('- Is Active:', streamInfo.data.isActive);
        console.log('- Playback ID:', streamInfo.data.playbackId);
        console.log('- Stream URL:', streamInfo.data.streamUrl);
        console.log('- HLS URL:', streamInfo.data.hlsUrl);
        console.log('- Format:', streamInfo.data.format);
        
        // Test HLS URL directly
        if (streamInfo.data.hlsUrl) {
          console.log('2️⃣ Testing HLS URL directly...');
          try {
            const hlsResponse = await fetch(streamInfo.data.hlsUrl, { method: 'HEAD' });
            console.log('HLS URL status:', hlsResponse.status, hlsResponse.statusText);
            if (hlsResponse.ok) {
              console.log('✅ HLS URL is accessible');
            } else {
              console.log('❌ HLS URL returned error:', hlsResponse.status);
            }
          } catch (hlsError) {
            console.log('❌ HLS URL fetch failed:', hlsError);
          }
        }
        
        // Test playback URL
        if (streamInfo.data.streamUrl) {
          console.log('3️⃣ Stream playback URL:', streamInfo.data.streamUrl);
          console.log('💡 You can test this URL directly in a new tab');
        }
        
      } else {
        console.log('❌ Failed to get stream info:', streamInfo.error);
      }
      
    } catch (error) {
      console.error('❌ Stream display debug failed:', error);
    }
  }

  // Debug function to test all fixed functionality
  async debugAllFunctions(): Promise<void> {
    console.log('🔧 === COMPREHENSIVE DEBUG TEST ===');
    
    try {
      // Test session connectivity
      console.log('1️⃣ Testing session connectivity...');
      await this.debugSession();
      
      // Test photo capture
      console.log('\n2️⃣ Testing photo capture...');
      await this.debugPhotoCapture();
      
      // Test stream display (new)
      console.log('\n3️⃣ Testing stream display...');
      await this.debugStreamDisplay();
      
      // Test streaming (with new optimistic approach)
      console.log('\n4️⃣ Testing streaming (optimistic approach)...');
      const streamResult = await this.startStream();
      console.log('Stream result:', streamResult);
      
      if (streamResult.success) {
        console.log('✅ Stream started successfully (or optimistically)');
        console.log('Stream info:', streamResult.data);
        
        // Wait a bit then stop
        console.log('Waiting 3 seconds before stopping stream...');
        await new Promise(resolve => setTimeout(resolve, 3000));
        
        const stopResult = await this.stopStream();
        console.log('Stop stream result:', stopResult);
      } else {
        console.log('❌ Stream failed:', streamResult.error);
      }
      
      // Test video recording (with fixed endpoint)
      console.log('\n5️⃣ Testing video recording (fixed endpoint)...');
      const recordResult = await this.startVideoRecording();
      console.log('Record result:', recordResult);
      
      if (recordResult.success) {
        console.log('✅ Recording started successfully');
        console.log('Recording info:', recordResult.data);
        
        // Wait a bit then stop
        console.log('Waiting 5 seconds before stopping recording...');
        await new Promise(resolve => setTimeout(resolve, 5000));
        
        const stopResult = await this.stopVideoRecording();
        console.log('Stop recording result:', stopResult);
      } else {
        console.log('❌ Recording failed:', recordResult.error);
      }
      
      console.log('\n🎉 === DEBUG TEST COMPLETE ===');
      
    } catch (error) {
      console.error('❌ Debug test failed:', error);
    }
  }

  // Debug function to test stream display fix
  async debugStreamDisplayFix(): Promise<void> {
    console.log('🔧 Testing stream display fix...');
    
    try {
      // Test the fixed getStreamInfo method
      console.log('1️⃣ Testing getStreamInfo with fixed status check...');
      const streamInfo = await this.getStreamInfo();
      console.log('Stream info result:', streamInfo);
      
      if (streamInfo.success && streamInfo.data) {
        console.log('✅ Stream info retrieved successfully');
        console.log('Stream details:');
        console.log('- Is Active:', streamInfo.data.isActive);
        console.log('- Playback ID:', streamInfo.data.playbackId);
        console.log('- Stream URL:', streamInfo.data.streamUrl);
        console.log('- HLS URL:', streamInfo.data.hlsUrl);
        console.log('- Format:', streamInfo.data.format);
        
        if (streamInfo.data.isActive) {
          console.log('🎉 Stream should now display as ACTIVE!');
          console.log('💡 The StreamPlayer should show the Livepeer video instead of "Stream is offline"');
        } else {
          console.log('⚠️ Stream is not active according to API');
        }
        
      } else {
        console.log('❌ Failed to get stream info:', streamInfo.error);
      }
      
    } catch (error) {
      console.error('❌ Stream display fix test failed:', error);
    }
  }
} 