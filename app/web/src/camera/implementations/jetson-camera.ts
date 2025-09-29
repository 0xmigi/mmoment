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
import { DeviceSignedResponse } from '../camera-types';
import { hasValidDeviceSignature, logDeviceSignature } from '../device-signature-utils';

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

  /**
   * Verify device signature on API responses for DePIN authentication
   */
  private verifyDeviceSignature(response: DeviceSignedResponse, endpoint: string): boolean {
    logDeviceSignature(response, `${this.cameraId.slice(0, 8)}:${endpoint}`);
    
    const isValid = hasValidDeviceSignature(response);
    if (!isValid) {
      this.log(`⚠️ Device signature verification failed for ${endpoint}`);
    } else {
      this.log(`✅ Device signature verified for ${endpoint}`);
    }
    
    return isValid;
  }

  /**
   * Get device public key for camera registration
   */
  async getDevicePublicKey(): Promise<string | null> {
    try {
      this.log('Fetching device public key for DePIN registration...');
      const response = await this.makeApiCall('/api/device-info', 'GET');
      
      if (response.ok) {
        const data = await response.json() as DeviceSignedResponse & { device_pubkey: string };
        
        // Verify device signature
        this.verifyDeviceSignature(data, 'device-info');
        
        if (data.device_pubkey) {
          this.log('Retrieved device public key:', data.device_pubkey);
          return data.device_pubkey;
        }
      }
      
      this.log('Failed to get device public key from /api/device-info');
      return null;
    } catch (error) {
      this.log('Error fetching device public key:', error);
      return null;
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
      }
      
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
        
        // Parse camera status from response - handle both /api/status and /api/camera/info formats
        let isOnline = true; // Default to online if we got a successful response
        if (data.camera_status?.online !== undefined) {
          isOnline = data.camera_status.online;
        } else if (data.camera_info?.camera_status?.online !== undefined) {
          isOnline = data.camera_info.camera_status.online;
        } else if (data.online !== undefined) {
          isOnline = data.online;
        }

        return {
          success: true,
          data: {
            isOnline: isOnline,
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
      
      // Check if already recording and stop if needed
      const currentlyRecording = await this.isCurrentlyRecording();
      if (currentlyRecording) {
        this.log('Camera is already recording, stopping existing recording first...');
        try {
          await this.stopVideoRecording();
          // Wait a moment for the stop to complete
          await new Promise(resolve => setTimeout(resolve, 1000));
        } catch (stopError) {
          this.log('Error stopping existing recording:', stopError);
          // Continue anyway - might be a stuck state
        }
      }
      
      // Use the standardized /api/record endpoint with action parameter
      const response = await this.makeApiCall('/api/record', 'POST', {
        action: 'start',
        wallet_address: this.currentSession.walletAddress,
        session_id: this.currentSession.sessionId
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
        
        // If it's a 400 "Already recording", return a more helpful error
        if (response.status === 400 && errorText.includes('Already recording')) {
          this.log('400 Already recording error - camera is already recording');
          return {
            success: false,
            error: 'Camera is already recording. Please wait for the current recording to finish or stop it manually.'
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
        
        if (data.success && data.filename) {
          this.log('Video recording stopped successfully:', data.filename);
          
          // Now fetch the actual video file from the API with retry logic
          try {
            this.log('Fetching video file with retry logic:', data.filename);
            console.log('API returned filename:', data.filename);
            console.log('Full API response:', data);
            
            const videoResponse = await this.getRecordedVideoWithRetry(data.filename);
            
            if (videoResponse.success && videoResponse.data?.blob) {
              this.log('Video file retrieved successfully:', videoResponse.data.blob.size, 'bytes');
              
              return {
                success: true,
                data: { 
                  blob: videoResponse.data.blob,
                  filename: data.filename,
                  timestamp: data.timestamp || Date.now(),
                  size: videoResponse.data.blob.size
                }
              };
            } else {
              this.log('Failed to retrieve video file:', videoResponse.error);
              return {
                success: false,
                error: `Recording stopped but failed to retrieve video file: ${videoResponse.error}`
              };
            }
          } catch (fetchError) {
            this.log('Error fetching video file:', fetchError);
            return {
              success: false,
              error: `Recording stopped but failed to fetch video: ${fetchError instanceof Error ? fetchError.message : 'Unknown error'}`
            };
          }
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

  async getRecordedVideoWithRetry(filename: string, maxRetries: number = 8): Promise<CameraActionResponse<CameraMediaResponse>> {
    let attempt = 0;
    let lastError: string = '';
    
    while (attempt < maxRetries) {
      try {
        this.log(`Attempt ${attempt + 1}/${maxRetries} to get video: ${filename}`);
        
        const result = await this.getRecordedVideo(filename);
        
        if (result.success) {
          this.log(`✅ Video retrieved successfully on attempt ${attempt + 1}`);
          return result;
        }
        
        lastError = result.error || 'Unknown error';
        
        // If the error indicates the MP4 file is not available, wait and retry
        if (lastError.includes('MP4 video file not found') || lastError.includes('not available')) {
          attempt++;
          
          if (attempt < maxRetries) {
            // Longer delays for video processing: 5s, 10s, 15s, 20s, 30s, 45s, 60s, 90s
            const delay = Math.min(5000 * attempt, 90000); // Start at 5s, cap at 90 seconds
            this.log(`⏳ Video not ready yet, waiting ${delay}ms before retry ${attempt + 1}/${maxRetries}`);
            await new Promise(resolve => setTimeout(resolve, delay));
            continue;
          }
        } else {
          // For other errors, fail immediately
          break;
        }
      } catch (error) {
        lastError = error instanceof Error ? error.message : 'Unknown error';
        break;
      }
    }
    
    this.log(`❌ Failed to get video after ${maxRetries} attempts: ${lastError}`);
    return {
      success: false,
      error: `Video processing timeout: The camera may still be processing the video file. Please try again in a few moments. (Last error: ${lastError})`
    };
  }

  async getRecordedVideo(filename: string): Promise<CameraActionResponse<CameraMediaResponse>> {
    try {
      this.log('Getting recorded video:', filename);
      console.log('Input filename:', filename);
      console.log('API URL:', this.apiUrl);
      
      // First try to get MP4 version if the filename is .mov
      let videoResponse: Response;
      let finalFilename = filename;
      
      // Always try to get MP4 version first for web compatibility
      let mp4Filename: string;
      if (filename.endsWith('.mov')) {
        mp4Filename = filename.replace('.mov', '.mp4');
      } else if (filename.endsWith('.mp4')) {
        mp4Filename = filename;
      } else {
        // If no extension, assume it's a base name and try .mp4
        mp4Filename = filename + '.mp4';
      }
      
      const mp4Url = `${this.apiUrl}/api/videos/${mp4Filename}`;
      console.log('Trying MP4 version:', mp4Url);
      
      const mp4Response = await this.makeApiCall(`/api/videos/${mp4Filename}`, 'GET');
      console.log('MP4 response status:', mp4Response.status);
      
      if (mp4Response.ok) {
        console.log('✅ MP4 version found, using it');
        videoResponse = mp4Response;
        finalFilename = mp4Filename;
      } else {
        console.log('❌ MP4 version not available - this is the issue!');
        throw new Error(`MP4 video file not found: ${mp4Filename}. The camera may need more time to process the video.`);
      }
      
      if (videoResponse.ok) {
        const videoBlob = await videoResponse.blob();
        console.log('Video blob retrieved - size:', videoBlob.size, 'bytes, type:', videoBlob.type);
        
        // Validate the video blob
        if (videoBlob.size === 0) {
          console.log('❌ Video file is empty!');
          throw new Error('Video file is empty');
        }
        
        console.log('✅ Video blob size:', videoBlob.size, 'bytes - processing...');
        
        // Determine the correct MIME type based on filename
        let mimeType: string;
        if (finalFilename.endsWith('.mp4')) {
          mimeType = 'video/mp4';
        } else if (finalFilename.endsWith('.mov')) {
          mimeType = 'video/quicktime';
        } else {
          // Use the blob's original type or default to mp4
          mimeType = videoBlob.type || 'video/mp4';
        }
        
        console.log('Using MIME type:', mimeType, 'for file:', finalFilename);
        
        // Create the video file with the correct MIME type
        const videoFile = new Blob([videoBlob], { type: mimeType });
        console.log('Final video file - size:', videoFile.size, 'bytes, type:', videoFile.type);
        
        return {
          success: true,
          data: { 
            blob: videoFile,
            filename: finalFilename,
            size: videoFile.size,
            timestamp: Date.now()
          }
        };
      } else {
        const errorText = await videoResponse.text();
        console.log('❌ Failed to get video:');
        console.log('- Status:', videoResponse.status, videoResponse.statusText);
        console.log('- Error text:', errorText);
        console.log('- Attempted filename:', finalFilename);
        throw new Error(`Failed to get video: ${videoResponse.status} - ${errorText}`);
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
      
      // Filter to video files (both .mp4 and .mov)
      const videoFiles = videos.filter(video => {
        const filename = video.filename || video;
        return filename.endsWith('.mp4') || filename.endsWith('.mov');
      });
      
      if (videoFiles.length === 0) {
        throw new Error('No video files available');
      }
      
      // Sort videos by timestamp/name to get the most recent
      const sortedVideos = videoFiles.sort((a, b) => {
        // Try to sort by timestamp if available, otherwise by filename
        if (a.timestamp && b.timestamp) {
          return b.timestamp - a.timestamp;
        }
        return b.filename?.localeCompare(a.filename || '') || 0;
      });
      
      // Prefer MP4 files over MOV files when available
      const mostRecentVideo = sortedVideos.find(video => {
        const filename = video.filename || video;
        return filename.endsWith('.mp4');
      }) || sortedVideos[0];
      
      const filename = mostRecentVideo.filename || mostRecentVideo;
      this.log('Most recent video:', filename);
      
      // Get the actual video file (this will handle MP4/MOV preference internally)
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

  /**
   * Extract facial embedding using the enhanced /api/face/extract-embedding endpoint
   * This method supports quality assessment, encryption, and personalized recommendations
   */
  async extractEmbedding(
    imageData: string,
    options: {
      encrypt?: boolean;
      qualityAssessment?: boolean;
    } = {}
  ): Promise<CameraActionResponse<{
    embedding: number[];
    quality?: {
      score: number;
      rating: 'excellent' | 'good' | 'acceptable' | 'poor' | 'very_poor';
      issues: string[];
      recommendations: string[];
    };
    encrypted?: boolean;
    sessionId?: string;
  }>> {
    try {
      this.log('Extracting facial embedding with enhanced endpoint');

      // Prepare the request payload
      const payload: any = {
        image_data: imageData.includes(',') ? imageData.split(',')[1] : imageData, // Remove data:image/jpeg;base64, prefix if present
        quality_assessment: options.qualityAssessment !== false, // Default to true unless explicitly false
      };

      // Add wallet address if we have an active session (needed for local enrollment)
      if (this.currentSession?.walletAddress) {
        payload.wallet_address = this.currentSession.walletAddress;
        this.log('Adding wallet address to embedding extraction for local enrollment:', this.currentSession.walletAddress);
      }

      // Add encryption option if requested
      if (options.encrypt) {
        payload.encrypt = true;
      }

      console.log('[JetsonCamera] Calling enhanced face embedding endpoint with options:', {
        hasImageData: !!payload.image_data,
        imageDataLength: payload.image_data?.length,
        qualityAssessment: payload.quality_assessment,
        encrypt: payload.encrypt
      });

      // Call the enhanced face embedding extraction endpoint
      const response = await this.makeApiCall('/api/face/extract-embedding', 'POST', payload);

      const data = await response.json();
      console.log('[JetsonCamera] Enhanced face embedding response:', data);
      console.log('[JetsonCamera] Response status:', response.status, response.statusText);

      if (response.ok && data.success) {
        this.log('✅ Enhanced face embedding extraction successful:', data);

        // Extract the response data
        const result: any = {
          embedding: data.embedding || data.face_embedding
        };

        // Add quality assessment if available
        if (data.quality_score !== undefined) {
          result.quality = {
            score: data.quality_score,
            rating: data.quality_rating || this.scoreToRating(data.quality_score),
            issues: data.quality_issues || [],
            recommendations: data.recommendations || []
          };
        }

        // Add encryption info if available
        if (data.encrypted !== undefined) {
          result.encrypted = data.encrypted;
        }

        // Add session ID if available
        if (data.biometric_session_id || data.session_id) {
          result.sessionId = data.biometric_session_id || data.session_id;
        }

        console.log('[JetsonCamera] Extracted embedding details:', {
          hasEmbedding: !!result.embedding,
          embeddingLength: result.embedding?.length,
          qualityScore: result.quality?.score,
          qualityRating: result.quality?.rating,
          encrypted: result.encrypted,
          hasSessionId: !!result.sessionId
        });

        return {
          success: true,
          data: result
        };
      } else {
        console.error('[JetsonCamera] Enhanced face embedding extraction failed:', {
          status: response.status,
          statusText: response.statusText,
          responseData: data
        });
        throw new Error(data.error || `HTTP ${response.status}: Failed to extract facial embedding`);
      }
    } catch (error) {
      this.log('Enhanced face embedding extraction error:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Failed to extract facial embedding'
      };
    }
  }

  /**
   * Helper method to convert quality score to rating
   */
  private scoreToRating(score: number): 'excellent' | 'good' | 'acceptable' | 'poor' | 'very_poor' {
    if (score >= 90) return 'excellent';
    if (score >= 80) return 'good';
    if (score >= 70) return 'acceptable';
    if (score >= 60) return 'poor';
    return 'very_poor';
  }
} 