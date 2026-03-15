import { useState, useEffect, useCallback, useRef } from 'react';
import { useDynamicContext } from '@dynamic-labs/sdk-react-core';
import { unifiedCameraService } from './unified-camera-service';
import {
  Play,
  Pause,
  SkipBack,
  SkipForward,
  RotateCcw,
  ChevronDown,
  ChevronUp,
  Film,
  Gauge,
  Repeat,
  Upload,
  Loader,
  RotateCw,
  Users,
  Link,
  Unlink,
  Trash2,
  RefreshCw,
  User,
} from 'lucide-react';

interface PlaybackState {
  playing: boolean;
  current_frame: number;
  total_frames: number;
  fps: number;
  speed: number;
  loop: boolean;
  progress: number;
  current_time: number;
  duration: number;
  rotation_enabled?: boolean;
}

interface Track {
  track_id: number;
  bbox: [number, number, number, number];
  confidence?: number;
}

interface TrackLink {
  track_id: number;
  wallet_address: string;
  display_name?: string;
}

interface CVDevPanelProps {
  cameraId: string;
  isExpanded?: boolean;
  onToggleExpand?: () => void;
}

const SPEED_OPTIONS = [0, 0.25, 0.5, 1, 1.5, 2, 4];

export function CVDevPanel({ cameraId, isExpanded = true, onToggleExpand }: CVDevPanelProps) {
  const { primaryWallet, user } = useDynamicContext();
  const [cvDevEnabled, setCvDevEnabled] = useState<boolean | null>(null); // null = loading
  const [videoLoaded, setVideoLoaded] = useState(false);
  const [videoPath, setVideoPath] = useState<string | null>(null);
  const [availableVideos, setAvailableVideos] = useState<string[]>([]);
  const [playbackState, setPlaybackState] = useState<PlaybackState | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(isExpanded);
  const pollingRef = useRef<NodeJS.Timeout | null>(null);

  // Track linking state
  const [tracks, setTracks] = useState<Track[]>([]);
  const [trackLinks, setTrackLinks] = useState<TrackLink[]>([]);
  const [showTrackPanel, setShowTrackPanel] = useState(false);
  const [linkingTrackId, setLinkingTrackId] = useState<number | null>(null);
  const [linkWalletAddress, setLinkWalletAddress] = useState('');
  const [linkDisplayName, setLinkDisplayName] = useState('');
  const [trackLoading, setTrackLoading] = useState(false);

  // Current user info
  const myWalletAddress = primaryWallet?.address;
  // NEVER use email as display name fallback - use username or generic 'Me'
  const myDisplayName = user?.username || 'Me';

  // Format time as MM:SS
  const formatTime = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  // Check CV dev status
  const checkStatus = useCallback(async () => {
    try {
      const response = await unifiedCameraService.getCVDevStatus(cameraId);
      console.log('[CVDevPanel] CV dev status response:', response);

      if (response.success && response.data) {
        // API responded - use the data
        setCvDevEnabled(response.data.enabled);
        setVideoLoaded(response.data.video_loaded);
        setVideoPath(response.data.video_path || null);
        if (response.data.playback_state) {
          setPlaybackState(response.data.playback_state);
        }
      } else {
        // API call failed - but the frontend toggle is on, so assume dev mode is available
        // and just show the controls. Errors will surface when user tries to use them.
        console.log('[CVDevPanel] Status check failed, assuming dev mode is available');
        setCvDevEnabled(true);
      }
    } catch (err) {
      console.error('[CVDevPanel] Error checking status:', err);
      // Same - assume dev mode is available since user toggled it on
      setCvDevEnabled(true);
    }
  }, [cameraId]);

  // List available videos
  const listVideos = useCallback(async () => {
    try {
      const response = await unifiedCameraService.listCVDevVideos(cameraId);
      if (response.success && response.data) {
        setAvailableVideos(response.data.videos || []);
      }
    } catch (err) {
      console.error('[CVDevPanel] Error listing videos:', err);
    }
  }, [cameraId]);

  // Poll playback state when video is loaded
  const pollPlaybackState = useCallback(async () => {
    if (!videoLoaded) return;

    try {
      const response = await unifiedCameraService.cvDevGetPlaybackState(cameraId);
      if (response.success && response.data) {
        setPlaybackState(response.data);
      }
    } catch (err) {
      // Silently ignore polling errors
    }
  }, [cameraId, videoLoaded]);

  // Initial check
  useEffect(() => {
    checkStatus();
    listVideos();
  }, [checkStatus, listVideos]);

  // Start polling when video is loaded
  useEffect(() => {
    if (videoLoaded && expanded && cvDevEnabled) {
      pollingRef.current = setInterval(pollPlaybackState, 500);
    }

    return () => {
      if (pollingRef.current) {
        clearInterval(pollingRef.current);
        pollingRef.current = null;
      }
    };
  }, [videoLoaded, expanded, cvDevEnabled, pollPlaybackState]);

  // Load a video
  const handleLoadVideo = async (path: string) => {
    setLoading(true);
    setError(null);

    try {
      const response = await unifiedCameraService.loadCVDevVideo(cameraId, path);
      if (response.success) {
        setVideoLoaded(true);
        setVideoPath(path);
        await pollPlaybackState();
      } else {
        throw new Error(response.error || 'Failed to load video');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load video');
    } finally {
      setLoading(false);
    }
  };

  // Playback controls
  const handlePlay = async () => {
    await unifiedCameraService.cvDevPlaybackControl(cameraId, 'play');
    await pollPlaybackState();
  };

  const handlePause = async () => {
    await unifiedCameraService.cvDevPlaybackControl(cameraId, 'pause');
    await pollPlaybackState();
  };

  const handleTogglePlay = async () => {
    if (playbackState?.playing) {
      await handlePause();
    } else {
      await handlePlay();
    }
  };

  const handleRestart = async () => {
    await unifiedCameraService.cvDevPlaybackControl(cameraId, 'restart');
    await pollPlaybackState();
  };

  const handleStepBack = async () => {
    await unifiedCameraService.cvDevStep(cameraId, 'backward');
    await pollPlaybackState();
  };

  const handleStepForward = async () => {
    await unifiedCameraService.cvDevStep(cameraId, 'forward');
    await pollPlaybackState();
  };

  const handleSeek = async (progress: number) => {
    await unifiedCameraService.cvDevSeek(cameraId, { progress });
    await pollPlaybackState();
  };

  const handleSetSpeed = async (speed: number) => {
    await unifiedCameraService.cvDevSetSpeed(cameraId, speed);
    await pollPlaybackState();
  };

  const handleToggleLoop = async () => {
    await unifiedCameraService.cvDevSetLoop(cameraId, !playbackState?.loop);
    await pollPlaybackState();
  };

  const handleToggleRotation = async () => {
    await unifiedCameraService.cvDevSetRotation(cameraId, !playbackState?.rotation_enabled);
    await pollPlaybackState();
  };

  const handleToggleExpand = () => {
    setExpanded(!expanded);
    onToggleExpand?.();
  };

  // Track linking handlers
  const refreshTracks = async () => {
    setTrackLoading(true);
    try {
      const [tracksResponse, linksResponse] = await Promise.all([
        unifiedCameraService.cvDevGetTracks(cameraId),
        unifiedCameraService.cvDevGetTrackLinks(cameraId)
      ]);

      if (tracksResponse.success && tracksResponse.data) {
        setTracks(tracksResponse.data.tracks);
      }
      if (linksResponse.success && linksResponse.data) {
        setTrackLinks(linksResponse.data.links);
      }
    } catch (err) {
      console.error('[CVDevPanel] Error refreshing tracks:', err);
    } finally {
      setTrackLoading(false);
    }
  };

  // Link track to current user (me)
  const handleLinkToMe = async (trackId: number) => {
    if (!myWalletAddress) {
      setError('No wallet connected');
      return;
    }

    setTrackLoading(true);
    try {
      const response = await unifiedCameraService.cvDevLinkTrack(
        cameraId,
        trackId,
        myWalletAddress,
        myDisplayName
      );

      if (response.success) {
        await refreshTracks();
      } else {
        setError(response.error || 'Failed to link track');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to link track');
    } finally {
      setTrackLoading(false);
    }
  };

  // Link track to another wallet (manual entry)
  const handleLinkTrack = async (trackId: number) => {
    if (!linkWalletAddress.trim()) return;

    setTrackLoading(true);
    try {
      const response = await unifiedCameraService.cvDevLinkTrack(
        cameraId,
        trackId,
        linkWalletAddress.trim(),
        linkDisplayName.trim() || undefined
      );

      if (response.success) {
        setLinkingTrackId(null);
        setLinkWalletAddress('');
        setLinkDisplayName('');
        await refreshTracks();
      } else {
        setError(response.error || 'Failed to link track');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to link track');
    } finally {
      setTrackLoading(false);
    }
  };

  const handleUnlinkTrack = async (trackId: number) => {
    setTrackLoading(true);
    try {
      const response = await unifiedCameraService.cvDevUnlinkTrack(cameraId, trackId);
      if (response.success) {
        await refreshTracks();
      } else {
        setError(response.error || 'Failed to unlink track');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to unlink track');
    } finally {
      setTrackLoading(false);
    }
  };

  const handleUnlinkAllTracks = async () => {
    setTrackLoading(true);
    try {
      const response = await unifiedCameraService.cvDevUnlinkAllTracks(cameraId);
      if (response.success) {
        await refreshTracks();
      } else {
        setError(response.error || 'Failed to unlink all tracks');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to unlink all tracks');
    } finally {
      setTrackLoading(false);
    }
  };

  // Load tracks when track panel is opened
  useEffect(() => {
    if (showTrackPanel && videoLoaded) {
      refreshTracks();
    }
  }, [showTrackPanel, videoLoaded]);

  return (
    <div className="bg-gray-900 text-white rounded-lg overflow-hidden shadow-lg border border-gray-700">
      {/* Header */}
      <button
        onClick={handleToggleExpand}
        className="w-full flex items-center justify-between px-4 py-3 bg-gray-800 hover:bg-gray-750 transition-colors"
      >
        <div className="flex items-center gap-2">
          <Film className="w-4 h-4 text-yellow-400" />
          <span className="font-medium text-sm">CV App Dev Mode</span>
          {cvDevEnabled === null && (
            <Loader className="w-3 h-3 animate-spin text-gray-400" />
          )}
          {videoLoaded && (
            <span className="text-xs bg-green-600 px-2 py-0.5 rounded">
              Video Loaded
            </span>
          )}
        </div>
        {expanded ? (
          <ChevronUp className="w-4 h-4" />
        ) : (
          <ChevronDown className="w-4 h-4" />
        )}
      </button>

      {expanded && (
        <div className="p-4 space-y-4">
          {/* Show loading state */}
          {cvDevEnabled === null && (
            <div className="flex items-center justify-center py-8">
              <Loader className="w-6 h-6 animate-spin text-yellow-400" />
              <span className="ml-2 text-gray-400">Checking CV dev status...</span>
            </div>
          )}

          {/* Show controls */}
          {cvDevEnabled !== null && (
            <>
              {/* Video Selector */}
              <div className="space-y-2">
                <label className="text-xs text-gray-400 uppercase tracking-wide">
                  Test Video
                </label>
                <div className="flex gap-2">
                  <select
                    value={videoPath || ''}
                    onChange={(e) => e.target.value && handleLoadVideo(e.target.value)}
                    disabled={loading}
                    className="flex-1 bg-gray-800 border border-gray-700 rounded px-3 py-2 text-sm focus:outline-none focus:border-yellow-500"
                  >
                    <option value="">Select a video...</option>
                    {availableVideos.map((video) => (
                      <option key={video} value={video}>
                        {video}
                      </option>
                    ))}
                  </select>
                  {loading && <Loader className="w-5 h-5 animate-spin text-yellow-400" />}
                </div>
                {error && (
                  <p className="text-xs text-red-400">{error}</p>
                )}
              </div>

              {videoLoaded && playbackState && (
                <>
                  {/* Playback Info */}
                  <div className="grid grid-cols-3 gap-2 text-xs">
                    <div className="bg-gray-800 rounded p-2 text-center">
                      <div className="text-gray-400">Frame</div>
                      <div className="font-mono text-yellow-400">
                        {playbackState.current_frame} / {playbackState.total_frames}
                      </div>
                    </div>
                    <div className="bg-gray-800 rounded p-2 text-center">
                      <div className="text-gray-400">Time</div>
                      <div className="font-mono text-yellow-400">
                        {formatTime(playbackState.current_time)} / {formatTime(playbackState.duration)}
                      </div>
                    </div>
                    <div className="bg-gray-800 rounded p-2 text-center">
                      <div className="text-gray-400">FPS</div>
                      <div className="font-mono text-yellow-400">
                        {playbackState.fps.toFixed(1)}
                      </div>
                    </div>
                  </div>

                  {/* Seek Bar */}
                  <div className="space-y-1">
                    <input
                      type="range"
                      min="0"
                      max="100"
                      value={playbackState.progress * 100}
                      onChange={(e) => handleSeek(parseInt(e.target.value) / 100)}
                      className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-yellow-400"
                    />
                  </div>

                  {/* Playback Controls */}
                  <div className="flex items-center justify-center gap-2">
                    <button
                      onClick={handleRestart}
                      className="p-2 hover:bg-gray-800 rounded transition-colors"
                      title="Restart"
                    >
                      <RotateCcw className="w-4 h-4" />
                    </button>
                    <button
                      onClick={handleStepBack}
                      className="p-2 hover:bg-gray-800 rounded transition-colors"
                      title="Step Back"
                    >
                      <SkipBack className="w-4 h-4" />
                    </button>
                    <button
                      onClick={handleTogglePlay}
                      className="p-3 bg-yellow-500 hover:bg-yellow-400 text-gray-900 rounded-full transition-colors"
                      title={playbackState.playing ? 'Pause' : 'Play'}
                    >
                      {playbackState.playing ? (
                        <Pause className="w-5 h-5" />
                      ) : (
                        <Play className="w-5 h-5" />
                      )}
                    </button>
                    <button
                      onClick={handleStepForward}
                      className="p-2 hover:bg-gray-800 rounded transition-colors"
                      title="Step Forward"
                    >
                      <SkipForward className="w-4 h-4" />
                    </button>
                    <button
                      onClick={handleToggleLoop}
                      className={`p-2 rounded transition-colors ${
                        playbackState.loop
                          ? 'bg-yellow-500 text-gray-900'
                          : 'hover:bg-gray-800'
                      }`}
                      title="Loop"
                    >
                      <Repeat className="w-4 h-4" />
                    </button>
                    <button
                      onClick={handleToggleRotation}
                      className={`p-2 rounded transition-colors ${
                        playbackState.rotation_enabled
                          ? 'bg-yellow-500 text-gray-900'
                          : 'hover:bg-gray-800'
                      }`}
                      title="Rotate 180°"
                    >
                      <RotateCw className="w-4 h-4" />
                    </button>
                  </div>

                  {/* Speed Control */}
                  <div className="space-y-2">
                    <div className="flex items-center gap-2 text-xs text-gray-400">
                      <Gauge className="w-3 h-3" />
                      <span>Speed: {playbackState.speed === 0 ? 'Max' : `${playbackState.speed}x`}</span>
                    </div>
                    <div className="flex gap-1">
                      {SPEED_OPTIONS.map((speed) => (
                        <button
                          key={speed}
                          onClick={() => handleSetSpeed(speed)}
                          className={`flex-1 py-1 text-xs rounded transition-colors ${
                            playbackState.speed === speed
                              ? 'bg-yellow-500 text-gray-900'
                              : 'bg-gray-800 hover:bg-gray-700'
                          }`}
                        >
                          {speed === 0 ? 'Max' : `${speed}x`}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Track Linking Section */}
                  <div className="border-t border-gray-700 pt-4 mt-4">
                    <button
                      onClick={() => {
                        setShowTrackPanel(!showTrackPanel);
                        if (!showTrackPanel) refreshTracks();
                      }}
                      className="w-full flex items-center justify-between text-xs text-gray-400 hover:text-white transition-colors"
                    >
                      <div className="flex items-center gap-2">
                        <Users className="w-3 h-3" />
                        <span>Track Linking (Identity Simulation)</span>
                      </div>
                      {showTrackPanel ? (
                        <ChevronUp className="w-3 h-3" />
                      ) : (
                        <ChevronDown className="w-3 h-3" />
                      )}
                    </button>

                    {showTrackPanel && (
                      <div className="mt-3 space-y-3">
                        {/* Refresh and Clear All buttons */}
                        <div className="flex gap-2">
                          <button
                            onClick={refreshTracks}
                            disabled={trackLoading}
                            className="flex-1 flex items-center justify-center gap-1 py-1.5 text-xs bg-gray-800 hover:bg-gray-700 rounded transition-colors disabled:opacity-50"
                          >
                            <RefreshCw className={`w-3 h-3 ${trackLoading ? 'animate-spin' : ''}`} />
                            Refresh Tracks
                          </button>
                          <button
                            onClick={handleUnlinkAllTracks}
                            disabled={trackLoading || trackLinks.length === 0}
                            className="flex items-center justify-center gap-1 px-3 py-1.5 text-xs bg-red-900/50 hover:bg-red-800/50 rounded transition-colors disabled:opacity-50"
                          >
                            <Trash2 className="w-3 h-3" />
                            Clear All
                          </button>
                        </div>

                        {/* Current Links */}
                        {trackLinks.length > 0 && (
                          <div className="space-y-1">
                            <div className="text-xs text-gray-500 uppercase tracking-wide">Active Links</div>
                            {trackLinks.map((link) => (
                              <div
                                key={link.track_id}
                                className="flex items-center justify-between bg-gray-800 rounded px-2 py-1.5 text-xs"
                              >
                                <div className="flex items-center gap-2">
                                  <span className="bg-green-600 px-1.5 py-0.5 rounded text-[10px]">
                                    Track {link.track_id}
                                  </span>
                                  <span className="text-gray-300 truncate max-w-[120px]">
                                    {link.display_name || link.wallet_address.slice(0, 8) + '...'}
                                  </span>
                                </div>
                                <button
                                  onClick={() => handleUnlinkTrack(link.track_id)}
                                  disabled={trackLoading}
                                  className="p-1 hover:bg-gray-700 rounded transition-colors"
                                  title="Unlink"
                                >
                                  <Unlink className="w-3 h-3 text-red-400" />
                                </button>
                              </div>
                            ))}
                          </div>
                        )}

                        {/* Detected Tracks */}
                        <div className="space-y-1">
                          <div className="text-xs text-gray-500 uppercase tracking-wide">
                            Detected Tracks ({tracks.length})
                          </div>
                          {tracks.length === 0 ? (
                            <div className="text-xs text-gray-600 text-center py-2">
                              No tracks detected. Make sure video is playing.
                            </div>
                          ) : (
                            tracks.map((track) => {
                              const existingLink = trackLinks.find(l => l.track_id === track.track_id);
                              const isLinking = linkingTrackId === track.track_id;
                              const isLinkedToMe = existingLink?.wallet_address === myWalletAddress;

                              return (
                                <div
                                  key={track.track_id}
                                  className="bg-gray-800 rounded px-2 py-1.5"
                                >
                                  <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-2 text-xs">
                                      <span className={`px-1.5 py-0.5 rounded text-[10px] ${
                                        existingLink ? 'bg-green-600' : 'bg-blue-600'
                                      }`}>
                                        Track {track.track_id}
                                      </span>
                                      {track.confidence && (
                                        <span className="text-gray-500">
                                          {(track.confidence * 100).toFixed(0)}%
                                        </span>
                                      )}
                                    </div>
                                    {existingLink ? (
                                      <span className={`text-xs ${isLinkedToMe ? 'text-green-400' : 'text-blue-400'}`}>
                                        {isLinkedToMe ? 'You' : existingLink.display_name || existingLink.wallet_address.slice(0, 6) + '...'}
                                      </span>
                                    ) : (
                                      <div className="flex items-center gap-1">
                                        {myWalletAddress && (
                                          <button
                                            onClick={() => handleLinkToMe(track.track_id)}
                                            disabled={trackLoading}
                                            className="flex items-center gap-1 text-xs text-green-400 hover:text-green-300 px-1.5 py-0.5 rounded bg-green-900/30 hover:bg-green-900/50"
                                            title={`Link to ${myDisplayName}`}
                                          >
                                            <User className="w-3 h-3" />
                                            Me
                                          </button>
                                        )}
                                        <button
                                          onClick={() => setLinkingTrackId(isLinking ? null : track.track_id)}
                                          className="flex items-center gap-1 text-xs text-blue-400 hover:text-blue-300 px-1.5 py-0.5 rounded bg-blue-900/30 hover:bg-blue-900/50"
                                        >
                                          <Link className="w-3 h-3" />
                                          {isLinking ? 'Cancel' : 'Other'}
                                        </button>
                                      </div>
                                    )}
                                  </div>

                                  {/* Link Form - for manual wallet entry */}
                                  {isLinking && (
                                    <div className="mt-2 space-y-2">
                                      <input
                                        type="text"
                                        value={linkWalletAddress}
                                        onChange={(e) => setLinkWalletAddress(e.target.value)}
                                        placeholder="Wallet address..."
                                        className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1 text-xs focus:outline-none focus:border-yellow-500"
                                      />
                                      <input
                                        type="text"
                                        value={linkDisplayName}
                                        onChange={(e) => setLinkDisplayName(e.target.value)}
                                        placeholder="Display name (optional)..."
                                        className="w-full bg-gray-700 border border-gray-600 rounded px-2 py-1 text-xs focus:outline-none focus:border-yellow-500"
                                      />
                                      <button
                                        onClick={() => handleLinkTrack(track.track_id)}
                                        disabled={trackLoading || !linkWalletAddress.trim()}
                                        className="w-full py-1.5 text-xs bg-yellow-500 hover:bg-yellow-400 text-gray-900 rounded transition-colors disabled:opacity-50"
                                      >
                                        {trackLoading ? 'Linking...' : 'Link Track'}
                                      </button>
                                    </div>
                                  )}
                                </div>
                              );
                            })
                          )}
                        </div>

                        {/* Help text */}
                        <div className="text-[10px] text-gray-600 leading-relaxed">
                          Click "Me" to link a track to your wallet, or "Other" for manual entry.
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Current Video */}
                  <div className="text-xs text-gray-500 truncate">
                    <Upload className="w-3 h-3 inline mr-1" />
                    {videoPath}
                  </div>
                </>
              )}

              {!videoLoaded && (
                <div className="text-center text-gray-500 py-4">
                  <Film className="w-8 h-8 mx-auto mb-2 opacity-50" />
                  <p className="text-sm">Select a video to start testing CV apps</p>
                </div>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}
