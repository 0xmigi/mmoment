"""
CV Dev API Routes

Flask blueprint providing API endpoints for CV development environment.
Includes dev-mode track linking for testing CV apps without face enrollment.
"""

import os
import logging
from typing import Dict, Optional
from flask import Blueprint, jsonify, request

logger = logging.getLogger("CVDevRoutes")

# Create blueprint
cv_dev_bp = Blueprint('cv_dev', __name__, url_prefix='/api/dev')

# Reference to video buffer service (set by register_blueprint)
_video_buffer_service = None

# Dev track-to-wallet linking state
# Maps track_id -> {wallet_address, display_name, linked_at}
_dev_track_links: Dict[int, Dict] = {}


def init_dev_routes(video_buffer_service):
    """Initialize routes with video buffer service reference."""
    global _video_buffer_service
    _video_buffer_service = video_buffer_service
    logger.info("CV Dev routes initialized with track linking support")


def get_dev_track_link(track_id: int) -> Optional[Dict]:
    """
    Get dev wallet link for a track_id (called by identity_tracker).

    Returns:
        Dict with wallet_address and display_name, or None if not linked
    """
    return _dev_track_links.get(track_id)


def get_all_dev_track_links() -> Dict[int, Dict]:
    """Get all dev track links."""
    return _dev_track_links.copy()


@cv_dev_bp.route('/status', methods=['GET'])
def get_dev_status():
    """Get CV dev environment status."""
    if _video_buffer_service is None:
        return jsonify({"error": "Dev mode not initialized"}), 500

    return jsonify({
        "mode": "cv_dev",
        "status": _video_buffer_service.get_status()
    })


# =========================================================================
# Video Library Management
# =========================================================================

@cv_dev_bp.route('/videos', methods=['GET'])
def list_videos():
    """List available videos in the dev_videos directory."""
    videos_dir = os.path.join(os.path.dirname(__file__), 'videos')

    if not os.path.exists(videos_dir):
        return jsonify({"videos": [], "directory": videos_dir})

    videos = []
    valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.webm'}

    for filename in os.listdir(videos_dir):
        ext = os.path.splitext(filename)[1].lower()
        if ext in valid_extensions:
            filepath = os.path.join(videos_dir, filename)
            videos.append({
                "name": filename,
                "path": filepath,
                "size_mb": round(os.path.getsize(filepath) / (1024 * 1024), 2)
            })

    return jsonify({
        "videos": videos,
        "directory": videos_dir,
        "count": len(videos)
    })


@cv_dev_bp.route('/load', methods=['POST'])
def load_video():
    """Load a video file for playback."""
    if _video_buffer_service is None:
        return jsonify({"error": "Dev mode not initialized"}), 500

    data = request.get_json() or {}
    video_path = data.get('path') or data.get('video_path')

    if not video_path:
        return jsonify({"error": "Missing 'path' parameter"}), 400

    # If just filename provided, look in videos directory
    if not os.path.isabs(video_path) and not video_path.startswith('/'):
        videos_dir = os.path.join(os.path.dirname(__file__), 'videos')
        video_path = os.path.join(videos_dir, video_path)

    success = _video_buffer_service.load_video(video_path)

    if success:
        # Auto-start playback
        _video_buffer_service.start()
        return jsonify({
            "success": True,
            "message": f"Loaded video: {video_path}",
            "playback": _video_buffer_service.get_playback_state()
        })
    else:
        return jsonify({
            "success": False,
            "error": f"Failed to load video: {video_path}"
        }), 400


# =========================================================================
# Playback Controls
# =========================================================================

@cv_dev_bp.route('/playback/state', methods=['GET'])
def get_playback_state():
    """Get current playback state."""
    if _video_buffer_service is None:
        return jsonify({"error": "Dev mode not initialized"}), 500

    return jsonify(_video_buffer_service.get_playback_state())


@cv_dev_bp.route('/playback/play', methods=['POST'])
def play():
    """Resume playback."""
    if _video_buffer_service is None:
        return jsonify({"error": "Dev mode not initialized"}), 500

    _video_buffer_service.play()
    return jsonify({
        "success": True,
        "playing": True,
        "playback": _video_buffer_service.get_playback_state()
    })


@cv_dev_bp.route('/playback/pause', methods=['POST'])
def pause():
    """Pause playback."""
    if _video_buffer_service is None:
        return jsonify({"error": "Dev mode not initialized"}), 500

    _video_buffer_service.pause()
    return jsonify({
        "success": True,
        "playing": False,
        "playback": _video_buffer_service.get_playback_state()
    })


@cv_dev_bp.route('/playback/toggle', methods=['POST'])
def toggle_playback():
    """Toggle play/pause."""
    if _video_buffer_service is None:
        return jsonify({"error": "Dev mode not initialized"}), 500

    state = _video_buffer_service.get_playback_state()
    if state['playing']:
        _video_buffer_service.pause()
    else:
        _video_buffer_service.play()

    return jsonify({
        "success": True,
        "playback": _video_buffer_service.get_playback_state()
    })


@cv_dev_bp.route('/playback/seek', methods=['POST'])
def seek():
    """Seek to a specific frame or time."""
    if _video_buffer_service is None:
        return jsonify({"error": "Dev mode not initialized"}), 500

    data = request.get_json() or {}

    if 'frame' in data:
        success = _video_buffer_service.seek(int(data['frame']))
    elif 'time' in data:
        success = _video_buffer_service.seek_time(float(data['time']))
    elif 'progress' in data:
        # Seek by progress (0.0 to 1.0)
        state = _video_buffer_service.get_playback_state()
        frame = int(float(data['progress']) * state['total_frames'])
        success = _video_buffer_service.seek(frame)
    else:
        return jsonify({"error": "Missing 'frame', 'time', or 'progress' parameter"}), 400

    return jsonify({
        "success": success,
        "playback": _video_buffer_service.get_playback_state()
    })


@cv_dev_bp.route('/playback/speed', methods=['POST'])
def set_speed():
    """Set playback speed."""
    if _video_buffer_service is None:
        return jsonify({"error": "Dev mode not initialized"}), 500

    data = request.get_json() or {}
    speed = data.get('speed', 1.0)

    _video_buffer_service.set_speed(float(speed))

    return jsonify({
        "success": True,
        "speed": speed,
        "playback": _video_buffer_service.get_playback_state()
    })


@cv_dev_bp.route('/playback/loop', methods=['POST'])
def set_loop():
    """Enable or disable looping."""
    if _video_buffer_service is None:
        return jsonify({"error": "Dev mode not initialized"}), 500

    data = request.get_json() or {}
    enabled = data.get('enabled', True)

    _video_buffer_service.set_loop(bool(enabled))

    return jsonify({
        "success": True,
        "loop": enabled,
        "playback": _video_buffer_service.get_playback_state()
    })


@cv_dev_bp.route('/playback/step', methods=['POST'])
def step():
    """Step forward or backward one frame."""
    if _video_buffer_service is None:
        return jsonify({"error": "Dev mode not initialized"}), 500

    data = request.get_json() or {}
    direction = data.get('direction', 'forward')

    if direction == 'forward' or direction == 1:
        _video_buffer_service.step_forward()
    elif direction == 'backward' or direction == -1:
        _video_buffer_service.step_backward()
    else:
        return jsonify({"error": "Invalid direction. Use 'forward' or 'backward'"}), 400

    # Small delay to let step complete
    import time
    time.sleep(0.1)

    return jsonify({
        "success": True,
        "direction": direction,
        "playback": _video_buffer_service.get_playback_state()
    })


@cv_dev_bp.route('/playback/rotation', methods=['POST'])
def set_rotation():
    """Enable or disable frame rotation.

    Pre-recorded videos (YouTube) don't need rotation - they're already correct.
    Only enable rotation if simulating camera behavior (camera is mounted sideways).
    """
    if _video_buffer_service is None:
        return jsonify({"error": "Dev mode not initialized"}), 500

    data = request.get_json() or {}
    enabled = data.get('enabled', False)

    _video_buffer_service.set_rotation(bool(enabled))

    return jsonify({
        "success": True,
        "rotation_enabled": enabled,
        "playback": _video_buffer_service.get_playback_state()
    })


# =========================================================================
# Convenience Endpoints
# =========================================================================

@cv_dev_bp.route('/restart', methods=['POST'])
def restart_video():
    """Restart video from beginning."""
    if _video_buffer_service is None:
        return jsonify({"error": "Dev mode not initialized"}), 500

    _video_buffer_service.seek(0)
    _video_buffer_service.play()

    return jsonify({
        "success": True,
        "message": "Video restarted",
        "playback": _video_buffer_service.get_playback_state()
    })


@cv_dev_bp.route('/help', methods=['GET'])
def get_help():
    """Get help on available endpoints."""
    return jsonify({
        "endpoints": {
            "GET /api/dev/status": "Get dev environment status",
            "GET /api/dev/videos": "List available videos",
            "POST /api/dev/load": "Load a video (body: {path: 'video.mp4'})",
            "GET /api/dev/playback/state": "Get playback state",
            "POST /api/dev/playback/play": "Resume playback",
            "POST /api/dev/playback/pause": "Pause playback",
            "POST /api/dev/playback/toggle": "Toggle play/pause",
            "POST /api/dev/playback/seek": "Seek (body: {frame: N} or {time: S} or {progress: 0.5})",
            "POST /api/dev/playback/speed": "Set speed (body: {speed: 1.0})",
            "POST /api/dev/playback/loop": "Set loop (body: {enabled: true})",
            "POST /api/dev/playback/step": "Step frame (body: {direction: 'forward'})",
            "POST /api/dev/restart": "Restart video from beginning",
            "GET /api/dev/tracks": "Get currently detected track_ids",
            "GET /api/dev/tracks/links": "Get current track-to-wallet links",
            "POST /api/dev/tracks/link": "Link a track_id to wallet (body: {track_id, wallet_address, display_name})",
            "POST /api/dev/tracks/unlink": "Unlink a track_id (body: {track_id})",
            "POST /api/dev/tracks/unlink-all": "Unlink all tracks"
        },
        "examples": {
            "load_video": "curl -X POST localhost:5002/api/dev/load -H 'Content-Type: application/json' -d '{\"path\": \"pushups.mp4\"}'",
            "pause": "curl -X POST localhost:5002/api/dev/playback/pause",
            "seek_to_frame": "curl -X POST localhost:5002/api/dev/playback/seek -H 'Content-Type: application/json' -d '{\"frame\": 100}'",
            "set_speed": "curl -X POST localhost:5002/api/dev/playback/speed -H 'Content-Type: application/json' -d '{\"speed\": 0.5}'",
            "link_track": "curl -X POST localhost:5002/api/dev/tracks/link -H 'Content-Type: application/json' -d '{\"track_id\": 1, \"wallet_address\": \"ABC...\", \"display_name\": \"Test User\"}'"
        }
    })


# =========================================================================
# Track Linking (Dev Mode Identity Simulation)
# =========================================================================

# Reference to services (set via init)
_gpu_face_service = None


def init_track_services(gpu_face_service):
    """Initialize track linking with GPU face service reference."""
    global _gpu_face_service
    _gpu_face_service = gpu_face_service
    logger.info("Track linking services initialized")


@cv_dev_bp.route('/tracks', methods=['GET'])
def get_detected_tracks():
    """
    Get currently detected track_ids from the video.

    Returns list of tracks with their bounding boxes and any linked wallet info.
    """
    if _gpu_face_service is None:
        return jsonify({"error": "GPU face service not initialized"}), 500

    try:
        # Get current detections from the face service
        detections = _gpu_face_service.get_current_detections()

        tracks = []
        for det in detections:
            if det.get('class') == 'person' and det.get('track_id') is not None:
                track_id = int(det['track_id'])
                track_info = {
                    'track_id': track_id,
                    'bbox': {
                        'x1': int(det.get('x1', 0)),
                        'y1': int(det.get('y1', 0)),
                        'x2': int(det.get('x2', 0)),
                        'y2': int(det.get('y2', 0))
                    },
                    'confidence': float(det.get('confidence', 0)),
                    'linked': track_id in _dev_track_links
                }

                # Add link info if linked
                if track_id in _dev_track_links:
                    link = _dev_track_links[track_id]
                    track_info['wallet_address'] = link['wallet_address']
                    track_info['display_name'] = link.get('display_name')

                tracks.append(track_info)

        return jsonify({
            "success": True,
            "tracks": tracks,
            "count": len(tracks)
        })

    except Exception as e:
        logger.error(f"Error getting tracks: {e}")
        return jsonify({"error": str(e)}), 500


@cv_dev_bp.route('/tracks/links', methods=['GET'])
def get_track_links():
    """Get all current track-to-wallet links."""
    links = []
    for track_id, link in _dev_track_links.items():
        links.append({
            'track_id': track_id,
            'wallet_address': link['wallet_address'],
            'display_name': link.get('display_name'),
            'linked_at': link.get('linked_at')
        })

    return jsonify({
        "success": True,
        "links": links,
        "count": len(links)
    })


@cv_dev_bp.route('/tracks/link', methods=['POST'])
def link_track():
    """
    Link a track_id to a wallet address.

    This simulates a user being "recognized" in the video for CV app testing.
    The linked wallet will receive all CV app events for that track.

    Body: {
        "track_id": 1,
        "wallet_address": "ABC123...",
        "display_name": "Test User"  // optional
    }
    """
    import time

    data = request.get_json() or {}

    track_id = data.get('track_id')
    wallet_address = data.get('wallet_address')
    display_name = data.get('display_name')

    if track_id is None:
        return jsonify({"error": "track_id is required"}), 400

    if not wallet_address:
        return jsonify({"error": "wallet_address is required"}), 400

    # Convert to int
    track_id = int(track_id)

    # Store the link
    _dev_track_links[track_id] = {
        'wallet_address': wallet_address,
        'display_name': display_name or wallet_address[:8],
        'linked_at': time.time()
    }

    logger.info(f"DEV: Linked track_id={track_id} to wallet={wallet_address[:8]}...")

    return jsonify({
        "success": True,
        "message": f"Linked track {track_id} to {wallet_address[:8]}...",
        "link": {
            "track_id": track_id,
            "wallet_address": wallet_address,
            "display_name": display_name or wallet_address[:8]
        }
    })


@cv_dev_bp.route('/tracks/unlink', methods=['POST'])
def unlink_track():
    """
    Unlink a track_id from its wallet.

    Body: {"track_id": 1}
    """
    data = request.get_json() or {}
    track_id = data.get('track_id')

    if track_id is None:
        return jsonify({"error": "track_id is required"}), 400

    track_id = int(track_id)

    if track_id in _dev_track_links:
        removed = _dev_track_links.pop(track_id)
        logger.info(f"DEV: Unlinked track_id={track_id} from wallet={removed['wallet_address'][:8]}...")
        return jsonify({
            "success": True,
            "message": f"Unlinked track {track_id}",
            "removed": {
                "track_id": track_id,
                "wallet_address": removed['wallet_address']
            }
        })
    else:
        return jsonify({
            "success": False,
            "error": f"Track {track_id} not linked"
        }), 404


@cv_dev_bp.route('/tracks/unlink-all', methods=['POST'])
def unlink_all_tracks():
    """Unlink all tracks."""
    count = len(_dev_track_links)
    _dev_track_links.clear()

    logger.info(f"DEV: Unlinked all {count} tracks")

    return jsonify({
        "success": True,
        "message": f"Unlinked {count} tracks",
        "count": count
    })
