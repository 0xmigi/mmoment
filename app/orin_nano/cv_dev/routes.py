"""
CV Dev API Routes

Flask blueprint providing API endpoints for CV development environment.
"""

import os
import logging
from flask import Blueprint, jsonify, request

logger = logging.getLogger("CVDevRoutes")

# Create blueprint
cv_dev_bp = Blueprint('cv_dev', __name__, url_prefix='/api/dev')

# Reference to video buffer service (set by register_blueprint)
_video_buffer_service = None


def init_dev_routes(video_buffer_service):
    """Initialize routes with video buffer service reference."""
    global _video_buffer_service
    _video_buffer_service = video_buffer_service
    logger.info("CV Dev routes initialized")


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
            "POST /api/dev/restart": "Restart video from beginning"
        },
        "examples": {
            "load_video": "curl -X POST localhost:5002/api/dev/load -H 'Content-Type: application/json' -d '{\"path\": \"pushups.mp4\"}'",
            "pause": "curl -X POST localhost:5002/api/dev/playback/pause",
            "seek_to_frame": "curl -X POST localhost:5002/api/dev/playback/seek -H 'Content-Type: application/json' -d '{\"frame\": 100}'",
            "set_speed": "curl -X POST localhost:5002/api/dev/playback/speed -H 'Content-Type: application/json' -d '{\"speed\": 0.5}'"
        }
    })
