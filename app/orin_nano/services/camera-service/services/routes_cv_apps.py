"""
Camera Service routes for CV Apps integration

Proxy endpoints to control CV apps from camera service API.
"""

from flask import Blueprint, request, jsonify
import logging
from services.cv_apps_client import get_cv_apps_client

logger = logging.getLogger("CVAppsRoutes")

bp = Blueprint('cv_apps', __name__, url_prefix='/api/cv-apps')


@bp.route('/health', methods=['GET'])
def health():
    """Check if cv-apps service is available"""
    client = get_cv_apps_client()
    is_healthy = client.check_health()

    return jsonify({
        'cv_apps_available': is_healthy,
        'active_app': client.active_app if is_healthy else None
    }), 200


@bp.route('/apps/<app_name>/activate', methods=['POST'])
def activate_app(app_name: str):
    """Activate a CV app"""
    client = get_cv_apps_client()

    if client.activate_app(app_name):
        return jsonify({
            'success': True,
            'active_app': app_name
        }), 200
    else:
        return jsonify({
            'error': 'Failed to activate app'
        }), 500


@bp.route('/apps/<app_name>/start', methods=['POST'])
def start_competition(app_name: str):
    """
    Start a competition.

    Body:
    {
        "competitors": [
            {"wallet_address": "...", "display_name": "..."},
            ...
        ],
        "duration_limit": 60,  // optional
        "bet_amount": 0.1,     // optional
        "target_reps": 50      // optional (pushup-specific)
    }
    """
    client = get_cv_apps_client()

    # Ensure app is activated
    if client.active_app != app_name:
        if not client.activate_app(app_name):
            return jsonify({'error': 'Failed to activate app'}), 500

    config = request.get_json()
    result = client.start_competition(app_name, config)

    if result:
        return jsonify(result), 200
    else:
        return jsonify({'error': 'Failed to start competition'}), 500


@bp.route('/apps/<app_name>/end', methods=['POST'])
def end_competition(app_name: str):
    """End competition"""
    client = get_cv_apps_client()
    result = client.end_competition(app_name)

    if result:
        return jsonify(result), 200
    else:
        return jsonify({'error': 'Failed to end competition'}), 500


@bp.route('/apps/<app_name>/state', methods=['GET'])
def get_state(app_name: str):
    """Get current competition state"""
    client = get_cv_apps_client()
    state = client.get_state(app_name)

    if state:
        return jsonify(state), 200
    else:
        return jsonify({'error': 'No state available'}), 404


@bp.route('/toggle', methods=['POST'])
def toggle_cv_apps():
    """Enable/disable CV apps processing"""
    client = get_cv_apps_client()

    data = request.get_json() or {}
    enabled = data.get('enabled', not client.enabled)

    client.enabled = enabled

    return jsonify({
        'success': True,
        'enabled': enabled
    }), 200


logger.info("CV Apps routes registered")
