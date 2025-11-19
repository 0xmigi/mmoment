"""
Flask routes for Push-up Competition App

Integrates with mmoment camera service to provide:
- Competition lifecycle endpoints
- Real-time state/stats
- Betting/escrow hooks
- WebSocket for live updates
"""

from flask import Blueprint, request, jsonify
import logging
from typing import Dict, Optional

logger = logging.getLogger("PushupRoutes")

# Global app instance (initialized by camera service)
pushup_app: Optional['PushupCompetitionApp'] = None


def init_routes(app_instance) -> Blueprint:
    """
    Initialize routes with app instance.

    Args:
        app_instance: PushupCompetitionApp instance

    Returns:
        Flask Blueprint
    """
    global pushup_app
    pushup_app = app_instance

    bp = Blueprint('pushup', __name__, url_prefix='/api/apps/pushup')

    @bp.route('/start', methods=['POST'])
    def start_competition():
        """
        Start a push-up competition.

        Body:
        {
            "competitors": [
                {"wallet_address": "...", "display_name": "..."},
                ...
            ],
            "duration_limit": 60,  // optional, seconds
            "bet_amount": 0.1,     // optional, SOL per competitor
            "target_reps": 50      // optional, first to reach wins
        }
        """
        try:
            data = request.get_json()

            competitors = data.get('competitors', [])
            if len(competitors) < 1:
                return jsonify({'error': 'Need at least 1 competitor'}), 400

            # Validate competitors have required fields
            for comp in competitors:
                if 'wallet_address' not in comp:
                    return jsonify({'error': 'Each competitor needs wallet_address'}), 400

            duration_limit = data.get('duration_limit')
            bet_amount = data.get('bet_amount', 0.0)
            target_reps = data.get('target_reps')

            # Update app config with target reps if provided
            if target_reps:
                pushup_app.config['target_reps'] = target_reps

            result = pushup_app.start_competition(
                competitors=competitors,
                duration_limit=duration_limit,
                bet_amount=bet_amount
            )

            return jsonify(result), 200

        except Exception as e:
            logger.error(f"Failed to start competition: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500

    @bp.route('/state', methods=['GET'])
    def get_state():
        """Get current competition state"""
        try:
            state = pushup_app._get_competition_state()
            return jsonify(state), 200
        except Exception as e:
            logger.error(f"Failed to get state: {e}")
            return jsonify({'error': str(e)}), 500

    @bp.route('/end', methods=['POST'])
    def end_competition():
        """Manually end the competition"""
        try:
            result = pushup_app.end_competition()
            return jsonify(result), 200
        except Exception as e:
            logger.error(f"Failed to end competition: {e}")
            return jsonify({'error': str(e)}), 500

    @bp.route('/attestation', methods=['GET'])
    def get_attestation():
        """Get on-chain attestation data"""
        try:
            attestation = pushup_app.get_attestation_data()
            if not attestation:
                return jsonify({'error': 'No competition data available'}), 404

            return jsonify(attestation), 200
        except Exception as e:
            logger.error(f"Failed to get attestation: {e}")
            return jsonify({'error': str(e)}), 500

    @bp.route('/config', methods=['GET'])
    def get_config():
        """Get current app configuration"""
        return jsonify({
            'down_angle_threshold': pushup_app.down_angle_threshold,
            'up_angle_threshold': pushup_app.up_angle_threshold,
            'min_rep_time': pushup_app.min_rep_time,
            'target_reps': pushup_app.config.get('target_reps'),
            'body_alignment_threshold': pushup_app.body_alignment_threshold
        }), 200

    @bp.route('/config', methods=['PUT'])
    def update_config():
        """
        Update app configuration.

        Body:
        {
            "down_angle_threshold": 90,
            "up_angle_threshold": 160,
            "min_rep_time": 0.5,
            "target_reps": 50
        }
        """
        try:
            data = request.get_json()

            if 'down_angle_threshold' in data:
                pushup_app.down_angle_threshold = float(data['down_angle_threshold'])
            if 'up_angle_threshold' in data:
                pushup_app.up_angle_threshold = float(data['up_angle_threshold'])
            if 'min_rep_time' in data:
                pushup_app.min_rep_time = float(data['min_rep_time'])
            if 'target_reps' in data:
                pushup_app.config['target_reps'] = int(data['target_reps'])
            if 'body_alignment_threshold' in data:
                pushup_app.body_alignment_threshold = float(data['body_alignment_threshold'])

            return jsonify({'success': True, 'config': get_config()[0].get_json()}), 200

        except Exception as e:
            logger.error(f"Failed to update config: {e}")
            return jsonify({'error': str(e)}), 500

    @bp.route('/health', methods=['GET'])
    def health():
        """Health check"""
        return jsonify({
            'status': 'healthy',
            'pose_estimation_available': pushup_app.pose_estimator is not None,
            'active_competition': pushup_app.state.active,
            'competitors_count': len(pushup_app.competitors)
        }), 200

    logger.info("Push-up competition routes initialized")
    return bp


# Example: Integration with escrow/betting system
def create_escrow_transaction(competitors: list, bet_amount: float) -> Dict:
    """
    Create on-chain escrow transaction for competition.

    This would integrate with Solana middleware to:
    1. Create PDA for competition
    2. Lock funds from each competitor
    3. Set competition parameters on-chain

    Args:
        competitors: List of competitor wallet addresses
        bet_amount: Amount per competitor (SOL)

    Returns:
        Transaction signature and escrow PDA
    """
    # TODO: Integrate with solana-middleware service
    # POST to http://solana-middleware:5001/api/escrow/create
    # with competition_id, competitors, amounts
    pass


def release_escrow_to_winner(competition_id: str, winner_wallet: str) -> Dict:
    """
    Release escrowed funds to competition winner.

    Args:
        competition_id: Competition ID
        winner_wallet: Winner's wallet address

    Returns:
        Transaction signature
    """
    # TODO: Integrate with solana-middleware service
    # POST to http://solana-middleware:5001/api/escrow/release
    # with competition_id and winner_wallet
    pass
