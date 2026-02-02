"""
Base CV App Class

All mmoment CV apps inherit from this base class.
Provides standard interface for frame processing and visualization.
"""

from typing import Dict, Any, Optional, List
import numpy as np
import logging

logger = logging.getLogger("BaseApp")


class BaseApp:
    """
    Base class for all CV apps.

    Apps implement process() to receive frame data and return:
    {
        'state': {...},  # App-specific state (reps, score, etc.)
        'visualization': {
            'skeleton': bool,  # Draw pose skeleton
            'text': [{'content': str, 'pos': (x, y), 'color': (r,g,b)}],
            'lines': [{'start': (x1,y1), 'end': (x2,y2), 'color': (r,g,b), 'thickness': int}],
            'circles': [{'center': (x,y), 'radius': int, 'color': (r,g,b)}]
        }
    }
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize app.

        Args:
            config: Optional configuration dict
        """
        self.config = config or {}
        self.enabled = True
        logger.info(f"{self.__class__.__name__} initialized")

    def on_activate(self):
        """Called when app is activated (becomes the active app)"""
        pass

    def on_deactivate(self):
        """Called when app is deactivated (another app becomes active)"""
        pass

    def process(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process frame data.

        Args:
            frame_data: Dict containing:
                - 'detections': List of person detections with bboxes, track_ids, wallet_addresses
                - 'keypoints': List of pose keypoints (17, 3) arrays
                - 'timestamp': Frame timestamp
                - 'frame': Optional - raw frame (np.ndarray) if app needs it

        Returns:
            {
                'state': Dict with app-specific state,
                'visualization': Dict with drawing commands
            }
        """
        raise NotImplementedError("Apps must implement process()")

    def get_state(self) -> Dict[str, Any]:
        """
        Get current app state.

        Returns:
            Dict with current app state
        """
        return {}

    def reset(self):
        """Reset app state"""
        pass


class CompetitionApp(BaseApp):
    """
    Base class for competition-style apps (push-ups, pull-ups, etc.)

    Adds common competition features:
    - Competitor tracking
    - Rep counting
    - Time limits
    - Leaderboard
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)

        self.active = False
        self.competitors = {}  # wallet -> competitor data
        self.start_time = None
        self.duration_limit = None
        self.competition_meta = None  # Escrow info for on-chain competitions

    def start_competition(self, competitors: List[Dict], duration_limit: Optional[int] = None, competition_meta: Optional[Dict] = None):
        """
        Start competition.

        Args:
            competitors: List of {'wallet_address': str, 'display_name': str}
            duration_limit: Optional time limit in seconds
            competition_meta: Optional escrow metadata {'escrow_pda': str, 'mode': str, ...}
        """
        import time
        self.active = True
        self.start_time = time.time()
        self.duration_limit = duration_limit
        self.competition_meta = competition_meta  # Store for non-initiators to sync

        for comp in competitors:
            wallet = comp['wallet_address']
            self.competitors[wallet] = {
                'wallet_address': wallet,
                'display_name': comp.get('display_name', wallet[:8]),
                'stats': self.init_competitor_stats(),
                'track_id': None,
                'last_seen': time.time()
            }

        logger.info(f"Competition started: {len(competitors)} competitors")

    def init_competitor_stats(self) -> Dict:
        """Initialize stats for a competitor (override in subclass)"""
        return {}

    def end_competition(self) -> Dict:
        """End competition and return results"""
        self.active = False
        logger.info("Competition ended")

        return {
            'active': False,
            'ended': True,
            'competitors': list(self.competitors.values())
        }

    def get_state(self) -> Dict:
        """Get competition state"""
        import time
        if not self.active:
            return {'active': False}

        elapsed = time.time() - self.start_time if self.start_time else 0
        time_remaining = self.duration_limit - elapsed if self.duration_limit else None

        state = {
            'active': True,
            'competitors': list(self.competitors.values()),
            'elapsed': elapsed,
            'time_remaining': time_remaining
        }
        # Include escrow_pda so non-initiators can sync (frontend expects state.escrow_pda)
        if self.competition_meta and self.competition_meta.get('escrow_pda'):
            state['escrow_pda'] = self.competition_meta['escrow_pda']
            state['competition_mode'] = self.competition_meta.get('mode')
        return state

    def reset(self):
        """Reset competition"""
        self.active = False
        self.competitors = {}
        self.start_time = None
