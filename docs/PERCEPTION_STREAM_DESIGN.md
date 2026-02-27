# Perception Stream as x402 Primitive

This document describes the architecture for treating MMOMENT's computer vision pipeline as a monetizable perception stream — a structured mathematical representation of physical reality that AI agents and developers can subscribe to via x402 micropayments.

## Core Insight

The camera system already produces a **mathematical abstraction of the physical world** before any application logic runs. The pushup counter never touches a pixel — it reads keypoint coordinates and computes angles. The basketball tracker works from bounding boxes and trajectories. Every CV app in the plugin system consumes the same structured `frame_data` dictionary:

```python
frame_data = {
    'detections': [
        {
            'track_id': int,           # Persistent person tracking ID
            'wallet_address': str,     # Solana wallet (identity-resolved)
            'x1': float, 'y1': float,
            'x2': float, 'y2': float,  # Bounding box
            'confidence': float,
        }
    ],
    'keypoints': [
        {
            'track_id': int,           # Matches detection track_id
            'keypoints': np.ndarray    # (17, 3) COCO format [x, y, conf]
        }
    ],
    'timestamp': float,
    'frame': np.ndarray               # Optional raw frame
}
```

This is not video. It's a **perception stream** — a real-time structured representation of who is where, doing what, with cryptographic identity attached. The raw pixels are the input to the pipeline, not its output.

## What Makes This Different

### Identity-Resolved Detections

Every detection carries a `wallet_address` from the identity pipeline:

1. Native C++ server detects faces and computes ArcFace embeddings (512-dim)
2. NativeIdentityService matches embeddings against enrolled faces
3. Track continuity maintains identity across frames
4. ReID (OSNet x0.25) re-acquires identity after brief occlusions

Result: each bounding box in the stream is tied to a specific Solana wallet. The stream doesn't just say "person at (x1,y1,x2,y2)" — it says "wallet `7xKp...` at (x1,y1,x2,y2) with 17-joint pose skeleton."

### Privacy by Architecture

Activities derived from the perception stream are encrypted at the Jetson before leaving the device:

1. AES-256-GCM encrypts the activity content
2. Each checked-in user gets an access grant (their pubkey encrypts the AES key)
3. Only encrypted bundles reach the backend
4. Users decrypt with their Solana private key at read time

The stream itself never leaves the camera. Only encrypted derivatives do.

### Scale-Invariant Math

The existing apps demonstrate how to build robust detections from the stream using pure geometry:

- **Angle calculation**: Dot product of vectors between keypoints (shoulder-elbow-wrist)
- **Position validation**: Torso-vertical / shoulder-width ratio (standing vs. push-up)
- **View detection**: Keypoint visibility confidence determines front/left/right
- **Anti-cheat**: Minimum rep timing, wrist-above-shoulder checks

No model retraining needed. New activities are just new geometric functions over the same keypoint stream.

## x402 Access Model

### The Primitive

The perception stream is the billable unit. An x402 payment grants time-bounded access to the structured `frame_data` output of a specific camera PDA. The subscriber receives:

- **Detections**: Identity-resolved bounding boxes at frame rate
- **Poses**: 17-joint COCO keypoints per tracked person
- **Events**: Buffered activity events from active CV apps
- **Metadata**: Camera PDA, timestamp, checked-in wallet list

The subscriber does NOT receive raw video frames unless explicitly requested and paid for at a higher tier. The mathematical stream is both more useful (structured, queryable) and more privacy-preserving (no pixel data) than raw video.

### Tiered Access

| Tier | Content | Use Case |
|------|---------|----------|
| **Events only** | Buffered activity events (encrypted) | Social feed, timeline aggregation |
| **Detections** | Real-time bounding boxes + identities | Occupancy analytics, crowd flow |
| **Full pose** | Detections + 17-joint keypoints | Fitness apps, gesture recognition, sports analytics |
| **Pose + frame** | Full stream including raw pixels | Custom CV models, AR/VR integration |

Each tier has a different x402 price per second of access. The camera owner (Solana wallet that registered the camera PDA) receives payment.

### Agent Consumption

AI agents are the primary expected consumer. An agent that wants to build a fitness coaching product:

1. Discovers cameras via on-chain camera registry
2. Pays x402 for "full pose" tier access to a gym camera
3. Receives the `frame_data` stream in real-time
4. Runs its own angle/position analysis (like pushup app does)
5. Sends coaching feedback to the user's wallet

The agent never needs to run its own computer vision. The Jetson already did the expensive GPU inference. The agent just does math on coordinates.

## Where Developers Plug In

### Current Plugin System (On-Camera)

Apps run directly on the Jetson as Python modules:

```
/opt/mmoment/apps/
└── myapp/
    ├── __init__.py
    └── app.py          # Must expose get_app() or App class
```

Activated via HTTP API:
```
POST /api/apps/load     {"app_name": "myapp"}
POST /api/apps/activate {"app_name": "myapp"}
```

Single active app at a time. App receives `frame_data`, returns `state` + `visualization` + optional `event`/`should_buffer` flags.

### Future: Remote App Execution via x402

The same `frame_data` dict that on-camera apps receive can be serialized and streamed to remote subscribers. A remote app would:

1. Subscribe to a camera's perception stream (x402 payment)
2. Receive `frame_data` dicts over WebSocket/SSE
3. Process them identically to an on-camera app
4. Return `state`/`visualization` back to the camera for overlay rendering (optional)
5. Or just consume silently for analytics/coaching

This means the same app code works both on-camera and remote — the only difference is latency and who pays for compute.

### App Development Contract

Whether on-camera or remote, every app implements the same interface:

```python
class MyApp(BaseApp):
    def process(self, frame_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Input:  frame_data with detections, keypoints, timestamp
        Output: {
            'state': {},           # App-specific state
            'visualization': {},   # Drawing commands (skeleton, text, boxes, lines, circles)
            'event': str,          # Optional discrete event name
            'should_buffer': bool  # Optional force-buffer flag
        }
        """
```

For competition apps (multi-player with escrow):

```python
class MyCompetitionApp(CompetitionApp):
    def init_competitor_stats(self) -> Dict:
        """Return initial stats dict for each competitor"""

    def process(self, frame_data) -> Dict:
        """Process frame, update self.competitors[wallet]['stats']"""
```

## Architectural Implications

### The Camera is an Oracle

Each camera PDA is a Solana-verifiable oracle for physical-world events. When the pushup app counts 15 reps for wallet `7xKp...`, that count is:

1. Computed on-device from raw sensor data
2. Encrypted with the user's public key
3. Buffered to the backend
4. Committed to the blockchain at checkout

The camera PDA signs the attestation. The on-chain program can verify the camera is registered and trusted. This is a credibly neutral count of push-ups — not self-reported, not from a phone the user controls.

### Composability

Because the stream is structured data (not pixels), it composes naturally:

- **Multi-camera**: Merge detections from overlapping cameras by wallet address
- **Multi-app**: Run a fitness tracker and a social proximity detector on the same stream
- **Temporal**: Query "show me all moments where wallet X was within 2m of wallet Y" from historical buffered events
- **Cross-chain**: Attestation data feeds into DeFi (escrow settlement), social (content creation), identity (reputation)

### What This Enables

- **Fitness-as-a-Service**: Gym cameras sell pose streams to coaching agents
- **Verified Competitions**: Escrow-backed 1v1 challenges with camera-attested results
- **Social Moments**: "You were both here" notifications from co-presence detection
- **Occupancy Intelligence**: Real-time anonymized crowd analytics (detections tier, no identity)
- **Content Attribution**: Camera-signed proof that a specific person was in a specific frame

## File References

| Component | Path |
|-----------|------|
| BaseApp / CompetitionApp | `app/orin_nano/apps/sdk/base_app.py` |
| AppManager (plugin loader) | `app/orin_nano/services/camera-service/services/app_manager.py` |
| Push-up app (reference) | `app/orin_nano/apps/pushup/app.py` |
| Basketball app (reference) | `app/orin_nano/apps/basketball/basketball_app.py` |
| Identity service | `app/orin_nano/services/camera-service/services/native_identity_service.py` |
| Activity encryption | `app/orin_nano/services/camera-service/services/activity_encryption_service.py` |
| Activity buffer client | `app/orin_nano/services/camera-service/services/activity_buffer_client.py` |
| Native buffer (frame flow) | `app/orin_nano/services/camera-service/services/native_buffer_service.py` |
| App API endpoints | `app/orin_nano/services/camera-service/routes.py` |
