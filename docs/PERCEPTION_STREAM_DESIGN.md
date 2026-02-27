# Perception Stream Architecture

The MMOMENT camera network produces a structured mathematical representation of physical reality — not video, but identity-resolved perception data. This document describes the architecture of that perception stream, how it flows through the network, and how consumers access it.

## Core Insight

The camera system produces a **mathematical abstraction of the physical world** before any application logic runs. The pushup counter never touches a pixel — it reads keypoint coordinates and computes angles. The basketball tracker works from bounding boxes and trajectories. Every consumer of the system operates on the same structured data:

```python
frame_data = {
    'detections': [
        {
            'track_id': int,           # Persistent person tracking ID
            'wallet_address': str,     # Solana wallet (identity-resolved, only if checked in)
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

This is not video. It's a **perception stream** — a real-time structured representation of who is where, doing what. The raw pixels are the input to the pipeline, not its output.

## Cryptographic Consent

Identity resolution is gated by a cryptographic act. A user signs a check-in transaction with their Solana wallet. That signature is what unlocks identity for their bounding box. Without it, the system cannot attribute data to them — they remain an anonymous detection.

This means:
- **No check-in, no identity.** The face embedding match has no wallet to map to.
- **Unconsented bodies are obfuscated.** YOLO detects them, but the network doesn't expose them as identified individuals.
- **Consent is verifiable.** Every identity-resolved data point has a corresponding on-chain check-in signature anyone can audit.
- **Privacy is architectural, not policy.** The system physically cannot produce identity-resolved data without the consent event.

## What Each Camera Produces

### The Perception Pipeline

1. Native C++ server runs YOLOv8 (person detection + tracking) and computes ArcFace embeddings (512-dim)
2. NativeIdentityService matches embeddings against enrolled, checked-in users
3. COCO 17-joint pose estimation runs per detected person
4. Track continuity + ReID (OSNet x0.25) maintains identity across frames and brief occlusions

Output: `frame_data` dicts at frame rate with identity-resolved detections and pose skeletons.

### What Makes Single-Camera Data Useful

The structured output supports pure geometric analysis without additional CV models:

- **Angle calculation**: Dot product of vectors between keypoints (shoulder-elbow-wrist)
- **Position validation**: Torso-vertical / shoulder-width ratio (standing vs. push-up position)
- **View detection**: Keypoint visibility confidence determines front/left/right camera angle
- **Proximity**: Bounding box distance between identified individuals

No model retraining needed. New analyses are just new geometric functions over the same keypoint stream.

## Network-Level Data

The unit of value is not a single camera's stream — it's the **network**. Individual camera #8282229 seeing a user do push-ups is useful. The network knowing that wallet `7xKp...` checked in at the gym at 6am, did 50 push-ups at camera #8282229, walked past camera #4419102 in the lobby, and checked in at the basketball court camera #7731004 at 7pm — that's an activity graph across physical space.

MMOMENT aggregates perception data across all cameras in the network. The API can answer questions no single camera can:

- "What has this user done across all cameras today?"
- "Who is at this location right now?"
- "Show me this user's basketball stats across every court they've played at"
- Cross-camera co-presence, movement patterns, aggregate activity history

All identity-resolved queries require the user's cryptographic check-in at each camera. The network sees more, but only what users have consented to at each point.

## Access Tiers

| Tier | Content | Example Use |
|------|---------|-------------|
| **Events** | Buffered activity events | Social feed, timeline, notifications |
| **Detections** | Real-time bounding boxes + identities | Occupancy, crowd flow, co-presence |
| **Full pose** | Detections + 17-joint keypoints | Fitness coaching, gesture recognition, sports analytics |
| **Pose + frame** | Full stream including raw pixels | Custom CV models, AR/VR, content creation |

Higher tiers include all data from lower tiers. Unconsented (non-checked-in) individuals are excluded from identity-resolved tiers and obfuscated in frame-level tiers.

## Consumer Patterns

Consumers access the network through MMOMENT's API, not through individual cameras. Typical consumption patterns:

**Real-time single-camera stream** — A fitness coaching agent subscribes to live pose data from one gym camera, computes form feedback, sends it to the user. Needs full pose tier.

**Cross-camera user history** — A personal analytics app queries a user's activity across all cameras they've checked into today/week/month. Needs events tier.

**Location intelligence** — A venue analytics service queries real-time occupancy and flow across cameras in a building. Needs detections tier (no individual identity required for aggregate counts).

**Content generation** — A social app detects co-presence (two users checked in at the same camera simultaneously) and generates shared moment content. Needs events or detections tier.

## On-Device vs. Remote Processing

The existing on-camera app system (BaseApp/CompetitionApp in `/opt/mmoment/apps/`) serves a narrow use case: **sub-frame-latency overlays on the camera's own stream.** Live competition rep counts appearing on the stream in real-time require on-device execution.

Most consumers will be remote. A coaching agent on a VPS doing trigonometry on keypoints doesn't need to run on the Jetson. The same `frame_data` contract works in both contexts — the only difference is latency and who pays for compute. Remote is the default path. On-device is the exception for latency-critical visualization.

## Camera as Oracle

Each camera PDA is a Solana-verifiable oracle for physical-world events. When the system counts 15 push-up reps for wallet `7xKp...`:

1. Computed on-device from raw sensor data
2. Encrypted with checked-in users' public keys (AES-256-GCM)
3. Buffered to the backend
4. Committed to the blockchain at checkout

The camera PDA signs the attestation. The on-chain program verifies the camera is registered. The count is camera-attested — not self-reported, not from a device the user controls.

## Composability

Because the stream is structured data (not pixels), it composes naturally:

- **Multi-camera**: Merge activity by wallet address across cameras
- **Temporal**: Query historical activity patterns from buffered events
- **Cross-domain**: Attestation data feeds into DeFi (escrow settlement), social (content creation), identity (activity reputation)

## File References

| Component | Path |
|-----------|------|
| BaseApp / CompetitionApp | `app/orin_nano/apps/sdk/base_app.py` |
| AppManager (plugin loader) | `app/orin_nano/services/camera-service/services/app_manager.py` |
| Push-up app (reference impl) | `app/orin_nano/apps/pushup/app.py` |
| Basketball app (reference impl) | `app/orin_nano/apps/basketball/basketball_app.py` |
| Identity service | `app/orin_nano/services/camera-service/services/native_identity_service.py` |
| Activity encryption | `app/orin_nano/services/camera-service/services/activity_encryption_service.py` |
| Activity buffer client | `app/orin_nano/services/camera-service/services/activity_buffer_client.py` |
| Native buffer (frame flow) | `app/orin_nano/services/camera-service/services/native_buffer_service.py` |
| App API endpoints | `app/orin_nano/services/camera-service/routes.py` |
