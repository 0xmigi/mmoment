# Identity Recognition System - Redesign

## Overview

The identity system follows these principles:
1. **Face is the ONLY way to initially assign identity** - No false positives from body matching
2. **Track continuity maintains identity** - ByteTracker keeps you recognized frame-to-frame
3. **ReID recovers identity after brief occlusions** - Same person, track was lost briefly
4. **ReID cannot assign to new people** - Prevents girlfriend false positive

---

## How It Works

### Initial Recognition (Face Only)

```
New person enters frame → Track created
        ↓
Face visible? ─── NO ──→ Stay UNRECOGNIZED (must face camera)
        │
       YES
        ↓
Face matches checked-in user? ─── NO ──→ Stay UNRECOGNIZED
        │
       YES (similarity > 0.65)
        ↓
RECOGNIZED ✓ (green box, wallet assigned)
        ↓
ReID embedding stored for this track
```

### Identity Persistence

Once recognized, identity persists through ByteTracker:
- **Face visible & matches**: Update confidence, refresh ReID embedding
- **Face visible & mismatch**: Revoke after 15 bad frames (wrong person!)
- **Face not visible** (turned away): Show ReID confidence, identity persists
- **Track continues**: ByteTracker Kalman filter maintains through brief gaps

### Track Loss & Recovery

When ByteTracker loses a track (occlusion, detection failure):

```
Track 5 lost (you went behind furniture)
        ↓
Store in pending_recovery:
  - last_bbox (where you were)
  - reid_embedding (your appearance)
  - lost_time (when track was lost)
        ↓
Track 6 appears...
        ↓
Is it in pending_recovery window? (< 5 seconds)
        ↓
Same spatial region? (IoU > 0.3)
        ↓
ReID matches? (similarity > 0.80)
        ↓
YES to all → RECOVERY: Track 6 gets your identity back
NO to any  → Stay UNRECOGNIZED (must face-match)
```

---

## Key Distinction: Recovery vs Assignment

**ReID Recovery (ALLOWED):**
```
You (recognized on Track 5) → walk behind pole → Track 5 lost
        ↓
Track 6 appears in same spot, same body appearance
        ↓
ReID confirms: "This is the same person who was on Track 5"
        ↓
Identity restored to Track 6
```

**ReID Assignment (BLOCKED):**
```
You (recognized) → leave frame completely
        ↓
Girlfriend enters (completely new person)
        ↓
ReID matches her body to your stored embedding
        ↓
BLOCKED: She was never recognized, no pending_recovery for her
        ↓
She stays UNRECOGNIZED until she faces camera
```

---

## Configuration

```python
# Face matching
FACE_RECOGNITION_THRESHOLD = 0.65      # Confidence to assign identity
FACE_REVOCATION_THRESHOLD = 0.30       # Below this = definitely wrong person
BAD_FRAME_COUNT_REVOCATION = 15        # ~500ms at 30fps

# ReID recovery
REID_SIMILARITY_THRESHOLD = 0.80       # High bar for body matching
REID_RECOVERY_MAX_TIME = 5.0           # Seconds since track lost
REID_RECOVERY_MIN_IOU = 0.3            # Spatial overlap required
```

---

## Data Flow

```
process_native_results()
    │
    ├── STEP 1: Face matching (assigns identity)
    │
    ├── STEP 2: Face mismatch revocation
    │
    ├── STEP 3: Cleanup stale tracks
    │   └── Store in pending_recovery if had identity
    │
    └── STEP 4: ReID maintenance & recovery
        ├── Recognized tracks: update features, show confidence
        └── Unrecognized tracks: attempt recovery from pending_recovery
```

---

## Key Functions

| Function | Purpose |
|----------|---------|
| `_cleanup_stale_tracks()` | Stores recovery data when track is lost |
| `_attempt_track_recovery()` | Recovers identity using ReID + IoU + time window |
| `_maintain_reid_for_recognized_tracks()` | Updates features, shows confidence, triggers recovery |
| `_store_reid_embedding()` | Stores embedding when face is confirmed |

---

## Edge Cases

### Person leaves and returns (> 5 seconds)
- pending_recovery expires
- Must face camera to be recognized again
- This is correct - prevents false positives

### Brief occlusion (< 5 seconds, same region)
- pending_recovery active
- ReID + IoU confirms same person
- Identity recovered automatically

### Two people swap positions
- IoU check prevents cross-assignment
- Each person's recovery data is tied to their last position

### Fast movement (standing up quickly)
- ByteTracker usually handles this
- If track is lost, recovery kicks in
- 5 second window gives plenty of time
