# Face Recognition & ReID Upgrade Plan

This document outlines improvements to MMOMENT's identity recognition system to achieve:
1. Better face recognition at distance (AdaFace)
2. More robust identity retention once recognized (Unified ReID)

---

## Implementation Progress (Jan 2026)

### Part 1: AdaFace - COMPLETED

**Status**: Deployed and running

**Key Implementation Notes**:

1. **MTCNN Alignment Landmarks Required**
   - AdaFace was trained with MTCNN face alignment, NOT InsightFace/ArcFace alignment
   - Reference landmarks differ by ~3px horizontally for eyes
   - Updated `native/src/preprocess.cu` with MTCNN landmarks:
   ```cpp
   // MTCNN landmarks for 112x112 (AdaFace training)
   {35.3437f, 51.6963f},   // Left eye  (vs ArcFace: 38.2946)
   {76.4538f, 51.5014f},   // Right eye (vs ArcFace: 73.5318)
   {56.0294f, 71.7366f},   // Nose (same)
   {39.1409f, 92.3655f},   // Left mouth (similar)
   {73.1849f, 92.2041f}    // Right mouth (similar)
   ```

2. **BGR Channel Order**
   - AdaFace official code expects BGR input (despite saying "RGB")
   - The warp kernel outputs BGR to match AdaFace training preprocessing

3. **Model Conversion**
   - PyTorch → ONNX: Perfect accuracy (1.0 similarity)
   - ONNX → TensorRT: Perfect accuracy (0.9999+ similarity)
   - Engine file: `native/adaface_ir50.engine` (~87MB)

**Files Modified**:
- `native/src/preprocess.cu` - MTCNN landmarks + BGR output
- `native/src/native_camera_server.cpp` - Engine path
- `docker-compose.yml` - Volume mount

### Part 2: SCRFD Face Detector - COMPLETED

**Status**: Deployed and running

**Implementation**:
- Model: SCRFD-10GF from InsightFace buffalo_l model pack
- Engine: `native/scrfd_10g.engine` (~9MB)
- New C++ class: `SCRFDEngine` in `insightface_engine.cpp`

**Key Technical Details**:
- 9 output tensors (score, bbox, kps × 3 FPN levels)
- Output tensor names renamed for compatibility: score_8, bbox_8, kps_8, etc.
- ~6.45ms latency, 154 QPS
- Confidence threshold: 0.35f (lowered to detect smaller/distant faces)

**Critical Implementation Notes** (discovered during debugging):

1. **Tensor Output Format is [N, features], NOT [C, H, W]**
   - score_8: [12800, 1], bbox_8: [12800, 4], kps_8: [12800, 10]
   - score_16: [3200, 1], bbox_16: [3200, 4], kps_16: [3200, 10]
   - score_32: [800, 1], bbox_32: [800, 4], kps_32: [800, 10]
   - Correct indexing: `anchorIdx = (y * featW + x) * numAnchors + a`
   - NOT: `a * featW * featH + y * featW + x` (wrong!)

2. **Landmark Format is Interleaved x,y Pairs**
   - Format: [x0, y0, x1, y1, x2, y2, x3, y3, x4, y4]
   - Access: `kpsData[anchorIdx * 10 + p * 2]` for x, `+ p * 2 + 1` for y
   - Verified against ncnn reference implementation

3. **Coordinate Transformation - No Padding Offset**
   - Image is placed at (0,0) with padding on RIGHT/BOTTOM
   - Do NOT subtract wpad/hpad from coordinates
   - Just divide by scaleRatio: `x0 = x0 / scaleRatio`

4. **BBox Decoding is Distance-Based**
   - dl, dt, dr, db = distances from anchor center
   - x0 = anchorCx - dl * stride, y0 = anchorCy - dt * stride
   - x1 = anchorCx + dr * stride, y1 = anchorCy + db * stride

**Files Modified**:
- `native/include/insightface_engine.h` - Added SCRFDEngine class
- `native/src/insightface_engine.cpp` - Full SCRFDEngine implementation
- `native/src/native_camera_server.cpp` - Use SCRFD engine, 0.35f threshold
- `docker-compose.yml` - Mount scrfd_10g.engine

### Part 3: Unified ReID - NOT YET STARTED

**Status**: Planned for future session

See detailed design in "Part 3: Unified Full-Body ReID" section below.

---

## Current State Summary

### Architecture (Updated Jan 2026)
```
Native C++ Pipeline:
├── YOLOv8-pose → Person detection + keypoints
├── ByteTracker → Track ID assignment
├── SCRFD-10GF → Face detection (640x640 input) [UPGRADED from RetinaFace]
├── AdaFace → Face embedding (112x112 → 512-dim) [UPGRADED from ArcFace]
└── OSNet → Body embedding (256x128 → 512-dim)

Python Identity Service:
├── Face matching (threshold: 0.65) → ONLY way to assign identity
├── Track continuity protection → Maintains identity during occlusion
├── ReID recovery → Re-acquires lost tracks within 5s window
└── Mismatch revocation → Removes wrong assignments proactively
```

### Key Files
| Component | File | Purpose |
|-----------|------|---------|
| Face Detection | `native/include/insightface_engine.h` | SCRFD TensorRT (SCRFDEngine class) |
| Face Detection Engine | `native/scrfd_10g.engine` | SCRFD-10GF TensorRT engine |
| Face Embedding | `native/include/insightface_engine.h` | AdaFace TensorRT (ArcFaceEngine class) |
| Face Embedding Engine | `native/adaface_ir50.engine` | AdaFace IR-50 TensorRT engine |
| Face Alignment | `native/src/preprocess.cu` | CUDA warp kernel with MTCNN landmarks |
| Body Embedding | `native/include/reid_engine.h` | OSNet TensorRT |
| Identity Logic | `services/native_identity_service.py` | Face matching, revocation |
| Identity Store | `services/identity_store.py` | UserIdentity dataclass, mappings |
| ByteTracker | `native/include/byte_tracker.h` | Track state management |

### Current Pain Points (Updated Jan 2026)
1. ~~**Face recognition requires ~3-4 feet distance**~~ - ADDRESSED with AdaFace + SCRFD
2. **Identity drops during movement** - Even when body is clearly the same person (Part 3: Unified ReID will address this)
3. ~~**Embedding similarity issue**~~ - ADDRESSED with correct MTCNN alignment landmarks
4. **ReID is supportive only** - Cannot help with initial recognition, only recovery (Part 3 will address this)

---

## Part 1: AdaFace Replacement

### Why AdaFace?

AdaFace was specifically designed for low-quality face recognition (surveillance, distance, blur):

| Model | Input | Output | Low-Quality Performance |
|-------|-------|--------|------------------------|
| ArcFace | 112x112 | 512-dim | Degrades significantly |
| AdaFace | 112x112 | 512-dim | Maintains accuracy |

AdaFace uses adaptive margin based on image quality - it knows when a face is hard to recognize and adjusts accordingly.

**Paper**: "AdaFace: Quality Adaptive Margin for Face Recognition" (CVPR 2022)

### Implementation Steps

#### Step 1: Obtain AdaFace Model

```bash
# Option A: Download pretrained from official repo
git clone https://github.com/mk-minchul/AdaFace
cd AdaFace
# Download IR-SE100 or IR-50 pretrained weights

# Option B: Use InsightFace model zoo (may have AdaFace variants)
pip install insightface
python -c "from insightface.model_zoo import get_model; print(get_model('buffalo_l'))"
```

**Recommended model**: `adaface_ir50_ms1mv2` (balanced accuracy/speed)

#### Step 2: Export to ONNX

```python
# export_adaface_onnx.py
import torch
from adaface import build_model

model = build_model('ir_50')
model.load_state_dict(torch.load('adaface_ir50_ms1mv2.ckpt'))
model.eval()

dummy_input = torch.randn(1, 3, 112, 112)
torch.onnx.export(
    model,
    dummy_input,
    "adaface_ir50.onnx",
    input_names=['input'],
    output_names=['embedding'],
    dynamic_axes={'input': {0: 'batch'}, 'embedding': {0: 'batch'}},
    opset_version=11
)
```

#### Step 3: Convert to TensorRT

```bash
# On Jetson (must match target architecture)
/usr/src/tensorrt/bin/trtexec \
    --onnx=adaface_ir50.onnx \
    --saveEngine=adaface_ir50.engine \
    --fp16 \
    --workspace=1024
```

#### Step 4: Update Native C++ Code

**File**: `native/include/insightface_engine.h`

No structural changes needed - AdaFace has same input (112x112 RGB) and output (512-dim embedding) as ArcFace.

```cpp
// Just update engine path in native_camera_server.cpp
// Line ~617 area
- if (!g_arcfaceEngine->loadEngine("arcface_r50.engine")) {
+ if (!g_arcfaceEngine->loadEngine("adaface_ir50.engine")) {
```

#### Step 5: Verify Preprocessing Compatibility

AdaFace expects same preprocessing as ArcFace:
- Input: 112x112 RGB
- Normalization: (pixel - 127.5) / 127.5 → [-1, 1] range
- Face alignment: Same 5-point landmark warping

**Verify in**: `native/src/insightface_engine.cpp` - the `launchWarpFaceToArcFace` CUDA kernel

#### Step 6: Update Embedding Storage

If switching models, existing stored embeddings become invalid:

```python
# In identity_store.py or migration script
def migrate_to_adaface():
    """
    Users will need to re-enroll (get new face embedding captured).
    Stored embeddings from ArcFace won't match AdaFace outputs.
    """
    # Option 1: Clear all stored embeddings, require re-enrollment
    # Option 2: Keep ArcFace for existing, use AdaFace for new (complex)
    pass
```

**Recommendation**: Clear stored embeddings and have users re-enroll. This is a one-time migration.

### Testing AdaFace

```python
# test_adaface_distance.py
# Compare ArcFace vs AdaFace at various distances

distances = ['3ft', '6ft', '10ft', '15ft']
for distance in distances:
    # Capture face at distance
    # Extract embedding with both models
    # Compare similarity to enrolled embedding
    # Log results
```

### Expected Improvement

| Distance | ArcFace Similarity | AdaFace Similarity (expected) |
|----------|-------------------|------------------------------|
| 3 feet | 0.75+ | 0.80+ |
| 6 feet | 0.55-0.65 | 0.70+ |
| 10 feet | 0.40-0.50 | 0.60+ |
| 15 feet | 0.30-0.40 | 0.50+ |

---

## Part 2: SCRFD Face Detector (Optional)

### Why SCRFD?

SCRFD (Sample and Computation Redistribution for Face Detection) is InsightFace's newer detector, optimized for:
- Small faces at distance
- Faster inference than RetinaFace
- Better accuracy/speed tradeoff

### Implementation

```bash
# SCRFD models available in InsightFace model zoo
# scrfd_10g_bnkps - highest accuracy
# scrfd_2.5g_bnkps - balanced
# scrfd_500m_bnkps - fastest
```

**Changes required**:
1. Export SCRFD to TensorRT
2. Update `insightface_engine.h` detector class (different output format)
3. Update landmark extraction (SCRFD uses different landmark indices)

**Effort**: Medium - detector output parsing differs from RetinaFace

---

## Part 3: Unified Full-Body ReID

### Goal

Once a user is recognized (via face), they should **never** drop identity as long as they remain in frame, regardless of:
- Face visibility
- Distance from camera
- Pose changes
- Partial occlusion

### Current Limitation

```
Current flow:
1. Face match → assign identity (wallet_address to track_id)
2. Face occluded → rely on track continuity
3. Track lost briefly → ReID recovery (5s window, 0.80 threshold)
4. Recovery fails OR track corrupted → identity lost

Problem: Identity can drop even when body is clearly visible and unchanged
```

### Proposed Design: Track-Level Identity Persistence

```
New flow:
1. Face match → assign identity + capture "identity anchor" (face + body)
2. Track continues → identity persists (body similarity > 0.70 sufficient)
3. Track lost → identity stays associated with appearance profile
4. Same appearance returns → immediate re-association (no face needed)
5. Different appearance with same face → update appearance profile

Key change: Once face-confirmed, body becomes sufficient for continuity
```

### Implementation Steps

#### Step 1: Extend UserIdentity with Appearance History

**File**: `services/identity_store.py`

```python
@dataclass
class UserIdentity:
    wallet_address: str

    # Existing
    face_embedding: Optional[np.ndarray] = None  # 512-dim ArcFace/AdaFace
    appearance_embedding: Optional[np.ndarray] = None  # 512-dim OSNet
    active_track_id: Optional[int] = None
    identity_confidence: float = 0.0
    last_face_seen: float = 0.0

    # NEW: Appearance history for robust matching
    appearance_history: List[np.ndarray] = field(default_factory=list)  # Last N embeddings
    appearance_history_max: int = 10  # Keep last 10 body snapshots

    # NEW: Identity confirmation level
    confirmation_level: str = 'none'  # 'none' | 'face_once' | 'face_confirmed' | 'body_maintained'
    face_confirmations: int = 0  # How many times face matched

    # NEW: Robust re-acquisition
    last_appearance_cluster: Optional[np.ndarray] = None  # Averaged appearance embedding
```

#### Step 2: Update Identity Assignment Logic

**File**: `services/native_identity_service.py`

```python
def _assign_identity_with_confirmation(self, person, wallet_address, face_similarity):
    """
    Assign identity with confirmation level tracking.
    """
    identity = self._identity_store.get_identity(wallet_address)

    # Update confirmation level
    if face_similarity > 0.65:
        identity.face_confirmations += 1

        if identity.face_confirmations >= 3:
            identity.confirmation_level = 'face_confirmed'
        else:
            identity.confirmation_level = 'face_once'

        # Capture body appearance as "anchor"
        body_embedding = person.get('reid_embedding')
        if body_embedding is not None:
            self._update_appearance_history(identity, body_embedding)

    # ... existing assignment logic
```

#### Step 3: Implement Appearance History

```python
def _update_appearance_history(self, identity: UserIdentity, embedding: np.ndarray):
    """
    Maintain rolling history of body appearances.
    """
    # Add to history
    identity.appearance_history.append(embedding.copy())

    # Trim to max size
    if len(identity.appearance_history) > identity.appearance_history_max:
        identity.appearance_history.pop(0)

    # Update cluster center (average of recent appearances)
    if len(identity.appearance_history) >= 3:
        identity.last_appearance_cluster = np.mean(
            identity.appearance_history, axis=0
        )
        # Re-normalize
        identity.last_appearance_cluster /= np.linalg.norm(identity.last_appearance_cluster)
```

#### Step 4: Relaxed Identity Maintenance for Confirmed Users

**File**: `services/native_identity_service.py`

Current revocation logic (line 464+) is aggressive. For confirmed users, relax it:

```python
def _should_revoke_identity(self, person, identity, face_visible, face_similarity):
    """
    Determine if identity should be revoked.

    Key change: Face-confirmed users get much more lenient treatment.
    """
    # If face not visible, check body only
    if not face_visible:
        body_sim = self._get_body_similarity(person, identity)

        # Face-confirmed users: body similarity alone is sufficient
        if identity.confirmation_level == 'face_confirmed':
            if body_sim > 0.60:  # Relaxed from 0.85
                return False  # Don't revoke

        # Face-once users: stricter body requirement
        elif identity.confirmation_level == 'face_once':
            if body_sim > 0.75:
                return False

    # Face visible but doesn't match
    if face_visible and face_similarity < 0.30:  # Clear mismatch
        # Even for confirmed users, clear face mismatch = revoke
        # But require multiple bad frames
        if identity.bad_face_frames < 30:  # ~1 second at 30fps
            identity.bad_face_frames += 1
            return False
        return True

    return False  # Default: don't revoke
```

#### Step 5: Enhanced Re-acquisition

**File**: `services/native_identity_service.py`

```python
def _attempt_reacquisition_unified(self, person):
    """
    Try to re-acquire identity using appearance cluster matching.

    This runs for all unidentified persons, checking against
    ALL checked-in users who don't currently have an active track.
    """
    person_embedding = person.get('reid_embedding')
    if person_embedding is None:
        return None

    best_match = None
    best_similarity = 0.65  # Threshold for appearance-based re-acquisition

    for wallet, identity in self._identity_store.get_all_checked_in():
        # Skip if already has active track
        if identity.active_track_id is not None:
            continue

        # Skip if never face-confirmed
        if identity.confirmation_level not in ('face_once', 'face_confirmed'):
            continue

        # Check against appearance cluster
        if identity.last_appearance_cluster is not None:
            similarity = cosine_similarity(person_embedding, identity.last_appearance_cluster)

            # Confirmed users: easier re-acquisition
            threshold = 0.60 if identity.confirmation_level == 'face_confirmed' else 0.70

            if similarity > threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match = wallet

    return best_match, best_similarity if best_match else (None, 0)
```

#### Step 6: Update ByteTracker Integration

**File**: `native/src/native_camera_server.cpp` (or wherever ReID is populated)

Ensure ReID embedding is extracted for ALL tracked persons every frame (or every N frames):

```cpp
// In the main processing loop
for (auto& track : activeTracks) {
    // Update ReID every 10 frames for efficiency
    if (frameCount - track.reidUpdateFrame >= 10) {
        // Extract person crop
        // Run through ReID engine
        // Store in track.reidEmbedding
        track.reidUpdateFrame = frameCount;
        track.hasReidEmbedding = true;
    }
}
```

### Configuration Changes

**File**: `services/native_identity_service.py`

```python
# Updated thresholds for unified approach
_similarity_threshold = 0.60  # Face match (was 0.65, AdaFace more reliable)
_reid_similarity_threshold = 0.65  # Body re-acquisition (was 0.80)
_reid_recovery_max_time = 30.0  # Extended from 5.0 seconds
_body_maintenance_threshold = 0.60  # NEW: body-only maintenance for confirmed users
_face_confirmed_threshold = 3  # NEW: faces matches needed for 'confirmed' status
```

---

## Part 4: Testing Plan

### Test 1: AdaFace Distance Recognition
```
Setup: User at 3ft, 6ft, 10ft, 15ft from camera
Measure: Face detection rate, embedding similarity, recognition success
Compare: ArcFace vs AdaFace
Pass criteria: AdaFace matches at 10ft where ArcFace fails
```

### Test 2: Identity Persistence During Movement
```
Setup: User recognized, then moves around room for 60 seconds
Measure: Identity drop rate, recovery time
Compare: Before vs after unified ReID changes
Pass criteria: <5% identity drops during continuous movement
```

### Test 3: Multi-Person Differentiation
```
Setup: 3 users in frame simultaneously, similar clothing
Measure: Cross-assignment rate, correct identification rate
Pass criteria: 0% cross-assignment, >95% correct ID
```

### Test 4: Occlusion Recovery
```
Setup: User recognized, walks behind obstacle for 5/10/30 seconds
Measure: Recovery success rate, time to re-acquire
Pass criteria: 100% recovery at 5s, >80% at 30s
```

---

## Implementation Order

### Phase 1: AdaFace (Estimated: 2-3 hours)
1. Download/export AdaFace model
2. Convert to TensorRT on Jetson
3. Swap engine file, test
4. Clear stored embeddings, test re-enrollment

### Phase 2: Unified ReID (Estimated: 4-6 hours)
1. Extend UserIdentity dataclass
2. Implement appearance history
3. Update confirmation level tracking
4. Relax revocation for confirmed users
5. Implement enhanced re-acquisition
6. Test and tune thresholds

### Phase 3: SCRFD Detector (Optional, Estimated: 3-4 hours)
1. Export SCRFD to TensorRT
2. Update native detector code
3. Test small face detection improvement

---

## Rollback Plan

If issues arise:
1. **AdaFace**: Swap back to `arcface_r50.engine` (keep both files)
2. **Unified ReID**: Feature flag the new logic, revert to old thresholds
3. **SCRFD**: Keep RetinaFace as fallback

---

## Files to Modify Summary

| File | Changes |
|------|---------|
| `native/src/native_camera_server.cpp` | Engine path for AdaFace |
| `services/identity_store.py` | UserIdentity extensions |
| `services/native_identity_service.py` | Confirmation levels, relaxed revocation, enhanced re-acquisition |
| `native/*.engine` | Add adaface_ir50.engine |

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Recognition distance | 3-4 feet | 8-10 feet |
| Identity drop rate (moving user) | ~20% | <5% |
| Re-acquisition after 10s occlusion | ~50% | >90% |
| False positive rate | <1% | <1% (maintain) |
