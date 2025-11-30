# Phase 3: Privacy Architecture

## CRITICAL: What is a Check-In?

**A check-in is a CRYPTOGRAPHIC HANDSHAKE between the user's wallet and the camera.**

It is NOT just clicking a button. It requires:
1. User's embedded wallet signs a challenge/message
2. Jetson verifies the Ed25519 signature
3. Only if signature is valid → session is created

```
USER CLIENT                              JETSON
     |                                       |
     |  1. Client constructs message:        |
     |     "{wallet}|{timestamp}|{nonce}"    |
     |                                       |
     |  2. Client signs with Ed25519         |
     |                                       |
     |  3. Single POST with everything:      |
     |     wallet_address, signature,        |
     |     timestamp, nonce                  |
     |-------------------------------------->|
     |                                       |
     |  4. Jetson reconstructs message       |
     |  5. Jetson VERIFIES Ed25519 signature |
     |     using wallet_address as pubkey    |
     |                                       |
     |  ✅ Valid → Create session            |
     |  ❌ Invalid → 401 Unauthorized        |
     |                                       |
```

**Single-shot flow** - no challenge endpoint needed. Client generates timestamp/nonce locally.

**Why this matters:**
- Prevents impersonation (can't check in as someone else)
- Proves cryptographic ownership of wallet
- Camera knows FOR CERTAIN who is checked in

---

## CRITICAL: Understanding the Camera Timeline

**The Camera Timeline is ONE thing at two stages:**
- **Real-time**: Activities being shown to live viewers NOW (cached on backend)
- **Historical**: The same activities, committed on-chain at checkout (permanent storage)

Think of the real-time timeline as a **working cache** that eventually gets committed on-chain.

---

## System Architecture: Three Entities

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  USER CLIENT    │     │  CAMERA NODE    │     │    BACKEND      │
│  (Web App)      │     │  (Jetson)       │     │  (Railway)      │
└────────┬────────┘     └────────┬────────┘     └────────┬────────┘
         │                       │                       │
         │ "Check me in"         │                       │
         │──────────────────────>│                       │
         │                       │                       │
         │                       │ PERFORMS check-in     │
         │                       │ Creates local session │
         │                       │                       │
         │                       │ "I did check-in"      │
         │                       │──────────────────────>│
         │                       │                       │ CACHES activity
         │                       │                       │ BROADCASTS to viewers
         │                       │                       │
         │<──────────────────────────────────────────────│ WebSocket: "check_in"
         │                       │                       │
         │ (Timeline shows it)   │                       │
         │                       │                       │
```

---

## Role of Each Entity

### JETSON (Camera Node)
**Job: PERFORM actions, TELL backend**

The Jetson is the actor. It:
1. PERFORMS check-ins (creates local session)
2. CAPTURES photos/videos
3. RUNS CV apps
4. PERFORMS check-outs

After DOING something, it tells the backend with a simple POST:
```python
requests.post(f"{BACKEND_URL}/api/session/activity", json={
    "sessionId": session_id,
    "cameraId": camera_pda,
    "activityType": 0,  # CHECK_IN
    "userId": wallet_address,
    "displayName": display_name,
    "timestamp": timestamp
}, timeout=5)
```

That's it. No queuing. No async buffers. No local caching of activities. Just POST and done.

### BACKEND (Railway)
**Job: CACHE, BROADCAST, and eventually COMMIT on-chain**

The backend is the reliable anchor. It:
1. RECEIVES activity notifications from Jetson
2. CACHES activities for the session (in database)
3. BROADCASTS to live viewers (WebSocket)
4. At checkout: PACKAGES, ENCRYPTS, and COMMITS to blockchain

**Why the backend?** It's the most reliable component:
- User's browser can close → backend still has the data
- Jetson can lose connection → backend still has the data
- Backend can properly close orphaned sessions and commit them on-chain

### USER CLIENT (Web App)
**Job: DISPLAY the timeline, INITIATE actions**

The client:
1. Connects to backend WebSocket
2. Joins the camera's "room"
3. Receives `timelineEvent` broadcasts
4. Displays activities in real-time

---

## Activity Flow: Check-In Example

```
1. User clicks "Check In" in web app

2. Frontend generates signed request (using request-signer.ts):
   - timestamp = Date.now()
   - nonce = crypto.randomUUID()
   - message = "{wallet_address}|{timestamp}|{nonce}"
   - signature = wallet.signMessage(message)  // Ed25519, base58 encoded

3. Frontend sends single POST to Jetson:
   POST https://{camera}.mmoment.xyz/api/checkin
   {
     "wallet_address": "RsLj...",
     "request_signature": "5abc...",      // base58 Ed25519 signature
     "request_timestamp": 1701234567890,  // Unix ms
     "request_nonce": "abc-123-def",      // UUID
     "display_name": "Alice",             // optional
     "username": "alice123"               // optional
   }

4. Jetson VERIFIES Ed25519 signature:
   - Reconstructs message: "{wallet_address}|{timestamp}|{nonce}"
   - Checks timestamp isn't >5 min old (replay protection)
   - Verifies signature using wallet_address as pubkey
   - If INVALID → 401 Unauthorized
   - If VALID → continue

5. Jetson creates local session (for face recognition, permissions)

6. Jetson TELLS backend (synchronous POST):
   POST https://backend/api/session/activity
   { sessionId, cameraId, activityType: CHECK_IN, userPubkey, displayName }

7. Backend receives it:
   - CACHES the activity in database
   - BROADCASTS to WebSocket room for this camera

8. All connected clients receive:
   WebSocket event: { type: "check_in", user: {...}, cameraId: "..." }

9. Timelines update in real-time
```

**The cryptographic verification is NOT optional.** Without it, anyone could impersonate any wallet.

---

## Activity Flow: Check-Out (Session End)

```
1. User clicks "Check Out" (or auto-checkout triggers)

2. Backend PACKAGES all cached activities for this session

3. Backend ENCRYPTS the package:
   - AES-256-GCM encryption
   - Access grants for users who were present

4. Backend COMMITS to blockchain:
   - Writes to CameraTimeline (no user info in transaction)
   - Stores access keys in user's UserSessionChain (no camera info)

5. On-chain result:
   - "Camera X had encrypted activity"
   - "User Y stored an access key"
   - NO visible link between Camera X and User Y
```

---

## What the Jetson DOES NOT Do

The Jetson does NOT:
- ❌ Buffer activities locally for later
- ❌ Use async queues to send activities
- ❌ Encrypt activities (backend does this at checkout)
- ❌ Write to blockchain directly (backend does this)
- ❌ Cache activities for historical storage

The Jetson ONLY:
- ✅ Performs the action
- ✅ Tells the backend immediately with a synchronous POST
- ✅ Manages local session state (for face recognition, permissions)

---

## Session Management on Jetson

The Jetson still manages LOCAL session state, but only for:
- Knowing who is checked in (for permissions)
- Face recognition (loading recognition tokens)
- Tracking which user performed an action

```python
class LocalSession:
    session_id: str           # Unique session ID
    wallet_address: str       # User's wallet pubkey
    display_name: str         # User's display name
    check_in_time: int        # Unix timestamp
    last_activity_time: int   # For auto-checkout detection
```

Note: NO `activities` list. NO `session_key`. Activities go straight to backend.

---

## Auto-Checkout

Even auto-checkout works through the backend:

1. Jetson detects inactivity timeout
2. Jetson tells backend: "User X should be checked out"
3. Backend handles packaging, encryption, on-chain commit

OR (more reliable):

1. Backend tracks last activity time
2. Backend detects inactivity
3. Backend auto-checkouts, packages, commits on-chain

The second approach is more reliable because backend is always online.

---

## Privacy: How It Works

**During session:**
- Activities are cached on backend (not encrypted yet)
- Broadcast to live viewers in real-time
- This is fine - viewers are seeing it happen live anyway

**At checkout:**
- Backend packages all activities
- Encrypts with session key (AES-256-GCM)
- Creates access grants for users who were present
- Commits encrypted blob to CameraTimeline on-chain
- Stores access keys in user's UserSessionChain (separate transaction)

**Result:**
- On-chain: "Camera X had activity" (no user info)
- On-chain: "User Y stored a key" (no camera info)
- Only users with access keys can decrypt the activities

---

## API Endpoints Summary

### Jetson Endpoints (called by web app)
- `POST /api/checkin` - Create signed session (single-shot, no challenge endpoint)
  - **Required**: `wallet_address`, `request_signature`, `request_timestamp`, `request_nonce`
  - **Optional**: `display_name`, `username`
  - Jetson VERIFIES Ed25519 signature before creating session
  - Returns 401 if signature invalid or timestamp >5 min old
- `POST /api/checkout` - End session
- `POST /api/capture` - Take photo
- `GET /api/status` - Camera status + activeSessionCount

### Backend Endpoints (called by Jetson)
- `POST /api/session/activity` - Report an activity (caches + broadcasts)
- `POST /api/session/checkout` - Trigger checkout (packages + commits on-chain)

### Backend WebSocket Events (received by web app)
- `timelineEvent` - Real-time activity notification

---

## Frontend Integration

**The frontend already has `request-signer.ts` ready to use!**

```typescript
// In your check-in handler:
import { createSignedRequest } from './request-signer';

const signedParams = await createSignedRequest(primaryWallet);
if (!signedParams) {
  throw new Error('Failed to sign check-in request');
}

const result = await unifiedCameraService.checkin(cameraId, {
  ...signedParams,  // wallet_address, request_signature, request_timestamp, request_nonce
  display_name: profile?.displayName,
  username: profile?.username
});
```

That's it. The signer handles message construction, signing, and base58 encoding.

---

## Implementation Checklist

### Check-In Security ✅ DONE
- [x] Jetson: Require `request_signature`, `request_timestamp`, `request_nonce` in `/api/checkin`
- [x] Jetson: VERIFY Ed25519 signature before creating session
- [x] Jetson: Reject requests with timestamps >5 min old (replay protection)
- [ ] Frontend: Use `createSignedRequest()` before calling checkin

### Timeline Flow (current focus)
- [x] Jetson: Remove ActivityBufferClient async queue
- [x] Jetson: Direct synchronous POST to backend on every activity
- [ ] Backend: Cache activities in database
- [ ] Backend: Broadcast to WebSocket on receive
- [ ] Backend: Package + encrypt + commit at checkout
- [ ] Frontend: Listen to WebSocket for real-time updates

### Cleanup
- [ ] Delete ActivityBufferClient and related async queue code
- [ ] Delete ActivityEncryptionService (encryption moves to backend)
- [ ] Remove `activities` list from LocalSession class
- [ ] Remove `session_key` from LocalSession class

---

## Key Principle

**KEEP IT SIMPLE:**

Jetson does action → Jetson tells backend → Backend broadcasts → Timeline shows it

No queues. No async buffers. No encryption until checkout. Just POST and broadcast.
