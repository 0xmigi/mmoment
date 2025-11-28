# Phase 3: Jetson Privacy Architecture Updates

## Overview

This document describes the changes needed on the Jetson camera device to implement the new privacy-preserving architecture. The goal is to break visible on-chain links between users and cameras while maintaining all functionality.

## Key Changes

### Before (Old Architecture)
- Check-in creates on-chain `UserSession` PDA with seeds `["session", user, camera]` - **directly links user to camera**
- Check-out closes session and commits activities to `CameraTimeline`
- Anyone can see which users are at which cameras by querying on-chain sessions

### After (New Architecture)
- Check-in is **fully off-chain** - Jetson manages sessions locally
- Check-out splits into two separate transactions:
  1. Jetson writes encrypted activities to `CameraTimeline` (no user info)
  2. Backend stores access keys in user's `UserSessionChain` (no camera info)
- On-chain records only show "User X interacted with their mmoment account" - not which camera

## Jetson Changes Required

### 1. Session Management (Off-Chain)

The Jetson must now manage user sessions entirely locally instead of relying on blockchain UserSession accounts.

```python
# Example session structure in Jetson memory
class LocalSession:
    session_id: str           # Unique session ID
    wallet_address: str       # User's wallet pubkey
    display_name: str         # User's display name
    check_in_time: int        # Unix timestamp
    last_activity_time: int   # Unix timestamp of last interaction (for auto-checkout)
    activities: List[Activity] # Buffered activities during session
    session_key: bytes        # AES-256 key for this session (encrypt activities)
```

**Session storage requirements:**
- Store active sessions in memory (or Redis for persistence)
- Generate unique session IDs (UUID or timestamp-based)
- Generate per-session AES-256 encryption key
- Track session activities for later encryption

### 2. Update `/api/checkin` Endpoint

The check-in endpoint no longer requires blockchain confirmation. It should:

```python
@app.route('/api/checkin', methods=['POST'])
def handle_checkin():
    data = request.json
    wallet_address = data['wallet_address']
    display_name = data.get('display_name')
    username = data.get('username')

    # NOTE: No more session_pda or transaction_signature needed!

    # 1. Generate session credentials
    session_id = generate_session_id()
    session_key = generate_aes_key()  # AES-256 key for encrypting activities

    # 2. Create local session record
    session = LocalSession(
        session_id=session_id,
        wallet_address=wallet_address,
        display_name=display_name or truncate_address(wallet_address),
        check_in_time=int(time.time()),
        activities=[],
        session_key=session_key
    )

    # 3. Store session locally
    active_sessions[wallet_address] = session

    # 4. Load user's recognition token (if available) for face recognition
    try:
        recognition_token = load_recognition_token_from_chain(wallet_address)
        if recognition_token:
            register_face_for_session(session_id, recognition_token)
    except Exception as e:
        print(f"No recognition token for user: {e}")

    # 5. Return success with session info
    return jsonify({
        'wallet_address': wallet_address,
        'display_name': session.display_name,
        'session_id': session_id,
        'camera_pda': CAMERA_PDA,
        'camera_url': CAMERA_API_URL,
        'message': 'Check-in successful'
    })
```

### 3. Update `/api/checkout` Endpoint

The checkout endpoint now orchestrates writing activities to the blockchain and delivering access keys:

```python
@app.route('/api/checkout', methods=['POST'])
def handle_checkout():
    data = request.json
    wallet_address = data['wallet_address']

    # 1. Get session from local storage
    session = active_sessions.get(wallet_address)
    if not session:
        return jsonify({'error': 'No active session found'}), 400

    # 2. Encrypt all session activities
    encrypted_activities = []
    for activity in session.activities:
        encrypted = encrypt_activity(activity, session.session_key)
        encrypted_activities.append(encrypted)

    # 3. Write encrypted activities to CameraTimeline (NO USER INFO!)
    if encrypted_activities:
        write_to_camera_timeline(encrypted_activities)

    # 4. Send access key to backend for user's UserSessionChain
    # Backend cron will store this in user's on-chain keychain
    send_access_key_to_backend(
        user_pubkey=wallet_address,
        session_key_ciphertext=encrypt_session_key_for_user(session.session_key, wallet_address),
        nonce=generate_nonce(),
        timestamp=session.check_in_time
    )

    # 5. Clean up local session
    del active_sessions[wallet_address]

    # 6. Remove face recognition data for user
    unregister_face_for_session(session.session_id)

    return jsonify({
        'wallet_address': wallet_address,
        'session_id': session.session_id,
        'message': 'Check-out successful'
    })
```

### 4. Write to CameraTimeline (New Instruction)

The Jetson must call the new `write_to_camera_timeline` instruction to commit activities:

```python
def write_to_camera_timeline(encrypted_activities):
    """
    Write encrypted activities to the camera's timeline.
    This transaction has NO USER ACCOUNTS - complete anonymity.
    """
    # Build transaction
    ix = program.methods.write_to_camera_timeline(
        activities=encrypted_activities
    ).accounts({
        'authority': JETSON_KEYPAIR.pubkey(),  # Jetson's signing key
        'camera': CAMERA_PDA,
        'camera_timeline': derive_camera_timeline_pda(CAMERA_PDA),
        'system_program': SYSTEM_PROGRAM_ID
    }).instruction()

    # Sign and send
    tx = Transaction().add(ix)
    signature = client.send_transaction(tx, JETSON_KEYPAIR)
    client.confirm_transaction(signature)

    return signature
```

**Important:** The `write_to_camera_timeline` instruction:
- Only requires Jetson's authority signature
- Has NO user account in the transaction
- Emits `TimelineUpdated` event (no user info)
- Anyone watching the blockchain only sees "Camera X had activity"

### 5. Send Access Keys to Backend

After checkout, send the encrypted session key to the backend for storage:

```python
def send_access_key_to_backend(user_pubkey, session_key_ciphertext, nonce, timestamp):
    """
    Send encrypted access key to backend.
    Backend cron will store this in user's UserSessionChain on-chain.
    """
    response = requests.post(
        f"{BACKEND_URL}/api/session/access-key",
        json={
            'user_pubkey': user_pubkey,
            'key_ciphertext': list(session_key_ciphertext),
            'nonce': list(nonce),
            'timestamp': timestamp
        }
    )

    if not response.ok:
        # Queue for retry if backend is unavailable
        queue_access_key_for_retry(user_pubkey, session_key_ciphertext, nonce, timestamp)
```

### 6. Update `/api/status` Endpoint

Add active session count to the status endpoint:

```python
@app.route('/api/status', methods=['GET'])
def get_status():
    return jsonify({
        'isOnline': True,
        'isStreaming': stream_manager.is_streaming(),
        'isRecording': recorder.is_recording(),
        'lastSeen': int(time.time() * 1000),
        'activeSessionCount': len(active_sessions)  # NEW: For privacy-preserving user count
    })
```

### 7. Activity Encryption Format

Each activity must be encrypted with the session's AES-256 key:

```python
def encrypt_activity(activity, session_key):
    """
    Encrypt activity content using AES-256-GCM.
    """
    nonce = os.urandom(12)  # 12-byte nonce for GCM

    # Serialize activity content
    plaintext = json.dumps({
        'type': activity.type,
        'data': activity.data,
        'metadata': activity.metadata
    }).encode()

    # Encrypt using AES-256-GCM
    cipher = Cipher(algorithms.AES(session_key), modes.GCM(nonce))
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(plaintext) + encryptor.finalize()

    return {
        'timestamp': activity.timestamp,        # Public (for overlap queries)
        'activity_type': activity.type_code,    # Public (for filtering)
        'encrypted_content': ciphertext + encryptor.tag,
        'nonce': nonce,
        'access_grants': []  # Populated for multi-user sessions
    }
```

### 8. Multi-User Session Handling

For sessions with multiple users (detected via face recognition), each activity needs access grants for all present users:

```python
def encrypt_activity_for_multiple_users(activity, session_key, present_users):
    """
    Create access grants so multiple users can decrypt the activity.
    """
    encrypted_activity = encrypt_activity(activity, session_key)

    # For each user present, encrypt the session key with their public key
    for user_pubkey in present_users:
        # Encrypt session key using X25519 key exchange
        user_grant = encrypt_session_key_for_user(session_key, user_pubkey)
        encrypted_activity['access_grants'].append(user_grant)

    return encrypted_activity
```

### 9. Auto-Checkout (Session Expiry)

The Jetson must automatically checkout users who become inactive. This is critical for:
- Freeing up resources when users forget to checkout
- Ensuring activities are written to the blockchain even if users leave abruptly
- Maintaining accurate `activeSessionCount`

```python
# Session timeout configuration
INACTIVITY_TIMEOUT = 60 * 60  # 1 hour - auto-checkout after no activity
DEV_INACTIVITY_TIMEOUT = 2 * 60 * 60  # 2 hours - extended timeout for dev/testing

# Use DEV_INACTIVITY_TIMEOUT while testing, INACTIVITY_TIMEOUT in production
CURRENT_TIMEOUT = DEV_INACTIVITY_TIMEOUT if IS_DEV_MODE else INACTIVITY_TIMEOUT

def check_expired_sessions():
    """
    Run periodically (e.g., every 5 minutes) to auto-checkout inactive users.
    """
    current_time = int(time.time())
    expired_wallets = []

    for wallet_address, session in active_sessions.items():
        time_since_activity = current_time - session.last_activity_time

        if time_since_activity > CURRENT_TIMEOUT:
            expired_wallets.append(wallet_address)
            print(f"Session expired for {wallet_address} - inactive for {time_since_activity}s")

    # Process expired sessions
    for wallet_address in expired_wallets:
        try:
            auto_checkout_user(wallet_address)
        except Exception as e:
            print(f"Auto-checkout failed for {wallet_address}: {e}")

def auto_checkout_user(wallet_address):
    """
    Perform automatic checkout for inactive user.
    Same as manual checkout but triggered by timeout.
    """
    session = active_sessions.get(wallet_address)
    if not session:
        return

    # 1. Encrypt all session activities
    encrypted_activities = []
    for activity in session.activities:
        encrypted = encrypt_activity(activity, session.session_key)
        encrypted_activities.append(encrypted)

    # 2. Write encrypted activities to CameraTimeline (NO USER INFO!)
    if encrypted_activities:
        write_to_camera_timeline(encrypted_activities)

    # 3. Send access key to backend for user's UserSessionChain
    send_access_key_to_backend(
        user_pubkey=wallet_address,
        session_key_ciphertext=encrypt_session_key_for_user(session.session_key, wallet_address),
        nonce=generate_nonce(),
        timestamp=session.check_in_time
    )

    # 4. Clean up local session
    del active_sessions[wallet_address]

    # 5. Remove face recognition data for user
    unregister_face_for_session(session.session_id)

    print(f"Auto-checkout completed for {wallet_address}")

def update_session_activity(wallet_address):
    """
    Call this whenever a user interacts with the camera (capture, etc.)
    to reset the inactivity timer.
    """
    if wallet_address in active_sessions:
        active_sessions[wallet_address].last_activity_time = int(time.time())
```

**Timeout values:**
- **Production**: 1 hour of inactivity triggers auto-checkout
- **Dev/Testing**: 2 hours of inactivity (extended for debugging)

**Important:** Update `last_activity_time` on every user interaction (photo capture, face detection, etc.) to prevent premature auto-checkout during active use.

## Backend Integration

The backend already has the endpoint to receive access keys from Jetson:

```
POST /api/session/access-key
{
    "user_pubkey": "UserWalletAddress...",
    "key_ciphertext": [encrypted bytes],
    "nonce": [12 bytes],
    "timestamp": 1699000000
}
```

The backend cron (`session-cleanup-cron.ts`) will:
1. Queue the access key
2. Call `store_session_access_keys` instruction on-chain
3. Retry if the transaction fails (user needs to create UserSessionChain first)

## Migration Notes

### Backwards Compatibility

The old flow (with on-chain UserSession) is being deprecated. During migration:

1. Web app no longer sends `session_pda` or `transaction_signature`
2. Jetson should handle both old and new clients gracefully
3. Old UserSession accounts will naturally expire (can be cleaned up later)

### Testing Checklist

- [ ] Check-in works without blockchain transaction
- [ ] Activities are encrypted and buffered locally
- [ ] Check-out writes activities to CameraTimeline (verify no user in tx)
- [ ] Access keys are sent to backend successfully
- [ ] Backend cron stores keys in UserSessionChain
- [ ] Multi-user sessions create proper access grants
- [ ] Face recognition still works (loads token from chain, not session)
- [ ] Status endpoint returns `activeSessionCount`
- [ ] Auto-checkout triggers after 1hr inactivity (2hr in dev mode)
- [ ] User interactions reset the inactivity timer
- [ ] Auto-checkout properly writes activities and sends access keys

## Security Considerations

1. **Session keys**: Generate fresh AES-256 key for each session
2. **Key delivery**: Use X25519 (or similar) to encrypt session key for user
3. **Nonce reuse**: Never reuse nonces for AES-GCM
4. **Local storage**: Consider encrypting sessions at rest on Jetson
5. **Backend trust**: Backend is trusted to store keys honestly (future: decentralized storage)

## Data Flow Summary

```
Check-in:
  Web App → Jetson /api/checkin
  Jetson creates local session (no blockchain)

During Session:
  Activities buffered locally, encrypted with session key

Check-out:
  Web App → Jetson /api/checkout
  Jetson → Blockchain (write_to_camera_timeline) - NO USER INFO
  Jetson → Backend /api/session/access-key
  Backend → Blockchain (store_session_access_keys) - NO CAMERA INFO

Result:
  On-chain: "Camera X had activity" + "User Y stored a key"
  No visible link between Camera X and User Y!
```
