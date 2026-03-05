# MMOMENT API

## One Sentence

The MMOMENT API answers: **"What is happening, where, and who is doing it?"** — with cryptographic proof, in real time.

How much you can see isn't determined by your subscription. It's determined by how much the people and places on the network consent to showing you.

---

## Current Reality

Each camera on the network is already a self-contained API node. When a camera registers, it gets a Cloudflare tunnel with a unique subdomain (derived from its on-chain PDA). Today that looks like:

```
https://arqxl9kzhz8qhjtnodnumvkd3hgdkwsstzbd4qd9qqkv.mmoment.xyz/api/...
```

The mmoment.xyz web app is just one client consuming these endpoints. It is an app on the network, not the network itself.

### What each camera already exposes

**Session management:**
- `POST /api/checkin` — ed25519 signature verification (wallet + timestamp + nonce)
- `POST /api/checkout` — end session, publish to chain
- `GET /api/session/status/<wallet>` — check if user is checked in

**CV apps:**
- `POST /api/apps/load` — load a CV app (pushup, basketball, etc.)
- `POST /api/apps/activate` / `deactivate` — start/stop the app
- `GET /api/apps/status` — what app is running, current state (scores, reps, etc.)
- `POST /api/apps/competition/start` / `end` — run escrow-backed competitions

**Media:**
- `POST /api/capture` — take a photo
- `POST /api/record` — start/stop video recording
- `GET /api/photos`, `GET /api/videos` — list captured media

**Streaming:**
- WebRTC, WHIP/WHEP, Livepeer endpoints for live video

**Identity:**
- `POST /api/face/enroll/confirm` — enroll face (one-time, network-wide)
- `POST /api/face/recognize` — list checked-in users

**Device info:**
- `GET /api/health` — health check with device signature
- `GET /api/camera/info` — camera specs and status
- `GET /api/stream/info` — streaming endpoint info

### What the backend adds (api.mmoment.xyz)

**Session history:**
- `GET /api/user/:wallet/sessions` — all sessions for a user
- `GET /api/user/:wallet/activities` — all user activities
- `GET /api/camera/:cameraId/activities` — all activities at a camera
- `GET /api/session/:sessionId/timeline` — timeline events for a session

**Real-time events (Socket.IO):**
- `joinCamera` / `leaveCamera` — subscribe to camera room
- `newTimelineEvent` — real-time activity events
- `recentEvents` — get recent events for a camera

**User profiles:**
- `POST /api/profile/save`, `GET /api/profile/:wallet` — user profiles

**Storage:**
- Walrus (Sui) and Pipe (Firestarter) for decentralized media storage
- Gallery endpoints per user

**Gas sponsorship:**
- `POST /api/sponsor-transaction` — users don't pay Solana tx fees

---

## Unified Gateway

The API gateway at `api.mmoment.xyz` unifies per-camera APIs into a single queryable network:

```
api.mmoment.xyz/cameras/{camera_pda}/...  →  proxies to that camera's tunnel URL
api.mmoment.xyz/events/...                →  queries across all cameras
api.mmoment.xyz/users/{wallet}/...        →  queries across all cameras for a user
```

The gateway knows about all registered cameras (they're on-chain PDAs). It routes per-camera requests to the right tunnel and fans out network-wide queries across cameras.

A developer hits one endpoint. The network handles routing.

---

## Visibility = Consent, Not Tiers

There are no subscription tiers. There are **relationships** between you and the data:

**You are the user** — You see everything about yourself. Always. Your data, your keys.

**You were there** — You were checked in at the same camera at the same time. You see what happened while you were present. This is the physical-world default.

**You are the host** — You own the camera. You see what happens at your camera.

**You have a grant** — A user or venue explicitly granted you access. OAuth-style delegation, revocable.

**It's public** — The camera/venue is set to open. Anyone can query.

The API resolves these relationships on every request. The same endpoint returns different data depending on who's asking and what the consent graph allows.

---

## What Makes This Different

Most APIs gate access by price. Pay more, see more.

This API gates access by **trust**. The humans and venues on the network decide what's visible. The API faithfully serves that reality.

- **A free indie dev** building a public leaderboard for an open basketball court has full access to that court's data. Because the court is open.
- **A funded startup** trying to analyze private gym data has zero access. Unless the gym and its users grant it.
- **A gym owner** sees everything at their gym. But nothing at the gym across town.

Payment is for **infrastructure** (rate limits, bandwidth, SLA), not data access. You don't pay to unlock data. You build relationships to earn visibility.

---

## Example Flows

### Friend group (the core growth loop)

1. 3 out of 20 friends buy a dev kit camera
2. Friends place cameras in their home gyms, garages, basketball hoops
3. Any of the 20 friends can check in at any of the 3 cameras (NFC tap + wallet sig)
4. After checking in, they can query their own data, see co-present friends' data
5. A dev in the friend group builds a custom leaderboard app using the API
6. That app works across all 3 cameras because the API is network-wide

### Indie dev builds a public court leaderboard

1. `GET api.mmoment.xyz/cameras?geo=boulder,5km` → finds cameras near them
2. Subscribes to events at an open court via Socket.IO
3. Builds a website showing live scores, player stats
4. All data is public because the court host set it to open

### Fitness app wants user workout data

1. User connects MMOMENT identity to the app (OAuth grant)
2. `GET api.mmoment.xyz/users/{wallet}/activities?type=rep_completed`
3. App gets workout history across every camera the user has visited
4. User revokes → app sees nothing

### Competition with escrow

1. Creates escrow on-chain: "50 pushups in 10 minutes, 0.1 SOL stake"
2. Subscribes to real-time events from the specific camera
3. Camera attests rep counts as they happen
4. At timeout: settle escrow based on camera-attested count

---

## Current Blockers

### Check-in must prove physical presence

Today, check-in is an ed25519 signature between wallet and camera. This proves **identity** but not **presence**. Someone who knows the camera URL can check in from anywhere.

For the network's core value prop ("proof of showing up") to hold, check-in must mean "I am physically here right now." This is an unsolved problem — see separate discussion on NFC/presence verification approaches.

### Network-wide query layer doesn't exist yet

Individual camera APIs exist. The backend stores session data. But there's no unified `api.mmoment.xyz` gateway that routes to cameras and fans out queries across the network. This is the infrastructure work needed to go from "each camera is an island" to "one queryable network."

### Visibility defaults need to be set

Camera hosts need a way to set their visibility floor (self-only, co-present, host, open). This determines what the API shows to different relationship types. The on-chain CameraAccount exists but doesn't have a `visibility_floor` field yet.
