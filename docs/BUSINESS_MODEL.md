# MMOMENT Business Model

## What MMOMENT Sells

User-consented, identity-resolved perception data across a network of cameras. Every identified data point is backed by a cryptographic check-in signature on Solana. The data ranges from single-camera real-time pose streams to cross-network activity histories spanning every camera a user has visited.

## Participants

### MMOMENT (the team)

Builds and maintains the perception engine, the Solana protocol, the network API, and the web application. Operates the API layer that aggregates and serves data from all cameras in the network.

**Revenue**: Protocol fee on all API access (percentage of x402 transactions flowing through the network).

**Incentive**: Increase total perception throughput — more cameras, better engine quality, more consumers buying access. Stream quality and network coverage directly determine revenue.

**Moat**: The identity layer. Person detection and pose estimation are commodity CV. Identity-resolved perception — knowing which wallet owns which bounding box, verified by cryptographic check-in — requires the enrollment flow, on-chain recognition tokens, and the full consent architecture. That's the protocol lock-in.

### Camera Hosts

Buy or receive camera hardware (Jetson rig), place it in a physical location, maintain power and connectivity. Run the MMOMENT perception engine. Do not configure APIs, set pricing, or manage subscriptions.

**Revenue**: Share of x402 fees proportional to activity flowing through their camera. More checked-in users doing interesting things at their camera = more revenue.

**Incentive**: Place cameras where physical activity happens — gyms, courts, event venues, public spaces. Maintain uptime. A camera that's offline or in an empty room earns nothing.

**Analogy**: Cell tower operators. They provide coverage. They don't sell minutes.

### Users

Enroll their face (create a RecognitionToken on-chain with encrypted embedding). Check in at cameras by signing a transaction with their Solana wallet. Do physical things. Check out.

**What they get**:
- Camera-attested activity data (not self-reported)
- Encrypted content ownership (photos, videos, stored on Walrus/IPFS, decryptable only by them)
- Proof of presence (cryptographic evidence of being at a specific place at a specific time)
- Access to escrow competitions (camera as neutral referee)
- Whatever services consumers build on top of the network data

**Incentive**: The camera sees things about them that they can't credibly self-report. A phone can count steps — but it can't verify 50 push-ups with proper form, prove you were at the gym at 6am (not GPS-spoofable), or referee a bet with a stranger.

**Critical role**: Users are what make the network valuable. Agents don't pay for empty cameras. They pay for identity-resolved perception of consenting humans doing things. The user's physical presence is the scarce resource.

### Consumers (agents, apps, services)

Access the MMOMENT API to query or subscribe to perception data. Pay via x402 micropayments.

**What they buy**: Time-bounded or query-based access to network data at their chosen tier (events, detections, full pose, pose + frame).

**Why**: The perception network eliminates the hardest part of building a physical-world AI product — getting reliable, identity-resolved sensor data from real spaces. A fitness coaching agent doesn't need to build cameras, run YOLO, solve identity, or handle privacy. It subscribes to the API and does math on coordinates.

**Types**:

| Consumer | Data Tier | What They Build |
|----------|-----------|----------------|
| Fitness coaching agent | Full pose | Form feedback, workout tracking |
| Sports analytics | Full pose | Game stats, performance tracking |
| Personal analytics | Events | "Your day across the built environment" |
| Social platform | Events / Detections | Co-presence detection, shared moments |
| Venue analytics | Detections (anonymous) | Occupancy, foot traffic, dwell time |
| Competition platform | Events + pose | Escrow-backed physical challenges |

### Developers

Build the consumers. Use MMOMENT's SDK, the `frame_data` contract, and reference implementations (pushup app, basketball app) to create products that run on network data.

**What they don't need**: Camera hardware, CV model training, edge deployment, identity infrastructure, encryption handling. All outsourced to the network.

## Consent Architecture

The consent mechanism is structural, not policy:

1. User enrolls face → encrypted embedding stored on-chain as RecognitionToken
2. User checks in at camera → signs Solana transaction → check-in recorded on-chain
3. Camera detects face → matches embedding → resolves identity ONLY for checked-in users
4. Non-checked-in individuals → detected but NOT identified, obfuscated in output
5. All identity-resolved data → encrypted at the Jetson (AES-256-GCM) before leaving device
6. Activity committed to blockchain at checkout → tamper-evident, user-owned

**No check-in, no identity.** The system cannot produce identity-resolved data without the cryptographic consent event. This is not "we promise to respect privacy" — it's "the architecture makes violation impossible."

**Auditability**: Any consumer querying identified data can verify: this user signed a check-in at this camera PDA at this timestamp. The consent proof is a Solana transaction, not a checkbox in a database.

## Money Flow

```
Users check in (free — their presence is what makes the network valuable)
    │
    ▼
Cameras perceive (Jetson GPU running perception engine)
    │
    ▼
Data aggregated at network level (MMOMENT API)
    │
    ▼
Consumers pay x402 for API access
    │
    ├──► Camera host receives share (proportional to their camera's activity)
    └──► MMOMENT receives protocol fee
```

Separately, consumers may charge end users for their products:

```
User wants coaching ──► Pays coaching agent (fiat/crypto)
                              │
                              ▼
                        Agent pays x402 for network data
                              │
                              ▼
                        Agent delivers coaching to user
```

The user doesn't pay the camera or the network directly. Consumers pay MMOMENT for data access. Revenue flows back to camera hosts proportionally.

## The Flywheel

1. Better perception engine → higher quality data
2. Higher quality data → more consumer subscriptions
3. More subscriptions → more revenue for camera hosts
4. More revenue → more cameras deployed in more locations
5. More cameras → more convenient for users to check in
6. More users → network data becomes more valuable (identity-resolved activity at scale)
7. More valuable data → consumers pay more → back to step 2

## Current State and Next Steps

**Built**: Perception engine (YOLO + ArcFace + pose + ReID), Solana protocol (camera registration, recognition tokens, session timelines, escrow), user check-in flow, on-device encryption, activity buffering, reference CV apps (pushup, basketball), web application.

**The gap**: Network-level API with x402 payment. Today, data lives on individual cameras and flows to the backend for storage. The missing piece is the consumer-facing API that aggregates across cameras and gates access via micropayments.

**Sequencing considerations**:
- Cold start requires cameras + users + consumers in the same location. First deployment should be vertically integrated (MMOMENT controls camera, recruits users, builds first consumer) in one venue type.
- Check-in friction is high (Solana wallet + face enrollment). First users will be crypto-native. Path to mainstream requires making check-in feel like tapping a subway turnstile.
- Biometric data triggers regulatory attention (BIPA, GDPR) even with encryption. The cryptographic consent architecture is the strongest legal defense, but the argument needs to be ready.
