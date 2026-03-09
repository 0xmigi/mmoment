# MMOMENT Business Model

## What MMOMENT Sells

User-consented, identity-resolved perception data across a network of cameras. Every identified data point is backed by a cryptographic consent signature (ed25519 between user wallet and camera), with the full session published on-chain at checkout. The data ranges from single-camera real-time pose streams to cross-network activity histories spanning every camera a user has visited.

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

Enroll their face once (create a RecognitionToken on-chain with encrypted embedding — works at every camera in the network). Check in at cameras by producing an ed25519 signature between their wallet and the camera (local, instant, no tx fees). Do physical things. Session published to chain at checkout.

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

**What they buy**: Access to perception data that users and venues have made visible. What's available depends on the visibility model — the camera's floor, the user's consent level, and whether the consumer has been explicitly authorized.

**Why**: The perception network eliminates the hardest part of building a physical-world AI product — getting reliable, identity-resolved sensor data from real spaces. A fitness coaching agent doesn't need to build cameras, run YOLO, solve identity, or handle privacy. It subscribes to the API and does math on coordinates.

**Types**:

| Consumer | What They Access | What They Build |
|----------|-----------------|----------------|
| Fitness coaching agent | Authorized user's pose stream | Form feedback, workout tracking |
| Sports analytics | Authorized user's pose + activity history | Game stats, performance tracking |
| Personal analytics | Authorized user's cross-camera sessions | "Your day across the built environment" |
| Social platform | Co-present users' shared sessions | Co-presence detection, shared moments |
| Venue analytics | Anonymous aggregate data (floor-level) | Occupancy, foot traffic, dwell time |
| Competition platform | Authorized participants' activity | Escrow-backed physical challenges |

### Developers

Build the consumers. Use MMOMENT's SDK, the `frame_data` contract, and reference implementations (pushup app, basketball app) to create products that run on network data.

**What they don't need**: Camera hardware, CV model training, edge deployment, identity infrastructure, encryption handling. All outsourced to the network.

## Consent Architecture

The consent mechanism is structural, not policy:

1. User enrolls face once → encrypted embedding stored on-chain as RecognitionToken (works network-wide)
2. User checks in at camera → ed25519 signature between wallet and camera PDA (local, instant, no tx fees)
3. Camera detects face → matches embedding → resolves identity ONLY for checked-in users
4. Non-checked-in individuals → detected but NOT identified, obfuscated in output
5. All identity-resolved data → encrypted at the Jetson (AES-256-GCM) before leaving device
6. Full session (check-in, activities, checkout) published to blockchain when user leaves → tamper-evident, user-owned

**No signature, no identity.** The system cannot produce identity-resolved data without the ed25519 consent signature. This is not "we promise to respect privacy" — it's "the architecture makes violation impossible."

**Auditability**: Any consumer querying identified data can verify the on-chain session record published at checkout. The consent proof is cryptographic, not a checkbox in a database.

**Check-in UX**: The ed25519 signature is local and instant — no blockchain latency at the moment of check-in. Target UX is NFC-triggered signing (tap phone near camera → wallet app signs → done). The chain interaction is deferred to checkout, asynchronous, invisible to the user.

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
- Face enrollment (RecognitionToken creation) is the main friction point, but it's one-time and network-wide. Once enrolled, every camera in the network recognizes the user. Check-in itself is an instant local signature — target UX is NFC tap.
- First users will be crypto-native (already have wallets). Path to mainstream is reducing check-in to a physical gesture (NFC, QR) that triggers signing under the hood.
- Biometric data triggers regulatory attention (BIPA, GDPR) even with encryption. The cryptographic consent architecture — provable ed25519 signatures, on-chain session records — is probably the strongest legal position anything in this space can have.

## Long-Term Hardware Vision

Venues already maintain two categories of hardware that don't talk to each other: **security cameras** (perception without identity or consent) and **payment terminals** (identity + payment without perception). MMOMENT can eventually replace both.

### The convergence

A MMOMENT terminal is a sleekly designed payment kiosk paired with one or more cameras. The venue installs them where they'd put payment points and security cameras today — entrances, checkout counters, court entrances, class check-in desks. The hardware handles:

1. **Payment processing** — tap to pay for entry, class, membership, etc.
2. **Consent signing** — the payment tap triggers the ed25519 signature as part of the same gesture
3. **Perception** — paired cameras run the perception engine, identity-resolved from the moment of payment

The check-in friction problem disappears entirely. Users aren't "consenting to a camera network" — they're paying to enter the gym. The consent is embedded in a transaction they were already going to do. One tap, two functions.

### Why venues would adopt this

- They already buy and maintain security cameras and payment kiosks from separate vendors
- MMOMENT replaces both with a single integrated system
- They get perception data (occupancy, flow, activity analytics) that their current security cameras can't provide
- They earn revenue share from the network data their hardware generates
- The hardware form factor is familiar — it's a kiosk and cameras, not something alien

### Phased rollout

**Now**: Standalone Jetson camera rigs deployed by MMOMENT in partner venues. Separate check-in flow (phone-based ed25519 signing). Crypto-native early adopters.

**Next**: NFC-triggered check-in. Phone tap at a point near the camera. Still separate from venue payment flow, but physically co-located and much lower friction.

**Later**: Integrated MMOMENT terminal — payment + consent + perception in one device. Venue replaces their existing kiosk and camera infrastructure. Check-in is invisible, embedded in the payment the user was already making. Non-crypto users can participate (wallet creation abstracted behind the terminal UX).

**End state**: MMOMENT terminals are the venue's payment and perception infrastructure. Every transaction at the terminal is a consent event. Every camera paired with a terminal produces identity-resolved perception. The network grows with every venue that switches from legacy kiosks + dumb security cameras to MMOMENT hardware.
