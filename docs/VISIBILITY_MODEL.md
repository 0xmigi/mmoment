# Visibility Model

## Core Principle

In the physical world, consent is governed by physics. If you walk into a room, everyone in the room can see you. If you leave, they can't. You consented to being seen by walking in. You revoked that consent by walking out. No terms of service, no checkboxes — just the laws of matter.

MMOMENT replicates this. The camera is the room. Checking in is walking through the door. Checking out is leaving. The default visibility of your session matches what physics would give you: the people who were there at the same time can see what happened.

The difference from the physical world is the **internet overlay**. Any party — you, the venue, an app — can opt to share *more* broadly than physics would allow. But the rules of that escalation are always visible before you walk in, and no one can make your data more visible without you knowing in advance.

**Physics is the default. Internet is the opt-in.**

---

## The Visibility Spectrum

Six ordered levels, from most private to most public. Each level includes all access from the levels below it.

| Level | Name | Physical-World Analogy | Who Can See |
|-------|------|----------------------|-------------|
| 0 | **Self** | Taking a selfie on your phone | Only you |
| 1 | **Co-Present** | Being in a room with other people | Everyone checked in at the same time |
| 2 | **Host** | A business with security cameras | Camera owner / venue operator |
| 3 | **Authorized** | Sharing your gym stats with a fitness app | Third-party apps you've explicitly granted access |
| 4 | **API** | Posting to a social feed | Any authenticated consumer of the MMOMENT API |
| 5 | **Internet** | Being on live television | Anyone, no authentication required |

Level 1 (Co-Present) is the **physics default**. It's what the real world gives you for free. Levels 2–5 are internet-overlay escalations that don't exist in physics — they require explicit decisions by someone.

---

## Floors and Escalation

### The Floor

Every camera has a **visibility floor**: the minimum visibility level that applies to all sessions at that camera. The floor is set by whoever operates the camera.

The floor answers the question: *"What's the least private this camera will be?"*

```
Camera: "Downtown Gym - Court 1"
Floor:  Level 1 (Co-Present)

→ Everyone checked in at the same time can see each other's sessions.
→ The gym owner cannot see your session content.
→ Nobody on the internet can see your session.
→ This is the physical-world default.
```

```
Camera: "Live Concert Stage"
Floor:  Level 4 (API)

→ Your session is accessible to any MMOMENT API consumer.
→ This is like being in the audience of a filmed event.
→ You see this BEFORE you check in. Don't like it? Don't check in.
```

```
Camera: "Private Physical Therapy Room"
Floor:  Level 0 (Self)

→ Only you can see your session. Not even other checked-in patients.
→ The therapist's camera captures your data, but only you hold the keys.
→ More private than physics — in a real room, the therapist sees you.
```

### Escalation

Any party can make their *own* data more visible than the floor. Never less.

- **You** can share your session publicly even if the floor is Co-Present.
- **The venue** can set the floor to API, making all sessions public-by-default.
- **You cannot** make yourself invisible at a venue with a Level 4 floor. If you don't accept the floor, you don't check in.

Escalation is always **upward** (more visible) and always **voluntary for the party whose data it is**.

### The Rules

1. **Floor is set before check-in.** The camera's visibility floor is fixed and disclosed to the user before they produce their check-in signature. It's part of the camera's on-chain metadata.

2. **Floor cannot increase on active sessions.** If a venue changes its floor from Level 1 to Level 4, existing checked-in sessions stay at Level 1. The new floor only applies to future check-ins. No retroactive escalation.

3. **Individuals escalate themselves.** A user can choose to share above the floor. "Make this session public" pushes their data from Co-Present to API. This only affects their data, not other co-present users.

4. **Venues set the floor, not the ceiling.** A Level 1 floor doesn't prevent a user from sharing their own session at Level 5. The floor is about the minimum, not the maximum.

5. **Each level is additive.** Level 3 (Authorized) means apps can see it, the host can see it, co-present users can see it, and you can see it. Nothing skips a level.

6. **Consent is the check-in signature.** The ed25519 signature between wallet and camera is the consent event. The user is signing with full knowledge of the floor. No signature, no session, no data.

---

## Party Hierarchy

Starting from the individual and scaling up. Each level adds parties, but never removes the ones below.

### Level 0: You

You check in. The camera runs CV apps, captures photos, records video. Everything is encrypted with a per-session AES-256 key. That key is encrypted for your wallet. You are the only person who can decrypt your session content.

**Access**: Only you, via your wallet.
**Analogy**: Taking photos on your phone. They're yours. Nobody else sees them.
**Current implementation**: Session key encrypted for user's pubkey, stored in UserSessionChain on-chain.

### Level 1: You + Co-Present Users

Three people check in at the same camera. The session key is distributed to all three via access grants (encrypted per-user). Everyone who was physically present during the overlapping time window can see the session content from that window.

Someone who checks in after you leave? They can't see your session. Someone who wasn't there? They can't see it. Presence equals access.

**Access**: All wallets that were checked in during overlapping time.
**Analogy**: Being in a room. Everyone in the room can see what happens. Leave the room, and new arrivals don't see what happened before.
**Current implementation**: Access grants in EncryptedActivity distribute session keys to all checked-in users at activity time.

### Level 2: You + Co-Present + Host

The camera owner can see session content from their camera. They don't need to be checked in — they own the hardware. This is like a business owner reviewing their security footage.

The host sees what happened at their camera. They don't see what you did at other cameras. Their visibility is scoped to their hardware.

**Access**: Co-present users + camera owner wallet.
**Analogy**: A store with security cameras. The owner can review footage. You knew there were cameras when you walked in.

### Level 3: You + Co-Present + Host + Authorized Apps

You connect a third-party app to your MMOMENT data. OAuth-style: "FitCoach wants to read your session activity at gym cameras." You approve. Now FitCoach can see your sessions at cameras you've authorized, for the scopes you've granted.

You revoke access, it's gone. The app never had the raw keys — it had delegated, time-bounded, revocable access.

**Access**: Co-present users + host + explicitly authorized third parties.
**Analogy**: Sharing your Apple Health data with a fitness app. You chose to. You can un-choose.

### Level 4: You + Co-Present + Host + Authorized + API Consumers

Your session data is accessible to any authenticated consumer of the MMOMENT API. Not just apps you've personally approved — anyone paying for API access can query it. This is a public social layer.

A concert venue might set their floor here: "If you check in at this show, your presence is part of the public event data." You know this before checking in. If you don't want to be in the public dataset, don't check in.

**Access**: Anyone with MMOMENT API credentials.
**Analogy**: Being in the audience of a televised event. You walked in knowing cameras were rolling for broadcast.

### Level 5: Internet

Fully public. No authentication. The session data is accessible to anyone on the internet, forever. The camera's stream and all session content are open.

This is the maximum. A camera set to this floor is essentially a public broadcast. A livestreamed event, an open community space, a public art installation.

**Access**: The entire internet. No auth, no paywall, no gating.
**Analogy**: Being on live television. Everyone can see you. You knew that when you walked on stage.

---

## Visual: Nested Visibility

```
┌──────────────────────────────────────────────────────────────┐
│  5: INTERNET                                                 │
│  Fully public. No authentication. Anyone on the internet.    │
│                                                              │
│  ┌───────────────────────────────────────────────────────┐   │
│  │  4: API                                               │   │
│  │  Any authenticated MMOMENT API consumer.              │   │
│  │                                                       │   │
│  │  ┌────────────────────────────────────────────────┐   │   │
│  │  │  3: AUTHORIZED                                 │   │   │
│  │  │  Third-party apps you explicitly granted.      │   │   │
│  │  │                                                │   │   │
│  │  │  ┌─────────────────────────────────────────┐   │   │   │
│  │  │  │  2: HOST                                │   │   │   │
│  │  │  │  Camera owner / venue operator.         │   │   │   │
│  │  │  │                                         │   │   │   │
│  │  │  │  ┌──────────────────────────────────┐   │   │   │   │
│  │  │  │  │  1: CO-PRESENT                   │   │   │   │   │
│  │  │  │  │  Everyone checked in at the      │   │   │   │   │
│  │  │  │  │  same time. This is physics.     │   │   │   │   │
│  │  │  │  │                                  │   │   │   │   │
│  │  │  │  │  ┌───────────────────────────┐   │   │   │   │   │
│  │  │  │  │  │  0: SELF                  │   │   │   │   │   │
│  │  │  │  │  │  Only you. Your keys.     │   │   │   │   │   │
│  │  │  │  │  │  Always yours.            │   │   │   │   │   │
│  │  │  │  │  └───────────────────────────┘   │   │   │   │   │
│  │  │  │  └──────────────────────────────────┘   │   │   │   │
│  │  │  └─────────────────────────────────────────┘   │   │   │
│  │  └────────────────────────────────────────────────┘   │   │
│  └───────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

---

## Visual: Floor + Escalation

```
VISIBILITY SPECTRUM
   0         1           2        3            4         5
   Self      Co-Present  Host     Authorized   API       Internet
   ├─────────┼───────────┼────────┼────────────┼─────────┤
   │         │           │        │            │         │
   │         ▲           │        │            │         │
   │         │           │        │            │         │
   │     FLOOR           │        │            │         │
   │  (set by venue)     │        │            │         │
   │         │           │        │            │         │
   │         ├───────────┴────────┴────────────┴─────────┤
   │         │                                           │
   │         │  ← USER CAN ESCALATE ANYWHERE IN HERE →   │
   │         │                                           │
   │         └───────────────────────────────────────────┘
   │
   ▼
   Below the floor: not possible.
   Don't like the floor? Don't check in.
```

---

## Example Configurations

### Private Gym (Default)

```
Floor: 1 (Co-Present)
Label: "Shared with people here"
```

You check in. Others check in. You can all see the overlapping session. The gym owner can't see your workout data. No one on the internet knows you're there. This is how a physical gym works — the people in the room see you, nobody else does.

You *can* choose to share your session with a fitness app (escalate to Level 3) or post it publicly (escalate to Level 4). But the default is physics.

### Open Basketball Court

```
Floor: 2 (Host)
Label: "Shared with court operator"
```

The court operator sees session data — who played, scores, highlights. Players see each other. This is like a rec center with a front desk that tracks court usage. You knew the court was monitored when you walked on.

### Livestreamed Competition

```
Floor: 4 (API)
Label: "Public event — visible to API consumers"
```

A fitness competition. The organizer sets the floor to API. Every check-in is part of the public event data. Leaderboards, live stats, audience engagement — all powered by the perception stream being openly queryable.

You see "Public event" before you check in. If you want to compete, you accept. If you don't, you watch from the stands (no check-in required).

### Private Health Clinic

```
Floor: 0 (Self)
Label: "Private — only you"
```

A physical therapy clinic. The camera captures your movement for analysis, but only you hold the decryption keys. Not even other patients in the room, not even the therapist's admin account. More private than physics — the digital system can enforce isolation that physical space can't.

The therapist accesses your session data only if you explicitly share it with them (escalate to Level 3 as an authorized party).

### Public Art Installation

```
Floor: 5 (Internet)
Label: "Fully public — open to the internet"
```

A public interactive art piece with cameras. Everything is open. No authentication to view. The artist wants the world to see. You walk up to it knowing it's a public broadcast. Like a webcam on Times Square.

---

## How It Maps to Existing Architecture

The current system already has the cryptographic primitives. The visibility model gives them structure.

| Visibility Concept | Existing Primitive | What Changes |
|--------------------|--------------------|-------------|
| Level 0 (Self) | Session key encrypted for user's pubkey only | Already works — just don't add other access grants |
| Level 1 (Co-Present) | Access grants in EncryptedActivity for all checked-in users | Current default. No change needed. |
| Level 2 (Host) | Camera owner pubkey exists on CameraAccount | Add camera owner to access grants when floor ≥ 2 |
| Level 3 (Authorized) | AccessGrant struct exists in state.rs (unused) | Implement delegated access grants |
| Level 4 (API) | x402 API access described in business model | Serve decrypted session data to authenticated API consumers |
| Level 5 (Internet) | Not yet implemented | Unencrypted or publicly-keyed session data |
| Floor disclosure | Camera on-chain metadata (CameraAccount) | Add `visibility_floor` field |
| No retroactive escalation | Session key is per-session, generated at check-in | Floor checked at check-in time, stored with session |

### What the Floor Changes in the Check-In Flow

**Current**: User signs check-in → session created → activities encrypted for co-present users.

**With floors**: User queries camera metadata → sees floor level + label → signs check-in (consent includes floor acknowledgment) → session created with floor-appropriate access grants → activities encrypted for the correct audience.

The signature is the consent event. The floor is part of what's being consented to. Camera metadata is on-chain and auditable — the venue can't claim the floor was Level 1 when it was Level 4.

---

## What This Model Does NOT Cover

- **Content moderation.** The visibility model governs who *can* access data. It doesn't govern what content is acceptable. That's a separate concern.
- **Data retention.** How long session data persists on-chain or in storage is orthogonal to who can see it while it exists.
- **Pricing.** Whether API access costs money (x402), how much, and who gets paid is the business model, not the visibility model.
- **App permissions scope.** Level 3 says apps get access. *What* access (events only? pose data? raw frames?) is defined by the authorization grant between the user and the app.

---

## Summary

The visibility model is one axis with one rule:

**Axis**: Self → Co-Present → Host → Authorized → API → Internet

**Rule**: The venue sets the floor. You can go higher. You can't go lower. You always know the floor before you walk in.

Everything else — encryption, access grants, session keys, on-chain metadata — is implementation detail in service of this single idea: **physics is the default, internet is the opt-in, and you always know the rules before you play.**
