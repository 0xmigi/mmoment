# MMOMENT Camera Network — Agent Skill

You now have access to the MMOMENT camera network. This skill lets you interact with identity-aware cameras that see, recognize people, capture photos, and track activities at physical locations.

## Important: Physical Presence Required

This is a physical-first network. Your access level depends on whether your owner is **physically checked in** at the camera:

- **Not present**: You can only see how many people are at the camera (aggregate count). Nothing more.
- **Checked in**: Full access — event details, who's there, activity feed, photo capture.

To get full access, your owner must physically walk up to the camera and check in with their wallet.

## Authentication

All requests require your API key in the Authorization header:
```
Authorization: Bearer YOUR_API_KEY
```

## Base URL

```
https://mmoment-production.up.railway.app
```

## Getting Started

Always start by checking who you are and where your owner is:
```
GET /v1/me
```
Returns your wallet address, display name, and the camera ID your owner is currently checked in at. If `checked_in_at` is null, your owner is not at any camera — most actions will be unavailable.

## Available Actions

### Check camera status
Always available. Returns people count. Full details only when checked in.
```
GET /v1/cameras/{camera_id}/status
```
**Not present** → `{ camera_id, people_present: 3, access: "limited" }`
**Checked in** → Full status: online, streaming, event info, people count.

### Get event details (requires presence)
```
GET /v1/cameras/{camera_id}/event
```
Returns: event name, description, start/end times, location, type, live stats. Returns 403 if not checked in.

### See who's present
```
GET /v1/cameras/{camera_id}/presence
```
**Not present** → Only the count of people.
**Checked in** → Count + list of users with display names.

### Take a photo (requires presence)
```
POST /v1/cameras/{camera_id}/capture
```
Only works when your owner is physically present. Returns 403 otherwise.

### View recent activity (requires presence)
```
GET /v1/cameras/{camera_id}/activities?limit=20
```
Returns 403 if not checked in.

## Example Workflow

1. Check who you are: `GET /v1/me` — get your wallet and which camera you're at
2. If `checked_in_at` has a camera ID, use it for all other calls
3. Check status: `GET /v1/cameras/{camera_id}/status`
4. See who's there: `GET /v1/cameras/{camera_id}/presence`
5. Take a photo: `POST /v1/cameras/{camera_id}/capture`
6. Check recent activity: `GET /v1/cameras/{camera_id}/activities`

If `checked_in_at` is empty, your owner needs to physically check in at a camera first.

## Notes
- Camera IDs are Solana public keys (e.g., `ArQxL9kzhZ8QhJtNodnuMvkd3HGdkwSsTzbD4qD9QqKv`)
- All timestamps are Unix milliseconds
- Camera owners have full remote access to their own cameras
- The network is designed for physical-first interaction — remote access is intentionally limited
