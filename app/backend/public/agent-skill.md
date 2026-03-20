# MMOMENT Camera Network

You can interact with physical cameras that see, recognize people, and capture photos. Photos are returned as publicly accessible URLs that you can display inline.

## Setup

Base URL: `https://mmoment-production.up.railway.app`

All requests need: `Authorization: Bearer YOUR_API_KEY`

## Step 1: Always call this first

```
GET /v1/me
```

This tells you who you are and which camera your owner is at. Example response:
```json
{
  "data": {
    "wallet": "8Pd3qgn...",
    "display_name": "Ash",
    "checked_in_at": "ArQxL9kzhZ8QhJtNodnuMvkd3HGdkwSsTzbD4qD9QqKv"
  }
}
```

Use the `checked_in_at` value as `{camera_id}` for all other calls. If it's null, your owner is not at a camera — tell them to check in first.

## Step 2: Use these endpoints

| Action | Method | Endpoint |
|--------|--------|----------|
| Camera status | GET | `/v1/cameras/{camera_id}/status` |
| Event details | GET | `/v1/cameras/{camera_id}/event` |
| Who's here | GET | `/v1/cameras/{camera_id}/presence` |
| Take a photo | POST | `/v1/cameras/{camera_id}/capture` |
| Start recording | POST | `/v1/cameras/{camera_id}/record/start` (optional `duration` in seconds) |
| Stop recording | POST | `/v1/cameras/{camera_id}/record/stop` |
| Recent activity | GET | `/v1/cameras/{camera_id}/activities` |

## Displaying photos

The `photo_url` returned by `/capture` is a publicly accessible URL hosted on this server. You can fetch it directly and display it inline to the user. No authentication is needed to view photos.

## Physical presence required

This is a physical-first network. Most endpoints return 403 or limited data unless your owner is physically checked in at the camera. Do not try to guess camera IDs — always get them from `/v1/me`.
