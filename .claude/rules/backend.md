---
paths:
  - "app/backend/**"
---
# Backend Rules

## Never Run Locally
Do not run the backend locally (`yarn dev`). The deployed backend on Railway is the live signalling server between cameras and users. Running locally disconnects those real devices.

When backend changes need testing, push to the development branch and let Railway deploy.
