# PIPE NETWORK API - CONFIRMED FACTS

**LAST UPDATED: 2025-11-07**

## CRITICAL: What DOESN'T Work

### ❌ `/listFiles` endpoint DOES NOT EXIST
- Returns 404
- DO NOT waste time trying to list files from Pipe
- We discovered this multiple times - it doesn't work

### ❌ Cannot be logged in from multiple locations simultaneously
- **CRITICAL**: Pipe API only allows ONE active session at a time
- If you login from location A, then login from location B, location A gets 401 errors
- This is why local backend testing fails when Railway backend is also running
- **DO NOT** try to test locally while Railway is deployed and running
- Must choose: either test locally OR use Railway, not both at once

### ❌ Downloading files returns "Failed to access file in storage"
- Both hash and filename return 500 error
- Pipe response: "Failed to access file in storage"
- Tested with:
  - Hash: `d4497c277294adae368c55326208237c2c230148f542404665447f176a53d534`
  - Filename: `RsLjCiEiHq3dyWeDpp1M8jSmAhpmaGamcVK32sJkdLT_photo_1762541096_17e582.jpg`

## What DOES Work

### ✅ Upload via `/priorityUpload`
- Uploads successfully
- Returns: "File uploaded (X bytes) - Background transfer in progress"
- File is stored on Pipe with Blake3/SHA256 hash as the stored filename

### ✅ Authentication
- **JWT Bearer Token** - CONFIRMED WORKING for uploads
  - Login: `POST /auth/login` with username/password
  - Use: `Authorization: Bearer {token}`
  - Account: `mmoment_jetson_1762380237` / `StrongPass123!@#`
  - User ID: `c14bbd63-3f06-4990-ac04-7101f38e2a27`

## The Mystery

### 🤔 Yesterday it worked - what was different?
- Downloads were working with commit `9fa746b`
- Gallery showed photos and they downloaded successfully
- Need to figure out what changed between then and now

**TODO: Check Railway logs or what files were ACTUALLY working yesterday**
