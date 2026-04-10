# Solana Transaction Verification Middleware

A simple middleware that sits between your frontend and camera API to verify Solana transactions.

## How It Works

1. Frontend sends transactions to Solana blockchain
2. After confirmation, frontend sends transaction signature to middleware
3. Middleware verifies:
   - Transaction is confirmed
   - Transaction involves the correct program
   - Transaction involves the correct camera PDA
4. If verified, middleware forwards the request to the camera API

## API Endpoints

The middleware exposes the same endpoints as your camera API, but requires a Solana transaction signature:

### Take Photo
```
POST /api/capture
{
  "tx_signature": "SOLANA_TRANSACTION_SIGNATURE"
}
```

### Start Stream
```
POST /api/stream/start
{
  "tx_signature": "SOLANA_TRANSACTION_SIGNATURE"
}
```

### Other Endpoints (Passthrough)
These endpoints don't require a transaction signature:
- `GET /api/health` - Check middleware and camera API health
- `GET /api/stream/info` - Get stream status 
- `POST /api/stream/stop` - Stop stream

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Configure in `.env`:
   ```
   SOLANA_PROGRAM_ID=your_program_id
   CAMERA_PDA=your_camera_pda
   ```

3. Run setup script:
   ```
   chmod +x setup.sh
   ./setup.sh
   ```

4. Test the middleware:
   ```
   curl http://localhost:5002/api/health
   ```

## Modify Your Frontend

Update your frontend to send transaction signatures with photo/stream requests:

```javascript
// After transaction confirmation:
const response = await fetch('http://localhost:5002/api/capture', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ tx_signature: signature })
});
``` 