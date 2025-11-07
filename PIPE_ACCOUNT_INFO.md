# Pipe Network Account Information

## 🟢 ACTIVE WORKING ACCOUNT

**Account:** `mmoment_jetson_1762380237`
**Status:** ✅ FULLY FUNCTIONAL
**Created:** Recently (has full JWT auth support)

### Credentials (SAVE THESE!)
```
Username: mmoment_jetson_1762380237
Password: StrongPass123!@#
user_id: c14bbd63-3f06-4990-ac04-7101f38e2a27
user_app_key: f7ad7c104508330c977ab1ab20b76aa1ef575327782d3e5e9380c81e5946bb6b
Deposit Address: Fhj4wUNc1NN76i2psAeBdBKTr3YhgudPvyuQUEY6UjMk
```

### Current Balance
- **PIPE Tokens:** 0 PIPE (ready to receive)
- **SOL:** 0 SOL

### To Fund This Account
Send PIPE tokens (SPL token on Solana mainnet) to:
```
Fhj4wUNc1NN76i2psAeBdBKTr3YhgudPvyuQUEY6UjMk
```

PIPE Token Mint Address: `7s9MoSt7VV1J3jVNnw2AyocsQDBdCkPYz5apQDPKy9i5`

---

## 🔴 BROKEN ACCOUNTS

### wallettest1762286471
- **Status:** ❌ 426 Error - Requires Migration
- **Issue:** Returns "Please set a password first" error
- **Cause:** Account flagged for password migration, missing `user_id`/`user_app_key`
- **On-chain Balance:** 1.0 PIPE (verified on Solana mainnet)
- **Recovery:** Not possible without original `user_id` and `user_app_key`

### RsLjCiEiHq3dyWeDpp1M (CLI Account)
- **Status:** ⚠️ Working but on old devnet
- **Has:** Full credentials (`user_id`, `user_app_key`)
- **Issue:** Created on devnet, has 70+ old uploads but 0 PIPE balance
- **Use Case:** Only useful for testing/development

---

## Best Practices to Avoid 426 Errors

### 1. Always Save Full Credentials
When creating a new Pipe account, ALWAYS save:
- ✅ `username`
- ✅ `password`
- ✅ `user_id` (from `/users` creation response)
- ✅ `user_app_key` (from `/users` creation response)
- ✅ `deposit_address` (from `/checkWallet` or creation response)

### 2. Implement JWT Token Caching
Don't login on every request:
```javascript
// BAD - Login every time
async function uploadFile() {
  const token = await login(); // ❌ Too many logins!
  await upload(token);
}

// GOOD - Cache and reuse tokens
let cachedToken = null;
let tokenExpiry = null;

async function getToken() {
  if (cachedToken && Date.now() < tokenExpiry) {
    return cachedToken; // ✅ Reuse existing token
  }
  const { access_token, expires_in } = await login();
  cachedToken = access_token;
  tokenExpiry = Date.now() + (expires_in * 1000);
  return cachedToken;
}
```

### 3. Use Refresh Tokens
JWT tokens expire after 15 minutes. Use refresh tokens instead of re-logging in:
```javascript
// If access token expires, use refresh token
if (response.status === 401) {
  const newToken = await refreshAccessToken(refreshToken);
}
```

### 4. Handle Migration Gracefully
If you get a 426 error, try password reset:
```javascript
if (response.status === 426) {
  // Try to reset password with saved credentials
  await fetch('/auth/set-password', {
    method: 'POST',
    body: JSON.stringify({
      user_id: SAVED_USER_ID,
      user_app_key: SAVED_USER_APP_KEY,
      new_password: PASSWORD
    })
  });
}
```

---

## Environment Variables

Updated `.env` file with working account:
```bash
PIPE_USERNAME=mmoment_jetson_1762380237
PIPE_PASSWORD=StrongPass123!@#
PIPE_USER_ID=c14bbd63-3f06-4990-ac04-7101f38e2a27
PIPE_USER_APP_KEY=f7ad7c104508330c977ab1ab20b76aa1ef575327782d3e5e9380c81e5946bb6b
PIPE_DEPOSIT_ADDRESS=Fhj4wUNc1NN76i2psAeBdBKTr3YhgudPvyuQUEY6UjMk
```

---

## Next Steps

1. **Fund the Account**
   - Send PIPE tokens to `Fhj4wUNc1NN76i2psAeBdBKTr3YhgudPvyuQUEY6UjMk`
   - Minimum recommended: 1 PIPE token
   - Check balance with `/checkCustomToken` endpoint

2. **Test Upload/Download**
   - Once funded, test with small file (~1KB)
   - Verify download integrity

3. **Deploy Updated Backend**
   - Updated `.env` is ready
   - Deploy to Railway with new credentials
   - Monitor for any 426 errors (shouldn't happen)

4. **Update Jetson Devices**
   - Update camera service with new credentials
   - Test direct uploads from Jetson

---

## Support & Troubleshooting

- **API Base URL:** `https://us-west-01-firestarter.pipenetwork.com`
- **API Documentation:** See `firestarter-sdk/API_REFERENCE.md`
- **Token Mint:** `7s9MoSt7VV1J3jVNnw2AyocsQDBdCkPYz5apQDPKy9i5`
- **Network:** Solana Mainnet-Beta

If you encounter 426 errors again:
1. Check that you have `user_id` and `user_app_key` saved
2. Use `/auth/set-password` endpoint to reset
3. Verify password meets requirements (8+ chars, uppercase, lowercase, numbers, symbols)
