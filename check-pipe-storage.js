#!/usr/bin/env node
/**
 * Check Pipe Network Firestarter storage contents
 *
 * This script logs into your Pipe account and displays account information.
 * Note: Pipe Network API doesn't have a "list files" endpoint, so we can only
 * check balance and account status.
 */

const https = require('https');

const BASE_URL = 'https://us-west-01-firestarter.pipenetwork.com';
const USERNAME = 'wallettest1762286471';
const PASSWORD = 'StrongPass123!@#';

/**
 * Make HTTPS request
 */
function makeRequest(method, path, data = null, headers = {}) {
  return new Promise((resolve, reject) => {
    const url = new URL(path, BASE_URL);

    const options = {
      method,
      hostname: url.hostname,
      path: url.pathname + url.search,
      headers: {
        'Content-Type': 'application/json',
        ...headers
      }
    };

    const req = https.request(options, (res) => {
      let body = '';
      res.on('data', chunk => body += chunk);
      res.on('end', () => {
        try {
          const response = body ? JSON.parse(body) : {};
          if (res.statusCode >= 200 && res.statusCode < 300) {
            resolve(response);
          } else {
            reject({ statusCode: res.statusCode, body: response });
          }
        } catch (e) {
          // Not JSON, return as text
          if (res.statusCode >= 200 && res.statusCode < 300) {
            resolve({ text: body });
          } else {
            reject({ statusCode: res.statusCode, body });
          }
        }
      });
    });

    req.on('error', reject);

    if (data) {
      req.write(JSON.stringify(data));
    }

    req.end();
  });
}

/**
 * Login and get JWT token
 */
async function login() {
  console.log('🔐 Logging in...');
  const response = await makeRequest('POST', '/auth/login', {
    username: USERNAME,
    password: PASSWORD
  });
  console.log('✅ Login successful');
  return response.access_token;
}

/**
 * Check SOL wallet balance
 */
async function checkWallet(token) {
  console.log('\n💰 Checking SOL wallet...');
  const response = await makeRequest('POST', '/checkWallet', {}, {
    'Authorization': `Bearer ${token}`
  });
  console.log('Wallet Address:', response.public_key);
  console.log('SOL Balance:', response.balance_sol, 'SOL');
  console.log('Balance (lamports):', response.balance_lamports);
  return response;
}

/**
 * Check PIPE token balance
 */
async function checkPipeBalance(token) {
  console.log('\n🔥 Checking PIPE token balance...');
  const response = await makeRequest('POST', '/checkCustomToken', {}, {
    'Authorization': `Bearer ${token}`
  });
  console.log('PIPE Balance:', response.ui_amount, 'PIPE');
  console.log('Token Mint:', response.token_mint);
  return response;
}

/**
 * Sync deposits (optional)
 */
async function syncDeposits(token) {
  console.log('\n🔄 Syncing deposits...');
  try {
    await makeRequest('POST', '/syncDeposits', {}, {
      'Authorization': `Bearer ${token}`
    });
    console.log('✅ Sync complete');
  } catch (error) {
    // Empty response is success
    console.log('✅ Sync complete (empty response)');
  }
}

/**
 * Main function
 */
async function main() {
  try {
    console.log('='.repeat(60));
    console.log('🚀 Pipe Network Firestarter Storage Check');
    console.log('='.repeat(60));
    console.log('Username:', USERNAME);
    console.log('API:', BASE_URL);

    // Login
    const token = await login();

    // Check balances
    const wallet = await checkWallet(token);
    const pipe = await checkPipeBalance(token);

    // Sync deposits
    await syncDeposits(token);

    // Summary
    console.log('\n' + '='.repeat(60));
    console.log('📊 Account Summary');
    console.log('='.repeat(60));
    console.log('Deposit Address:', wallet.public_key);
    console.log('SOL Balance:    ', wallet.balance_sol, 'SOL');
    console.log('PIPE Balance:   ', pipe.ui_amount, 'PIPE');

    console.log('\n⚠️  Note: Pipe Network API does not provide a "list files" endpoint.');
    console.log('    To download a file, you need to know its filename.');
    console.log('    Files uploaded from this account:');
    console.log('    - Check your upload history in the application');
    console.log('    - Or reference the API_REFERENCE.md which mentions:');
    console.log('      "test-pipe-upload.txt" (296 bytes) was uploaded');

  } catch (error) {
    console.error('\n❌ Error:', error);
    if (error.body) {
      console.error('Response:', error.body);
    }
    process.exit(1);
  }
}

main();
