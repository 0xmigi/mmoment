# Pipe SDK (Simple) - CLI Wrapper for Programmatic Access

A lightweight Rust SDK that wraps the Pipe CLI to enable programmatic access to Pipe Network's decentralized storage. Built specifically for multi-user applications like camera networks, IoT devices, and dApps.

## 🎯 Why This SDK?

While the Pipe CLI is excellent for manual operations, modern applications need:
- **Programmatic control** - Call Pipe from your code, not shell commands
- **Multi-user sessions** - Manage thousands of user accounts concurrently
- **Deterministic accounts** - Derive accounts from existing identities (wallet addresses)
- **Automatic management** - Handle account creation, funding, and token swaps

## 🚀 Key Features

### Multi-User Session Management
Each user gets their own isolated Pipe account, perfect for:
- Camera networks where users own their captured media
- IoT devices serving multiple users
- dApps with user-controlled storage

### Simple API
```rust
// Initialize
let manager = PipeSessionManager::new(None);

// Auto-create account for user (using their wallet as ID)
let account = manager.get_or_create_account(wallet_address).await?;

// Upload user's data
let file_id = manager.upload_bytes_for_user(
    wallet_address,
    photo_bytes,
    "photo.jpg",
    true  // encrypt
).await?;
```

### Real-World Use Case: MMOMENT Camera Network

MMOMENT uses this SDK to give users true ownership of their camera captures:

1. User checks in with Solana wallet → Camera creates Pipe account
2. Camera captures photo → Uploads to user's Pipe storage
3. User owns their media → Can access from anywhere with their credentials

## 📦 Installation

```toml
[dependencies]
pipe-sdk-simple = { path = "../pipe-sdk-simple" }
```

## 🔧 Prerequisites

- Pipe CLI installed (`pipe --version`)
- Rust 1.70+

## 💡 Usage Example

```rust
use pipe_sdk_simple::PipeSessionManager;

#[tokio::main]
async fn main() -> Result<()> {
    let manager = PipeSessionManager::new(None);
    
    // When user interacts with your device/app
    let user_id = "user_wallet_or_identifier";
    
    // Upload their data (account created automatically if needed)
    let file_id = manager.upload_bytes_for_user(
        user_id,
        b"user data",
        "data.bin",
        true  // encrypt for privacy
    ).await?;
    
    println!("Stored to user's Pipe account: {}", file_id);
    Ok(())
}
```

## 🏗️ Architecture

```
Your App
    ↓
Pipe SDK (this library)
    ↓
Pipe CLI (subprocess calls)
    ↓
Pipe Network API
    ↓
Decentralized Storage
```

The SDK manages:
- Config files per user
- Password generation (deterministic from user ID)
- CLI command construction
- Output parsing
- Error handling

## 🔐 Security

- Each user gets unique credentials
- Passwords derived from user ID (deterministic but secure)
- Config files stored separately per user
- Optional encryption for all uploads

## 🤝 Contributing

This SDK is designed as a community contribution to the Pipe ecosystem. We built it for our camera network but believe it can help many other projects.

### Future Enhancements
- [ ] Direct API calls (skip CLI)
- [ ] Wallet signature authentication
- [ ] Python/JS bindings
- [ ] Batch operations
- [ ] Progress callbacks

## 📊 Production Stats

Currently managing:
- X active user accounts
- Y daily uploads
- Z GB stored

## 🙏 Credits

Built with ❤️ for the Pipe Network community. Special thanks to the Pipe team for building the infrastructure that makes user-owned storage possible.

## 📄 License

MIT - Use freely in your projects!

---

**Note to Pipe Team**: We built this because we needed programmatic access for our camera network. Happy to contribute this back to the community or help integrate similar functionality into the official CLI. The wrapper approach works well for us but we're open to collaborating on a more integrated solution!