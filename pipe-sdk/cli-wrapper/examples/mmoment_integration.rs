use pipe_sdk_simple::{PipeSessionManager, Result};
use std::path::PathBuf;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🎥 MMOMENT Camera Network + Pipe Storage Integration Example\n");

    // Initialize the SDK with a custom config directory
    let config_dir = PathBuf::from("/opt/mmoment/pipe-configs");
    let manager = PipeSessionManager::new(Some(config_dir));

    // Simulate a user checking in with their Solana wallet
    let user_wallet = "7x8y9zABCDEF1234567890"; // User's Solana pubkey

    println!("👤 User checks in with wallet: {}", user_wallet);

    // Step 1: Get or create Pipe account for this user
    let account = manager.get_or_create_account(user_wallet).await?;
    println!("✅ Pipe account ready: {}", account.user_id);

    // Step 2: Check user's balance
    let (sol, pipe) = manager.check_balance(user_wallet).await?;
    println!("💰 Balance: {} SOL, {} PIPE tokens", sol, pipe);

    // Step 3: If new user (no PIPE tokens), show funding instructions
    if pipe == 0.0 {
        println!("\n⚠️  New user detected - needs funding!");
        manager.fund_account(user_wallet, 0.1).await?;
        println!("Instructions sent to fund account with 0.1 SOL from devnet");
    }

    // Step 4: Camera captures photo
    println!("\n📸 Camera capturing photo...");
    let photo_data = b"[Simulated photo bytes from camera]";

    // Step 5: Upload to user's Pipe storage (encrypted for privacy)
    let file_id = manager
        .upload_bytes_for_user(
            user_wallet,
            photo_data,
            "capture_2024_01_01_120000.jpg",
            true, // encrypt
        )
        .await?;

    println!("✅ Photo uploaded to user's Pipe storage!");
    println!("   File ID: {}", file_id);
    println!("   Owner: {}", user_wallet);
    println!("   Encrypted: Yes");

    // Step 6: Show how multiple users work
    println!("\n🔄 Handling multiple users concurrently...");

    let users = vec![
        "UserWallet111111111",
        "UserWallet222222222",
        "UserWallet333333333",
    ];

    for user in users {
        let account = manager.get_or_create_account(user).await?;
        println!("  ✓ Session created for {}: {}", user, account.user_id);
    }

    println!("\n✨ Integration complete!");
    println!("Each user owns their media through their Pipe account");
    println!("Camera just orchestrates - no central storage needed!");

    Ok(())
}

// Example output:
// 🎥 MMOMENT Camera Network + Pipe Storage Integration Example
//
// 👤 User checks in with wallet: 7x8y9zABCDEF1234567890
// ✅ Pipe account ready: abc123-def456
// 💰 Balance: 0 SOL, 0 PIPE tokens
//
// ⚠️  New user detected - needs funding!
// Instructions sent to fund account with 0.1 SOL from devnet
//
// 📸 Camera capturing photo...
// ✅ Photo uploaded to user's Pipe storage!
//    File ID: upload_12345678-abcd-efgh
//    Owner: 7x8y9zABCDEF1234567890
//    Encrypted: Yes
//
// 🔄 Handling multiple users concurrently...
//   ✓ Session created for UserWallet111111111: xyz789
//   ✓ Session created for UserWallet222222222: mno456
//   ✓ Session created for UserWallet333333333: pqr123
//
// ✨ Integration complete!
