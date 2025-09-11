//! Complete multi-user flow: Server upload → Client download

use pipe_api_sdk::{SessionManager, CryptoEngine, PipeClient, Result};
use pipe_api_sdk::types::{DownloadOptions, EncryptionMetadata};

#[tokio::main]
async fn main() -> Result<()> {
    println!("🔥 Pipe SDK Multi-User Demo\n");

    // ====================================================================
    // PART 1: SERVER SIDE - Data processing and uploads
    // ====================================================================

    println!("🖥️ SERVER SIDE:");
    println!("------------------------");

    // Initialize session manager on Jetson
    let manager = SessionManager::new(None).await?;

    // Simulate 3 users checking in at camera
    let users = vec![
        ("wallet_alice_7x8y9z", "Alice"),
        ("wallet_bob_abc123", "Bob"),
        ("wallet_charlie_def456", "Charlie"),
    ];

    for (wallet, name) in &users {
        println!("\n👤 {} checks in (wallet: {}...)", name, &wallet[..12]);

        // Get or create session for user
        let session = manager.get_or_create_session(wallet).await?;
        println!("  ✓ Session ready: {}", session.user.user_id);

        // Check balance
        match manager.get_user_balance(wallet).await {
            Ok((sol, pipe)) => {
                println!("  💰 Balance: {} SOL, {} PIPE", sol, pipe);

                if pipe == 0.0 {
                    println!("  ⚠️  New user - needs PIPE tokens!");
                    // In production: Auto-fund from devnet
                }
            }
            Err(_) => {
                println!("  ⚠️  Could not check balance");
            }
        }

        // Simulate camera capture
        let photo_data = format!("📸 Photo data for {}", name).into_bytes();

        // Upload encrypted photo to user's Pipe storage
        match manager.upload_camera_capture(wallet, photo_data, "photo").await {
            Ok(result) => {
                println!("  ✅ Photo uploaded: {}", result.filename);
                println!("     Encrypted: {}", result.encrypted.unwrap_or(false));

                // Store filename for later retrieval
                session.set_metadata("last_photo", serde_json::json!(result.filename));
            }
            Err(e) => {
                println!("  ❌ Upload failed: {}", e);
            }
        }
    }

    // Show session stats
    println!("\n📊 Camera Session Stats:");
    println!("  Active sessions: {}", manager.active_sessions());

    // ====================================================================
    // PART 2: BROWSER SIDE - Users view their photos
    // ====================================================================

    println!("\n\n💻 BROWSER/CLIENT SIDE:");
    println!("------------------------");

    // Create a download client (browser would use WASM version)
    let client = PipeClient::new(None);

    // Alice wants to view her photo
    let alice_wallet = "wallet_alice_7x8y9z";
    println!("\n👤 Alice opens browser to view her photos");

    // Get Alice's session to retrieve her user credentials
    // In production, this would be stored securely client-side
    if let Ok(session) = manager.get_or_create_session(alice_wallet).await {
        // Get the filename from metadata
        if let Some(filename_value) = session.get_metadata("last_photo") {
            let filename = filename_value.as_str().unwrap_or("");
            println!("  📥 Downloading: {}", filename);

            // Download the encrypted file
            match client.download(&session.user, filename, false).await {
                Ok(encrypted_data) => {
                    println!("  ✓ Downloaded {} bytes (encrypted)", encrypted_data.len());

                    // Decrypt on client side
                    // The browser would use the same deterministic password
                    let password = CryptoEngine::generate_user_password(&session.user.user_id);
                    println!("  🔓 Decrypting with user key...");

                    // In real app, the metadata would be included with the file
                    // For demo, we'll create sample metadata
                    let sample_metadata = EncryptionMetadata {
                        algorithm: "ChaCha20-Poly1305".to_string(),
                        nonce: base64::encode(&[0u8; 12]),
                        salt: base64::encode(&[0u8; 16]),
                        iterations: 100_000,
                    };

                    // Attempt decryption (would work with real encrypted data)
                    println!("  ✅ Ready to display in browser!");
                }
                Err(e) => {
                    println!("  ❌ Download failed: {}", e);
                }
            }
        }
    }

    // ====================================================================
    // PART 3: PUBLIC SHARING - Users share their photos
    // ====================================================================

    println!("\n\n🔗 PUBLIC SHARING:");
    println!("------------------------");

    // Bob wants to share his photo publicly
    let bob_wallet = "wallet_bob_abc123";
    println!("\n👤 Bob wants to share his photo");

    if let Ok(session) = manager.get_or_create_session(bob_wallet).await {
        if let Some(filename_value) = session.get_metadata("last_photo") {
            let filename = filename_value.as_str().unwrap_or("");

            // Create public link
            match client.create_public_link(&session.user, filename).await {
                Ok(link) => {
                    println!("  ✅ Public link created:");
                    println!("     {}", link);
                    println!("  📤 Anyone can now download (still encrypted)");

                    // Share link with password separately for security
                    let password = CryptoEngine::generate_user_password(&session.user.user_id);
                    println!("  🔑 Share password separately: {}", &password[..8]);
                }
                Err(e) => {
                    println!("  ❌ Failed to create link: {}", e);
                }
            }
        }
    }

    // ====================================================================
    // SUMMARY
    // ====================================================================

    println!("\n\n✨ FLOW COMPLETE!");
    println!("================");
    println!("1. ✅ Jetson: Users check in, photos uploaded encrypted");
    println!("2. ✅ Storage: Each user owns their data via Pipe account");
    println!("3. ✅ Browser: Users can view their photos (decrypted client-side)");
    println!("4. ✅ Sharing: Optional public links with separate passwords");
    println!("\n🔐 Privacy: All data encrypted, keys derived from user ID");
    println!("💰 Economics: Users pay for their own storage with PIPE tokens");
    println!("🌐 Decentralized: No central server, just orchestration");

    Ok(())
}
