//! Test with real Pipe API using existing credentials

use pipe_api_sdk::{PipeClient, Result};
use pipe_api_sdk::types::User;
use std::fs;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🔥 Testing Pipe SDK with Real API\n");

    // Read existing credentials
    let creds_path = dirs::home_dir()
        .unwrap()
        .join(".pipe-cli.json");

    if !creds_path.exists() {
        println!("❌ No Pipe credentials found at ~/.pipe-cli.json");
        println!("   Run: pipe new-user your_username");
        return Ok(());
    }

    let creds_content = fs::read_to_string(&creds_path).map_err(|e| pipe_api_sdk::PipeError::Crypto(e.to_string()))?;
    let creds: serde_json::Value = serde_json::from_str(&creds_content)?;

    let user = User {
        user_id: creds["user_id"].as_str().unwrap().to_string(),
        user_app_key: creds["user_app_key"].as_str().unwrap().to_string(),
        solana_pubkey: None,
        username: None,
    };

    println!("👤 Using existing user: {}...", &user.user_id[..8]);

    // Initialize client
    let client = PipeClient::new(None);

    // Test 1: Check SOL balance
    println!("\n💰 Checking SOL balance...");
    match client.check_sol_balance(&user).await {
        Ok(balance) => {
            println!("   ✅ SOL: {} ({} lamports)", balance.balance_sol, balance.balance_lamports);
            println!("   Wallet: {}", balance.public_key);
        },
        Err(e) => println!("   ❌ Failed: {}", e),
    }

    // Test 2: Check PIPE balance
    println!("\n🪙 Checking PIPE balance...");
    match client.check_pipe_balance(&user).await {
        Ok(balance) => {
            println!("   ✅ PIPE: {}", balance.ui_amount);
            println!("   Raw: {}", balance.amount);
        },
        Err(e) => println!("   ❌ Failed: {}", e),
    }

    // Test 3: Upload a test file
    println!("\n📤 Testing file upload...");
    let test_data = format!("Hello from MMOMENT! Timestamp: {}", chrono::Utc::now()).into_bytes();
    let filename = format!("mmoment_test_{}.txt", chrono::Utc::now().timestamp());

    match client.upload(&user, test_data, &filename, false).await {
        Ok(uploaded_name) => {
            println!("   ✅ Upload successful: {}", uploaded_name);
            
            // Test 4: Download it back
            println!("\n📥 Testing download...");
            match client.download(&user, &uploaded_name, false).await {
                Ok(data) => {
                    let content = String::from_utf8_lossy(&data);
                    println!("   ✅ Downloaded: {}", content);
                    
                    // Test 5: Delete it
                    println!("\n🗑️ Cleaning up...");
                    match client.delete_file(&user, &uploaded_name).await {
                        Ok(_) => println!("   ✅ File deleted"),
                        Err(e) => println!("   ⚠️ Could not delete: {}", e),
                    }
                },
                Err(e) => println!("   ❌ Download failed: {}", e),
            }
        },
        Err(e) => println!("   ❌ Upload failed: {}", e),
    }

    println!("\n✨ API tests complete!");
    Ok(())
}