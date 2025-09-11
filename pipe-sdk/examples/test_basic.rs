//! Quick test to verify Pipe SDK works with the actual API

use pipe_api_sdk::{PipeClient, Result};

#[tokio::main]
async fn main() -> Result<()> {
    println!("🧪 Testing Pipe SDK Connection...\n");

    // Initialize client
    let client = PipeClient::new(None);
    println!("✅ Client initialized");

    // Test 1: Create a test user
    let username = format!("mmoment_test_{}", uuid::Uuid::new_v4().simple());
    println!("\n📝 Creating test user: {}", username);
    
    match client.create_user(&username).await {
        Ok(user) => {
            println!("✅ User created successfully!");
            println!("   User ID: {}", user.user_id);
            println!("   App Key: {}...", &user.user_app_key[..8]);
            if let Some(pubkey) = &user.solana_pubkey {
                println!("   Solana: {}", pubkey);
            }

            // Test 2: Check balances
            println!("\n💰 Checking balances...");
            match client.check_sol_balance(&user).await {
                Ok(balance) => {
                    println!("   SOL: {} ({} lamports)", balance.balance_sol, balance.balance_lamports);
                },
                Err(e) => println!("   ⚠️ Could not check SOL: {}", e),
            }

            match client.check_pipe_balance(&user).await {
                Ok(balance) => {
                    println!("   PIPE: {}", balance.ui_amount);
                },
                Err(e) => println!("   ⚠️ Could not check PIPE: {}", e),
            }

            // Test 3: Upload a small test file
            println!("\n📤 Testing file upload...");
            let test_data = b"Hello from MMOMENT Camera Network!";
            let filename = "test_message.txt";
            
            match client.upload(&user, test_data.to_vec(), filename, false).await {
                Ok(uploaded_name) => {
                    println!("✅ Upload successful: {}", uploaded_name);
                    
                    // Test 4: Download it back
                    println!("\n📥 Testing download...");
                    match client.download(&user, &uploaded_name, false).await {
                        Ok(data) => {
                            let content = String::from_utf8_lossy(&data);
                            println!("✅ Downloaded: {}", content);
                        },
                        Err(e) => println!("❌ Download failed: {}", e),
                    }

                    // Test 5: Delete the test file
                    println!("\n🗑️ Cleaning up...");
                    match client.delete_file(&user, &uploaded_name).await {
                        Ok(_) => println!("✅ File deleted"),
                        Err(e) => println!("⚠️ Could not delete: {}", e),
                    }
                },
                Err(e) => println!("❌ Upload failed: {}", e),
            }

        },
        Err(e) => {
            println!("❌ Failed to create user: {}", e);
            println!("\n🔍 Possible issues:");
            println!("   - Check internet connection");
            println!("   - Verify API is accessible");
            println!("   - API might be rate limited");
        }
    }

    println!("\n✨ Test complete!");
    Ok(())
}