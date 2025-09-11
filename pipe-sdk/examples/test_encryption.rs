//! Test encryption/decryption functionality without API calls

use pipe_api_sdk::{CryptoEngine, Result, pack_encrypted_file, unpack_encrypted_file};

#[tokio::main]
async fn main() -> Result<()> {
    println!("🔐 Testing Pipe SDK Encryption/Decryption\n");

    // Initialize crypto engine
    let crypto = CryptoEngine::new();
    
    // Test data
    let original_data = b"Hello from MMOMENT Camera! This is a test photo.";
    let user_id = "test_user_wallet_123";
    
    println!("📝 Original data: {} bytes", original_data.len());
    println!("   Content: {}", String::from_utf8_lossy(original_data));
    
    // Generate deterministic password from user ID
    let password = CryptoEngine::generate_user_password(user_id);
    println!("\n🔑 Generated password from user ID: {}...", &password[..16]);
    
    // Encrypt
    println!("\n🔒 Encrypting...");
    let (encrypted_data, metadata) = crypto.encrypt(original_data, &password)?;
    println!("   Encrypted size: {} bytes", encrypted_data.len());
    println!("   Metadata: {:?}", metadata);
    
    // Pack into file format
    let packed = pack_encrypted_file(encrypted_data.clone(), &metadata)?;
    println!("   Packed file size: {} bytes", packed.len());
    
    // Simulate file storage/retrieval
    println!("\n📦 Simulating file storage and retrieval...");
    
    // Unpack
    let (unpacked_data, unpacked_metadata) = unpack_encrypted_file(&packed)?;
    println!("   Unpacked successfully");
    assert_eq!(encrypted_data, unpacked_data);
    
    // Decrypt
    println!("\n🔓 Decrypting...");
    let decrypted = crypto.decrypt(&unpacked_data, &password, &unpacked_metadata)?;
    println!("   Decrypted size: {} bytes", decrypted.len());
    println!("   Content: {}", String::from_utf8_lossy(&decrypted));
    
    // Verify
    if original_data == decrypted.as_slice() {
        println!("\n✅ SUCCESS: Encryption/decryption works correctly!");
    } else {
        println!("\n❌ FAILED: Data doesn't match!");
    }
    
    // Test wrong password
    println!("\n🧪 Testing wrong password...");
    match crypto.decrypt(&unpacked_data, "wrong_password", &unpacked_metadata) {
        Ok(_) => println!("   ❌ Should have failed with wrong password!"),
        Err(e) => println!("   ✅ Correctly rejected: {}", e),
    }
    
    println!("\n✨ Encryption tests complete!");
    Ok(())
}