//! Encryption module - Compatible with both Jetson and browser
//! Uses deterministic encryption so the same user always gets the same key

use crate::error::{PipeError, Result};
use crate::types::EncryptionMetadata;
use base64::{decode as b64_decode, encode as b64_encode};
use ring::aead::{Aad, LessSafeKey, Nonce, UnboundKey};
use ring::{aead, pbkdf2, rand};

const SALT_LEN: usize = 16;
const NONCE_LEN: usize = 12;
const TAG_LEN: usize = 16;
const ITERATIONS: u32 = 100_000;

/// Crypto engine for encrypting/decrypting files
#[derive(Clone)]
pub struct CryptoEngine {
    iterations: u32,
}

impl CryptoEngine {
    pub fn new() -> Self {
        Self {
            iterations: ITERATIONS,
        }
    }

    /// Generate a deterministic password from user ID
    /// This allows the same user to decrypt their files from any device
    pub fn generate_user_password(user_id: &str) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(user_id.as_bytes());
        hasher.update(b"mmoment-pipe-encryption-2024");
        format!("{:x}", hasher.finalize())
    }

    /// Encrypt data with password
    /// Returns: encrypted_data + metadata
    pub fn encrypt(&self, data: &[u8], password: &str) -> Result<(Vec<u8>, EncryptionMetadata)> {
        // Generate random salt and nonce
        let rng = rand::SystemRandom::new();
        let mut salt = [0u8; SALT_LEN];
        let mut nonce_bytes = [0u8; NONCE_LEN];

        rand::SecureRandom::fill(&rng, &mut salt)
            .map_err(|_| PipeError::Crypto("Failed to generate salt".into()))?;
        rand::SecureRandom::fill(&rng, &mut nonce_bytes)
            .map_err(|_| PipeError::Crypto("Failed to generate nonce".into()))?;

        // Derive key from password
        let mut key = [0u8; 32];
        pbkdf2::derive(
            pbkdf2::PBKDF2_HMAC_SHA256,
            std::num::NonZeroU32::new(self.iterations).unwrap(),
            &salt,
            password.as_bytes(),
            &mut key,
        );

        // Setup AEAD cipher
        let unbound_key = UnboundKey::new(&aead::CHACHA20_POLY1305, &key)
            .map_err(|e| PipeError::Crypto(format!("Key setup failed: {:?}", e)))?;
        let sealing_key = LessSafeKey::new(unbound_key);

        // Encrypt
        let nonce = Nonce::assume_unique_for_key(nonce_bytes);
        let mut encrypted = data.to_vec();

        sealing_key
            .seal_in_place_append_tag(nonce, Aad::empty(), &mut encrypted)
            .map_err(|_| PipeError::Crypto("Encryption failed".into()))?;

        // Create metadata
        let metadata = EncryptionMetadata {
            algorithm: "ChaCha20-Poly1305".to_string(),
            nonce: b64_encode(&nonce_bytes),
            salt: b64_encode(&salt),
            iterations: self.iterations,
        };

        Ok((encrypted, metadata))
    }

    /// Decrypt data with password and metadata
    pub fn decrypt(
        &self,
        encrypted_data: &[u8],
        password: &str,
        metadata: &EncryptionMetadata,
    ) -> Result<Vec<u8>> {
        // Decode metadata
        let salt = b64_decode(&metadata.salt)
            .map_err(|e| PipeError::Crypto(format!("Invalid salt: {}", e)))?;
        let nonce_bytes = b64_decode(&metadata.nonce)
            .map_err(|e| PipeError::Crypto(format!("Invalid nonce: {}", e)))?;

        if nonce_bytes.len() != NONCE_LEN {
            return Err(PipeError::Crypto("Invalid nonce length".into()));
        }

        // Derive key from password
        let mut key = [0u8; 32];
        pbkdf2::derive(
            pbkdf2::PBKDF2_HMAC_SHA256,
            std::num::NonZeroU32::new(metadata.iterations).unwrap(),
            &salt,
            password.as_bytes(),
            &mut key,
        );

        // Setup AEAD cipher
        let unbound_key = UnboundKey::new(&aead::CHACHA20_POLY1305, &key)
            .map_err(|e| PipeError::Crypto(format!("Key setup failed: {:?}", e)))?;
        let opening_key = LessSafeKey::new(unbound_key);

        // Create nonce array
        let mut nonce_array = [0u8; NONCE_LEN];
        nonce_array.copy_from_slice(&nonce_bytes);
        
        // Decrypt
        let nonce = Nonce::assume_unique_for_key(nonce_array);

        let mut decrypted = encrypted_data.to_vec();
        opening_key
            .open_in_place(nonce, Aad::empty(), &mut decrypted)
            .map_err(|_| PipeError::Crypto("Decryption failed - wrong password?".into()))?;

        // Remove auth tag
        decrypted.truncate(decrypted.len().saturating_sub(TAG_LEN));

        Ok(decrypted)
    }

    /// Encrypt data for a specific user (uses deterministic password)
    pub fn encrypt_for_user(
        &self,
        data: &[u8],
        user_id: &str,
    ) -> Result<(Vec<u8>, EncryptionMetadata)> {
        let password = Self::generate_user_password(user_id);
        self.encrypt(data, &password)
    }

    /// Decrypt data for a specific user (uses deterministic password)
    pub fn decrypt_for_user(
        &self,
        encrypted_data: &[u8],
        user_id: &str,
        metadata: &EncryptionMetadata,
    ) -> Result<Vec<u8>> {
        let password = Self::generate_user_password(user_id);
        self.decrypt(encrypted_data, &password, metadata)
    }
}

impl Default for CryptoEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper to combine encrypted data with metadata for storage
pub fn pack_encrypted_file(
    encrypted_data: Vec<u8>,
    metadata: &EncryptionMetadata,
) -> Result<Vec<u8>> {
    // Format: [metadata_len:4][metadata_json][encrypted_data]
    let metadata_json = serde_json::to_vec(metadata)?;
    let metadata_len = metadata_json.len() as u32;

    let mut packed = Vec::with_capacity(4 + metadata_json.len() + encrypted_data.len());
    packed.extend_from_slice(&metadata_len.to_le_bytes());
    packed.extend_from_slice(&metadata_json);
    packed.extend_from_slice(&encrypted_data);

    Ok(packed)
}

/// Helper to extract encrypted data and metadata from packed file
pub fn unpack_encrypted_file(packed_data: &[u8]) -> Result<(Vec<u8>, EncryptionMetadata)> {
    if packed_data.len() < 4 {
        return Err(PipeError::Crypto("Invalid packed data".into()));
    }

    // Read metadata length
    let metadata_len = u32::from_le_bytes(
        packed_data[0..4]
            .try_into()
            .map_err(|_| PipeError::Crypto("Invalid metadata length".into()))?,
    ) as usize;

    if packed_data.len() < 4 + metadata_len {
        return Err(PipeError::Crypto("Truncated packed data".into()));
    }

    // Extract metadata
    let metadata: EncryptionMetadata = serde_json::from_slice(&packed_data[4..4 + metadata_len])?;

    // Extract encrypted data
    let encrypted_data = packed_data[4 + metadata_len..].to_vec();

    Ok((encrypted_data, metadata))
}
