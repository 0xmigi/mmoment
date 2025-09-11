use serde::{Deserialize, Serialize};

/// User account on Pipe Network
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct User {
    pub user_id: String,
    pub user_app_key: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub solana_pubkey: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub username: Option<String>,
}

/// Result of uploading a file
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UploadResult {
    pub filename: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub file_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub encrypted: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size: Option<u64>,
}

/// File information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    pub filename: String,
    pub size: u64,
    pub created_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expires_at: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub public_link: Option<String>,
}

/// Wallet balance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WalletBalance {
    pub user_id: String,
    pub public_key: String,
    pub balance_lamports: u64,
    pub balance_sol: f64,
}

/// Token balance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenBalance {
    pub user_id: String,
    pub public_key: String,
    pub token_mint: String,
    pub amount: String,
    pub ui_amount: f64,
}

/// Upload options
#[derive(Debug, Clone, Default)]
pub struct UploadOptions {
    pub encrypt: bool,
    pub password: Option<String>,
    pub priority: bool,
    pub epochs: Option<u32>,
}

/// Download options
#[derive(Debug, Clone, Default)]
pub struct DownloadOptions {
    pub decrypt: bool,
    pub password: Option<String>,
    pub priority: bool,
}

/// Encryption metadata (stored with encrypted files)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncryptionMetadata {
    pub algorithm: String,
    pub nonce: String, // Base64
    pub salt: String,  // Base64
    pub iterations: u32,
}
