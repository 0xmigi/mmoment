//! Pipe API SDK - Built for MMOMENT Camera Network
//!
//! Priorities:
//! 1. Jetson: Multi-user encrypted uploads
//! 2. Browser: Download and decrypt for viewing

pub mod client;
pub mod session;
pub mod storage;
pub mod crypto;
pub mod types;
pub mod error;

pub use client::PipeClient;
pub use session::{SessionManager, UserSession};
pub use storage::StorageClient;
pub use crypto::{CryptoEngine, pack_encrypted_file, unpack_encrypted_file};
pub use types::{User, UploadResult, FileInfo, EncryptionMetadata};
pub use error::{PipeError, Result};

/// Quick start for Jetson/server use
pub async fn create_session_manager() -> Result<SessionManager> {
    SessionManager::new(None).await
}

/// Quick start for browser/client use
pub fn create_download_client() -> PipeClient {
    PipeClient::new(None)
}
