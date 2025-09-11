//! Storage operations with automatic encryption

use crate::client::PipeClient;
use crate::crypto::{CryptoEngine, pack_encrypted_file, unpack_encrypted_file};
use crate::error::Result;
use crate::types::{User, UploadOptions, DownloadOptions, UploadResult};
use std::sync::Arc;

/// Storage client with encryption support
pub struct StorageClient {
    client: Arc<PipeClient>,
    crypto: CryptoEngine,
}

impl StorageClient {
    pub fn new(client: Arc<PipeClient>) -> Self {
        Self {
            client,
            crypto: CryptoEngine::new(),
        }
    }

    /// Upload a file with optional encryption
    pub async fn upload(
        &self,
        user: &User,
        data: Vec<u8>,
        filename: &str,
        options: UploadOptions,
    ) -> Result<UploadResult> {
        let (upload_data, encrypted_filename) = if options.encrypt {
            // Encrypt the data
            let password = options.password.as_deref()
                .unwrap_or(&CryptoEngine::generate_user_password(&user.user_id));

            let (encrypted_data, metadata) = self.crypto.encrypt(&data, password)?;
            let packed = pack_encrypted_file(encrypted_data, &metadata)?;

            // Add .enc extension to indicate encryption
            let enc_filename = format!("{}.enc", filename);
            (packed, enc_filename)
        } else {
            (data, filename.to_string())
        };

        // Upload to Pipe
        let uploaded_name = self.client
            .upload(user, upload_data, &encrypted_filename, options.priority)
            .await?;

        Ok(UploadResult {
            filename: uploaded_name.clone(),
            file_id: Some(uploaded_name),
            encrypted: Some(options.encrypt),
            size: Some(data.len() as u64),
        })
    }

    /// Download a file with automatic decryption
    pub async fn download(
        &self,
        user: &User,
        filename: &str,
        options: DownloadOptions,
    ) -> Result<Vec<u8>> {
        // Download from Pipe
        let data = self.client
            .download(user, filename, options.priority)
            .await?;

        // Check if file needs decryption (ends with .enc or decrypt requested)
        if options.decrypt || filename.ends_with(".enc") {
            let password = options.password.as_deref()
                .unwrap_or(&CryptoEngine::generate_user_password(&user.user_id));

            // Unpack and decrypt
            let (encrypted_data, metadata) = unpack_encrypted_file(&data)?;
            self.crypto.decrypt(&encrypted_data, password, &metadata)
        } else {
            Ok(data)
        }
    }

    /// Upload media for MMOMENT camera (always encrypted)
    pub async fn upload_camera_capture(
        &self,
        user: &User,
        image_data: Vec<u8>,
        capture_type: &str,
    ) -> Result<UploadResult> {
        use chrono::Utc;

        // Generate filename with timestamp
        let timestamp = Utc::now().format("%Y%m%d_%H%M%S");
        let filename = format!("mmoment_{}_{}.jpg", capture_type, timestamp);

        // Always encrypt camera captures for privacy
        let options = UploadOptions {
            encrypt: true,
            password: None,  // Use deterministic password from user_id
            priority: capture_type == "video",  // Videos get priority
            epochs: None,
        };

        self.upload(user, image_data, &filename, options).await
    }

    /// Create a shareable link (works even for encrypted files)
    pub async fn create_shareable_link(
        &self,
        user: &User,
        filename: &str,
        include_password: bool,
    ) -> Result<String> {
        let public_link = self.client.create_public_link(user, filename).await?;

        if include_password && filename.ends_with(".enc") {
            // Append the password as a URL fragment (never sent to server)
            let password = CryptoEngine::generate_user_password(&user.user_id);
            Ok(format!("{}#{}", public_link, password))
        } else {
            Ok(public_link)
        }
    }

    /// Batch upload multiple files
    pub async fn batch_upload(
        &self,
        user: &User,
        files: Vec<(Vec<u8>, String)>,
        encrypt: bool,
    ) -> Result<Vec<UploadResult>> {
        let mut results = Vec::new();

        for (data, filename) in files {
            let options = UploadOptions {
                encrypt,
                password: None,
                priority: false,
                epochs: None,
            };

            let result = self.upload(user, data, &filename, options).await?;
            results.push(result);
        }

        Ok(results)
    }
}
