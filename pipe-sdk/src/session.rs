//! Multi-user session management for camera networks

use crate::client::PipeClient;
use crate::storage::StorageClient;
use crate::error::{PipeError, Result};
use crate::types::{User, UploadResult};
use dashmap::DashMap;
use parking_lot::RwLock;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Session for a single user
#[derive(Clone)]
pub struct UserSession {
    pub user: User,
    pub storage: Arc<StorageClient>,
    pub created_at: Instant,
    pub last_activity: Arc<RwLock<Instant>>,
    metadata: Arc<RwLock<serde_json::Value>>,
}

impl UserSession {
    fn new(user: User, storage: Arc<StorageClient>) -> Self {
        Self {
            user,
            storage,
            created_at: Instant::now(),
            last_activity: Arc::new(RwLock::new(Instant::now())),
            metadata: Arc::new(RwLock::new(serde_json::json!({}))),
        }
    }

    /// Update last activity time
    pub fn touch(&self) {
        *self.last_activity.write() = Instant::now();
    }

    /// Get session age
    pub fn age(&self) -> Duration {
        self.created_at.elapsed()
    }

    /// Get time since last activity
    pub fn idle_time(&self) -> Duration {
        self.last_activity.read().elapsed()
    }

    /// Store custom metadata
    pub fn set_metadata(&self, key: &str, value: serde_json::Value) {
        self.metadata.write()[key] = value;
    }

    /// Get custom metadata
    pub fn get_metadata(&self, key: &str) -> Option<serde_json::Value> {
        self.metadata.read().get(key).cloned()
    }
}

/// Manages multiple user sessions for camera network
pub struct SessionManager {
    client: Arc<PipeClient>,
    sessions: Arc<DashMap<String, UserSession>>,
    max_idle_time: Duration,
}

impl SessionManager {
    /// Create a new session manager
    pub async fn new(base_url: Option<&str>) -> Result<Self> {
        let client = Arc::new(PipeClient::new(base_url));

        Ok(Self {
            client,
            sessions: Arc::new(DashMap::new()),
            max_idle_time: Duration::from_secs(3600),  // 1 hour
        })
    }

    /// Set maximum idle time before session cleanup
    pub fn set_max_idle_time(&mut self, duration: Duration) {
        self.max_idle_time = duration;
    }

    /// Get or create a session for a user (identified by wallet address)
    pub async fn get_or_create_session(&self, user_id: &str) -> Result<UserSession> {
        // Check if session exists
        if let Some(session) = self.sessions.get(user_id) {
            session.touch();
            return Ok(session.clone());
        }

        // Create new session
        self.create_session(user_id).await
    }

    /// Create a new session
    async fn create_session(&self, user_id: &str) -> Result<UserSession> {
        // Generate username from user_id (for deterministic account creation)
        let username = format!("mmoment_{}", &user_id[..16.min(user_id.len())]);

        // Try to create user (might already exist)
        let user = match self.client.create_user(&username).await {
            Ok(user) => user,
            Err(PipeError::Api { message, .. }) if message.contains("exists") => {
                // User already exists, we need to get their credentials
                // For now, we'll create a placeholder (in production, store credentials)
                User {
                    user_id: username.clone(),
                    user_app_key: String::new(),  // Would be retrieved from storage
                    solana_pubkey: None,
                    username: Some(username),
                }
            }
            Err(e) => return Err(e),
        };

        // Create storage client for this user
        let storage = Arc::new(StorageClient::new(self.client.clone()));

        // Create session
        let session = UserSession::new(user, storage);

        // Store in map
        self.sessions.insert(user_id.to_string(), session.clone());

        Ok(session)
    }

    /// Upload file for a specific user
    pub async fn upload_for_user(
        &self,
        user_id: &str,
        data: Vec<u8>,
        filename: &str,
        encrypt: bool,
    ) -> Result<UploadResult> {
        let session = self.get_or_create_session(user_id).await?;
        session.touch();

        let options = crate::types::UploadOptions {
            encrypt,
            password: None,  // Use deterministic password
            priority: false,
            epochs: None,
        };

        session.storage.upload(&session.user, data, filename, options).await
    }

    /// Upload camera capture for user
    pub async fn upload_camera_capture(
        &self,
        user_id: &str,
        image_data: Vec<u8>,
        capture_type: &str,
    ) -> Result<UploadResult> {
        let session = self.get_or_create_session(user_id).await?;
        session.touch();

        session.storage
            .upload_camera_capture(&session.user, image_data, capture_type)
            .await
    }

    /// Download file for a specific user
    pub async fn download_for_user(
        &self,
        user_id: &str,
        filename: &str,
        decrypt: bool,
    ) -> Result<Vec<u8>> {
        let session = self.get_or_create_session(user_id).await?;
        session.touch();

        let options = crate::types::DownloadOptions {
            decrypt,
            password: None,
            priority: false,
        };

        session.storage.download(&session.user, filename, options).await
    }

    /// Get balance for a user
    pub async fn get_user_balance(&self, user_id: &str) -> Result<(f64, f64)> {
        let session = self.get_or_create_session(user_id).await?;
        session.touch();

        let sol = self.client.check_sol_balance(&session.user).await?;
        let pipe = self.client.check_pipe_balance(&session.user).await?;

        Ok((sol.balance_sol, pipe.ui_amount))
    }

    /// Fund user account (transfer SOL and swap to PIPE)
    pub async fn fund_user(&self, user_id: &str, sol_amount: f64) -> Result<()> {
        let session = self.get_or_create_session(user_id).await?;
        session.touch();

        // Get user's Pipe wallet address
        let wallet = self.client.check_sol_balance(&session.user).await?;

        println!("To fund user {}, send {} SOL to: {}",
                 user_id, sol_amount, wallet.public_key);

        // In production, integrate with Solana to actually transfer
        // For now, just attempt the swap (will fail if no SOL)

        // Swap 90% to PIPE, keep 10% for fees
        let swap_amount = sol_amount * 0.9;
        self.client.swap_sol_for_pipe(&session.user, swap_amount).await?;

        Ok(())
    }

    /// Clean up inactive sessions
    pub fn cleanup_inactive(&self) {
        let now = Instant::now();
        self.sessions.retain(|_, session| {
            session.idle_time() < self.max_idle_time
        });
    }

    /// Get number of active sessions
    pub fn active_sessions(&self) -> usize {
        self.sessions.len()
    }

    /// Get session info for monitoring
    pub fn get_session_info(&self, user_id: &str) -> Option<serde_json::Value> {
        self.sessions.get(user_id).map(|session| {
            serde_json::json!({
                "user_id": session.user.user_id,
                "created_at": session.created_at.elapsed().as_secs(),
                "last_activity": session.idle_time().as_secs(),
                "metadata": *session.metadata.read(),
            })
        })
    }
}

/// Batch operations for multiple users
impl SessionManager {
    /// Upload files for multiple users in parallel
    pub async fn batch_upload(
        &self,
        operations: Vec<(String, Vec<u8>, String)>,  // (user_id, data, filename)
    ) -> Vec<Result<UploadResult>> {
        use futures::future::join_all;

        let futures = operations.into_iter().map(|(user_id, data, filename)| {
            async move {
                self.upload_for_user(&user_id, data, &filename, true).await
            }
        });

        join_all(futures).await
    }

    /// Process camera captures for multiple users
    pub async fn process_camera_batch(
        &self,
        captures: Vec<(String, Vec<u8>)>,  // (user_id, image_data)
    ) -> Vec<Result<UploadResult>> {
        use futures::future::join_all;

        let futures = captures.into_iter().map(|(user_id, image_data)| {
            async move {
                self.upload_camera_capture(&user_id, image_data, "photo").await
            }
        });

        join_all(futures).await
    }
}
