//! Simple Pipe SDK - Wraps the Pipe CLI for programmatic access
//! Perfect for MMOMENT camera network integration

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;
use thiserror::Error;
use tokio::sync::RwLock;

#[derive(Error, Debug)]
pub enum PipeError {
    #[error("Command failed: {0}")]
    CommandFailed(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Account not found")]
    AccountNotFound,

    #[error("Insufficient tokens")]
    InsufficientTokens,

    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, PipeError>;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipeAccount {
    pub user_id: String,
    pub user_app_key: String,
    pub config_path: PathBuf,
}

/// Multi-user session manager for camera networks
pub struct PipeSessionManager {
    base_config_dir: PathBuf,
    sessions: Arc<RwLock<HashMap<String, PipeAccount>>>,
}

impl PipeSessionManager {
    pub fn new(base_config_dir: Option<PathBuf>) -> Self {
        let base_config_dir =
            base_config_dir.unwrap_or_else(|| PathBuf::from("/tmp/pipe-sdk-configs"));

        std::fs::create_dir_all(&base_config_dir).ok();

        Self {
            base_config_dir,
            sessions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get or create a Pipe account for a user (identified by wallet address)
    pub async fn get_or_create_account(&self, user_id: &str) -> Result<PipeAccount> {
        // Check if we already have this user
        {
            let sessions = self.sessions.read().await;
            if let Some(account) = sessions.get(user_id) {
                return Ok(account.clone());
            }
        }

        // Create new account
        let username = format!("user_{}", &user_id[..16.min(user_id.len())]);
        let config_path = self.base_config_dir.join(format!("{}.json", username));

        // Check if config already exists
        if config_path.exists() {
            let content = std::fs::read_to_string(&config_path)?;
            let config: serde_json::Value = serde_json::from_str(&content)?;

            let account = PipeAccount {
                user_id: config["user_id"].as_str().unwrap_or("").to_string(),
                user_app_key: config["user_app_key"].as_str().unwrap_or("").to_string(),
                config_path: config_path.clone(),
            };

            let mut sessions = self.sessions.write().await;
            sessions.insert(user_id.to_string(), account.clone());
            return Ok(account);
        }

        // Create new account via CLI
        let password = self.generate_password(user_id);
        let output = Command::new("pipe")
            .args(&[
                "new-user",
                &username,
                "--config",
                config_path.to_str().unwrap(),
            ])
            .env("PIPE_PASSWORD", &password)
            .output()?;

        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);

            // Check if user already exists
            if error.contains("already exists") {
                // Try to login instead
                return self.login_account(user_id, &username, &password).await;
            }

            return Err(PipeError::CommandFailed(error.to_string()));
        }

        // Read the created config
        let content = std::fs::read_to_string(&config_path)?;
        let config: serde_json::Value = serde_json::from_str(&content)?;

        let account = PipeAccount {
            user_id: config["user_id"].as_str().unwrap_or("").to_string(),
            user_app_key: config["user_app_key"].as_str().unwrap_or("").to_string(),
            config_path: config_path.clone(),
        };

        let mut sessions = self.sessions.write().await;
        sessions.insert(user_id.to_string(), account.clone());
        Ok(account)
    }

    /// Upload a file for a specific user
    pub async fn upload_for_user(
        &self,
        user_id: &str,
        file_path: &Path,
        filename: &str,
        encrypt: bool,
    ) -> Result<String> {
        let account = self.get_or_create_account(user_id).await?;

        let mut command = Command::new("pipe");
        command.args(&[
            "upload-file",
            file_path.to_str().unwrap(),
            filename,
            "--config",
            account.config_path.to_str().unwrap(),
        ]);

        if encrypt {
            command.arg("--encrypt");
            let password = self.generate_password(user_id);
            command.args(&["--password", &password]);
        }

        let output = command.output()?;

        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            if error.contains("Insufficient tokens") {
                return Err(PipeError::InsufficientTokens);
            }
            return Err(PipeError::CommandFailed(error.to_string()));
        }

        // Parse file ID from output
        let stdout = String::from_utf8_lossy(&output.stdout);
        self.parse_file_id(&stdout)
    }

    /// Upload raw bytes for a user
    pub async fn upload_bytes_for_user(
        &self,
        user_id: &str,
        data: &[u8],
        filename: &str,
        encrypt: bool,
    ) -> Result<String> {
        // Write to temp file
        let temp_path = self
            .base_config_dir
            .join(format!("temp_{}.tmp", uuid::Uuid::new_v4()));
        std::fs::write(&temp_path, data)?;

        let result = self
            .upload_for_user(user_id, &temp_path, filename, encrypt)
            .await;

        // Clean up temp file
        std::fs::remove_file(temp_path).ok();

        result
    }

    /// Check user's balance
    pub async fn check_balance(&self, user_id: &str) -> Result<(f64, f64)> {
        let account = self.get_or_create_account(user_id).await?;

        // Check SOL balance
        let output = Command::new("pipe")
            .args(&[
                "check-sol",
                "--config",
                account.config_path.to_str().unwrap(),
            ])
            .output()?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let sol_balance = self.parse_sol_balance(&stdout);

        // Check PIPE balance
        let output = Command::new("pipe")
            .args(&[
                "check-token",
                "--config",
                account.config_path.to_str().unwrap(),
            ])
            .output()?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let pipe_balance = self.parse_pipe_balance(&stdout);

        Ok((sol_balance, pipe_balance))
    }

    /// Auto-fund account with SOL and swap to PIPE
    pub async fn fund_account(&self, user_id: &str, sol_amount: f64) -> Result<()> {
        let account = self.get_or_create_account(user_id).await?;

        // First get the Pipe wallet address
        let output = Command::new("pipe")
            .args(&[
                "check-sol",
                "--config",
                account.config_path.to_str().unwrap(),
            ])
            .output()?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let wallet_address = self.parse_wallet_address(&stdout);

        println!(
            "Fund this wallet with {} SOL: {}",
            sol_amount, wallet_address
        );

        // TODO: Integrate with your Solana middleware to actually transfer SOL

        // After funding, swap to PIPE
        let swap_amount = sol_amount * 0.9; // Keep 10% as SOL for fees

        let output = Command::new("pipe")
            .args(&[
                "swap-sol-for-pipe",
                &swap_amount.to_string(),
                "--config",
                account.config_path.to_str().unwrap(),
            ])
            .output()?;

        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            return Err(PipeError::CommandFailed(error.to_string()));
        }

        Ok(())
    }

    // Helper methods

    fn generate_password(&self, seed: &str) -> String {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(seed.as_bytes());
        hasher.update(b"pipe-sdk-password");
        format!("{:x}", hasher.finalize())
    }

    async fn login_account(
        &self,
        user_id: &str,
        username: &str,
        password: &str,
    ) -> Result<PipeAccount> {
        let config_path = self.base_config_dir.join(format!("{}.json", username));

        let output = Command::new("pipe")
            .args(&["login", username, "--config", config_path.to_str().unwrap()])
            .env("PIPE_PASSWORD", password)
            .output()?;

        if !output.status.success() {
            let error = String::from_utf8_lossy(&output.stderr);
            return Err(PipeError::CommandFailed(error.to_string()));
        }

        // Read the config
        let content = std::fs::read_to_string(&config_path)?;
        let config: serde_json::Value = serde_json::from_str(&content)?;

        let account = PipeAccount {
            user_id: config["user_id"].as_str().unwrap_or("").to_string(),
            user_app_key: config["user_app_key"].as_str().unwrap_or("").to_string(),
            config_path: config_path.clone(),
        };

        let mut sessions = self.sessions.write().await;
        sessions.insert(user_id.to_string(), account.clone());
        Ok(account)
    }

    fn parse_file_id(&self, output: &str) -> Result<String> {
        // Look for file ID in output
        for line in output.lines() {
            if line.contains("File ID:") || line.contains("file_id") {
                let parts: Vec<&str> = line.split(':').collect();
                if parts.len() > 1 {
                    return Ok(parts[1].trim().to_string());
                }
            }
        }

        // If no explicit file ID, assume success and return a placeholder
        Ok(format!("upload_{}", uuid::Uuid::new_v4()))
    }

    fn parse_sol_balance(&self, output: &str) -> f64 {
        for line in output.lines() {
            if line.contains("SOL:") {
                let parts: Vec<&str> = line.split(':').collect();
                if parts.len() > 1 {
                    return parts[1].trim().parse().unwrap_or(0.0);
                }
            }
        }
        0.0
    }

    fn parse_pipe_balance(&self, output: &str) -> f64 {
        for line in output.lines() {
            if line.contains("Balance:") || line.contains("PIPE:") {
                let parts: Vec<&str> = line.split(':').collect();
                if parts.len() > 1 {
                    return parts[1].trim().parse().unwrap_or(0.0);
                }
            }
        }
        0.0
    }

    fn parse_wallet_address(&self, output: &str) -> String {
        for line in output.lines() {
            if line.contains("Pubkey:") {
                let parts: Vec<&str> = line.split(':').collect();
                if parts.len() > 1 {
                    return parts[1].trim().to_string();
                }
            }
        }
        String::new()
    }
}
