//! Core Pipe API client

use crate::error::{PipeError, Result};
use crate::types::{User, WalletBalance, TokenBalance};
use reqwest::{Client as HttpClient, multipart};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Duration;

const DEFAULT_BASE_URL: &str = "https://us-west-00-firestarter.pipenetwork.com";
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Core client for Pipe Network API
pub struct PipeClient {
    http: Arc<HttpClient>,
    base_url: String,
}

impl PipeClient {
    /// Create a new Pipe client
    pub fn new(base_url: Option<&str>) -> Self {
        let http = HttpClient::builder()
            .timeout(DEFAULT_TIMEOUT)
            .build()
            .expect("Failed to create HTTP client");

        Self {
            http: Arc::new(http),
            base_url: base_url.unwrap_or(DEFAULT_BASE_URL).to_string(),
        }
    }

    /// Create a new user account
    pub async fn create_user(&self, username: &str) -> Result<User> {
        #[derive(Serialize)]
        struct CreateUserRequest {
            username: String,
        }

        #[derive(Deserialize)]
        struct CreateUserResponse {
            user_id: String,
            user_app_key: String,
            solana_pubkey: String,
        }

        let response = self.http
            .post(format!("{}/createUser", self.base_url))
            .json(&CreateUserRequest {
                username: username.to_string(),
            })
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let message = response.text().await.unwrap_or_default();
            return Err(PipeError::Api { message, status });
        }

        let resp: CreateUserResponse = response.json().await?;

        Ok(User {
            user_id: resp.user_id,
            user_app_key: resp.user_app_key,
            solana_pubkey: Some(resp.solana_pubkey),
            username: Some(username.to_string()),
        })
    }

    /// Upload a file for a user
    pub async fn upload(
        &self,
        user: &User,
        data: Vec<u8>,
        filename: &str,
        priority: bool,
    ) -> Result<String> {
        let endpoint = if priority { "/priorityUpload" } else { "/upload" };

        let form = multipart::Form::new()
            .part("file", multipart::Part::bytes(data)
                .file_name(filename.to_string()));

        let response = self.http
            .post(format!("{}{}", self.base_url, endpoint))
            .header("X-User-Id", &user.user_id)
            .header("X-User-App-Key", &user.user_app_key)
            .query(&[("file_name", filename)])
            .multipart(form)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let message = response.text().await.unwrap_or_default();

            // Check for insufficient tokens
            if message.contains("insufficient") || message.contains("tokens") {
                // Try to parse the amounts
                return Err(PipeError::InsufficientTokens {
                    required: 0.0,  // Would parse from message
                    available: 0.0,
                });
            }

            return Err(PipeError::Api { message, status });
        }

        // The API returns the filename as a plain string
        let filename = response.text().await?;
        Ok(filename)
    }

    /// Download a file for a user
    pub async fn download(
        &self,
        user: &User,
        filename: &str,
        priority: bool,
    ) -> Result<Vec<u8>> {
        let endpoint = if priority { "/priorityDownload" } else { "/download" };

        let response = self.http
            .get(format!("{}{}", self.base_url, endpoint))
            .query(&[
                ("user_id", user.user_id.as_str()),
                ("user_app_key", user.user_app_key.as_str()),
                ("file_name", filename),
            ])
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let message = response.text().await.unwrap_or_default();

            if status == 404 {
                return Err(PipeError::FileNotFound(filename.to_string()));
            }

            return Err(PipeError::Api { message, status });
        }

        // Priority download returns base64, regular returns raw bytes
        if priority {
            let base64_data = response.text().await?;
            base64::decode(&base64_data)
                .map_err(|e| PipeError::InvalidResponse(e.to_string()))
        } else {
            Ok(response.bytes().await?.to_vec())
        }
    }

    /// Delete a file
    pub async fn delete_file(&self, user: &User, filename: &str) -> Result<()> {
        #[derive(Serialize)]
        struct DeleteRequest {
            user_id: String,
            user_app_key: String,
            file_name: String,
        }

        let response = self.http
            .post(format!("{}/deleteFile", self.base_url))
            .json(&DeleteRequest {
                user_id: user.user_id.clone(),
                user_app_key: user.user_app_key.clone(),
                file_name: filename.to_string(),
            })
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let message = response.text().await.unwrap_or_default();
            return Err(PipeError::Api { message, status });
        }

        Ok(())
    }

    /// Create a public link for a file
    pub async fn create_public_link(&self, user: &User, filename: &str) -> Result<String> {
        #[derive(Serialize)]
        struct PublicLinkRequest {
            user_id: String,
            user_app_key: String,
            file_name: String,
        }

        #[derive(Deserialize)]
        struct PublicLinkResponse {
            link_hash: String,
        }

        let response = self.http
            .post(format!("{}/createPublicLink", self.base_url))
            .json(&PublicLinkRequest {
                user_id: user.user_id.clone(),
                user_app_key: user.user_app_key.clone(),
                file_name: filename.to_string(),
            })
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let message = response.text().await.unwrap_or_default();
            return Err(PipeError::Api { message, status });
        }

        let resp: PublicLinkResponse = response.json().await?;
        Ok(format!("{}/publicDownload?hash={}", self.base_url, resp.link_hash))
    }

    /// Download a file using a public link
    pub async fn public_download(&self, hash: &str) -> Result<Vec<u8>> {
        let response = self.http
            .get(format!("{}/publicDownload", self.base_url))
            .query(&[("hash", hash)])
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let message = response.text().await.unwrap_or_default();
            return Err(PipeError::Api { message, status });
        }

        Ok(response.bytes().await?.to_vec())
    }

    /// Check SOL balance
    pub async fn check_sol_balance(&self, user: &User) -> Result<WalletBalance> {
        let response = self.http
            .post(format!("{}/checkWallet", self.base_url))
            .header("X-User-Id", &user.user_id)
            .header("X-User-App-Key", &user.user_app_key)
            .json(&serde_json::json!({}))
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let message = response.text().await.unwrap_or_default();
            return Err(PipeError::Api { message, status });
        }

        Ok(response.json().await?)
    }

    /// Check PIPE token balance
    pub async fn check_pipe_balance(&self, user: &User) -> Result<TokenBalance> {
        #[derive(Serialize)]
        struct TokenRequest {
            user_id: String,
            user_app_key: String,
        }

        let response = self.http
            .post(format!("{}/getCustomTokenBalance", self.base_url))
            .json(&TokenRequest {
                user_id: user.user_id.clone(),
                user_app_key: user.user_app_key.clone(),
            })
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let message = response.text().await.unwrap_or_default();
            return Err(PipeError::Api { message, status });
        }

        Ok(response.json().await?)
    }

    /// Swap SOL for PIPE tokens
    pub async fn swap_sol_for_pipe(&self, user: &User, amount_sol: f64) -> Result<f64> {
        #[derive(Serialize)]
        struct SwapRequest {
            user_id: String,
            user_app_key: String,
            amount_sol: f64,
        }

        #[derive(Deserialize)]
        struct SwapResponse {
            #[allow(dead_code)]
            user_id: String,
            #[allow(dead_code)]
            sol_spent: f64,
            tokens_minted: f64,
        }

        let response = self.http
            .post(format!("{}/swapSolForPipe", self.base_url))
            .json(&SwapRequest {
                user_id: user.user_id.clone(),
                user_app_key: user.user_app_key.clone(),
                amount_sol,
            })
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let message = response.text().await.unwrap_or_default();
            return Err(PipeError::Api { message, status });
        }

        let resp: SwapResponse = response.json().await?;
        Ok(resp.tokens_minted)
    }
}
