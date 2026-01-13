use anchor_lang::prelude::*;

declare_id!("E67WTa1NpFVoapXwYYQmXzru3pyhaN9Kj3wPdZEyyZsL"); // Deployed program ID

mod error;
mod state;
mod instructions;

use instructions::*;
use state::*;

#[program]
pub mod camera_network {
    use super::*;

    // ==================== Camera Management ====================

    /// Initialize the camera registry (admin only)
    pub fn initialize(ctx: Context<Initialize>) -> Result<()> {
        instructions::initialize::handler(ctx)
    }

    /// Register a new camera to the network
    pub fn register_camera(ctx: Context<RegisterCamera>, args: RegisterCameraArgs) -> Result<()> {
        instructions::register_camera::handler(ctx, args)
    }

    /// Update camera information
    pub fn update_camera(ctx: Context<UpdateCamera>, args: UpdateCameraArgs) -> Result<()> {
        instructions::update_camera::handler(ctx, args)
    }

    /// Deregister a camera from the network
    pub fn deregister_camera(ctx: Context<DeregisterCamera>) -> Result<()> {
        instructions::deregister_camera::handler(ctx)
    }

    /// Set camera active or inactive status
    pub fn set_camera_active(ctx: Context<SetCameraActive>, is_active: bool) -> Result<()> {
        instructions::set_camera_active::handler(ctx, is_active)
    }

    // ==================== Recognition Tokens ====================

    /// Create or regenerate a recognition token (stores encrypted facial embedding)
    pub fn upsert_recognition_token(
        ctx: Context<UpsertRecognitionToken>,
        encrypted_embedding: Vec<u8>,
        display_name: Option<String>,
        source: u8,
    ) -> Result<()> {
        instructions::enroll_face::handler(ctx, encrypted_embedding, display_name, source)
    }

    /// Delete recognition token and reclaim rent
    pub fn delete_recognition_token(ctx: Context<DeleteRecognitionToken>) -> Result<()> {
        instructions::delete_recognition_token::handler(ctx)
    }

    // ==================== Privacy-Preserving Session Management ====================
    //
    // New architecture:
    // - Check-in is OFF-CHAIN (ed25519 handshake with Jetson)
    // - Jetson writes encrypted activities to CameraTimeline (write_to_camera_timeline)
    // - User stores access keys in their UserSessionChain (store_session_access_keys)
    // - No on-chain link between user and camera

    /// Create a user's session chain for storing encrypted access keys
    /// This is the user's "keychain" for accessing their session history
    pub fn create_user_session_chain(ctx: Context<CreateUserSessionChain>) -> Result<()> {
        instructions::create_user_session_chain::handler(ctx)
    }

    /// Store encrypted session access keys in a user's chain
    /// Can be called by the user OR the mmoment authority (cron bot fallback)
    pub fn store_session_access_keys(
        ctx: Context<StoreSessionAccessKeys>,
        keys: Vec<EncryptedSessionKey>,
    ) -> Result<()> {
        instructions::store_session_access_keys::handler(ctx, keys)
    }

    /// Write encrypted activities to a camera's timeline
    /// Called by Jetson (device key) or camera owner - NO user account involved
    pub fn write_to_camera_timeline(
        ctx: Context<WriteToCameraTimeline>,
        activities: Vec<ActivityData>,
    ) -> Result<()> {
        instructions::write_to_camera_timeline::handler(ctx, activities)
    }
} 