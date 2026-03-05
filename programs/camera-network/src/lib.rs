use anchor_lang::prelude::*;

declare_id!("E67WTa1NpFVoapXwYYQmXzru3pyhaN9Kj3wPdZEyyZsL"); // Deployed program ID

mod error;
mod state;
mod instructions;

use instructions::*;
use state::*;
use light_sdk::{
    cpi::CpiSigner,
    derive_light_cpi_signer,
    instruction::{PackedAddressTreeInfo, ValidityProof},
};

pub const LIGHT_CPI_SIGNER: CpiSigner =
    derive_light_cpi_signer!("E67WTa1NpFVoapXwYYQmXzru3pyhaN9Kj3wPdZEyyZsL");

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
    // - At checkout, a compressed TimelineEntry is created (create_timeline_entry)
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

    /// Create a compressed timeline entry for a camera session
    /// Called at checkout — device signs, backend pays gas
    pub fn create_timeline_entry<'info>(
        ctx: Context<'_, '_, '_, 'info, CreateTimelineEntry<'info>>,
        proof: ValidityProof,
        address_tree_info: PackedAddressTreeInfo,
        output_merkle_tree_index: u8,
        encrypted_payload: Vec<u8>,
        nonce: [u8; 12],
        access_grants_blob: Vec<u8>,
        activity_count: u8,
    ) -> Result<()> {
        instructions::create_timeline_entry::handler(
            ctx, proof, address_tree_info, output_merkle_tree_index,
            encrypted_payload, nonce, access_grants_blob, activity_count,
        )
    }
} 