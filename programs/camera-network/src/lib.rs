use anchor_lang::prelude::*;

declare_id!("E67WTa1NpFVoapXwYYQmXzru3pyhaN9Kj3wPdZEyyZsL"); // Updated program ID

mod error;
mod state;
mod instructions;

use instructions::*;
use state::*;

#[program]
pub mod camera_network {
    use super::*;

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

    /// Check in user to a camera
    pub fn check_in(ctx: Context<CheckIn>, use_face_recognition: bool) -> Result<()> {
        instructions::check_in::handler(ctx, use_face_recognition)
    }

    /// Check out user from a camera with optional activity bundle
    pub fn check_out(ctx: Context<CheckOut>, activities: Vec<ActivityData>) -> Result<()> {
        instructions::check_out::handler(ctx, activities)
    }
    
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

    /// Record a camera activity (photo, video, stream)
    pub fn record_activity(ctx: Context<RecordActivity>, args: RecordActivityArgs) -> Result<()> {
        instructions::record_activity::handler(ctx, args)
    }
} 