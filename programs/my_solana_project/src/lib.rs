use anchor_lang::prelude::*;

declare_id!("7BYuxsNyaxxsxwzcRzFd6UJGnUctN6V1vDQxjGPaK2L4");

pub mod error;
pub mod instructions;
pub mod state;

pub use instructions::*;
pub use state::*;

#[program]
pub mod my_solana_project {
    use super::*;

    // Initialize the camera registry
    pub fn initialize(ctx: Context<Initialize>) -> Result<()> {
        instructions::initialize::handler(ctx)
    }

    // Register a new camera
    pub fn register_camera(ctx: Context<RegisterCamera>, args: RegisterCameraArgs) -> Result<()> {
        instructions::register_camera::handler(ctx, args)
    }

    // Update camera info
    pub fn update_camera(ctx: Context<UpdateCamera>, args: UpdateCameraArgs) -> Result<()> {
        instructions::update_camera::handler(ctx, args)
    }

    // Record camera activity (photo/video capture)
    pub fn record_activity(ctx: Context<RecordActivity>, args: RecordActivityArgs) -> Result<()> {
        instructions::record_activity::handler(ctx, args)
    }

    // Set camera active/inactive
    pub fn set_camera_active(ctx: Context<SetCameraActive>, args: SetCameraActiveArgs) -> Result<()> {
        instructions::set_camera_active::handler(ctx, args)
    }

    // Deregister a camera (remove it from the registry)
    pub fn deregister_camera(ctx: Context<DeregisterCamera>) -> Result<()> {
        instructions::deregister_camera::handler(ctx)
    }
}