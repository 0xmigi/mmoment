use anchor_lang::prelude::*;
use crate::state::{
    CameraRegistry, CameraAccount, CameraMetadata, RegisterCameraArgs,
    camera_registry_seeds, CameraRegistered
};
use crate::error::CameraNetworkError;

#[derive(Accounts)]
#[instruction(args: RegisterCameraArgs)]
pub struct RegisterCamera<'info> {
    #[account(mut)]
    pub owner: Signer<'info>,
    
    #[account(
        mut,
        seeds = [camera_registry_seeds()],
        bump = camera_registry.bump
    )]
    pub camera_registry: Account<'info, CameraRegistry>,
    
    #[account(
        init,
        payer = owner,
        space = 8 + 32 + 200 + 1 + 8 + 8 + 1 + 8 + 5 + 1, // Discriminator + owner + metadata + is_active + counters + features + bump
        seeds = [
            b"camera",
            args.name.as_bytes(),
            owner.key().as_ref()
        ],
        bump
    )]
    pub camera: Account<'info, CameraAccount>,
    
    pub system_program: Program<'info, System>,
}

pub fn handler(ctx: Context<RegisterCamera>, args: RegisterCameraArgs) -> Result<()> {
    // Get accounts
    let camera = &mut ctx.accounts.camera;
    let owner = &ctx.accounts.owner;
    let camera_registry = &mut ctx.accounts.camera_registry;
    
    // Validate inputs
    if args.name.trim().is_empty() || args.model.trim().is_empty() {
        return err!(CameraNetworkError::InvalidCameraData);
    }
    
    // Set camera data
    camera.owner = owner.key();
    camera.is_active = true;
    camera.activity_counter = 0;
    camera.last_activity_at = Clock::get()?.unix_timestamp;
    camera.last_activity_type = 0; // Default activity type
    camera.access_count = 0;
    camera.features = args.features;
    camera.bump = ctx.bumps.camera;
    
    // Set metadata
    camera.metadata = CameraMetadata {
        name: args.name.clone(),
        model: args.model,
        location: args.location,
        registration_date: Clock::get()?.unix_timestamp,
        description: args.description,
    };
    
    // Increment camera count in registry
    camera_registry.camera_count = camera_registry.camera_count.checked_add(1)
        .ok_or(error!(CameraNetworkError::InvalidCameraData))?;
    
    // Emit event
    emit!(CameraRegistered {
        camera: camera.key(),
        owner: owner.key(),
        name: args.name,
        model: camera.metadata.model.clone(),
        timestamp: Clock::get()?.unix_timestamp,
    });
    
    msg!("Camera registered: {}", camera.key());
    
    Ok(())
} 