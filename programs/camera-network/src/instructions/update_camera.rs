use anchor_lang::prelude::*;
use crate::state::{
    CameraAccount, UpdateCameraArgs
};
use crate::error::CameraNetworkError;

#[derive(Accounts)]
pub struct UpdateCamera<'info> {
    #[account(mut)]
    pub owner: Signer<'info>,
    
    #[account(
        mut,
        constraint = camera.owner == owner.key() @ CameraNetworkError::Unauthorized
    )]
    pub camera: Account<'info, CameraAccount>,
}

pub fn handler(ctx: Context<UpdateCamera>, args: UpdateCameraArgs) -> Result<()> {
    let camera = &mut ctx.accounts.camera;
    
    // Update camera metadata if provided
    if let Some(name) = args.name {
        if !name.trim().is_empty() {
            camera.metadata.name = name;
        }
    }
    
    if let Some(location) = args.location {
        camera.metadata.location = Some(location);
    }
    
    if let Some(description) = args.description {
        if !description.trim().is_empty() {
            camera.metadata.description = description;
        }
    }
    
    // Update features if provided
    if let Some(features) = args.features {
        camera.features = features;
    }

    // Update timestamp
    camera.last_activity_at = Clock::get()?.unix_timestamp;
    
    msg!("Camera {} updated", camera.key());
    
    Ok(())
} 