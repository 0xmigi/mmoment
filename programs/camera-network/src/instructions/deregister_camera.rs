use anchor_lang::prelude::*;
use crate::state::{
    CameraRegistry, CameraAccount, camera_registry_seeds
};
use crate::error::CameraNetworkError;

#[derive(Accounts)]
pub struct DeregisterCamera<'info> {
    #[account(mut)]
    pub owner: Signer<'info>,
    
    #[account(
        mut,
        seeds = [camera_registry_seeds()],
        bump = camera_registry.bump
    )]
    pub camera_registry: Account<'info, CameraRegistry>,
    
    #[account(
        mut,
        close = owner,
        constraint = camera.owner == owner.key() @ CameraNetworkError::Unauthorized
    )]
    pub camera: Account<'info, CameraAccount>,
}

pub fn handler(ctx: Context<DeregisterCamera>) -> Result<()> {
    let owner = &ctx.accounts.owner;
    let camera = &ctx.accounts.camera;
    let camera_registry = &mut ctx.accounts.camera_registry;
    
    // Decrement camera count in registry
    camera_registry.camera_count = camera_registry.camera_count.checked_sub(1)
        .ok_or(error!(CameraNetworkError::InvalidCameraData))?;
    
    msg!("Camera {} deregistered by owner {}", camera.key(), owner.key());
    
    Ok(())
} 