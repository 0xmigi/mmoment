use anchor_lang::prelude::*;
use crate::state::{CameraAccount};
use crate::error::CameraNetworkError;

#[derive(Accounts)]
pub struct SetCameraActive<'info> {
    #[account(mut)]
    pub owner: Signer<'info>,
    
    #[account(
        mut,
        constraint = camera.owner == owner.key() @ CameraNetworkError::Unauthorized
    )]
    pub camera: Account<'info, CameraAccount>,
}

pub fn handler(ctx: Context<SetCameraActive>, is_active: bool) -> Result<()> {
    let camera = &mut ctx.accounts.camera;
    
    // Update activity status
    camera.is_active = is_active;
    
    // Update timestamp
    camera.last_activity_at = Clock::get()?.unix_timestamp;
    
    msg!("Camera {} is now {}", camera.key(), if is_active { "active" } else { "inactive" });
    
    Ok(())
} 