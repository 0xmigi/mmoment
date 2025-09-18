use anchor_lang::prelude::*;
use crate::state::{
    CameraAccount, UserSession,
    UserCheckedOut, ActivityType, ActivityRecorded
};
use crate::error::CameraNetworkError;

#[derive(Accounts)]
pub struct CheckOut<'info> {
    #[account(mut)]
    pub user: Signer<'info>,
    
    #[account(mut)]
    pub camera: Account<'info, CameraAccount>,
    
    #[account(
        mut,
        close = user,
        seeds = [
            b"session",
            user.key().as_ref(),
            camera.key().as_ref()
        ],
        bump = session.bump,
        constraint = session.user == user.key() @ CameraNetworkError::Unauthorized,
        constraint = session.camera == camera.key() @ CameraNetworkError::InvalidCameraData
    )]
    pub session: Account<'info, UserSession>,
}

pub fn handler(ctx: Context<CheckOut>) -> Result<()> {
    let user = &ctx.accounts.user;
    let camera = &mut ctx.accounts.camera;
    let session = &ctx.accounts.session;
    
    // Calculate session duration
    let now = Clock::get()?.unix_timestamp;
    let session_duration = now.checked_sub(session.check_in_time)
        .ok_or(error!(CameraNetworkError::InvalidCameraData))?;
    
    // Update camera activity stats
    camera.activity_counter = camera.activity_counter.checked_add(1)
        .ok_or(error!(CameraNetworkError::InvalidCameraData))?;
    camera.last_activity_at = now;
    camera.last_activity_type = ActivityType::CheckOut as u8;
    
    // Emit events
    emit!(UserCheckedOut {
        user: user.key(),
        camera: camera.key(),
        session: session.key(),
        duration: session_duration,
        timestamp: now,
    });
    
    emit!(ActivityRecorded {
        camera: camera.key(),
        user: user.key(),
        activity_type: ActivityType::CheckOut as u8,
        timestamp: now,
    });
    
    msg!("User {} checked out from camera {} after {} seconds", 
        user.key(), camera.key(), session_duration);
    
    Ok(())
} 