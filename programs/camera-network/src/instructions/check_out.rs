use anchor_lang::prelude::*;
use crate::state::{
    CameraAccount, UserSession,
    UserCheckedOut, ActivityType, ActivityRecorded
};
use crate::error::CameraNetworkError;

#[derive(Accounts)]
pub struct CheckOut<'info> {
    #[account(mut)]
    pub closer: Signer<'info>,  // Can be anyone if session expired

    #[account(mut)]
    pub camera: Account<'info, CameraAccount>,

    #[account(
        mut,
        close = session_user,  // Rent goes to original user
        seeds = [
            b"session",
            session.user.as_ref(),
            camera.key().as_ref()
        ],
        bump = session.bump,
        constraint = session.camera == camera.key() @ CameraNetworkError::InvalidCameraData
    )]
    pub session: Account<'info, UserSession>,

    /// CHECK: Original session creator (for rent reclamation)
    #[account(mut)]
    pub session_user: UncheckedAccount<'info>,
}

pub fn handler(ctx: Context<CheckOut>) -> Result<()> {
    let closer = &ctx.accounts.closer;
    let camera = &mut ctx.accounts.camera;
    let session = &ctx.accounts.session;
    let now = Clock::get()?.unix_timestamp;

    // Check authorization: session owner OR expired session
    let is_owner = closer.key() == session.user;
    let is_expired = now > session.auto_checkout_at;

    require!(
        is_owner || is_expired,
        CameraNetworkError::Unauthorized
    );

    // Calculate session duration
    let session_duration = now.checked_sub(session.check_in_time)
        .ok_or(error!(CameraNetworkError::InvalidCameraData))?;

    // Update camera activity stats
    camera.activity_counter = camera.activity_counter.checked_add(1)
        .ok_or(error!(CameraNetworkError::InvalidCameraData))?;
    camera.last_activity_at = now;
    camera.last_activity_type = ActivityType::CheckOut as u8;

    // Emit events
    emit!(UserCheckedOut {
        user: session.user,
        camera: camera.key(),
        session: session.key(),
        duration: session_duration,
        timestamp: now,
    });

    emit!(ActivityRecorded {
        camera: camera.key(),
        user: session.user,
        activity_type: ActivityType::CheckOut as u8,
        timestamp: now,
    });

    if is_expired && !is_owner {
        msg!("Expired session cleaned up by {} for user {}", closer.key(), session.user);
    } else {
        msg!("User {} checked out from camera {} after {} seconds",
            session.user, camera.key(), session_duration);
    }

    Ok(())
} 