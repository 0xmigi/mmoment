use anchor_lang::prelude::*;
use crate::state::{
    CameraAccount, UserSession,
    UserCheckedOut, ActivityType, ActivityRecorded
};
use crate::error::CameraNetworkError;

#[derive(Accounts)]
pub struct CheckOut<'info> {
    #[account(mut)]
    pub closer: Signer<'info>,  // User checking out themselves OR anyone if expired

    #[account(mut)]
    pub camera: Account<'info, CameraAccount>,

    #[account(
        mut,
        close = rent_destination,  // Rent goes to designated recipient
        seeds = [
            b"session",
            session.user.as_ref(),
            camera.key().as_ref()
        ],
        bump = session.bump,
        constraint = session.camera == camera.key() @ CameraNetworkError::InvalidCameraData
    )]
    pub session: Account<'info, UserSession>,

    /// CHECK: Original session creator (for self-checkout)
    #[account(mut)]
    pub session_user: UncheckedAccount<'info>,

    /// CHECK: Destination for rent reclamation (user if self-checkout, closer if expired cleanup)
    #[account(mut)]
    pub rent_destination: UncheckedAccount<'info>,
}

pub fn handler(ctx: Context<CheckOut>) -> Result<()> {
    let closer = &ctx.accounts.closer;
    let camera = &mut ctx.accounts.camera;
    let session = &ctx.accounts.session;
    let rent_destination = &ctx.accounts.rent_destination;
    let session_user = &ctx.accounts.session_user;
    let now = Clock::get()?.unix_timestamp;

    // Check authorization
    let is_user_checkout = closer.key() == session.user;
    let is_expired = now > session.auto_checkout_at;

    // Allow checkout if:
    // 1. User is checking out themselves, OR
    // 2. Session is expired (anyone can cleanup for rent reward)
    require!(
        is_user_checkout || is_expired,
        CameraNetworkError::Unauthorized
    );

    // Validate rent destination:
    // - If user self-checkout: rent must go to user
    // - If expired cleanup: rent goes to closer (incentive for running cleanup bots)
    if is_user_checkout {
        require!(
            rent_destination.key() == session_user.key(),
            CameraNetworkError::Unauthorized
        );
    } else if is_expired {
        require!(
            rent_destination.key() == closer.key(),
            CameraNetworkError::Unauthorized
        );
    }

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

    if is_expired && !is_user_checkout {
        msg!("Expired session cleaned up by {} for user {} - rent collected as reward", closer.key(), session.user);
    } else {
        msg!("User {} checked out from camera {} after {} seconds",
            session.user, camera.key(), session_duration);
    }

    Ok(())
} 