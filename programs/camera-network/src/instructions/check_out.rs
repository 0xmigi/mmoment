use anchor_lang::prelude::*;
use crate::state::{
    CameraAccount, UserSession, CameraTimeline, ActivityData, EncryptedActivity,
    UserCheckedOut, ActivityType, ActivityRecorded
};
use crate::error::CameraNetworkError;

#[derive(Accounts)]
#[instruction(activities: Vec<ActivityData>)]
pub struct CheckOut<'info> {
    #[account(mut)]
    pub closer: Signer<'info>,  // User checking out themselves OR anyone if expired

    #[account(mut)]
    pub camera: Account<'info, CameraAccount>,

    /// Camera timeline - created lazily on first checkout with activities
    #[account(
        init_if_needed,
        payer = closer,
        space = 8 + 32 + 4 + 8 + 1 + 10240,  // Start with space for ~30 activities, can be reallocated
        seeds = [b"camera-timeline", camera.key().as_ref()],
        bump
    )]
    pub camera_timeline: Account<'info, CameraTimeline>,

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

    pub system_program: Program<'info, System>,
}

pub fn handler(ctx: Context<CheckOut>, activities: Vec<ActivityData>) -> Result<()> {
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

    // Initialize timeline if needed
    let timeline = &mut ctx.accounts.camera_timeline;
    if timeline.camera == Pubkey::default() {
        // First time initialization
        timeline.camera = camera.key();
        timeline.activity_count = 0;
        timeline.bump = ctx.bumps.camera_timeline;
    }

    // Process activities if any were provided
    // Activities arrive pre-encrypted from Jetson (which handles AES encryption + access grants)
    if !activities.is_empty() {
        msg!("Storing {} encrypted activities to timeline", activities.len());

        // Convert ActivityData (from Jetson) to EncryptedActivity (timeline storage)
        for activity in activities {
            let encrypted_activity = EncryptedActivity {
                timestamp: activity.timestamp,
                activity_type: activity.activity_type,
                encrypted_content: activity.encrypted_content,
                nonce: activity.nonce,
                access_grants: activity.access_grants,
            };

            timeline.encrypted_activities.push(encrypted_activity);
        }

        timeline.activity_count += timeline.encrypted_activities.len() as u64;

        msg!("Timeline now contains {} total activities", timeline.activity_count);
    }

    Ok(())
} 