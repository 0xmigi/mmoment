use anchor_lang::prelude::*;
use crate::state::{
    CameraAccount, CameraTimeline, ActivityData, EncryptedActivity,
    ActivityType
};
use crate::error::CameraNetworkError;

/// Event emitted when activities are written to timeline
/// Note: No user info exposed - just camera-level stats
#[event]
pub struct TimelineUpdated {
    pub camera: Pubkey,
    pub activities_added: u64,
    pub total_activities: u64,
    pub timestamp: i64,
}

#[derive(Accounts)]
#[instruction(activities: Vec<ActivityData>)]
pub struct WriteToCameraTimeline<'info> {
    /// The signer - must be the camera's device key or owner
    #[account(mut)]
    pub signer: Signer<'info>,

    #[account(
        mut,
        constraint = (
            camera.device_pubkey == Some(signer.key()) ||
            camera.owner == signer.key()
        ) @ CameraNetworkError::Unauthorized
    )]
    pub camera: Account<'info, CameraAccount>,

    /// Camera timeline - created lazily on first write
    #[account(
        init_if_needed,
        payer = signer,
        space = 8 + 32 + 4 + 8 + 1 + 10187,  // 10240 bytes total
        seeds = [b"camera-timeline", camera.key().as_ref()],
        bump
    )]
    pub camera_timeline: Account<'info, CameraTimeline>,

    pub system_program: Program<'info, System>,
}

pub fn handler(ctx: Context<WriteToCameraTimeline>, activities: Vec<ActivityData>) -> Result<()> {
    let camera = &mut ctx.accounts.camera;
    let timeline = &mut ctx.accounts.camera_timeline;
    let now = Clock::get()?.unix_timestamp;

    // Validate we have activities to write
    require!(!activities.is_empty(), CameraNetworkError::InvalidCameraData);

    // Initialize timeline if needed (first write)
    if timeline.camera == Pubkey::default() {
        timeline.camera = camera.key();
        timeline.activity_count = 0;
        timeline.bump = ctx.bumps.camera_timeline;
    }

    let activities_count = activities.len() as u64;

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

    timeline.activity_count = timeline.activity_count.saturating_add(activities_count);

    // Update camera activity stats (anonymous - no user link)
    camera.activity_counter = camera.activity_counter.saturating_add(1);
    camera.last_activity_at = now;
    camera.last_activity_type = ActivityType::CheckOut as u8;

    // Emit event without user info
    emit!(TimelineUpdated {
        camera: camera.key(),
        activities_added: activities_count,
        total_activities: timeline.activity_count,
        timestamp: now,
    });

    msg!(
        "Wrote {} activities to camera {} timeline (total: {})",
        activities_count,
        camera.key(),
        timeline.activity_count
    );

    Ok(())
}
