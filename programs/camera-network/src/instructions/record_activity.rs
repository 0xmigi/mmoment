use anchor_lang::prelude::*;
use crate::state::{
    CameraAccount, UserSession, ActivityType, ActivityRecorded
};
use crate::error::CameraNetworkError;

// Define the activity type enum for the instruction
#[derive(AnchorSerialize, AnchorDeserialize, Clone, Debug)]
pub enum CameraActionType {
    PhotoCapture,
    VideoRecord,
    StreamStart,
    StreamStop,
    Custom,
}

// Define the struct for activity arguments
#[derive(AnchorSerialize, AnchorDeserialize, Clone)]
pub struct RecordActivityArgs {
    pub action_type: CameraActionType,
    pub metadata: String, // Can include IPFS hash or other metadata
}

#[derive(Accounts)]
pub struct RecordActivity<'info> {
    #[account(mut)]
    pub user: Signer<'info>,
    
    #[account(
        mut,
        constraint = camera.is_active @ CameraNetworkError::CameraInactive
    )]
    pub camera: Account<'info, CameraAccount>,
    
    // Require that the user is currently checked in
    #[account(
        constraint = session.user == user.key() @ CameraNetworkError::Unauthorized,
        constraint = session.camera == camera.key() @ CameraNetworkError::NoActiveSession,
    )]
    pub session: Account<'info, UserSession>,
}

pub fn handler(ctx: Context<RecordActivity>, args: RecordActivityArgs) -> Result<()> {
    let camera = &mut ctx.accounts.camera;
    let user = &ctx.accounts.user;
    let session = &ctx.accounts.session;
    
    // Map the activity type
    let activity_type = match args.action_type {
        CameraActionType::PhotoCapture => ActivityType::PhotoCapture,
        CameraActionType::VideoRecord => ActivityType::VideoRecord,
        CameraActionType::StreamStart | CameraActionType::StreamStop => ActivityType::LiveStream,
        CameraActionType::Custom => ActivityType::Other,
    };
    
    // Update camera activity stats
    camera.activity_counter = camera.activity_counter.checked_add(1)
        .ok_or(error!(CameraNetworkError::InvalidCameraData))?;
    camera.last_activity_at = Clock::get()?.unix_timestamp;
    camera.last_activity_type = activity_type as u8;
    
    // Update session last activity timestamp
    // We don't need to update the session account since it's not marked as mut
    // but this would be the code if we wanted to:
    // session.last_activity = Clock::get()?.unix_timestamp;
    
    // Emit an event for indexing
    emit!(ActivityRecorded {
        camera: camera.key(),
        user: user.key(),
        activity_type: activity_type as u8,
        timestamp: Clock::get()?.unix_timestamp,
    });
    
    msg!("Activity recorded: {:?}", args.action_type);
    
    Ok(())
} 