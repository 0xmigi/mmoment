use anchor_lang::prelude::*;
use crate::state::*;
use crate::error::ErrorCode;

#[derive(AnchorSerialize, AnchorDeserialize)]
pub struct RecordActivityArgs {
    pub activity_type: ActivityType,
    pub metadata: String,
}

#[derive(Accounts)]
#[instruction(args: RecordActivityArgs)]
pub struct RecordActivity<'info> {
    #[account(mut)]
    pub owner: Signer<'info>,
    
    #[account(
        mut,
        constraint = camera.is_active @ ErrorCode::CameraInactive
    )]
    pub camera: Account<'info, CameraAccount>,
    
    pub system_program: Program<'info, System>,
}

pub fn handler(ctx: Context<RecordActivity>, args: RecordActivityArgs) -> Result<()> {
    let camera = &mut ctx.accounts.camera;
    let timestamp = Clock::get()?.unix_timestamp;
    
    // Increment activity counter
    camera.activity_counter += 1;
    
    // Update last activity type and timestamp
    camera.last_activity_type = Some(args.activity_type);
    camera.metadata.last_activity = timestamp;
    
    // Get activity name for logging
    let activity_name = match args.activity_type {
        ActivityType::PhotoCapture => "photo capture",
        ActivityType::VideoRecord => "video recording",
        ActivityType::LiveStream => "live stream",
        ActivityType::Custom => "custom activity",
    };
    
    // Emit event with activity details
    emit!(ActivityRecorded {
        camera: camera.key(),
        name: camera.metadata.name.clone(),
        activity_number: camera.activity_counter,
        activity_type: args.activity_type,
        timestamp,
        metadata: args.metadata.clone(),
    });
    
    msg!("Recorded {} activity #{} for camera {}", 
         activity_name, 
         camera.activity_counter,
         camera.metadata.name);
    
    Ok(())
}