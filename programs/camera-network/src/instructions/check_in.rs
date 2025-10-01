use anchor_lang::prelude::*;
use crate::state::{
    CameraAccount, UserSession, SessionFeatures, RecognitionToken,
    UserCheckedIn, ActivityType, ActivityRecorded
};
use crate::error::CameraNetworkError;

#[derive(Accounts)]
pub struct CheckIn<'info> {
    #[account(mut)]
    pub user: Signer<'info>,

    #[account(
        mut,
        constraint = camera.is_active @ CameraNetworkError::CameraInactive
    )]
    pub camera: Account<'info, CameraAccount>,

    /// Optional recognition token - required if use_face_recognition is true
    pub recognition_token: Option<Account<'info, RecognitionToken>>,

    #[account(
        init,
        payer = user,
        space = 8 + 32 + 32 + 8 + 8 + 8 + 5 + 1, // Added auto_checkout_at = 102 bytes
        seeds = [
            b"session",
            user.key().as_ref(),
            camera.key().as_ref()
        ],
        bump
    )]
    pub session: Account<'info, UserSession>,

    pub system_program: Program<'info, System>,
}

pub fn handler(ctx: Context<CheckIn>, use_face_recognition: bool) -> Result<()> {
    let user = &ctx.accounts.user;
    let camera = &mut ctx.accounts.camera;
    let session = &mut ctx.accounts.session;
    let now = Clock::get()?.unix_timestamp;

    // Validate recognition token if face recognition requested
    if use_face_recognition {
        // Check camera supports face recognition
        require!(
            camera.features.face_recognition,
            CameraNetworkError::FeatureNotAvailable
        );

        // Check user has recognition token
        let token = ctx.accounts.recognition_token.as_ref()
            .ok_or(CameraNetworkError::NoRecognitionToken)?;

        // Validate token belongs to user
        require!(
            token.user == user.key(),
            CameraNetworkError::Unauthorized
        );
    }

    // Initialize session data
    session.user = user.key();
    session.camera = camera.key();
    session.check_in_time = now;
    session.last_activity = now;
    session.auto_checkout_at = now + 7200; // 2 hour fallback timeout
    session.bump = ctx.bumps.session;

    // Configure features based on user choice
    // Only face_recognition and gesture_control are optional and tied together
    // All other features are always enabled by default
    if use_face_recognition && camera.features.face_recognition {
        // Enable facial recognition and gesture control if the camera supports it and user requests it
        session.enabled_features = SessionFeatures {
            face_recognition: true,
            gesture_control: true,
            // These are always available after check-in
            video_recording: true,
            live_streaming: true,
            messaging: false, // Removed messaging functionality
        };
    } else {
        // Basic features without facial recognition
        session.enabled_features = SessionFeatures {
            face_recognition: false,
            gesture_control: false,
            // These are always available after check-in
            video_recording: true,
            live_streaming: true,
            messaging: false, // Removed messaging functionality
        };
    }
    
    // Update camera activity stats
    camera.activity_counter = camera.activity_counter.checked_add(1)
        .ok_or(error!(CameraNetworkError::InvalidCameraData))?;
    camera.access_count = camera.access_count.checked_add(1)
        .ok_or(error!(CameraNetworkError::InvalidCameraData))?;
    camera.last_activity_at = Clock::get()?.unix_timestamp;
    camera.last_activity_type = ActivityType::CheckIn as u8;
    
    // Emit events
    emit!(UserCheckedIn {
        user: user.key(),
        camera: camera.key(),
        session: session.key(),
        timestamp: Clock::get()?.unix_timestamp,
    });
    
    emit!(ActivityRecorded {
        camera: camera.key(),
        user: user.key(),
        activity_type: ActivityType::CheckIn as u8,
        timestamp: Clock::get()?.unix_timestamp,
    });
    
    msg!("User {} checked in to camera {}", user.key(), camera.key());
    if use_face_recognition {
        msg!("Face recognition and gesture control enabled for this session");
    }
    
    Ok(())
} 