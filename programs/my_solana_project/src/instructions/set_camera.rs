use anchor_lang::prelude::*;
use crate::state::*;
use crate::error::ErrorCode;

#[derive(AnchorSerialize, AnchorDeserialize)]
pub struct SetCameraActiveArgs {
    pub is_active: bool,
}

#[derive(Accounts)]
pub struct SetCameraActive<'info> {
    #[account(mut)]
    pub owner: Signer<'info>,
    
    #[account(
        mut,
        seeds = [b"camera", camera.camera_id.as_bytes(), owner.key().as_ref()],
        bump = camera.bump,
        constraint = camera.owner == owner.key() @ ErrorCode::Unauthorized
    )]
    pub camera: Account<'info, CameraAccount>,
}

pub fn handler(ctx: Context<SetCameraActive>, args: SetCameraActiveArgs) -> Result<()> {
    let camera = &mut ctx.accounts.camera;
    camera.is_active = args.is_active;
    camera.metadata.last_activity = Clock::get()?.unix_timestamp;
    
    let status = if args.is_active { "activated" } else { "deactivated" };
    msg!("Camera {} {} by owner {}", camera.camera_id, status, ctx.accounts.owner.key());
    
    Ok(())
}