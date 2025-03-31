use anchor_lang::prelude::*;
use crate::state::*;
use crate::error::ErrorCode;

#[derive(AnchorSerialize, AnchorDeserialize)]
pub struct RegisterCameraArgs {
    pub name: String,
    pub model: String,
    pub location: Option<[i64; 2]>,
    pub fee: u64,
}

#[derive(Accounts)]
#[instruction(args: RegisterCameraArgs)]
pub struct RegisterCamera<'info> {
    #[account(mut)]
    pub owner: Signer<'info>,
    
    #[account(
        mut,
        seeds = [camera_registry_seeds()],
        bump = registry.bump
    )]
    pub registry: Account<'info, CameraRegistry>,
    
    #[account(
        init,
        payer = owner,
        space = 8 + 32 + 1 + 8 + 1 + 8 + 256, // Adjusted size for Camera account
        seeds = [b"camera", args.name.as_bytes(), owner.key().as_ref()],
        bump
    )]
    pub camera: Account<'info, CameraAccount>,
    
    pub system_program: Program<'info, System>,
}

pub fn handler(ctx: Context<RegisterCamera>, args: RegisterCameraArgs) -> Result<()> {
    let registry = &mut ctx.accounts.registry;
    let camera = &mut ctx.accounts.camera;
    let owner = &ctx.accounts.owner;
    
    // Ensure minimum fee is provided
    require!(args.fee >= 100, ErrorCode::InsufficientFee);
    
    // Initialize camera account
    camera.owner = *owner.key;
    camera.is_active = true;
    camera.activity_counter = 0;
    camera.last_activity_type = None;
    camera.bump = ctx.bumps.camera;
    
    // Initialize metadata
    camera.metadata.name = args.name.clone();
    camera.metadata.model = args.model;
    camera.metadata.location = args.location;
    camera.metadata.registration_date = Clock::get()?.unix_timestamp;
    camera.metadata.last_activity = Clock::get()?.unix_timestamp;
    
    // Update registry
    registry.camera_count += 1;
    
    msg!("Camera {} registered by {}", args.name, owner.key());
    
    Ok(())
}