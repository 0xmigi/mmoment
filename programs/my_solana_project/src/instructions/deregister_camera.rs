use anchor_lang::prelude::*;
use crate::state::*;
use crate::error::ErrorCode;

#[derive(Accounts)]
pub struct DeregisterCamera<'info> {
    #[account(mut)]
    pub owner: Signer<'info>,
    
    #[account(
        mut,
        seeds = [camera_registry_seeds()],
        bump = registry.bump
    )]
    pub registry: Account<'info, CameraRegistry>,
    
    #[account(
        mut,
        seeds = [b"camera", camera.metadata.name.as_bytes(), owner.key().as_ref()],
        bump = camera.bump,
        constraint = camera.owner == owner.key() @ ErrorCode::Unauthorized,
        close = owner
    )]
    pub camera: Account<'info, CameraAccount>,
    
    pub system_program: Program<'info, System>,
}

pub fn handler(ctx: Context<DeregisterCamera>) -> Result<()> {
    let registry = &mut ctx.accounts.registry;
    let camera = &ctx.accounts.camera;
    
    // Update the camera count in the registry
    if registry.camera_count > 0 {
        registry.camera_count -= 1;
    }
    
    msg!("Camera {} deregistered by owner {}", camera.metadata.name, ctx.accounts.owner.key());
    
    // The camera account will be closed automatically due to the `close = owner` directive
    // This will return the rent-exempt SOL to the owner
    
    Ok(())
}
