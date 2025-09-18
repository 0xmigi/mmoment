use anchor_lang::prelude::*;
use crate::state::{CameraRegistry, camera_registry_seeds};

#[derive(Accounts)]
pub struct Initialize<'info> {
    #[account(mut)]
    pub authority: Signer<'info>,
    
    #[account(
        init,
        payer = authority,
        space = 8 + 32 + 8 + 32 + 1, // Discriminator + authority + camera_count + fee_account + bump
        seeds = [camera_registry_seeds()],
        bump
    )]
    pub camera_registry: Account<'info, CameraRegistry>,
    
    pub system_program: Program<'info, System>,
}

pub fn handler(ctx: Context<Initialize>) -> Result<()> {
    let registry = &mut ctx.accounts.camera_registry;
    let authority = &ctx.accounts.authority;
    
    registry.authority = authority.key();
    registry.camera_count = 0;
    registry.fee_account = authority.key(); // Default fee receiver is the authority
    registry.bump = ctx.bumps.camera_registry;
    
    msg!("Camera registry initialized with authority: {}", authority.key());
    
    Ok(())
} 