use anchor_lang::prelude::*;
use crate::state::*;

#[derive(Accounts)]
pub struct Initialize<'info> {
    #[account(mut)]
    pub authority: Signer<'info>,
    
    #[account(
        init,
        payer = authority,
        space = 8 + 32 + 8 + 1, // 8 bytes for discriminator + 32 for pubkey + 8 for count + 1 for bump
        seeds = [camera_registry_seeds()],
        bump
    )]
    pub registry: Account<'info, CameraRegistry>,
    
    pub system_program: Program<'info, System>,
}

pub fn handler(ctx: Context<Initialize>) -> Result<()> {
    let registry = &mut ctx.accounts.registry;
    registry.authority = ctx.accounts.authority.key();
    registry.camera_count = 0;
    registry.bump = ctx.bumps.registry;
    
    msg!("Camera registry initialized with authority: {}", registry.authority);
    
    Ok(())
}