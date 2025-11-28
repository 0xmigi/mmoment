use anchor_lang::prelude::*;
use crate::state::UserSessionChain;

#[derive(Accounts)]
pub struct CreateUserSessionChain<'info> {
    #[account(mut)]
    pub user: Signer<'info>,

    /// The mmoment authority (cron bot) that can also write to this chain
    /// CHECK: This is validated as a trusted authority address
    pub authority: UncheckedAccount<'info>,

    #[account(
        init,
        payer = user,
        space = 8 + 32 + 32 + 4 + 8 + 1, // 85 bytes initial (empty vec)
        seeds = [b"user-session-chain", user.key().as_ref()],
        bump
    )]
    pub user_session_chain: Account<'info, UserSessionChain>,

    pub system_program: Program<'info, System>,
}

pub fn handler(ctx: Context<CreateUserSessionChain>) -> Result<()> {
    let user = &ctx.accounts.user;
    let authority = &ctx.accounts.authority;
    let chain = &mut ctx.accounts.user_session_chain;

    chain.user = user.key();
    chain.authority = authority.key();
    chain.encrypted_keys = Vec::new();
    chain.session_count = 0;
    chain.bump = ctx.bumps.user_session_chain;

    msg!("Created user session chain for user {}", user.key());

    Ok(())
}
